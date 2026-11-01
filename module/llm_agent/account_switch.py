"""切换游戏账号（MCP 工具 switch_account 的流程主体）。

背景：手机菜单上有「跃迁」这个危险词，护栏禁止此时 click_grid；退出游戏的电源图标又没有文字，只能按格子点，所以模型自己切不了号。
护栏不放宽，改成一个专用工具：模型只说切到哪个号，这里按固定流程走，每一步先确认画面对了再点，出意外就停下来报告。
不走 M7A 自带的注册表多账号功能（需要重启游戏、每个号先导出一次、登录过期会卡在输密码）。

流程（坐标是 1920×1080 截图像素，实际一律用 OCR / 模板定位，不硬编码坐标）：
 1. 主界面 → 手机菜单（M7A 的界面跳转图）→ 点右下角电源图标 → 弹窗「是否返回登录页面？」
 2. 点「确认」→（警告页、加载，约 20 秒）→ 登录页（中间「开始游戏」，右侧竖排「公告 / 设置 / 登出 / 退出」）
 3. 点「登出」→ 弹窗「退出登录」：先确认选中的是「退出并保留登录记录」再点「退出」（选错了下次要重新输密码）
 4. miHoYo 登录框：读下拉框里当前号码，不是目标就展开列表、点目标号码那一行，再点「进入游戏」
 5. 回登录页点「开始游戏」→「点击进入」→ 主界面；读左下角 UID（AccountTracker）确认就是目标账号

安全（写在代码里，tests/test_module/test_llm_agent_account_switch.py 有测试）：
- 绝不点：「退出并清除登录记录」、下拉列表里每行右边的 ⊗（删除登录记录）、登录页右侧的「退出」「更新」、「登录其他账号」。
  每次点击前都要过 _guard_click：点位落在这些文字上就拒绝；落在屏幕右侧竖排按钮区只有电源图标和「登出」允许；
  下拉列表的点击横坐标限制在左半边（⊗ 在右边）。各步查找的区域也都限定在对应的弹窗范围里（例如弹窗里的「退出」按钮区域不含登录页右侧的「退出」）。
- 任何一步在超时内没等到预期画面就立刻停，不乱点补救；整个流程最多 4 分钟。
- 只有目标账号在 accounts.json 里登记了 login（登录框下拉里显示的打码手机号），才会尝试切换。

识别函数（find_* / classify）只依赖一个 Observation（图片 + 元素），不发任何输入，所以可以拿截图离线检查。
"""
import functools
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from module.llm_agent import mcp_tools as mt
from module.llm_agent.actions import ActionResult
from module.llm_agent.errors import reraise_passthrough
from module.llm_agent.observe import ROOT, Element, Observation

# ---------------------------------------------------------------- 常量 ----

TOOL_NAME = "switch_account"
# 整个流程的总时长上限（秒）
TOTAL_TIMEOUT = 240.0
# 各步等待预期画面的超时（秒）；每步实际超时 = min(这里的值, 总时长剩余)
WAIT_MENU = 10.0
WAIT_DIALOG = 10.0
WAIT_LOGIN_PAGE = 60.0  # 确认后要经过警告页和加载，约 20 秒
WAIT_LOGIN_BOX = 15.0
WAIT_DROPDOWN = 8.0
WAIT_AFTER_ENTER = 30.0
WAIT_CLICK_ENTER = 90.0  # 开始游戏后加载，约 15–30 秒
WAIT_MAIN = 90.0
POLL_SECONDS = 1.0
# 下拉列表展开、登录框切换号码有动画：连续这么多次观察的结果一样才算稳定，两次观察之间隔 STABLE_GAP 秒
STABLE_LOOKS = 2
STABLE_GAP = 1.0
SLOW_POLL_SECONDS = 2.0
# 回到主界面后读 UID 的重试次数和间隔（秒）
UID_CONFIRM_TRIES = 4
UID_CONFIRM_GAP = 2.0
UID_CONFIRM_LOOKS = 2  # 连续新帧都唯一识别到目标 UID 才确认，不能拿 tracker 的历史缓存当证据
# 点了「保留登录记录」那一行之后等单选状态刷新的时间（秒）
RADIO_SETTLE = 0.5

# 识别区域（相对整张截图的比例：左、上、右、下），按 1920×1080 实机截图 a1–a15 量出来
REGION_POWER = (0.90, 0.80, 1.00, 1.00)  # 手机菜单右下角的电源图标
REGION_DIALOG_TITLE = (0.30, 0.30, 0.70, 0.55)
REGION_DIALOG_CONFIRM = (0.50, 0.55, 0.75, 0.72)  # 「是否返回登录页面」的确认按钮
REGION_DIALOG_CANCEL = (0.25, 0.55, 0.50, 0.72)
REGION_LOGIN_START = (0.35, 0.65, 0.65, 0.90)  # 登录页中间的「开始游戏」
REGION_LOGIN_LOGOUT = (0.90, 0.10, 1.00, 0.50)  # 登录页右侧竖排按钮里的「登出」
REGION_LOGOUT_TITLE = (0.40, 0.30, 0.60, 0.45)  # 「退出登录」弹窗
REGION_LOGOUT_CLEAR = (0.30, 0.38, 0.70, 0.52)
REGION_LOGOUT_KEEP = (0.30, 0.50, 0.70, 0.62)
REGION_LOGOUT_CANCEL = (0.30, 0.58, 0.50, 0.75)
REGION_LOGOUT_EXIT = (0.50, 0.58, 0.70, 0.75)  # 弹窗里的「退出」按钮，不含右侧竖排的「退出」
REGION_BOX_ENTER = (0.40, 0.50, 0.60, 0.66)  # 登录框的「进入游戏」按钮
REGION_BOX_HEADER = (0.35, 0.40, 0.65, 0.49)  # 登录框下拉框里当前显示的号码
REGION_LIST_ROWS = (0.30, 0.49, 0.65, 0.80)  # 展开后的各行号码
REGION_CLICK_ENTER = (0.30, 0.88, 0.70, 1.00)
# 右侧竖排按钮区（登出 / 退出 / 更新…）：x 超过这个比例的点击只有电源图标和「登出」允许
RIGHT_STRIP_X = 0.80
# 下拉箭头的横向位置（比例）；纵向跟随当前号码那一行
DROPDOWN_ARROW_X = 0.624
# 下拉列表行的点击横坐标上限（比例）：每行右边 x≈0.62 处是 ⊗ 删除登录记录
LIST_CLICK_X_MAX = 0.55
# 「登出」图标在文字上方，点图标和文字之间（和手动走通时点的位置一致）
LOGOUT_CLICK_DY = -24

POWER_TEMPLATE = "assets/images/llm_agent/power_icon.png"
POWER_THRESHOLD = 0.8
CLICK_ENTER_TEMPLATE = "assets/images/screen/click_enter.png"
CLICK_ENTER_THRESHOLD = 0.9

# 单选圈：在「保留登录记录」文字左边找橙色填充；橙色像素数 ≥ SELECTED 算选中，≤ UNSELECTED 算没选中，之间判断不了
RADIO_SELECTED_MIN = 100
RADIO_UNSELECTED_MAX = 15

# 打码手机号：开头几位数字 + 星号 + 结尾几位数字；OCR 会把星号个数认错、偶尔丢掉开头的一位数字，所以不比较整串
MASK_RE = re.compile(r"(\d{2,4})\s*[*＊xX×•·.]{2,}\s*(\d{2,4})")

# 点击点落在这些文字上一律拒绝
FORBIDDEN_TEXT_PARTS = ("清除登录记录", "登录其他账号", "其他账号")
FORBIDDEN_STRIP_TEXTS = ("退出", "更新")
FORBIDDEN_PAD = 10
FORBIDDEN_STRIP_PAD = 30

# 画面类型（classify 的返回值）
KIND_PHONE_MENU = "phone_menu"
KIND_BACK_DIALOG = "back_dialog"
KIND_LOGIN_PAGE = "login_page"
KIND_LOGOUT_DIALOG = "logout_dialog"
KIND_LOGIN_BOX = "login_box"
KIND_DROPDOWN = "dropdown"
KIND_CLICK_ENTER = "click_enter"
KIND_WARNING = "warning"
KIND_LOADING = "loading"
KIND_UNKNOWN = "unknown"

# 失败时给模型看的画面说明和建议
KIND_HINTS = {
    KIND_PHONE_MENU: ("手机菜单", "还没有退出登录"),
    KIND_BACK_DIALOG: ("「是否返回登录页面」确认弹窗", "还没有退出登录；可以点「取消」关掉"),
    KIND_LOGIN_PAGE: ("登录页（开始游戏）", "已经回到登录页，需要用户确认登录的是哪个号"),
    KIND_LOGOUT_DIALOG: ("「退出登录」弹窗", "还没有退出登录；可以点「取消」关掉"),
    KIND_LOGIN_BOX: ("miHoYo 登录框", "停在登录框，需要用户手动选号进入"),
    KIND_DROPDOWN: ("登录框的账号下拉列表", "停在账号下拉列表，需要用户手动选号进入"),
    KIND_CLICK_ENTER: ("「点击进入」页", "已经登录到点击进入这一页，可以手动点击进入"),
    KIND_WARNING: ("「警告：游戏前详阅」页", "游戏还在加载"),
    KIND_LOADING: ("加载中", "游戏还在加载"),
    KIND_UNKNOWN: ("认不出的画面", "先看附图"),
}
GENERIC_ADVICE = ("不要去点登录页右侧的「退出」「更新」、登录框里的「登录其他账号」、账号列表里的 ⊗、「退出并清除登录记录」；"
                  "搞不定就 leave_for_user 留给用户。")

PLAN_STEPS = (
    "确认当前画面和账号（已经是目标账号就直接返回，什么都不点）",
    "回到手机菜单，点右下角电源图标",
    "在「是否返回登录页面？」弹窗点「确认」，等回到登录页",
    "点登录页右侧的「登出」",
    "在「退出登录」弹窗确认选中的是「退出并保留登录记录」，再点「退出」",
    "在登录框里读当前号码；不是目标就展开下拉列表，点目标号码那一行",
    "点「进入游戏」，回登录页点「开始游戏」，再点「点击进入」，等回到主界面",
    "读左下角 UID，确认就是目标账号",
)


# ---------------------------------------------------------------- 数据 ----

@dataclass(frozen=True)
class Hit:
    """识别到的一个点击目标（截图坐标）。"""
    x: int
    y: int
    text: str = ""

    @property
    def point(self) -> Tuple[int, int]:
        return self.x, self.y


@dataclass(frozen=True)
class Mask:
    """打码手机号拆出来的开头和结尾数字。"""
    prefix: str
    suffix: str
    raw: str = ""

    def __str__(self):
        return f"{self.prefix}…{self.suffix}"


@dataclass(frozen=True)
class BackDialog:
    confirm: Hit
    cancel: Optional[Hit]


@dataclass(frozen=True)
class LoginPage:
    start: Hit
    logout: Optional[Hit]  # 「登出」的点击点；没认到时是 None


@dataclass(frozen=True)
class LogoutDialog:
    keep: Element  # 「退出并保留登录记录」
    clear: Optional[Element]  # 「退出并清除登录记录」
    exit: Hit
    cancel: Hit


@dataclass(frozen=True)
class LoginBox:
    enter: Hit  # 「进入游戏」
    mask: Optional[Mask]  # 下拉框里当前显示的号码
    arrow: Hit  # 下拉箭头


@dataclass(frozen=True)
class Row:
    mask: Mask
    hit: Hit


@dataclass(frozen=True)
class Dropdown:
    rows: List[Row]


class StepFailed(Exception):
    """某一步没有走通；流程立刻停下。"""


class AlreadyCurrent(Exception):
    """中途发现当前账号已经是目标账号（已经处理完画面），流程直接结束。"""


# ---------------------------------------------------------------- 手机号 ----

def parse_mask(text: str) -> Optional[Mask]:
    match = MASK_RE.search(text or "")
    if not match:
        return None
    return Mask(match.group(1), match.group(2), match.group(0))


def masks_match(a: Mask, b: Mask) -> bool:
    """开头数字、结尾数字分别比较（一边是另一边的前缀 / 后缀就算相同，容忍 OCR 丢掉一位）。"""
    return ((a.prefix.startswith(b.prefix) or b.prefix.startswith(a.prefix))
            and (a.suffix.endswith(b.suffix) or b.suffix.endswith(a.suffix)))


# ---------------------------------------------------------------- 账号表 ----

@dataclass(frozen=True)
class Account:
    uid: str
    name: str
    login: Optional[str]  # accounts.json 里的原始 login 字段


def load_accounts(path: str) -> List[Account]:
    """读 accounts.json：{"<uid>": {"name": "...", "note": "...", "login": "111******11"}}；不存在或格式错误返回空表。"""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return []
    accounts = []
    if isinstance(data, dict):
        for uid, entry in data.items():
            if isinstance(entry, dict):
                name, login = entry.get("name"), entry.get("login")
            else:
                name, login = entry, None
            accounts.append(Account(str(uid).strip(), (str(name).strip() if isinstance(name, str) else "") or "未命名账号",
                                    login.strip() if isinstance(login, str) and login.strip() else None))
    return accounts


def find_account(accounts: List[Account], query) -> Tuple[Optional[Account], str]:
    """按 UID 或账号名查；名字先精确匹配，再唯一的包含匹配。返回 (账号, 错误说明)。"""
    text = str(query).strip() if isinstance(query, (str, int)) and not isinstance(query, bool) else ""
    if not text:
        return None, "account 必须是账号名或 UID（accounts.json 里登记的）。"
    if not accounts:
        return None, "还没有账号表（logs/mcp_agent/accounts.json 不存在或为空），无法切换账号；这件事请 leave_for_user 留给用户。"
    for account in accounts:
        if account.uid == text:
            return account, ""
    exact = [account for account in accounts if account.name == text]
    if len(exact) == 1:
        return exact[0], ""
    loose = exact or [account for account in accounts if text.lower() in account.name.lower()]
    if len(loose) == 1:
        return loose[0], ""
    names = "、".join(f"{account.name}（UID {account.uid}）" for account in accounts)
    if loose:
        return None, f"「{text}」匹配到多个账号：{names}。请写完整的账号名或 UID。"
    return None, f"账号表里没有「{text}」。已登记的账号：{names}。"


# ---------------------------------------------------------------- 识别：通用 ----

def _norm(text: str) -> str:
    return "".join((text or "").split())


def _in_region(element: Element, size, region) -> bool:
    width, height = size
    cx, cy = element.center
    left, top, right, bottom = region
    return left * width <= cx <= right * width and top * height <= cy <= bottom * height


def _find_text(observation: Observation, region, match: Callable[[str], bool]) -> Optional[Element]:
    """区域里第一个文字满足条件的元素（文字去掉空白后再比）。"""
    for element in observation.elements:
        if _in_region(element, observation.size, region) and match(_norm(element.text)):
            return element
    return None


def _is(word: str) -> Callable[[str], bool]:
    return lambda text: text == word


def _has(word: str) -> Callable[[str], bool]:
    return lambda text: word in text


def _hit(element: Element, dy: int = 0) -> Hit:
    x, y = element.center
    return Hit(x, y + dy, element.text)


@functools.lru_cache(maxsize=8)
def _load_gray(path: str):
    import numpy as np
    from PIL import Image
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    return np.asarray(Image.open(full).convert("L"))


def match_template(image, template_path: str, region, threshold: float) -> Optional[Hit]:
    """在区域内找模板（灰度 + TM_CCOEFF_NORMED，和 M7A 的做法一致），返回命中中心；找不到返回 None。"""
    import cv2
    import numpy as np
    try:
        template = _load_gray(template_path)
    except OSError:
        return None
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    left, top, right, bottom = int(region[0] * width), int(region[1] * height), int(region[2] * width), int(region[3] * height)
    crop = gray[top:bottom, left:right]
    th, tw = template.shape
    if crop.shape[0] < th or crop.shape[1] < tw:
        return None
    result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
    _, best, _, loc = cv2.minMaxLoc(result)
    if best < threshold:
        return None
    return Hit(left + loc[0] + tw // 2, top + loc[1] + th // 2, f"{os.path.basename(template_path)} {best:.2f}")


def radio_selected(image, text: Element) -> Optional[bool]:
    """文字左边的单选圈是不是选中的（选中是橙色填充）：True / False / 判断不了 None。"""
    import numpy as np
    left, top, _, bottom = text.rect
    cy = (top + bottom) // 2
    box = (max(0, left - 44), max(0, cy - 16), max(1, left - 2), cy + 16)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    pixels = np.asarray(image.convert("RGB").crop(box)).astype(int)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    orange = (red >= 200) & (green >= 140) & (red - blue >= 60) & (red >= green) & (green > blue)
    count = int(orange.sum())
    if count >= RADIO_SELECTED_MIN:
        return True
    if count <= RADIO_UNSELECTED_MAX:
        return False
    return None


# ---------------------------------------------------------------- 识别：各画面 ----

def find_phone_menu(observation: Observation) -> Optional[Hit]:
    """手机菜单：右下角的电源图标（没有文字，用模板）。返回图标中心。"""
    return match_template(observation.image, POWER_TEMPLATE, REGION_POWER, POWER_THRESHOLD)


def find_back_dialog(observation: Observation) -> Optional[BackDialog]:
    """弹窗「提示 / 是否返回登录页面？」，按钮「取消」「确认」。"""
    if not _find_text(observation, REGION_DIALOG_TITLE, _has("返回登录页面")):
        return None
    confirm = _find_text(observation, REGION_DIALOG_CONFIRM, _is("确认"))
    if confirm is None:
        return None
    cancel = _find_text(observation, REGION_DIALOG_CANCEL, _is("取消"))
    return BackDialog(_hit(confirm), _hit(cancel) if cancel else None)


def find_logout_dialog(observation: Observation) -> Optional[LogoutDialog]:
    """弹窗「退出登录」：两个单选项 + 「取消」「退出」按钮。"""
    if not _find_text(observation, REGION_LOGOUT_TITLE, _is("退出登录")):
        return None
    keep = _find_text(observation, REGION_LOGOUT_KEEP, _has("保留登录记录"))
    exit_button = _find_text(observation, REGION_LOGOUT_EXIT, _is("退出"))
    cancel = _find_text(observation, REGION_LOGOUT_CANCEL, _is("取消"))
    if keep is None or exit_button is None or cancel is None:
        return None
    clear = _find_text(observation, REGION_LOGOUT_CLEAR, _has("清除登录记录"))
    return LogoutDialog(keep, clear, _hit(exit_button), _hit(cancel))


def find_login_box(observation: Observation) -> Optional[LoginBox]:
    """miHoYo 登录框：「进入游戏」按钮 + 下拉框（显示当前号码）。展开下拉列表时按钮被盖住，不算。"""
    enter = _find_text(observation, REGION_BOX_ENTER, _is("进入游戏"))
    if enter is None:
        return None
    header = None
    for element in observation.elements:
        if _in_region(element, observation.size, REGION_BOX_HEADER) and parse_mask(element.text):
            header = element
            break
    width, height = observation.size
    arrow = Hit(int(width * DROPDOWN_ARROW_X), header.center[1] if header else int(height * 0.458), "下拉箭头")
    return LoginBox(_hit(enter), parse_mask(header.text) if header else None, arrow)


def find_dropdown(observation: Observation) -> Optional[Dropdown]:
    """展开的账号下拉列表：下方有号码行，且「进入游戏」被盖住。只返回号码行（文字所在位置，不含右边的 ⊗）。"""
    if _find_text(observation, REGION_BOX_ENTER, _is("进入游戏")):
        return None
    rows = []
    for element in observation.elements:
        if _in_region(element, observation.size, REGION_LIST_ROWS):
            mask = parse_mask(element.text)
            if mask:
                rows.append(Row(mask, _hit(element)))
    return Dropdown(rows) if rows else None


def find_login_page(observation: Observation) -> Optional[LoginPage]:
    """登录页：中间「开始游戏」，右侧竖排有「登出」。被弹窗或登录框盖住时（背景里还能 OCR 到）不算。"""
    start = _find_text(observation, REGION_LOGIN_START, _is("开始游戏"))
    if start is None:
        return None
    if find_logout_dialog(observation) or find_back_dialog(observation) or find_login_box(observation):
        return None
    logout = _find_text(observation, REGION_LOGIN_LOGOUT, _is("登出"))
    return LoginPage(_hit(start), _hit(logout, LOGOUT_CLICK_DY) if logout else None)


def find_click_enter(observation: Observation) -> Optional[Hit]:
    """「点击进入」页：下方中间的「点击进入」（OCR 优先，认不出再用模板）。"""
    element = _find_text(observation, REGION_CLICK_ENTER, _is("点击进入"))
    if element is not None:
        return _hit(element)
    return match_template(observation.image, CLICK_ENTER_TEMPLATE, REGION_CLICK_ENTER, CLICK_ENTER_THRESHOLD)


def classify(observation: Observation) -> str:
    """当前画面是哪一类（失败时告诉模型停在哪）。弹窗和登录框先判，因为背景里的登录页文字还能 OCR 到。"""
    if find_logout_dialog(observation):
        return KIND_LOGOUT_DIALOG
    if find_back_dialog(observation):
        return KIND_BACK_DIALOG
    if find_dropdown(observation):
        return KIND_DROPDOWN
    if find_login_box(observation):
        return KIND_LOGIN_BOX
    if find_click_enter(observation):
        return KIND_CLICK_ENTER
    if find_login_page(observation):
        return KIND_LOGIN_PAGE
    if find_phone_menu(observation):
        return KIND_PHONE_MENU
    texts = [_norm(element.text) for element in observation.elements]
    if any("游戏前详阅" in text for text in texts):
        return KIND_WARNING
    if any("加载中" in text for text in texts):
        return KIND_LOADING
    return KIND_UNKNOWN


# ---------------------------------------------------------------- 点击安全 ----

def forbidden_reason(observation: Observation, point: Tuple[int, int]) -> Optional[str]:
    """点位落在不能点的文字上就返回原因：清除登录记录、登录其他账号、右侧竖排的「退出」「更新」。"""
    x, y = point
    width, _ = observation.size
    for element in observation.elements:
        text = _norm(element.text)
        left, top, right, bottom = element.rect
        in_strip = element.center[0] >= width * RIGHT_STRIP_X and text in FORBIDDEN_STRIP_TEXTS
        pad = FORBIDDEN_STRIP_PAD if in_strip else FORBIDDEN_PAD
        if not (left - pad <= x <= right + pad and top - pad <= y <= bottom + pad):
            continue
        for part in FORBIDDEN_TEXT_PARTS:
            if part in text:
                return f"点位落在「{element.text}」上，这个不能点"
        if in_strip:
            return f"点位在右侧竖排的「{element.text}」附近，这个不能点"
    return None


# ---------------------------------------------------------------- 流程 ----

class AccountSwitcher:
    """执行一次切换。tools 是 McpTools；调用方已经抢到互斥锁、确认没有 M7A 任务在跑。"""

    def __init__(self, tools, account: Account):
        self.tools = tools
        self.account = account
        self.target = parse_mask(account.login or "")
        self.steps: List[dict] = []
        self.obs: Optional[Observation] = None
        self.first_obs: Optional[Observation] = None
        self._step_name = "开始"
        self._step_started = tools._clock()
        self._started = tools._clock()

    # -------------------------------------------------------- 小工具 ----

    def _begin(self, name: str):
        self._step_name = name
        self._step_started = self.tools._clock()

    def _done(self, detail: str = ""):
        self.steps.append({"step": self._step_name, "ok": True, "detail": detail,
                           "seconds": round(self.tools._clock() - self._step_started, 1)})

    def _fail(self, reason: str):
        self.steps.append({"step": self._step_name, "ok": False, "detail": reason,
                           "seconds": round(self.tools._clock() - self._step_started, 1)})
        raise StepFailed(reason)

    def _look(self) -> Observation:
        self.obs = self.tools._observe(False)
        return self.obs

    def _remaining(self) -> float:
        return TOTAL_TIMEOUT - (self.tools._clock() - self._started)

    def _wait_for(self, finder: Callable[[Observation], object], timeout: float, what: str,
                  interval: float = POLL_SECONDS):
        """反复观察，直到 finder 认出预期画面（返回非空）；超时或总时长用完就停，不做任何补救点击。"""
        deadline = self.tools._clock() + min(timeout, max(self._remaining(), 0))
        while True:
            found = finder(self._look())
            if found:
                return found
            if self._remaining() <= 0:
                self._fail(f"整个流程已超过 {TOTAL_TIMEOUT:g} 秒，停止（还在等：{what}）")
            if self.tools._clock() >= deadline:
                self._fail(f"{timeout:g} 秒内没等到「{what}」")
            self.tools._sleep(interval)

    def _guard_click(self, hit: Hit, label: str, *, right_strip: bool, x_max: Optional[float]):
        observation = self.obs
        width, height = observation.size
        if not (0 <= hit.x < width and 0 <= hit.y < height):
            self._fail(f"安全检查：点位 {hit.point} 超出画面，不点「{label}」")
        reason = forbidden_reason(observation, hit.point)
        if reason:
            self._fail(f"安全检查：{reason}，不点「{label}」")
        if hit.x >= width * RIGHT_STRIP_X and not right_strip:
            self._fail(f"安全检查：「{label}」的点位在右侧竖排按钮区（那里有「退出」「更新」），不点")
        if x_max is not None and hit.x > width * x_max:
            self._fail(f"安全检查：「{label}」的点位太靠右（右边是删除登录记录的 ⊗），不点")

    def _click(self, hit: Hit, label: str, *, right_strip: bool = False, x_max: Optional[float] = None):
        """点击 hit（必须来自最近一次观察 self.obs）。先过 _guard_click，再确认窗口在前台。"""
        self._guard_click(hit, label, right_strip=right_strip, x_max=x_max)
        problem = self.tools._ensure_front(True)
        if problem:
            self._fail(problem)
        result = self.tools.executor.click_point(self.obs, hit.point, label)
        if not result.ok:
            self._fail(result.message)

    # -------------------------------------------------------- 稳定等待 ----

    def _wait_stable(self, finder: Callable[[Observation], object], signature: Callable, timeout: float, what: str,
                     seed=None, need: int = STABLE_LOOKS, interval: float = STABLE_GAP):
        """等 finder 认出预期画面，并且连续 need 次观察的 signature 完全一样才返回（避开展开、切换时的动画中间帧）。

        seed 是已经看到的一次结果（算第一次）。中间有一次认不出就重新计数；超时或总时长用完就停，不补点。
        """
        deadline = self.tools._clock() + min(timeout, max(self._remaining(), 0))
        found = seed if seed is not None else finder(self._look())
        last = signature(found) if found else None
        same = 1 if found else 0
        while not (found and same >= need):
            if self._remaining() <= 0:
                self._fail(f"整个流程已超过 {TOTAL_TIMEOUT:g} 秒，停止（还在等：{what}）")
            if self.tools._clock() >= deadline:
                self._fail(f"{timeout:g} 秒内没等到「{what}」" + ("稳定下来（画面一直在变）" if found else ""))
            self.tools._sleep(interval)
            found = finder(self._look())
            if not found:
                last, same = None, 0
                continue
            current = signature(found)
            same = same + 1 if current == last else 1
            last = current
        return found

    @staticmethod
    def _box_signature(box: LoginBox):
        return (box.mask.prefix[:2], box.mask.suffix) if box.mask else None

    @staticmethod
    def _rows_signature(dropdown: Dropdown):
        return tuple((row.mask.prefix[:2], row.mask.suffix) for row in dropdown.rows)

    def _target_visible(self) -> bool:
        """当前这张画面左下角能读到目标账号的 UID（不靠 tracker 里可能过时的记录）。"""
        return self.tools.tracker._candidates(self.obs) == [self.account.uid]

    # -------------------------------------------------------- 流程 ----

    def run(self) -> str:
        """走完整个流程，返回成功说明；任何一步失败抛 StepFailed。

        先看当前画面决定从哪一步接着走（中途停下的流程可以继续）：每个 _s_* 步骤接收「这一步要点的东西」，
        点完后等下一步的画面出现，把它交给下一步。流程是线性的、不回头，所以不会死循环。
        """
        tools = self.tools
        self._begin("确认当前画面和账号")
        self.first_obs = self.obs = tools._take_observation(False)
        kind = classify(self.obs)
        label = KIND_HINTS[kind][0]
        if kind in (KIND_UNKNOWN, KIND_PHONE_MENU) and self._target_visible():
            detail = "当前已经是目标账号"
            if kind == KIND_PHONE_MENU:  # 手机菜单里也看得到左下角 UID：把菜单关掉
                result = tools.executor.press("esc", 0)
                detail += "，已按 esc 关闭手机菜单" + ("" if result.ok else "（关闭失败）")
            self._done(detail)
            return "已经是这个号"
        self._done(f"当前账号：{tools.tracker.describe()}；画面：{label}")

        stage, found = self._pick_start(kind)
        order = [self._s_open_menu, self._s_power, self._s_confirm, self._s_logout, self._s_exit, self._s_select, self._s_enter,
                 self._s_start, self._s_click_enter, self._s_verify]
        for step in order[order.index(stage):]:
            found = step(found)
        return "切换成功"

    def _pick_start(self, kind: str):
        """按当前画面选起点，返回 (从哪一步开始, 这一步要点的东西)。"""
        obs = self.obs
        if kind == KIND_PHONE_MENU:
            return self._s_power, find_phone_menu(obs)
        if kind == KIND_BACK_DIALOG:
            return self._s_confirm, find_back_dialog(obs)
        if kind == KIND_LOGIN_PAGE:
            # 登录页有两种：还没登出，或者已经选好号回到了登录页。分不出来，一律从「登出」走：
            # 登出后回到登录框，重新读号、选号，线性往下走，不会兜圈
            return self._s_logout, find_login_page(obs)
        if kind == KIND_LOGOUT_DIALOG:
            return self._s_exit, find_logout_dialog(obs)
        if kind == KIND_LOGIN_BOX:
            return self._s_select, find_login_box(obs)
        if kind == KIND_DROPDOWN:
            return self._s_select, find_dropdown(obs)
        if kind == KIND_CLICK_ENTER:
            return self._s_click_enter, find_click_enter(obs)
        if kind in (KIND_WARNING, KIND_LOADING):
            self._begin("确认当前画面和账号")
            self._fail(f"游戏正在加载（{KIND_HINTS[kind][0]}），不知道加载的是哪一步，先别动；等加载完看到登录页或主界面再试")
        return self._s_open_menu, None

    def _s_open_menu(self, _):
        """主界面（或 M7A 认得的其它界面）→ 手机菜单，返回电源图标。"""
        tools = self.tools
        self._begin("打开手机菜单")
        screen_id, screen_name = tools._last_screen
        if not screen_id:
            self._fail(f"当前画面是{KIND_HINTS[classify(self.obs)][0]}，不是主界面也不是手机菜单；"
                       f"先用 goto_screen(main) 回到主界面再试")
        try:
            tools.screen.change_to("menu")
        except Exception as e:
            reraise_passthrough(e)
            self._fail(f"从「{screen_name}」跳转到手机菜单失败：{tools._safe(e)}")
        power = self._wait_for(find_phone_menu, WAIT_MENU, "手机菜单（右下角电源图标）")
        tools.tracker.update(self.obs)
        if self._target_visible():
            result = tools.executor.press("esc", 0)  # 其实已经是目标账号，把菜单关掉
            self._done("手机菜单里发现已经是目标账号，已按 esc 关闭菜单" + ("" if result.ok else "（关闭失败）"))
            raise AlreadyCurrent()
        self._done()
        return power

    def _s_power(self, power):
        """点电源图标 → 「是否返回登录页面」弹窗。"""
        self._begin("点电源图标")
        self._click(power, "电源图标", right_strip=True)
        dialog = self._wait_for(find_back_dialog, WAIT_DIALOG, "弹窗「是否返回登录页面？」")
        self._done()
        return dialog

    def _s_confirm(self, dialog):
        """点「确认」→ 登录页（带「登出」）。"""
        self._begin("确认返回登录页面")
        self._click(dialog.confirm, "确认")
        page = self._wait_for(lambda o: (p := find_login_page(o)) and p.logout and p, WAIT_LOGIN_PAGE,
                              "登录页（开始游戏 + 右侧的登出）", SLOW_POLL_SECONDS)
        self._done()
        return page

    def _s_logout(self, page):
        """点「登出」→ 「退出登录」弹窗。"""
        self._begin("点登出")
        if not page.logout:  # 从中途接上时登录页可能还没把「登出」画出来
            page = self._wait_for(lambda o: (p := find_login_page(o)) and p.logout and p, WAIT_LOGIN_PAGE,
                                  "登录页（开始游戏 + 右侧的登出）", SLOW_POLL_SECONDS)
        self._click(page.logout, "登出", right_strip=True)
        dialog = self._wait_for(find_logout_dialog, WAIT_DIALOG, "弹窗「退出登录」")
        self._done()
        return dialog

    def _s_exit(self, dialog):
        """确认选的是「保留登录记录」，点「退出」→ 登录框（号码稳定读到）。"""
        self._begin("确认保留登录记录并退出")
        dialog = self._ensure_keep_selected(dialog)
        self._click(dialog.exit, "退出（弹窗按钮）")
        box = self._wait_stable(lambda o: (b := find_login_box(o)) and b.mask and b, self._box_signature,
                                WAIT_LOGIN_BOX, "登录框（下拉框里能读到当前号码）")
        self._done(f"登录框当前号码：{box.mask}")
        return box

    def _s_select(self, found):
        """登录框里选目标账号，返回显示着目标号码的登录框。found 是登录框，或者（从展开的列表接上时）下拉列表。"""
        self._begin("在登录框里选目标账号")
        if isinstance(found, LoginBox):
            box = self._wait_stable(lambda o: (b := find_login_box(o)) and b.mask and b, self._box_signature,
                                    WAIT_LOGIN_BOX, "登录框（下拉框里能读到当前号码）", seed=found if found.mask else None)
            if masks_match(box.mask, self.target):
                self._done(f"登录框当前就是目标号码 {self.target}，不用展开列表")
                return box
            self._click(box.arrow, "下拉箭头")
            found = self._wait_for(find_dropdown, WAIT_DROPDOWN, "展开的账号下拉列表")
        # 展开有动画：连续两次看到一样的行才拿来判断；目标不在里面时再等一轮确认
        dropdown = self._wait_stable(find_dropdown, self._rows_signature, WAIT_DROPDOWN, "账号下拉列表", seed=found)
        matches = self._matches(dropdown)
        if not matches:
            dropdown = self._wait_stable(find_dropdown, self._rows_signature, WAIT_DROPDOWN, "账号下拉列表", seed=dropdown)
            matches = self._matches(dropdown)
        if not matches:
            shown = "、".join(str(row.mask) for row in dropdown.rows)
            self._fail(f"账号列表里没有目标号码 {self.target}（列表里是：{shown}）；不点「登录其他账号」，请用 leave_for_user 留给用户")
        if len(matches) > 1:
            self._fail(f"账号列表里有 {len(matches)} 行都像目标号码 {self.target}，分不清，不点；请用 leave_for_user 留给用户")
        self._click(matches[0].hit, f"号码 {matches[0].mask}", x_max=LIST_CLICK_X_MAX)
        box = self._wait_stable(lambda o: (b := find_login_box(o)) and b.mask and masks_match(b.mask, self.target) and b,
                                self._box_signature, WAIT_DROPDOWN, f"登录框显示目标号码 {self.target}")
        self._done(f"已选中 {box.mask}")
        return box

    def _matches(self, dropdown: Dropdown) -> List[Row]:
        return [row for row in dropdown.rows if masks_match(row.mask, self.target)]

    def _s_enter(self, box):
        """点「进入游戏」→ 登录页（开始游戏）。"""
        self._begin("点进入游戏")
        self._click(box.enter, "进入游戏")
        page = self._wait_for(find_login_page, WAIT_AFTER_ENTER, "登录页（开始游戏）", SLOW_POLL_SECONDS)
        self._done()
        return page

    def _s_start(self, page):
        """点「开始游戏」→「点击进入」页。"""
        self._begin("点开始游戏")
        self._click(page.start, "开始游戏")
        enter = self._wait_for(find_click_enter, WAIT_CLICK_ENTER, "「点击进入」页", SLOW_POLL_SECONDS)
        self._done()
        return enter

    def _s_click_enter(self, enter):
        """点「点击进入」→ 主界面。"""
        self._begin("点击进入")
        self._click(enter, "点击进入")
        self._wait_for_main()
        self._done()

    def _s_verify(self, _):
        """读左下角 UID，必须等于目标账号（从「点击进入」接上时，账号可能不是目标，这里不对就失败，不会再切一轮）。"""
        tools = self.tools
        self._begin("确认账号")
        same = 0
        for attempt in range(UID_CONFIRM_TRIES):
            self.obs = tools._take_observation(False)
            same = same + 1 if self._target_visible() else 0
            if same >= UID_CONFIRM_LOOKS:
                self._done(f"左下角 UID 是 {self.account.uid}")
                return
            if attempt + 1 < UID_CONFIRM_TRIES:
                tools._sleep(UID_CONFIRM_GAP)
        candidates = tools.tracker._candidates(self.obs)
        current = "、".join(tools.tracker.label(uid) for uid in candidates) if candidates else "未识别到 UID"
        self._fail(f"回到主界面了，但未能连续确认目标 UID；当前帧左下角：{current}，不是目标账号的稳定确认")

    def _ensure_keep_selected(self, dialog: LogoutDialog) -> LogoutDialog:
        """点「退出」之前确认选中的是「保留登录记录」：没选中或判断不了就点一下那一行，再判断一次；还不行点「取消」并中止。"""
        if self._keep_ok(dialog):
            return dialog
        self._click(_hit(dialog.keep), "退出并保留登录记录")
        self.tools._sleep(RADIO_SETTLE)
        again = find_logout_dialog(self._look())
        if again is not None and self._keep_ok(again):
            return again
        if again is None:
            self._fail("判断不了「退出并保留登录记录」是否选中，点那一行之后弹窗也看不到了")
        self._click(again.cancel, "取消")
        self._fail("判断不了「退出并保留登录记录」是否选中，已点「取消」中止，还没有退出登录")

    def _keep_ok(self, dialog: LogoutDialog) -> bool:
        image = self.obs.image
        keep = radio_selected(image, dialog.keep)
        clear = radio_selected(image, dialog.clear) if dialog.clear is not None else False
        return keep is True and clear is not True

    def _wait_for_main(self):
        """点击进入之后等回到主界面。M7A 认得的其它界面（例如月卡弹窗）交给 M7A 的界面跳转图回主界面；认不出的只等不点。"""
        tools = self.tools
        deadline = tools._clock() + min(WAIT_MAIN, max(self._remaining(), 0))
        jumps = 0
        while True:
            screen_id, screen_name = tools._detect_screen()
            if screen_id == "main":
                return
            if screen_id and jumps < 2:
                jumps += 1
                try:
                    tools.screen.change_to("main")
                except Exception as e:
                    reraise_passthrough(e)
                    self._fail(f"从「{screen_name}」回主界面失败：{tools._safe(e)}")
                continue
            if self._remaining() <= 0:
                self._fail(f"整个流程已超过 {TOTAL_TIMEOUT:g} 秒，停止（还在等：主界面）")
            if tools._clock() >= deadline:
                self._fail(f"{WAIT_MAIN:g} 秒内没等到主界面")
            tools._sleep(SLOW_POLL_SECONDS)


# ---------------------------------------------------------------- 入口 ----

def _steps_text(steps: List[dict]) -> str:
    return "；".join(f"{'✓' if step['ok'] else '✗'}{step['step']}（{step['seconds']:g}s）" for step in steps)


def _plan_text(account: Account) -> str:
    lines = [f"离线模式：只列出计划，没有执行。目标账号：{account.name}（UID {account.uid}），登录框里显示 {account.login}。"]
    lines += [f"{index}. {step}" for index, step in enumerate(PLAN_STEPS, 1)]
    return "\n".join(lines)


def switch_account(tools, account) -> "mt.ToolOutput":
    """switch_account 工具的实现：校验 → 抢锁 → 走流程 → 写一条记录 → 返回最终观察。"""
    args = {"account": account}
    started = tools._clock()
    found, error = find_account(load_accounts(tools.tracker.registry_path), account)
    if found is None:
        return tools._refuse(TOOL_NAME, args, started, mt.RULE_INVALID_ARGUMENT, error)
    if not found.login:
        return tools._refuse(
            TOOL_NAME, args, started, mt.RULE_INVALID_ARGUMENT,
            f"账号「{found.name}」在 accounts.json 里没有 login 字段（登录框下拉里显示的打码手机号），无法自动选号。"
            f"这件事请用 leave_for_user 留给用户，不要自己去点登录页的按钮。")
    if parse_mask(found.login) is None:
        return tools._refuse(
            TOOL_NAME, args, started, mt.RULE_INVALID_ARGUMENT,
            f"账号「{found.name}」的 login 字段格式不对（应该形如 111******11：开头几位数字 + 星号 + 结尾几位数字），"
            f"这件事请用 leave_for_user 留给用户。")
    if tools.offline:
        return tools._refuse(TOOL_NAME, args, started, mt.RULE_OFFLINE, _plan_text(found))

    try:
        with tools._exclusive():
            return _run_locked(tools, args, found, started)
    except mt.ToolBusy as e:
        return mt.ToolOutput(f"【未执行】{e}")


def _run_locked(tools, args: dict, account: Account, started: float) -> "mt.ToolOutput":
    task_text = tools._active_task()
    if task_text:
        return tools._refuse(TOOL_NAME, args, started, mt.RULE_TASK_RUNNING, task_text)
    problem = tools._ensure_front(True)
    if problem:
        return tools._refuse(TOOL_NAME, args, started, mt.RULE_INVALID_ARGUMENT, problem)

    switcher = AccountSwitcher(tools, account)
    failure = None
    outcome = ""
    try:
        outcome = switcher.run()
    except AlreadyCurrent:
        outcome = "已经是这个号"
    except StepFailed as e:
        failure = str(e)
    except Exception as e:
        reraise_passthrough(e)  # 窗口丢失、需要用户处理的后台输入异常不能吞掉
        failure = f"执行出错：{tools._safe(e)}"
        switcher.steps.append({"step": switcher._step_name, "ok": False, "detail": failure,
                               "seconds": round(tools._clock() - switcher._step_started, 1)})

    # 流程结束：重新观察一次，让最近一次观察和现在的画面一致（之后的 click 编号才对）
    try:
        final = tools._take_observation(False)
    except Exception as e:
        reraise_passthrough(e)
        final = switcher.obs
        tools._warn(f"切换账号结束后的观察失败：{tools._safe(e)}")
    final_uid_uncertain = False
    if failure is None and final is not None:
        candidates = tools.tracker._candidates(final)
        final_uid_uncertain = not candidates
        if candidates and candidates != [account.uid]:
            switcher._begin("确认账号")
            try:
                switcher._fail("最终观察左下角读到的 UID 与目标账号不一致，不能确认切换成功")
            except StepFailed as e:
                failure = str(e)
                outcome = ""
    elapsed = round(tools._clock() - started, 1)
    ok = failure is None
    extra = {"target_uid": account.uid, "steps": switcher.steps, "outcome": outcome or failure,
             "stopped_at": None if ok else switcher._step_name}
    message = outcome if ok else f"失败：停在「{switcher._step_name}」：{failure}"
    tools._write_record(TOOL_NAME, args, started, result=ActionResult(ok, message), before=switcher.first_obs,
                        before_screen=(None, None), extra=extra)

    steps = _steps_text(switcher.steps)
    if ok and outcome == "已经是这个号":
        head = f"当前已经是「{account.name}」（UID {account.uid}），没有做任何切换。"
    elif ok:
        head = f"已切换到「{account.name}」（UID {account.uid}），用时约 {elapsed:g} 秒。步骤：{steps}"
    else:
        kind = classify(final) if final is not None else KIND_UNKNOWN
        label, advice = KIND_HINTS[kind]
        head = (f"【切换账号失败】停在「{switcher._step_name}」：{failure}\n"
                f"现在的画面大概是：{label}。{advice}。{GENERIC_ADVICE}\n已走的步骤：{steps}")
    if ok and final_uid_uncertain:
        head += "\n最终附图未识别到 UID；目标账号已在流程内的新截图中核验，附图中的账号暂不确定。"
    if final is None:
        return mt.ToolOutput(head)
    return tools._observation_output(final, head)
