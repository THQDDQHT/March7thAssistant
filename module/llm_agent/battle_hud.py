"""战斗状态识别：从一张战斗截图里读出「轮到谁、选了什么、技能点几个、终结技亮没亮」等关键状态。

回合制战斗里这些信息都是 1920×1080 画面上的小图标，模型看缩小后的标注图经常读错（技能点 4 读成 5、
按了键没生效也不知道）。这里用固定区域上的像素运算判断，只对很小的几块裁剪图做一次 OCR。

- 纯函数 read_battle_hud(image, ocr=...)：输入任意 16:9 截图，先缩放到 1920×1080 参考分辨率再按固定区域判断；
  不依赖游戏窗口、mcp 包和 observe，离线截图就能跑。
- 每个字段都允许是 None（= 不确定）；宁可不确定，也不报错的值。各项原始测量值放在 BattleHud.debug，方便以后调阈值。
- 坐标全是 1920×1080 下的像素位置，来自 1920×1080 实机截图（活动「一镜到底」战斗，3 人和 4 人队）。
  UI 是固定布局：右下 Q/E 两个技能按钮各自位置不变、选中的那个变大；技能点在按钮左边（数字 + 5 颗星）；
  左下角色栏每人一个血条、一个终结技圆盘；左上是波次和行动顺序，沙漏图标旁的数字是剩余轮次。
"""
import base64
import io
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

REF_SIZE = (1920, 1080)
ASPECT_TOLERANCE = 0.02  # 宽高比和 16:9 相差超过这个值就不认

PHASE_COMMAND = "command"        # 轮到我方选动作（Q/E 按钮在）
PHASE_ULT_PENDING = "ult_pending"  # 终结技待释放（大按钮上方显示「空格」）
PHASE_RESULT = "result"          # 结算页
PHASE_OTHER = "other"            # 动画、角色专属特殊模式、高光过场等

SELECTED_BASIC = "basic"
SELECTED_SKILL = "skill"

# ---------------------------------------------------------------- 固定区域（1920×1080） ----

# 右上角暂停键 ‖：两根白竖条，中间和两侧是暗的
PAUSE_BARS = ((1854, 1857), (1866, 1869))  # 两根竖条的 x 范围
PAUSE_Y = (41, 55)
PAUSE_GAP = (1860, 1864)
PAUSE_SIDES = ((1846, 1851), (1872, 1877))
PAUSE_MIN_BAR = 190      # 竖条最小通道亮度均值
PAUSE_MIN_CONTRAST = 70  # 竖条比周围至少亮这么多

# 倍速 / 自动按钮：开着时外面有白框，框上沿比外侧亮得多
SPEED_BOX = (1625, 1698)
AUTO_BOX = (1725, 1798)
BOX_ON_MIN = 50
BOX_OFF_MAX = 10

# 左下角色栏：每个角色一个青色血条（y 970-973），第 k 个（从 0 起）血条左端在 166 + 225k
HP_BAR_X0 = 166
HP_BAR_PITCH = 225
HP_BAR_Y = (970, 974)
MAX_TEAM = 4

# Q/E 按钮右上角的小圆徽标（深色圆 + 米黄字母），大按钮和小按钮的位置、大小都不同：(x, y, 半径)
BADGES = {
    "Ql": (1699, 867, 13),  # Q 大（选中）
    "Qs": (1706, 908, 10),  # Q 小
    "El": (1863, 763, 13),  # E 大（选中）
    "Es": (1845, 802, 10),  # E 小
}
BADGE_SEARCH = 3  # 徽标位置允许偏移的像素（动画里按钮会晃）
BADGE_MIN_DARK = 0.5
BADGE_CREAM = (0.05, 0.45)  # 下限留宽：JPEG 压缩后小徽标的字母笔画会变淡

# 终结技待释放：大按钮右上角的「空格」药丸（深底米黄字）
ULT_PILL = (1762, 784, 1822, 816)
ULT_PILL_MIN_CREAM = 0.10
ULT_PILL_MIN_DARK = 0.40

# 「我方行动中 / 敌方行动中」状态框：深色底 + 白字，在技能点右侧
STATUS_BOX = (1700, 984, 1900, 1014)
STATUS_MIN_DARK = 0.60
STATUS_WHITE = (0.05, 0.12)

# 技能点：数字 + 5 颗星 + 数字下面一条横线
SP_DIGIT_BOX = (1385, 945, 1430, 995)
SP_STAR_X = (1451, 1469, 1488, 1507, 1526)
SP_STAR_Y = (969, 976)
SP_LINE_ROWS = (992, 994)
SP_LINE_REF_ROWS = ((988, 990), (996, 998))
SP_LINE_MIN_CONTRAST = 15
SP_STAR_LIT = 218          # 亮星的最小通道亮度均值（白）
SP_MAX = 5
SP_DIGIT_MIN_CONF = 0.9

# 左上：波次（旗帜图标 + 「1/2」）和行动顺序竖列（沙漏 + 剩余轮次）
WAVE_BOX = (92, 8, 160, 40)
QUEUE_X = (70, 130)       # 在这个横向范围内找沙漏图标
QUEUE_Y = (60, 780)
HOURGLASS_MIN_SCORE = 0.8
ROUNDS_BOX_OFFSET = (30, -4, 66, 26)  # 相对沙漏模板左上角的数字裁剪框 (左, 上, 右, 下)

# 终结技圆盘：第 k 个角色圆盘中心横向 ≈ 291 + 225k，纵向在 880-910 之间晃
ULT_CX0 = 291
ULT_CY_RANGE = (850, 935)
ULT_HALF = 42
ULT_DISC_RADIUS = 44
ULT_READY_BRIGHT = 0.30      # 亮色像素占比 ≥ 此值：粉彩液面填满或徽记发光
ULT_FULL_DISC = (4650, 0.85)  # 圆盘范围内有色像素数 ≥ 此值且集中在圆盘里：圆盘被填满
ULT_PARTIAL_DISC = 4400      # 有色像素数 ≤ 此值且不亮：只填了一部分 / 灰暗进度环
ULT_COLOR_PARTIAL_MIN = 2000  # 有色像素数 ≥ 此值（又 ≤ 上面那个）：确实有一块彩色液面，只是没填满；< 此值是灰盘
ULT_CLEAN_COMPACT = 0.80     # 有色像素集中在圆盘里的比例 ≥ 此值，才相信「液面只填一块」
ULT_LIQUID_MIN_HUE = 0.20    # 液面的主色相占窗口比例 ≥ 此值
ULT_CYAN_HUE = (150, 215)    # 主演徽记（青色线条徽记，不是实心圆盘）的色相范围
ULT_CYAN_DIM = 0.05          # 青色徽记亮色占比 ≤ 此值：暗淡的线条徽记（没满）
ULT_CYAN_PARTIAL_DISC = 2900  # 青色液面的有色像素数 ≤ 此值：只填了一小块（没满）
ULT_WHITE_LEVEL = 240        # 「纯白」像素：三个通道都 ≥ 此值
# 白 / 银色圆盘：满了是一整盘纯白的徽记（中段纯白占比 ≥ 0.2，圆盘自己的银边会让窗口角上也有一点白，最多到 0.13）；
# 没满是灰盘加一块下沿的白色液面（含液面上扫过的反光），中段纯白占比是 0。两者差得很远，所以满的阈值取 0.10。
# 背景发白（过场闪光）时角上的纯白也多，或者窗口里有大片彩色，都不当作满盘。
ULT_WHITE_READY = (0.10, 0.16, 1500)  # 圆盘中段纯白占比 ≥ 第一个值、角上纯白占比 ≤ 第二个值、有色像素数 ≤ 第三个值
ULT_WHITE_PARTIAL = 0.06     # 纯白占比（圆盘内和角上）都 ≤ 此值：没有满盘的白，才可能是没满
ULT_GRAY_MAX_HUE = 0.10      # 灰盘的有色像素占比上限

# 结算页：深色背景 + 居中标题（米黄色的「挑战结束」、红色的「战斗失败」等）
RESULT_TITLE_BOX = (780, 200, 1140, 380)
RESULT_MAX_BRIGHTNESS = 70
RESULT_MIN_CREAM = 0.004  # 标题是米黄色（挑战结束）
RESULT_MIN_RED = 0.02     # 或者红色（战斗失败）
RESULT_WORDS = ("挑战结束", "战斗失败", "战斗胜利", "挑战失败", "挑战成功", "战斗结束", "挑战完成")

OCR_SCALE = 3
OCR_GAP = 24

# battle_state 给模型看的图：1280×720 的干净画面，下面拼一条放大的 HUD 局部
PANEL_SIZE = (1280, 720)
PANEL_JPEG_QUALITY = 60
PANEL_QUEUE_BOX = (0, 0, 250, 700)        # 左上：波次、行动顺序、沙漏
PANEL_QUEUE_SCALE = 0.6
PANEL_TEAM_BOX = (100, 820, 1060, 1010)   # 左下：角色栏（血条、终结技圆盘）
PANEL_SKILL_BOX = (1340, 780, 1920, 1000)  # 右下：Q/E 技能按钮和技能点
PANEL_TOP_RIGHT_BOX = (1580, 20, 1920, 80)  # 右上：倍速、自动、暂停

# 沙漏图标（16×21 灰度，取自 1920×1080 截图）的 PNG，base64；用模板匹配在行动顺序竖列里找它
_HOURGLASS_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAVCAAAAABqVTEOAAABNklEQVR42gXBXSvDYRgH4N99P/fj"
    "b7OtTImlzY6kHEjKoRPlYyifxRfwKZxKHEgOSJyQcKCM5HVtpBZr/2fPy+26aEUiRSaNNiqTt2K3"
    "FopEBIVS/Ns+NjMDV64aY9hwvDu47gnO8/ossRIoPJ28ilCSRAwgEYeYgYPJoAookRJYxQYPAoig"
    "MYUoDOn/KSVVIL0zByb9fO4lhapr3/wGb+pwZAulRD+toysXciHwfRirVFz3dC+FQYHBAylloe9j"
    "uZrnkpkGZHWjcXn4NrVYO0tFSIbafOV092v0cX1p6Q5OCvnUZGe/xb+3o7XmU4+5H+A+HmC4fdMd"
    "HwkkQ/GdnrMp9l3LJuvYeOu+yyXjy9lPkqGKik4058iHopl+8VFFN5crRQcyCaY5s3NBa0OPkJTh"
    "s0jq7D/ikqA4wvgZWQAAAABJRU5ErkJggg=="
)


@dataclass
class SlotHud:
    """我方一个位置的状态。"""
    index: int                       # 1-4
    ult_ready: Optional[bool] = None  # 终结技是否可放；None = 不确定


@dataclass
class BattleHud:
    """一张截图里的战斗状态。所有字段允许 None（= 不确定）。"""
    in_battle: Optional[bool] = None
    phase: Optional[str] = None            # command / ult_pending / result / other
    selected: Optional[str] = None         # basic / skill（只在 command 阶段给）
    skill_points: Optional[int] = None
    team_size: Optional[int] = None
    slots: List[SlotHud] = field(default_factory=list)
    rounds_left: Optional[int] = None
    wave: Optional[Tuple[int, int]] = None  # (当前, 总数)
    auto_on: Optional[bool] = None
    speed_on: Optional[bool] = None
    result_text: Optional[str] = None
    debug: dict = field(default_factory=dict)

    def ult_ready_slots(self) -> List[int]:
        return [slot.index for slot in self.slots if slot.ult_ready]

    def to_dict(self) -> dict:
        return {
            "in_battle": self.in_battle, "phase": self.phase, "selected": self.selected,
            "skill_points": self.skill_points, "team_size": self.team_size,
            "slots": [{"index": slot.index, "ult_ready": slot.ult_ready} for slot in self.slots],
            "rounds_left": self.rounds_left, "wave": list(self.wave) if self.wave else None,
            "auto_on": self.auto_on, "speed_on": self.speed_on, "result_text": self.result_text,
        }


# ---------------------------------------------------------------- 像素小工具 ----

def _to_ref_array(image: Image.Image) -> Optional[np.ndarray]:
    """任意 16:9 截图 → 1920×1080 的 RGB 数组（int16，方便做差）；宽高比不对返回 None。"""
    width, height = image.size
    if width < 320 or height < 180 or abs(width / height - REF_SIZE[0] / REF_SIZE[1]) > ASPECT_TOLERANCE:
        return None
    image = image.convert("RGB")
    if image.size != REF_SIZE:
        image = image.resize(REF_SIZE, Image.Resampling.BOX if width > REF_SIZE[0] else Image.Resampling.BICUBIC)
    return np.asarray(image).astype(np.int16)


def _min_ch(block: np.ndarray) -> np.ndarray:
    return block.min(axis=-1)


def _cream_mask(pixels: np.ndarray) -> np.ndarray:
    """米黄色（徽标字母、「空格」、结算页标题的颜色）。"""
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    return (r > 150) & (g > 115) & (b > 55) & (r >= g) & (g >= b + 15)


_DISC_CACHE: Dict[int, np.ndarray] = {}


def _disc_mask(radius: int) -> np.ndarray:
    mask = _DISC_CACHE.get(radius)
    if mask is None:
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        mask = (yy ** 2 + xx ** 2) <= radius * radius
        _DISC_CACHE[radius] = mask
    return mask


def _badge_stats(a: np.ndarray, x: int, y: int, r: int) -> Tuple[float, float]:
    """圆形区域内的 (深色占比, 米黄占比)。"""
    block = a[y - r:y + r + 1, x - r:x + r + 1]
    pixels = block[_disc_mask(r)]
    dark = float((pixels.max(axis=1) < 75).mean())
    cream = float(_cream_mask(pixels).mean())
    return dark, cream


def _find_badge(a: np.ndarray, x: int, y: int, r: int) -> Tuple[bool, Tuple[float, float]]:
    """在 (x, y) 附近 ±BADGE_SEARCH 像素里找深色圆 + 米黄字母的徽标。返回 (是否找到, 最佳 (深色, 米黄))。"""
    best, best_score = (0.0, 0.0), -1.0
    for dy in range(-BADGE_SEARCH, BADGE_SEARCH + 1):
        for dx in range(-BADGE_SEARCH, BADGE_SEARCH + 1):
            dark, cream = _badge_stats(a, x + dx, y + dy, r)
            score = dark * min(cream, BADGE_CREAM[1])
            if score > best_score:
                best, best_score = (dark, cream), score
    found = best[0] >= BADGE_MIN_DARK and BADGE_CREAM[0] <= best[1] <= BADGE_CREAM[1]
    return found, (round(best[0], 2), round(best[1], 2))


def _red_mask(pixels: np.ndarray) -> np.ndarray:
    """鲜红色（「战斗失败」标题的颜色）。"""
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    return (r > 170) & (g < 110) & (b < 110) & (r - g > 90)


def _rect_stats(a: np.ndarray, rect) -> Tuple[float, float]:
    left, top, right, bottom = rect
    pixels = a[top:bottom, left:right].reshape(-1, 3)
    return float(_cream_mask(pixels).mean()), float((pixels.max(axis=1) < 75).mean())


# ---------------------------------------------------------------- 各项判断 ----

def _pause_contrast(a: np.ndarray) -> float:
    """暂停键两根白竖条比周围亮多少（越大越像）；同时返回两根竖条里较暗的一根的亮度。"""
    y0, y1 = PAUSE_Y
    bars = [float(_min_ch(a[y0:y1, x0:x1]).mean()) for x0, x1 in PAUSE_BARS]
    around = [float(_min_ch(a[y0:y1, x0:x1]).mean()) for x0, x1 in (PAUSE_GAP,) + PAUSE_SIDES]
    return min(bars) - max(around), min(bars)


def _read_team(a: np.ndarray) -> Tuple[int, List[int]]:
    """血条在的位置 → (队伍人数, 有血条的位置列表)。队员倒下后血条没了，所以人数取最大的位置号。"""
    y0, y1 = HP_BAR_Y
    present = []
    for k in range(MAX_TEAM):
        x0 = HP_BAR_X0 + HP_BAR_PITCH * k + 4
        block = a[y0:y1, x0:x0 + 12]
        r, g, b = block[..., 0], block[..., 1], block[..., 2]
        cyan = (g > 170) & (b > 170) & (r < g - 25)
        if cyan.mean() >= 0.6:
            present.append(k + 1)
    return (max(present) if present else 0), present


def _box_contrast(a: np.ndarray, box) -> float:
    """倍速 / 自动按钮外框的上沿比外侧亮多少。开着时有白框（+60 以上），关着时是负数。"""
    x0, x1 = box
    top = _min_ch(a[30:34, x0 + 4:x1 - 4]).mean(axis=1)
    outside = float(_min_ch(a[26:29, x0 + 4:x1 - 4]).mean())
    return float(top.max()) - outside


def _status_box(a: np.ndarray) -> bool:
    """右下「我方行动中 / 敌方行动中」状态框在不在（动画阶段才有，没有可选动作）。"""
    left, top, right, bottom = STATUS_BOX
    block = a[top:bottom, left:right]
    dark = float((block.max(axis=2) < 85).mean())
    white = float((block.min(axis=2) > 200).mean())
    return dark >= STATUS_MIN_DARK and STATUS_WHITE[0] <= white <= STATUS_WHITE[1]


def _switch_state(contrast: float, settled: bool) -> Optional[bool]:
    """倍速 / 自动开关：框清楚可见是开；没有框只有在界面稳定（能选动作）时才是关。"""
    if contrast >= BOX_ON_MIN:
        return True
    if settled and contrast <= BOX_OFF_MAX:
        return False
    return None


def _sp_panel(a: np.ndarray) -> float:
    """技能点数字下面那条横线比上下亮多少（有技能点面板就是正数）。"""
    x0, x1 = 1400, 1540
    line = float(_min_ch(a[SP_LINE_ROWS[0]:SP_LINE_ROWS[1], x0:x1]).mean())
    refs = [float(_min_ch(a[r0:r1, x0:x1]).mean()) for r0, r1 in SP_LINE_REF_ROWS]
    return line - max(refs)


def _star_cores(a: np.ndarray) -> List[Tuple[int, float]]:
    """5 颗星的 (中心亮度, 饱和度)。亮星是纯白（≥218），空心星和预览星更暗。"""
    cores = []
    y0, y1 = SP_STAR_Y
    for cx in SP_STAR_X:
        block = a[y0:y1, cx - 2:cx + 3].reshape(-1, 3)
        mn = block.min(axis=1)
        mx = block.max(axis=1)
        sat = float(((mx - mn) / np.maximum(mx, 1)).mean())
        cores.append((int(mn.mean()), round(sat, 2)))
    return cores


def _lit_prefix(cores) -> int:
    count = 0
    for brightness, _ in cores:
        if brightness >= SP_STAR_LIT:
            count += 1
        else:
            break
    return count


def _sp_consistent(digit: int, lit: int, selected: Optional[str]) -> bool:
    """数字和亮星数交叉验证。

    普攻选中时多出一颗「将获得」的预览星（d+1 颗，可能暗也可能因扫光特效变白），所以亮星数 ∈ {d, d+1}；
    战技选中时第 d 颗是「将消耗」的预览星（红橙色，暗），亮星数 ∈ {d-1, d}；没有选中（终结技待释放等）∈ {d-1, d, d+1}。
    技能点上限 5，所以 d=5 时普攻没有预览星，亮星数 = 5。
    """
    if not 0 <= digit <= SP_MAX:
        return False
    if selected == SELECTED_BASIC:
        allowed = {digit, min(digit + 1, SP_MAX)}
    elif selected == SELECTED_SKILL:
        allowed = {digit - 1, digit}
    else:
        allowed = {digit - 1, digit, min(digit + 1, SP_MAX)}
    return lit in allowed


_HOURGLASS_TEMPLATE: Optional[np.ndarray] = None


def _find_hourglass(a: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
    """在行动顺序竖列里找沙漏图标（灰度模板匹配）。返回 (左上角 (x, y), 匹配分)；没找到 (None, 最高分)。"""
    global _HOURGLASS_TEMPLATE
    import cv2
    if _HOURGLASS_TEMPLATE is None:
        _HOURGLASS_TEMPLATE = np.asarray(Image.open(io.BytesIO(base64.b64decode(_HOURGLASS_PNG))).convert("L"))
    x0, x1 = QUEUE_X
    y0, y1 = QUEUE_Y
    gray = a[y0:y1, x0:x1].astype(np.uint8).mean(axis=2).astype(np.uint8)
    scores = cv2.matchTemplate(gray, _HOURGLASS_TEMPLATE, cv2.TM_CCOEFF_NORMED)
    _, best, _, pos = cv2.minMaxLoc(scores)
    if best < HOURGLASS_MIN_SCORE:
        return None, round(float(best), 2)
    return (x0 + pos[0], y0 + pos[1]), round(float(best), 2)


# ---------------------------------------------------------------- 终结技圆盘 ----

def _hue_degrees(pixels: np.ndarray) -> np.ndarray:
    """RGB（N×3）→ 色相角（0-360）。"""
    r, g, b = (pixels[:, i].astype(np.float32) for i in range(3))
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = np.maximum(mx - mn, 1e-3)
    hue = np.where(mx == r, ((g - b) / delta) % 6, np.where(mx == g, (b - r) / delta + 2, (r - g) / delta + 4))
    return hue * 60.0


def _ult_disc_stats(a: np.ndarray, k: int) -> dict:
    """第 k 个（从 0 起）角色的终结技圆盘窗口里的颜色统计。

    - bright：亮色像素（饱和且很亮）占窗口比例。粉彩液面填满一整个圆盘、主演徽记发光时 ≥ 0.30。
    - hue / hue_frac：最多的那种色相角度，和同色相（±30°）饱和像素占窗口比例（青色徽记在 150-215）。
    - disc / compact：在圆盘范围（半径 44，纵向位置在窗口内搜）里的有色像素数，和它占窗口全部有色像素的比例。
      圆盘被填满时 disc ≈ 5000 以上、compact ≥ 0.9；半满时 disc 小很多；背景有大片颜色时 compact 低。
    """
    cx = ULT_CX0 + HP_BAR_PITCH * k
    y0, y1 = ULT_CY_RANGE
    window = a[y0:y1, cx - ULT_HALF - 6:cx + ULT_HALF + 6]
    mx = window.max(axis=2)
    mn = window.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    colored = (sat >= 0.28) & (mx >= 90)
    total = int(colored.sum())

    inner = window[:, 6:-6].reshape(-1, 3)
    imx = mx[:, 6:-6].reshape(-1)
    isat = sat[:, 6:-6].reshape(-1)
    area = float(inner.shape[0])
    stats = {"bright": round(float(((isat >= 0.25) & (imx >= 215)).sum()) / area, 3),
             "hue_frac": 0.0, "hue": None, "disc": 0, "compact": 0.0}
    icolored = colored[:, 6:-6].reshape(-1)
    if icolored.sum() >= 30:
        hue = _hue_degrees(inner[icolored])
        hist, _ = np.histogram(hue, bins=36, range=(0, 360))
        wrapped = np.concatenate([hist[-1:], hist, hist[:1]])
        smooth = wrapped[:-2] + wrapped[1:-1] + wrapped[2:]
        top = int(smooth.argmax())
        stats["hue_frac"] = round(float(smooth[top]) / area, 3)
        stats["hue"] = top * 10 + 5
    # 无彩色（白 / 银）圆盘：统计纯白像素占比（发光的徽记和满盘是一大片纯白），圆盘四角外面的纯白占比（背景是白的就不可信），
    # 以及圆盘上沿和下沿两条横带的平均亮度（只填了一部分时上沿明显比下沿暗）
    pure = window.min(axis=2)
    core = pure[:, 10:-10]
    stats["white"] = round(float((core >= ULT_WHITE_LEVEL).mean()), 3)
    stats["white_mid"] = round(float((core[20:60] >= ULT_WHITE_LEVEL).mean()), 3)  # 中间一段：满盘有整个徽记，液面只在最下面
    corners = np.concatenate([pure[:10, 10:20].ravel(), pure[:10, -20:-10].ravel(),
                              pure[-10:, 10:20].ravel(), pure[-10:, -20:-10].ravel()])
    stats["white_outside"] = round(float((corners >= ULT_WHITE_LEVEL).mean()), 3)
    stats["top"] = int(mx[8:24, 20:-20].mean())
    stats["bottom"] = int(mx[60:76, 20:-20].mean())
    if total:
        rows, cols = np.mgrid[0:window.shape[0], 0:window.shape[1]]
        best = 0
        for cy in range(ULT_DISC_RADIUS - 5, window.shape[0] - ULT_DISC_RADIUS + 6, 3):
            for cxx in range(window.shape[1] // 2 - 6, window.shape[1] // 2 + 7, 3):
                inside = (rows - cy) ** 2 + (cols - cxx) ** 2 <= ULT_DISC_RADIUS ** 2
                best = max(best, int((colored & inside).sum()))
        stats["disc"] = best
        stats["compact"] = round(best / total, 2)
    return stats


def _ult_ready(stats: dict) -> Optional[bool]:
    """圆盘统计 → 终结技是否可放。能量满时圆盘整个被填满（粉彩、深色实心盘、白 / 银色盘）或徽记发光；
    没满时是灰暗的进度环 / 液面只填一部分。

    True 要有明确的「满」特征；False 只给确认没满的（灰暗、明显只填一部分，或认得的暗淡徽记）；
    圆盘风格没见过、判断不了的一律 None——宁可不确定，也不把能放说成不能放。
    """
    bright, hue = stats["bright"], stats["hue"]
    if bright >= ULT_READY_BRIGHT:
        return True
    if hue is not None and ULT_CYAN_HUE[0] <= hue <= ULT_CYAN_HUE[1]:
        # 青色徽记：线条徽记暗淡是没满；液面只填了一小块也是没满。淡蓝色的液面填满、但徽记没有发光，
        # 是「快满了还是满了」说不准（实机上这种盘之后才会发光），报 None
        return False if bright <= ULT_CYAN_DIM or stats["disc"] <= ULT_CYAN_PARTIAL_DISC else None
    if stats["disc"] >= ULT_FULL_DISC[0] and stats["compact"] >= ULT_FULL_DISC[1]:
        return True
    if (stats["white_mid"] >= ULT_WHITE_READY[0] and stats["white_outside"] <= ULT_WHITE_READY[1]
            and stats["disc"] <= ULT_WHITE_READY[2]):
        return True
    if stats["white"] > ULT_WHITE_PARTIAL or stats["white_outside"] > ULT_WHITE_PARTIAL:
        return None  # 有不少白但不够「满」，也可能是背景在发白
    if bright < 0.25:
        if (ULT_COLOR_PARTIAL_MIN <= stats["disc"] <= ULT_PARTIAL_DISC and stats["compact"] >= ULT_CLEAN_COMPACT
                and stats["hue_frac"] >= ULT_LIQUID_MIN_HUE):
            return False  # 彩色液面只填了一块，圆盘范围里干干净净（背景没有大片颜色）
        if stats["disc"] < ULT_COLOR_PARTIAL_MIN and stats["hue_frac"] <= ULT_GRAY_MAX_HUE:
            return False  # 灰暗的进度环 / 放完之后的空盘：几乎没有彩色像素
    return None


# ---------------------------------------------------------------- OCR ----

def default_ocr(image: np.ndarray):
    """M7A 现有 OCR（recognize_multi_lines）。懒加载，Mac 上也能跑。"""
    from module.ocr import ocr
    return ocr.recognize_multi_lines(image) or []


def _stack_crops(crops: List[Tuple[str, np.ndarray]]) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """把几块小裁剪图放大后竖着拼成一张黑底图（一次 OCR 读完），返回 (画布, {名字: (上, 下)})。"""
    scaled = []
    for name, crop in crops:
        image = Image.fromarray(np.ascontiguousarray(crop.astype(np.uint8)))
        image = image.resize((image.width * OCR_SCALE, image.height * OCR_SCALE), Image.Resampling.BICUBIC)
        scaled.append((name, np.asarray(image)))
    width = max(item.shape[1] for _, item in scaled) + OCR_GAP * 2
    height = sum(item.shape[0] + OCR_GAP for _, item in scaled) + OCR_GAP
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    bands = {}
    y = OCR_GAP
    for name, item in scaled:
        canvas[y:y + item.shape[0], OCR_GAP:OCR_GAP + item.shape[1]] = item
        bands[name] = (y, y + item.shape[0])
        y += item.shape[0] + OCR_GAP
    return canvas, bands


def _ocr_texts(ocr: Callable, crops: List[Tuple[str, np.ndarray]]) -> Dict[str, Tuple[str, float]]:
    """对几块裁剪图一次 OCR，返回 {名字: (文字, 最低置信度)}；某块没读到就不在结果里。"""
    if not crops:
        return {}
    canvas, bands = _stack_crops(crops)
    found: Dict[str, list] = {}
    for entry in ocr(canvas) or []:
        try:
            box, (text, conf) = entry
        except (TypeError, ValueError):
            continue
        ys = [float(point[1]) for point in box]
        xs = [float(point[0]) for point in box]
        cy = (min(ys) + max(ys)) / 2
        for name, (top, bottom) in bands.items():
            if top - OCR_GAP // 2 <= cy <= bottom + OCR_GAP // 2:
                found.setdefault(name, []).append((min(xs), str(text or "").strip(), float(conf)))
                break
    result = {}
    for name, items in found.items():
        items.sort()
        result[name] = ("".join(item[1] for item in items), min(item[2] for item in items))
    return result


def _parse_digit(text: str) -> Optional[int]:
    digits = re.findall(r"\d", text or "")
    return int(digits[0]) if len(digits) == 1 else None


def _parse_wave(text: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"(\d)\s*[/／]\s*(\d)", text or "")
    if not match:
        return None
    current, total = int(match.group(1)), int(match.group(2))
    return (current, total) if 1 <= current <= total <= 9 else None


def _parse_rounds(text: str) -> Optional[int]:
    match = re.fullmatch(r"\D*(\d{1,2})\D*", (text or "").strip())
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------- 主函数 ----

def read_battle_hud(image: Image.Image, ocr: Optional[Callable] = None) -> BattleHud:
    """读一张截图的战斗状态。

    :param image: 任意 16:9 截图（PIL）
    :param ocr: 对 numpy RGB 图返回 [[box, (text, conf)], ...] 的函数；默认用 M7A 的 OCR（懒加载）。
                只会对几块很小的裁剪图（技能点数字、波次、剩余轮次、结算标题）拼成一张图调用一次。
    """
    started = time.perf_counter()
    hud = BattleHud()
    debug = hud.debug
    a = _to_ref_array(image)
    if a is None:
        debug["error"] = "not_16x9"
        return hud

    # ---- 是不是战斗界面：暂停键 + 血条
    contrast, bar_brightness = _pause_contrast(a)
    pause = contrast >= PAUSE_MIN_CONTRAST and bar_brightness >= PAUSE_MIN_BAR
    team_size, present = _read_team(a)
    debug.update(pause_contrast=round(contrast, 1), team_bars=present)
    if pause and team_size:
        hud.in_battle = True
    elif not pause and not team_size:
        hud.in_battle = False
    else:
        hud.in_battle = None  # 只有一项对得上（半透明过场等）
    hud.team_size = team_size or None

    if hud.in_battle is False:
        _read_result(a, hud, ocr)
        debug["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return hud
    if hud.in_battle is None:
        debug["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return hud

    # ---- 倍速 / 自动
    speed_c, auto_c = _box_contrast(a, SPEED_BOX), _box_contrast(a, AUTO_BOX)
    debug.update(speed_box=round(speed_c, 1), auto_box=round(auto_c, 1))
    # 动画、过场里这两个按钮会变暗、白框也不显示，「没有框」不等于「关着」：只在能选动作的阶段才报关，
    # 其它阶段只有框清楚可见时才报开，否则 None（报错的「关」会让模型去按 b，反而把倍速关掉）

    # ---- 阶段和选中
    badges = {}
    for name, (x, y, r) in BADGES.items():
        badges[name] = _find_badge(a, x, y, r)
    debug["badges"] = {name: {"found": found, "dark_cream": stats} for name, (found, stats) in badges.items()}
    pill_cream, pill_dark = _rect_stats(a, ULT_PILL)
    debug["ult_pill"] = {"cream": round(pill_cream, 3), "dark": round(pill_dark, 3)}
    sp_line = _sp_panel(a)
    cores = _star_cores(a)
    debug.update(sp_line=round(sp_line, 1), star_cores=cores)
    has_sp = sp_line >= SP_LINE_MIN_CONTRAST

    big_q, big_e = badges["Ql"][0], badges["El"][0]
    small_q, small_e = badges["Qs"][0], badges["Es"][0]
    pill = pill_cream >= ULT_PILL_MIN_CREAM and pill_dark >= ULT_PILL_MIN_DARK
    # 选中的按钮会变大，大徽标位置固定；另一个小按钮的徽标字母细，压缩后容易丢，所以只当佐证记进 debug
    if big_q and not big_e and has_sp:
        hud.phase, hud.selected = PHASE_COMMAND, SELECTED_BASIC
        debug["partner_badge"] = small_e
    elif big_e and not big_q and has_sp:
        hud.phase, hud.selected = PHASE_COMMAND, SELECTED_SKILL
        debug["partner_badge"] = small_q
    elif pill and has_sp and not (big_q or big_e):
        hud.phase = PHASE_ULT_PENDING
    elif big_q or big_e or small_q or small_e:
        hud.phase = None  # 按钮在变（切换动画）或只认出一部分：不确定
    elif not has_sp or _status_box(a):
        hud.phase = PHASE_OTHER  # 没有技能点面板（飞霄专属模式、过场），或有「行动中」状态框
    else:
        hud.phase = None  # 技能点面板在、按钮和状态框都没认出：可能是按钮正在滑入，不下结论

    settled = hud.phase in (PHASE_COMMAND, PHASE_ULT_PENDING)
    hud.speed_on = _switch_state(speed_c, settled)
    hud.auto_on = _switch_state(auto_c, settled)

    # ---- 终结技圆盘
    slots = []
    for k in range(team_size):
        stats = _ult_disc_stats(a, k)
        slots.append(SlotHud(index=k + 1, ult_ready=_ult_ready(stats)))
        debug.setdefault("ult_discs", []).append(stats)
    hud.slots = slots

    # ---- 需要 OCR 的几块：技能点数字、波次、剩余轮次
    crops = []
    if has_sp and hud.phase in (PHASE_COMMAND, PHASE_ULT_PENDING):
        left, top, right, bottom = SP_DIGIT_BOX
        crops.append(("sp", a[top:bottom, left:right]))
    left, top, right, bottom = WAVE_BOX
    crops.append(("wave", a[top:bottom, left:right]))
    hourglass, hourglass_score = _find_hourglass(a)
    debug["hourglass"] = {"pos": hourglass, "score": hourglass_score}
    if hourglass:
        hx, hy = hourglass
        left, top, right, bottom = ROUNDS_BOX_OFFSET
        crops.append(("rounds", a[hy + top:hy + bottom, hx + left:hx + right]))
    try:
        texts = _ocr_texts(ocr or default_ocr, crops)
    except Exception as e:  # OCR 出错只影响数字和文字类字段，像素判断的结果照常返回
        texts = {}
        debug["ocr_error"] = f"{type(e).__name__}: {e}"
    debug["ocr"] = {name: [text, round(conf, 3)] for name, (text, conf) in texts.items()}

    if "wave" in texts and texts["wave"][1] >= 0.8:
        hud.wave = _parse_wave(texts["wave"][0])
    if "rounds" in texts and texts["rounds"][1] >= 0.8:
        hud.rounds_left = _parse_rounds(texts["rounds"][0])
    if "sp" in texts and texts["sp"][1] >= SP_DIGIT_MIN_CONF:
        digit = _parse_digit(texts["sp"][0])
        lit = _lit_prefix(cores)
        debug["sp_lit"] = lit
        if digit is not None and _sp_consistent(digit, lit, hud.selected):
            hud.skill_points = digit

    debug["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return hud


def _read_result(a: np.ndarray, hud: BattleHud, ocr: Optional[Callable]):
    """结算页：背景很暗、标题区有米黄色或红色文字时，对标题区 OCR 一次。"""
    left, top, right, bottom = RESULT_TITLE_BOX
    region = a[top:bottom, left:right]
    brightness = float(a[::16, ::16].mean())
    pixels = region.reshape(-1, 3)
    cream = float(_cream_mask(pixels).mean())
    red = float(_red_mask(pixels).mean())
    hud.debug["result_gate"] = {"brightness": round(brightness, 1), "cream": round(cream, 4), "red": round(red, 4)}
    if brightness > RESULT_MAX_BRIGHTNESS or (cream < RESULT_MIN_CREAM and red < RESULT_MIN_RED):
        return
    try:
        texts = _ocr_texts(ocr or default_ocr, [("title", region)])
    except Exception as e:
        hud.debug["ocr_error"] = f"{type(e).__name__}: {e}"
        return
    text = texts.get("title", ("", 0.0))
    hud.debug["ocr"] = {"title": [text[0], round(text[1], 3)]}
    for word in RESULT_WORDS:
        if word in text[0]:
            hud.phase = PHASE_RESULT
            hud.result_text = word
            return


# ---------------------------------------------------------------- 给模型看的图 ----

def render_hud_panel(image: Image.Image) -> bytes:
    """1280×720 的干净画面（不画框），下面拼一条放大的 HUD 局部，JPEG 字节。

    局部都是原图 1920×1080 上的裁剪，不放大插值之外不做任何处理：左边是行动顺序竖列（波次、沙漏），
    右边上面是角色栏（血条、终结技圆盘），下面是 Q/E 技能按钮、技能点和右上角的倍速 / 自动 / 暂停。
    """
    full = image.convert("RGB")
    if full.size != REF_SIZE:
        full = full.resize(REF_SIZE, Image.Resampling.LANCZOS)
    queue = full.crop(PANEL_QUEUE_BOX)
    queue = queue.resize((round(queue.width * PANEL_QUEUE_SCALE), round(queue.height * PANEL_QUEUE_SCALE)),
                         Image.Resampling.LANCZOS)
    team = full.crop(PANEL_TEAM_BOX)
    skill = full.crop(PANEL_SKILL_BOX)
    top_right = full.crop(PANEL_TOP_RIGHT_BOX)

    gap = 8
    strip_height = max(queue.height, team.height + gap + skill.height)
    strip = Image.new("RGB", (PANEL_SIZE[0], strip_height), (20, 20, 20))
    strip.paste(queue, (0, 0))
    strip.paste(team, (queue.width + gap, 0))
    strip.paste(skill, (queue.width + gap, team.height + gap))
    if queue.width + gap + skill.width + gap + top_right.width <= PANEL_SIZE[0]:
        strip.paste(top_right, (queue.width + gap + skill.width + gap, team.height + gap))

    canvas = Image.new("RGB", (PANEL_SIZE[0], PANEL_SIZE[1] + strip_height))
    canvas.paste(full.resize(PANEL_SIZE, Image.Resampling.LANCZOS), (0, 0))
    canvas.paste(strip, (0, PANEL_SIZE[1]))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=PANEL_JPEG_QUALITY)
    return buffer.getvalue()


# ---------------------------------------------------------------- 文字 ----

def describe_hud(hud: BattleHud) -> Optional[str]:
    """给模型看的一行「战斗状态：…」；不在战斗中（也不是结算页）返回 None。"""
    def yes_no(value):
        return "不确定" if value is None else ("开" if value else "关")

    if hud.phase == PHASE_RESULT:
        return f"战斗状态：结算页（{hud.result_text or '未知'}）"
    if not hud.in_battle:
        return None
    phase = {PHASE_COMMAND: "轮到我方出手", PHASE_ULT_PENDING: "终结技待释放（按空格释放）",
             PHASE_OTHER: "动画或特殊模式（没有可选的动作）"}.get(hud.phase, "不确定")
    parts = [phase]
    if hud.phase == PHASE_COMMAND:
        parts.append("选中：" + {SELECTED_BASIC: "普攻", SELECTED_SKILL: "战技"}.get(hud.selected, "不确定"))
    if hud.phase in (PHASE_COMMAND, PHASE_ULT_PENDING) or hud.skill_points is not None:
        parts.append("技能点 " + ("不确定" if hud.skill_points is None else str(hud.skill_points)))
    if hud.slots:
        ready = [slot for slot in hud.slots if slot.ult_ready]
        unknown = [slot for slot in hud.slots if slot.ult_ready is None]
        if ready:
            text = "、".join(f"{slot.index}号" for slot in ready)
            if unknown:
                text += "（其余不确定）"
        elif unknown:
            text = "不确定"
        else:
            text = "无"
        parts.append("终结技就绪：" + text)
    parts.append("剩余轮次 " + ("不确定" if hud.rounds_left is None else str(hud.rounds_left)))
    parts.append("波次 " + ("不确定" if hud.wave is None else f"{hud.wave[0]}/{hud.wave[1]}"))
    parts.append("自动：" + yes_no(hud.auto_on))
    parts.append("倍速：" + yes_no(hud.speed_on))
    return "战斗状态：" + "｜".join(parts)


def format_hud_details(hud: BattleHud) -> str:
    """battle_state 在「战斗状态」那行后面补的几行：各位置终结技、读不准的提醒。"""
    lines = []
    if hud.slots:
        names = {True: "可放", False: "未满", None: "不确定"}
        lines.append("各位置终结技：" + "；".join(f"{slot.index}号 {names[slot.ult_ready]}" for slot in hud.slots))
    if hud.phase == PHASE_COMMAND:
        if hud.skill_points is None:
            lines.append("技能点没读准（数字和星星对不上，或者数字太淡）：看下方放大图右下角。")
        if hud.selected is None:
            lines.append("选中的是普攻还是战技没认出来（按钮可能正在切换）：隔一会再读一次。")
    elif hud.phase == PHASE_ULT_PENDING:
        lines.append("终结技已经进入待释放：按空格释放。")
    elif hud.phase is None:
        lines.append("阶段不确定（动画或按钮正在切换）：隔一会再读一次。")
    lines.append("图：上面是干净的 1280×720 画面，下面是放大的 HUD（左：行动顺序和沙漏，上：角色栏和终结技圆盘，下：技能按钮、技能点、倍速自动暂停）。")
    return "\n".join(lines)
