# coding:utf-8
"""switch_account：识别函数、点击安全、整条切换流程（用一个按点击切换画面的假游戏）。

全部 mock：auto、screen、game 窗口、runs。不需要游戏，也不需要 OCR（元素由假游戏直接给出）。
账号、UID、手机号都是虚构的（111111111 / 账号甲 / 111******11 …）。
"""
import glob
import json
import os
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from module.llm_agent import account_switch as sw
from module.llm_agent import guard as gd
from module.llm_agent import mcp_tools as mt
from module.llm_agent.observe import Element, Observation

UID_A, UID_B, UID_C = "111111111", "222222222", "333333333"
MASK_A, MASK_B, MASK_C = "111******11", "222******22", "333******33"
TEMPLATE = os.path.join(sw.ROOT, sw.POWER_TEMPLATE)

# 不管什么情况都不能被点到的目标（假游戏里这些区域被点中就记下这个名字）
FORBIDDEN_NAMES = {"quit_game", "other_account", "delete_row1", "delete_row2", "update", "clear_row", "close_box", "stray"}


# ---------------------------------------------------------------- 假游戏画面 ----

def elem(index, text, cx, cy, width=None, height=24):
    width = width or max(40, len(text) * 22)
    return Element(index, text, (cx - width // 2, cy - height // 2, cx + width // 2, cy + height // 2))


def build_obs(texts, image=None):
    """texts: [(文字, 中心 x, 中心 y)]，1920×1080。"""
    elements = [elem(i + 1, text, cx, cy) for i, (text, cx, cy) in enumerate(texts)]
    return Observation(image=image or Image.new("RGB", (1920, 1080), (20, 24, 40)), elements=elements,
                       screenshot_pos=(0, 0, 1920, 1080), scale=1.0)


RIGHT_STRIP_LOGGED = [("公告", 1829, 116), ("设置", 1829, 208), ("登出", 1829, 304), ("退出", 1828, 401)]
RIGHT_STRIP_BOX = [("公告", 1829, 116), ("登入", 1828, 208), ("退出", 1828, 305)]
LOGIN_PAGE = RIGHT_STRIP_LOGGED + [("开始游戏", 969, 838), ("星穹列车", 980, 911)]
BACK_DIALOG = [("提示", 960, 413), ("是否返回登录页面？", 956, 526), ("取消", 769, 671), ("确认", 1187, 671)]
LOGOUT_DIALOG = (LOGIN_PAGE + [("退出登录", 960, 411), ("退出并清除登录记录", 802, 476), ("下次在该设备需要重新登录当前账号", 861, 511),
                               ("退出并保留登录记录", 811, 568), ("取消", 812, 679), ("退出", 1088, 678)])
CLICK_ENTER = [("更新", 1829, 208), ("设置", 1829, 304), ("登出", 1829, 402), ("退出", 1828, 496), ("点击进入", 960, 1014)]


def box_texts(mask_text):
    return RIGHT_STRIP_BOX + [("miHoYo", 959, 398), (mask_text, 842, 495), ("进入游戏", 961, 626), ("登录其他账号", 960, 703)]


def dropdown_texts(header, rows):
    out = RIGHT_STRIP_BOX + [("miHoYo", 959, 398), (header, 842, 495)]
    for index, row in enumerate(rows):
        out += [(row, 842, 563 + 88 * index), ("刚刚", 823, 603 + 88 * index)]
    return out + [("登录其他账号", 960, 750 + 88 * max(0, len(rows) - 2))]


def menu_image():
    image = Image.new("RGB", (1920, 1080), (20, 24, 40))
    image.paste(Image.open(TEMPLATE).convert("RGB"), (1846, 954))
    return image


def radio_image(base, keep, clear):
    """在「退出并保留登录记录」「退出并清除登录记录」文字左边画单选圈。state: True 橙色填充 / False 空圈 / None 一小块橙色（判断不了）。"""
    draw = ImageDraw.Draw(base)
    for center_y, state in ((568, keep), (476, clear)):
        left = 811 - 99  # 假文字框的左边（宽 198）
        cx = left - 18 if center_y == 568 else 802 - 99 - 18
        if state is True:
            draw.ellipse((cx - 11, center_y - 11, cx + 11, center_y + 11), fill=(249, 209, 139))
        elif state is None:
            draw.ellipse((cx - 3, center_y - 3, cx + 3, center_y + 3), fill=(249, 209, 139))
        else:
            draw.ellipse((cx - 11, center_y - 11, cx + 11, center_y + 11), outline=(200, 200, 230), width=2)
    return base


def inside(point, rect):
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


class World:
    """一个按点击切换画面的假游戏。点击区域按 1920×1080 真实布局画；点中不该点的地方会记下对应的名字。"""

    # 画面 → [(名字, 区域, 点中后的处理方法名)]
    CLICKABLE = {
        "menu": [("power", (1840, 950, 1895, 1005), "_power")],
        "back": [("back.confirm", (1000, 640, 1340, 700), "_confirm"), ("back.cancel", (580, 640, 930, 700), "_cancel_back")],
        "login_page": [("login.logout", (1800, 240, 1860, 330), "_logout"), ("quit_game", (1800, 370, 1860, 430), "_quit"),
                       ("login.start", (880, 810, 1060, 860), "_start")],
        "logout_dialog": [("logout.keep", (700, 545, 1000, 592), "_pick_keep"), ("clear_row", (700, 455, 1000, 498), "_pick_clear"),
                          ("logout.cancel", (690, 650, 940, 710), "_cancel_logout"), ("logout.exit", (960, 650, 1220, 710), "_exit")],
        "login_box": [("box.arrow", (1100, 470, 1230, 520), "_open_list"), ("box.enter", (690, 600, 1230, 655), "_enter"),
                      ("other_account", (880, 685, 1040, 725), "_other"), ("close_box", (1240, 340, 1285, 380), "_close_box"),
                      ("quit_game", (1800, 280, 1860, 330), "_quit")],
        "dropdown": [("delete_row1", (1180, 550, 1220, 620), "_delete"), ("delete_row2", (1180, 640, 1220, 700), "_delete"),
                     ("list.row1", (700, 540, 1100, 620), "_pick_row0"), ("list.row2", (700, 628, 1100, 708), "_pick_row1"),
                     ("other_account", (880, 730, 1040, 770), "_other")],
        "click_enter": [("click_enter", (700, 990, 1220, 1040), "_click_enter"), ("update", (1800, 190, 1860, 230), "_update"),
                        ("quit_game", (1800, 470, 1860, 520), "_quit")],
    }
    # 画面 → (观察多少次后自动变成的画面)：模拟加载
    AUTO = {"warning": ("login_page", 3), "loading": ("click_enter", 3), "entering": ("main", 2)}

    def __init__(self, uid=UID_A, mask=MASK_A, others=((UID_B, MASK_B),), *, keep_state=True, clear_state=False,
                 radio_fixed=None, first_login_mask=None, click_enter_to="main", final_uid=None, stuck=None,
                 dropdown_rows=None, ocr_box_mask=None, dropdown_anim=0, box_first_look_mask=None, no_logout_looks=0):
        self.state = "main"
        self.uid, self.mask = uid, mask
        self.accounts = {uid: mask}
        self.accounts.update(dict(others))
        self.keep, self.clear = keep_state, clear_state
        self.radio_fixed = radio_fixed  # 点保留那一行也改变不了单选圈（模拟认不出）
        self.first_login_mask = first_login_mask  # 退出登录后登录框下拉里的号码（默认就是刚退出的号）
        self.click_enter_to = click_enter_to  # 点击进入之后的画面：main 或 monthly
        self.final_uid = final_uid  # 回到主界面左下角显示的 UID（默认是登录框选中的号）；"" 表示读不到
        self.stuck = stuck  # 卡在这个画面不再自动前进
        self.dropdown_rows = dropdown_rows  # 下拉列表里显示的号码（默认是所有已登录的号）
        self.ocr_box_mask = ocr_box_mask  # OCR 把号码认成的文字（默认原样）
        self.dropdown_anim = dropdown_anim  # 展开动画：前几次观察只画出第一行
        self.box_first_look_mask = box_first_look_mask  # 登录框第一次观察时号码是这个（动画中间帧）
        self.no_logout_looks = no_logout_looks  # 登录页前几次观察还没画出「登出」
        self.box_mask = None
        self.selected_uid = None
        self.cleared_login = False
        self.count = 0
        self.clicks = []  # (画面, 名字, 点位)
        self.pending = None
        self.change_calls = []

    # ---- 状态 ----
    def go(self, state):
        self.state, self.count = state, 0

    def tick(self):
        self.count += 1
        target = self.AUTO.get(self.state)
        if target and self.stuck != self.state and self.count > target[1]:
            self.go(target[0])

    def box_label(self, mask):
        return self.ocr_box_mask or mask

    def rows(self):
        return self.dropdown_rows or list(self.accounts.values())

    # ---- 观察 ----
    def observation(self):
        self.tick()
        state = self.state
        if state == "main":
            uid = self.uid if self.final_uid is None else self.final_uid
            return build_obs(([(f"UID:{uid}", 97, 1052)] if uid else []) + [("任务", 1601, 926)])
        if state == "menu":
            return build_obs([("跃迁", 1601, 667), ("商店", 1323, 408), (f"UID:{self.uid}", 97, 1052)], menu_image())
        if state == "back":
            return build_obs(BACK_DIALOG + [(f"UID:{self.uid}", 97, 1051)])
        if state == "warning":
            return build_obs([("警告：游戏前详阅", 960, 426)])
        if state == "login_page":
            return build_obs([item for item in LOGIN_PAGE if not (item[0] == "登出" and self.count <= self.no_logout_looks)])
        if state == "logout_dialog":
            image = radio_image(Image.new("RGB", (1920, 1080), (20, 24, 40)), self.keep, self.clear)
            return build_obs(LOGOUT_DIALOG, image)
        if state == "login_box":
            mask = self.box_first_look_mask if self.count == 1 and self.box_first_look_mask else self.box_mask
            return build_obs(box_texts(self.box_label(mask)))
        if state == "dropdown":
            rows = self.rows()[:1] if self.count <= self.dropdown_anim else self.rows()
            return build_obs(dropdown_texts(self.box_label(self.box_mask), rows))
        if state == "loading":
            return build_obs([("数值加载中（43.64%）", 348, 947)])
        if state == "click_enter":
            return build_obs(CLICK_ENTER)
        if state == "monthly":
            return build_obs([("月卡", 960, 500)])
        return build_obs([])  # entering：黑屏

    def recognized_screen(self):
        self.tick()
        return {"main": "main", "menu": "menu", "monthly": "monthly_card"}.get(self.state)

    # ---- 点击 ----
    def click(self, point):
        for name, rect, handler in self.CLICKABLE.get(self.state, []):
            if inside(point, rect):
                self.clicks.append((self.state, name, point))
                getattr(self, handler)()
                return
        self.clicks.append((self.state, "stray", point))

    def names(self):
        return [name for _, name, _ in self.clicks]

    def _power(self):
        self.go("back")

    def _confirm(self):
        self.go("warning")

    def _cancel_back(self):
        self.go("menu")

    def _logout(self):
        self.go("logout_dialog")

    def _quit(self):
        pass

    def _start(self):
        self.go("loading")

    def _pick_keep(self):
        if self.radio_fixed is None:
            self.keep, self.clear = True, False

    def _pick_clear(self):
        self.keep, self.clear = False, True

    def _cancel_logout(self):
        self.go("login_page")

    def _exit(self):
        self.cleared_login = bool(self.clear)
        self.box_mask = self.first_login_mask or self.accounts[self.uid]
        self.go("login_box")

    def _open_list(self):
        self.go("dropdown")

    def _enter(self):
        self.selected_uid = next(uid for uid, mask in self.accounts.items() if mask == self.box_mask)
        self.go("login_page")

    def _other(self):
        pass

    def _close_box(self):
        pass

    def _delete(self):
        pass

    def _pick_row(self, index):
        canonical = list(self.accounts.values())  # 列表里显示的可能是 OCR 认错的写法，选中后登录框里是真实号码
        self.box_mask = canonical[index] if index < len(canonical) else self.rows()[index]
        self.go("login_box")

    def _pick_row0(self):
        self._pick_row(0)

    def _pick_row1(self):
        self._pick_row(1)

    def _click_enter(self):
        self.uid = self.selected_uid
        self.go("entering" if self.click_enter_to == "main" else "monthly")

    def _update(self):
        pass


class FakeAuto:
    def __init__(self, world):
        self.world = world
        self.calls = []
        self.input_handler = SimpleNamespace(is_background=False)

    def calculate_text_position(self, box, relative):
        return tuple(box[0]), tuple(box[2])

    def click_element_with_pos(self, coordinates, offset=(0, 0), action="click", cnt=1, press_duration=0.0):
        self.calls.append(("click", coordinates, action))
        if action == "click":
            self.world.click(coordinates[0])
        return True

    def press_key(self, key, wait_time=0.2):
        self.calls.append(("press_key", key))
        return True


class FakeScreen:
    def __init__(self, world):
        self.world = world
        self.current_screen = None
        self.names = {"main": "主界面", "menu": "手机菜单", "monthly_card": "月卡"}

    def get_current_screen(self, autotry=True, max_retries=10):
        assert autotry is False
        self.current_screen = self.world.recognized_screen()
        return bool(self.current_screen)

    def get_name(self, screen_id):
        return self.names.get(screen_id, screen_id)

    def change_to(self, target, max_recursion=2):
        self.world.change_calls.append(target)
        if target == "menu" and self.world.state == "main":
            self.world.go("menu")
        elif target == "main":
            self.world.go("main")


class FakeGame:
    def is_running(self):
        return True

    def is_front(self):
        return True

    def bring_front(self):
        return True

    def resolution(self):
        return (1920, 1080)


class FakeRuns:
    def __init__(self):
        self.active = None

    def read_active_run(self):
        return self.active

    def describe_run(self, info):
        return f"任务 {info.get('task')}（运行编号 {info.get('run_id')}）"


DEFAULT_ACCOUNTS = {
    UID_A: {"name": "账号甲", "note": "", "login": MASK_A},
    UID_B: {"name": "账号乙", "note": "", "login": MASK_B},
}


class Env:
    def __init__(self, tmp_path, world=None, accounts=DEFAULT_ACCOUNTS, offline_image=None):
        self.tmp_path = str(tmp_path)
        self.world = world or World()
        self.auto = FakeAuto(self.world)
        self.runs = FakeRuns()
        self.now = 1000.0
        self.sleeps = []
        env = self

        def clock():
            return env.now

        def sleep(seconds):
            env.sleeps.append(seconds)
            env.now += seconds

        base = os.path.join(self.tmp_path, "rec")
        if accounts is not None:
            os.makedirs(base, exist_ok=True)
            with open(os.path.join(base, "accounts.json"), "w", encoding="utf-8") as file:
                json.dump(accounts, file, ensure_ascii=False)
        self.tools = mt.McpTools(
            offline_image=offline_image, root=self.tmp_path, record_base=base, auto=self.auto,
            screen=FakeScreen(self.world), game=FakeGame(), observer=lambda grid: self.world.observation(),
            guard=gd.Guard((), mt.MCP_ALLOWED_KEYS), runs=self.runs,
            config_get=lambda key, default=None: {"use_background_screenshot": True, "cloud_game_enable": False}.get(key, default),
            sleep=sleep, clock=clock, settle_seconds=0)

    def records(self):
        pattern = os.path.join(self.tmp_path, "rec", "*", "*-switch_account.json")
        return [json.load(open(path, encoding="utf-8")) for path in sorted(glob.glob(pattern))]


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


HAPPY_CLICKS = ["power", "back.confirm", "login.logout", "logout.exit", "box.arrow", "list.row2", "box.enter",
                "login.start", "click_enter"]


# ---------------------------------------------------------------- 流程 ----

class TestFlow:
    def test_happy_path_click_order(self, env):
        out = env.tools.switch_account("账号乙")
        assert env.world.names() == HAPPY_CLICKS
        assert env.world.state == "main" and env.world.uid == UID_B
        assert "已切换到「账号乙」" in out.text and out.image[:3] == b"\xff\xd8\xff"
        assert "当前账号已从「账号甲（UID 111111111）」切换为「账号乙（UID 222222222）」" in out.text
        assert env.tools.tracker.uid == UID_B and env.tools.tracker.confirmed
        assert not env.world.cleared_login and env.world.keep and not env.world.clear
        assert env.world.change_calls == ["menu"]

    def test_only_power_and_logout_are_clicked_in_right_strip(self, env):
        env.tools.switch_account("账号乙")
        strip = [name for _, name, point in env.world.clicks if point[0] >= 1536]
        assert strip == ["power", "login.logout"]

    def test_no_forbidden_target_is_ever_clicked(self, env):
        env.tools.switch_account("账号乙")
        assert not FORBIDDEN_NAMES & set(env.world.names())

    def test_row_click_stays_left_of_delete_icon(self, env):
        env.tools.switch_account("账号乙")
        row = next(point for _, name, point in env.world.clicks if name == "list.row2")
        assert row[0] < 1920 * sw.LIST_CLICK_X_MAX

    def test_selects_by_uid_too(self, env):
        env.tools.switch_account(UID_B)
        assert env.world.names() == HAPPY_CLICKS

    def test_start_from_phone_menu_does_not_jump(self, tmp_path):
        env = Env(tmp_path, World())
        env.world.go("menu")
        env.tools.switch_account("账号乙")
        assert env.world.change_calls == [] and env.world.names() == HAPPY_CLICKS

    def test_login_box_already_shows_target_skips_dropdown(self, tmp_path):
        env = Env(tmp_path, World(first_login_mask=MASK_B))
        out = env.tools.switch_account("账号乙")
        assert env.world.names() == [name for name in HAPPY_CLICKS if name not in ("box.arrow", "list.row2")]
        assert "已切换到「账号乙」" in out.text

    def test_ocr_misreads_mask_stars_and_a_digit(self, tmp_path):
        # OCR 把星号认成 5 个、把开头的 222 认成 22：只比较开头和结尾的数字，仍然认得出
        env = Env(tmp_path, World(dropdown_rows=["111*****11", "22*****22"]))
        out = env.tools.switch_account("账号乙")
        assert env.world.names() == HAPPY_CLICKS and "已切换到" in out.text

    def test_phone_menu_shows_already_target_closes_menu(self, tmp_path):
        # 主界面左下角没读到 UID，进了手机菜单才发现已经是目标账号：只按 esc 关掉菜单，不退出登录
        env = Env(tmp_path, World(uid=UID_B, mask=MASK_B, others=((UID_A, MASK_A),), final_uid=""))
        out = env.tools.switch_account("账号乙")
        assert env.world.clicks == [] and env.auto.calls == [("press_key", "esc")]
        assert "当前已经是「账号乙」" in out.text

    def test_monthly_card_popup_goes_back_through_screen_graph(self, tmp_path):
        env = Env(tmp_path, World(click_enter_to="monthly"))
        out = env.tools.switch_account("账号乙")
        assert env.world.change_calls == ["menu", "main"] and "已切换到" in out.text
        assert env.world.names() == HAPPY_CLICKS

    def test_record_has_each_step_with_result_and_seconds(self, env):
        env.tools.switch_account("账号乙")
        records = env.records()
        assert len(records) == 1
        record = records[0]
        assert record["tool"] == "switch_account" and record["result"]["ok"] is True
        steps = record["steps"]
        assert [step["step"] for step in steps][0] == "确认当前画面和账号" and steps[-1]["step"] == "确认账号"
        assert all(step["ok"] and "seconds" in step for step in steps)
        assert any(step["step"] == "点登出" for step in steps)
        assert record["image"].endswith(".jpg")


class TestAnimation:
    def test_dropdown_first_look_has_only_one_row(self, tmp_path):
        # 实机现象：展开动画中第一次观察只画出第一行，不能拿来判断目标不在列表里
        env = Env(tmp_path, World(dropdown_anim=1))
        out = env.tools.switch_account("账号乙")
        assert env.world.names() == HAPPY_CLICKS and "已切换到「账号乙」" in out.text

    def test_dropdown_animation_lasting_two_looks(self, tmp_path):
        env = Env(tmp_path, World(dropdown_anim=2))
        env.tools.switch_account("账号乙")
        assert env.world.names() == HAPPY_CLICKS

    def test_target_really_absent_is_confirmed_by_extra_look(self, tmp_path):
        env = Env(tmp_path, World(dropdown_rows=[MASK_A, MASK_C]))
        out = env.tools.switch_account("账号乙")
        assert env.world.names()[-1] == "box.arrow" and "账号列表里没有目标号码" in out.text
        assert env.world.count >= 3  # 稳定的两次之后又看了一次才下结论

    def test_box_misread_in_first_frame_does_not_skip_dropdown(self, tmp_path):
        # 登录框第一帧号码显示成目标（动画中间帧），下一帧才是真正的号码：不能因此跳过选号
        env = Env(tmp_path, World(box_first_look_mask=MASK_B))
        env.tools.switch_account("账号乙")
        assert env.world.names() == HAPPY_CLICKS

    def test_dropdown_that_never_settles_fails(self, tmp_path):
        world = World()
        env = Env(tmp_path, world)
        base = world.observation
        flip = {"n": 0}

        def flicker():
            obs = base()
            if world.state == "dropdown":
                flip["n"] += 1
                rows = [MASK_A, MASK_B, MASK_C][: 1 + flip["n"] % 3]
                obs = build_obs(dropdown_texts(MASK_A, rows))
            return obs

        env.tools._observer = lambda grid: flicker()
        out = env.tools.switch_account("账号乙")
        assert "list.row1" not in world.names() and "list.row2" not in world.names() and "稳定下来" in out.text


class TestResume:
    """从中途接着走：按当前画面直接跳到对应的那一步，之后线性往下走（不回头、不兜圈）。"""

    def _start(self, tmp_path, state, **world_kwargs):
        env = Env(tmp_path, World(**world_kwargs))
        env.world.go(state)
        return env

    def test_from_expanded_dropdown(self, tmp_path):
        env = self._start(tmp_path, "dropdown", dropdown_anim=1)
        env.world.box_mask = MASK_A
        out = env.tools.switch_account("账号乙")
        assert env.world.names() == ["list.row2", "box.enter", "login.start", "click_enter"]
        assert "已切换到「账号乙」" in out.text and env.tools.tracker.uid == UID_B

    def test_from_login_box_other_account(self, tmp_path):
        env = self._start(tmp_path, "login_box")
        env.world.box_mask = MASK_A
        env.tools.switch_account("账号乙")
        assert env.world.names() == ["box.arrow", "list.row2", "box.enter", "login.start", "click_enter"]

    def test_from_login_box_already_target(self, tmp_path):
        env = self._start(tmp_path, "login_box")
        env.world.box_mask = MASK_B
        env.tools.switch_account("账号乙")
        assert env.world.names() == ["box.enter", "login.start", "click_enter"]

    def test_from_login_page_goes_through_logout_again(self, tmp_path):
        # 登录页可能是还没登出，也可能是选好号回来的；分不出就从登出走，重新选号，只走一遍
        env = self._start(tmp_path, "login_page")
        out = env.tools.switch_account("账号乙")
        assert env.world.names() == ["login.logout", "logout.exit", "box.arrow", "list.row2", "box.enter", "login.start", "click_enter"]
        assert "已切换到" in out.text and "quit_game" not in env.world.names()

    def test_from_login_page_without_logout_icon_waits_for_it(self, tmp_path):
        env = self._start(tmp_path, "login_page", no_logout_looks=3)
        env.tools.switch_account("账号乙")
        assert env.world.names()[0] == "login.logout" and env.world.names()[-1] == "click_enter"

    def test_from_logout_dialog(self, tmp_path):
        env = self._start(tmp_path, "logout_dialog")
        env.tools.switch_account("账号乙")
        assert env.world.names() == ["logout.exit", "box.arrow", "list.row2", "box.enter", "login.start", "click_enter"]

    def test_from_logout_dialog_still_checks_radio(self, tmp_path):
        env = self._start(tmp_path, "logout_dialog", keep_state=None, radio_fixed=True)
        out = env.tools.switch_account("账号乙")
        assert env.world.names() == ["logout.keep", "logout.cancel"] and "已点「取消」中止" in out.text

    def test_from_back_dialog(self, tmp_path):
        env = self._start(tmp_path, "back")
        env.tools.switch_account("账号乙")
        assert env.world.names() == HAPPY_CLICKS[1:]

    def test_from_click_enter_on_target(self, tmp_path):
        env = self._start(tmp_path, "click_enter")
        env.world.selected_uid = UID_B
        out = env.tools.switch_account("账号乙")
        assert env.world.names() == ["click_enter"] and "已切换到" in out.text and env.tools.tracker.uid == UID_B

    def test_from_click_enter_on_wrong_account_fails_without_switching_again(self, tmp_path):
        env = self._start(tmp_path, "click_enter")
        env.world.selected_uid = UID_A
        out = env.tools.switch_account("账号乙")
        assert env.world.names() == ["click_enter"]  # 不会自动再切一轮
        assert "【切换账号失败】停在「确认账号」" in out.text and "不是目标账号" in out.text

    def test_resume_never_clicks_forbidden_targets(self, tmp_path):
        for state in ("login_page", "logout_dialog", "login_box", "dropdown", "click_enter", "back"):
            env = self._start(tmp_path / state, state)
            env.world.box_mask = MASK_A
            env.world.selected_uid = UID_B
            env.tools.switch_account("账号乙")
            assert not FORBIDDEN_NAMES & set(env.world.names()), state


class TestAlreadyCurrent:
    def test_target_is_current_does_nothing(self, env):
        out = env.tools.switch_account("账号甲")
        assert "当前已经是「账号甲」" in out.text
        assert env.world.clicks == [] and env.auto.calls == [] and env.world.change_calls == []
        assert env.records()[0]["result"]["message"] == "已经是这个号"

    def test_stale_target_cache_without_visible_uid_does_not_skip_switch(self, tmp_path):
        env = Env(tmp_path, World(final_uid=""))
        env.tools.tracker.update(build_obs([(f"UID:{UID_B}", 97, 1052)]))
        out = env.tools.switch_account("账号乙")
        assert env.world.names() == HAPPY_CLICKS
        assert "当前已经是「账号乙」" not in out.text
        assert env.records()[0]["result"]["ok"] is False

    def test_visible_target_overrides_stale_other_account_cache(self, env):
        env.tools.tracker.update(build_obs([(f"UID:{UID_B}", 97, 1052)]))
        out = env.tools.switch_account("账号甲")
        assert "当前已经是「账号甲」" in out.text
        assert env.world.clicks == [] and env.auto.calls == []


class TestUidVerification:
    @pytest.mark.parametrize("final_uid", ["", UID_A, UID_C])
    def test_resume_does_not_confirm_using_stale_target_cache(self, tmp_path, final_uid):
        env = Env(tmp_path, World(final_uid=final_uid))
        env.tools.tracker.update(build_obs([(f"UID:{UID_B}", 97, 1052)]))
        env.world.go("click_enter")
        env.world.selected_uid = UID_A
        out = env.tools.switch_account("账号乙")
        assert env.world.names() == ["click_enter"]
        assert "【切换账号失败】停在「确认账号」" in out.text
        assert "已切换到「账号乙」" not in out.text
        assert env.records()[0]["result"]["ok"] is False
        if not final_uid:
            assert env.tools.tracker.confirmed and env.tools.tracker.uid == UID_B
            assert "当前帧左下角：未识别到 UID" in out.text

    @pytest.mark.parametrize("interruption", ["", UID_A])
    def test_confirmation_requires_consecutive_fresh_target_frames(self, env, interruption):
        # 间隔出现一张目标帧会更新 tracker，但不能靠缓存把下一张空白/其他账号帧当成确认。
        reads = iter([UID_B, interruption, UID_B, interruption])

        def observe(grid):
            uid = next(reads)
            return build_obs([(f"UID:{uid}", 97, 1052)] if uid else [])

        env.tools._observer = observe
        switcher = sw.AccountSwitcher(env.tools, sw.Account(UID_B, "账号乙", MASK_B))
        with pytest.raises(sw.StepFailed, match="未能连续确认目标 UID"):
            switcher._s_verify(None)
        assert switcher.steps[-1]["ok"] is False

    @pytest.mark.parametrize("final_uid", ["", UID_A])
    def test_final_picture_uid_is_checked_after_successful_verification(self, tmp_path, final_uid):
        env = Env(tmp_path)
        env.world.go("click_enter")
        env.world.selected_uid = UID_B
        observe = env.world.observation
        main_reads = 0

        def final_picture(grid):
            nonlocal main_reads
            obs = observe()
            if env.world.state == "main":
                main_reads += 1
                if main_reads > sw.UID_CONFIRM_LOOKS:
                    return build_obs([(f"UID:{final_uid}", 97, 1052)] if final_uid else [])
            return obs

        env.tools._observer = final_picture
        out = env.tools.switch_account("账号乙")
        assert main_reads == sw.UID_CONFIRM_LOOKS + 1
        assert env.world.names() == ["click_enter"]
        if final_uid:
            assert "最终观察左下角读到的 UID 与目标账号不一致" in out.text
            assert "已切换到「账号乙」" not in out.text
            assert env.records()[0]["result"]["ok"] is False
            assert "最终观察左下角读到的 UID 与目标账号不一致" in env.records()[0]["outcome"]
        else:
            assert "已切换到「账号乙」" in out.text
            assert "最终附图未识别到 UID" in out.text
            assert env.records()[0]["result"]["ok"] is True


class TestRefusals:
    def test_unknown_account(self, env):
        out = env.tools.switch_account("账号丙")
        assert "账号表里没有「账号丙」" in out.text and "账号甲" in out.text
        assert env.auto.calls == [] and env.world.clicks == []

    def test_no_accounts_file(self, tmp_path):
        env = Env(tmp_path, accounts=None)
        out = env.tools.switch_account("账号乙")
        assert "没有账号表" in out.text and "leave_for_user" in out.text and env.auto.calls == []

    def test_target_without_login_field(self, tmp_path):
        env = Env(tmp_path, accounts={UID_A: {"name": "账号甲", "login": MASK_A}, UID_B: {"name": "账号乙", "note": ""}})
        out = env.tools.switch_account("账号乙")
        assert "没有 login 字段" in out.text and "leave_for_user" in out.text
        assert env.auto.calls == [] and env.world.clicks == []

    def test_malformed_login_field(self, tmp_path):
        env = Env(tmp_path, accounts={UID_B: {"name": "账号乙", "login": "不是手机号"}})
        out = env.tools.switch_account("账号乙")
        assert "login 字段格式不对" in out.text and env.auto.calls == []

    @pytest.mark.parametrize("value", ["", "  ", None, 3.5, True])
    def test_bad_argument(self, env, value):
        assert "必须是账号名或 UID" in env.tools.switch_account(value).text
        assert env.auto.calls == []

    def test_refused_while_m7a_task_runs(self, env):
        env.runs.active = {"task": "daily", "run_id": "r1"}
        out = env.tools.switch_account("账号乙")
        assert "被拒绝" in out.text and "M7A 任务" in out.text
        assert env.auto.calls == [] and env.world.clicks == []

    def test_busy_when_lock_is_held(self, env, monkeypatch):
        monkeypatch.setattr(mt, "LOCK_TIMEOUT", 0.01)
        with env.tools._lock:
            out = env.tools.switch_account("账号乙")
        assert "【未执行】" in out.text and env.auto.calls == []

    def test_offline_returns_plan_only(self, tmp_path):
        image = tmp_path / "x.png"
        Image.new("RGB", (1920, 1080)).save(image)
        env = Env(tmp_path, offline_image=str(image))
        out = env.tools.switch_account("账号乙")
        assert "只列出计划" in out.text and "电源图标" in out.text and "保留登录记录" in out.text
        assert env.tools._offline_auto.calls == [] and env.world.clicks == []


# ---------------------------------------------------------------- 单选判断 ----

class TestKeepRadio:
    def test_unselected_clicks_keep_row_then_exits(self, tmp_path):
        env = Env(tmp_path, World(keep_state=False))
        env.tools.switch_account("账号乙")
        names = env.world.names()
        assert names[:4] == ["power", "back.confirm", "login.logout", "logout.keep"] and names[4] == "logout.exit"
        assert not env.world.cleared_login

    def test_clear_selected_clicks_keep_row_first(self, tmp_path):
        env = Env(tmp_path, World(keep_state=False, clear_state=True))
        env.tools.switch_account("账号乙")
        assert env.world.names().index("logout.keep") < env.world.names().index("logout.exit")
        assert not env.world.cleared_login and "clear_row" not in env.world.names()

    def test_undecidable_cancels_and_aborts(self, tmp_path):
        env = Env(tmp_path, World(keep_state=None, radio_fixed=True))
        out = env.tools.switch_account("账号乙")
        names = env.world.names()
        assert names == ["power", "back.confirm", "login.logout", "logout.keep", "logout.cancel"]
        assert "logout.exit" not in names and "clear_row" not in names
        assert "【切换账号失败】" in out.text and "已点「取消」中止" in out.text and "确认保留登录记录并退出" in out.text
        assert out.image is not None and env.world.state == "login_page"

    def test_unselected_that_cannot_be_changed_cancels(self, tmp_path):
        env = Env(tmp_path, World(keep_state=False, radio_fixed=True))
        env.tools.switch_account("账号乙")
        assert env.world.names()[-1] == "logout.cancel" and "logout.exit" not in env.world.names()

    def test_radio_selected_function(self):
        text = elem(1, "退出并保留登录记录", 811, 568)
        on = radio_image(Image.new("RGB", (1920, 1080), (20, 24, 40)), True, False)
        off = radio_image(Image.new("RGB", (1920, 1080), (20, 24, 40)), False, False)
        unsure = radio_image(Image.new("RGB", (1920, 1080), (20, 24, 40)), None, False)
        assert sw.radio_selected(on, text) is True
        assert sw.radio_selected(off, text) is False
        assert sw.radio_selected(unsure, text) is None


# ---------------------------------------------------------------- 选号 ----

class TestDropdown:
    def test_target_not_in_list_aborts_without_clicking_other_account(self, tmp_path):
        env = Env(tmp_path, World(dropdown_rows=[MASK_A, MASK_C]))
        out = env.tools.switch_account("账号乙")
        names = env.world.names()
        assert names == ["power", "back.confirm", "login.logout", "logout.exit", "box.arrow"]
        assert not FORBIDDEN_NAMES & set(names)
        assert "账号列表里没有目标号码" in out.text and "leave_for_user" in out.text and "不点「登录其他账号」" in out.text
        assert "停在账号下拉列表" in out.text and env.world.state == "dropdown"
        assert out.image is not None

    def test_two_rows_look_like_target_aborts(self, tmp_path):
        env = Env(tmp_path, World(dropdown_rows=[MASK_B, "222*****22"]))
        out = env.tools.switch_account("账号乙")
        assert "分不清" in out.text and env.world.names()[-1] == "box.arrow"

    def test_selection_that_does_not_stick_stops(self, tmp_path):
        # 点了第二行但登录框显示的还是别的号：停下，不去点「进入游戏」
        world = World()
        env = Env(tmp_path, world)
        world._pick_row = lambda index: (setattr(world, "box_mask", MASK_C), world.go("login_box"))[-1]
        out = env.tools.switch_account("账号乙")
        assert "box.enter" not in world.names() and "没等到" in out.text and "登录框" in out.text

    def test_unreadable_box_mask_stops_before_clicking_arrow(self, tmp_path):
        env = Env(tmp_path, World(ocr_box_mask="账号"))
        out = env.tools.switch_account("账号乙")
        assert env.world.names() == ["power", "back.confirm", "login.logout", "logout.exit"]
        assert "没等到" in out.text and "登录框" in out.text and "停在登录框，需要用户手动选号进入" in out.text


# ---------------------------------------------------------------- 超时和安全 ----

class TestTimeouts:
    def test_step_timeout_stops_without_further_clicks(self, tmp_path):
        env = Env(tmp_path, World(stuck="warning"))
        out = env.tools.switch_account("账号乙")
        assert env.world.names() == ["power", "back.confirm"]
        assert "【切换账号失败】停在「确认返回登录页面」" in out.text and "60 秒内没等到" in out.text
        assert "游戏还在加载" in out.text and out.image is not None
        record = env.records()[0]
        assert record["result"]["ok"] is False and record["stopped_at"] == "确认返回登录页面"
        assert record["steps"][-1]["ok"] is False

    def test_popup_never_appears_after_power_click(self, tmp_path):
        world = World()
        env = Env(tmp_path, world)
        world._power = lambda: None  # 点了没反应
        out = env.tools.switch_account("账号乙")
        assert world.names() == ["power"] and "没等到" in out.text and "弹窗「是否返回登录页面？」" in out.text

    def test_total_time_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sw, "TOTAL_TIMEOUT", 8)
        env = Env(tmp_path, World(stuck="warning"))
        out = env.tools.switch_account("账号乙")
        assert "整个流程已超过 8 秒" in out.text and env.world.names() == ["power", "back.confirm"]

    def test_unknown_screen_at_start_does_not_click(self, tmp_path):
        env = Env(tmp_path, World())
        env.world.go("entering")  # 黑屏：M7A 认不出，也没有任何文字
        env.world.stuck = "entering"
        out = env.tools.switch_account("账号乙")
        assert env.world.clicks == [] and "不是主界面也不是手机菜单" in out.text

    @pytest.mark.parametrize("state", ["warning", "loading"])
    def test_loading_screen_at_start_does_not_click(self, tmp_path, state):
        env = Env(tmp_path, World(stuck=state))
        env.world.go(state)
        out = env.tools.switch_account("账号乙")
        assert env.world.clicks == [] and "游戏正在加载" in out.text

    def test_wrong_uid_at_the_end_fails(self, tmp_path):
        env = Env(tmp_path, World(final_uid=UID_C))
        out = env.tools.switch_account("账号乙")
        assert "【切换账号失败】停在「确认账号」" in out.text and "不是目标账号" in out.text
        assert env.world.names() == HAPPY_CLICKS

    def test_unreadable_uid_at_the_end_fails(self, tmp_path):
        env = Env(tmp_path, World(final_uid=""))
        out = env.tools.switch_account("账号乙")
        assert "停在「确认账号」" in out.text and out.image is not None

    def test_stuck_before_click_enter_page(self, tmp_path):
        env = Env(tmp_path, World(stuck="loading"))
        out = env.tools.switch_account("账号乙")
        assert env.world.names() == HAPPY_CLICKS[:-1] and "停在「点开始游戏」" in out.text


class TestClickSafety:
    def _switcher(self, env, texts):
        switcher = sw.AccountSwitcher(env.tools, sw.Account(UID_B, "账号乙", MASK_B))
        switcher.obs = build_obs(texts)
        return switcher

    @pytest.mark.parametrize("texts,point", [
        (box_texts(MASK_A), (960, 703)),  # 登录其他账号
        (dropdown_texts(MASK_A, [MASK_A, MASK_B]), (960, 750)),  # 列表底部的登录其他账号
        (LOGOUT_DIALOG, (802, 476)),  # 退出并清除登录记录
        (LOGOUT_DIALOG, (1828, 401)),  # 登录页右侧的退出
        (CLICK_ENTER, (1829, 208)),  # 更新
        (CLICK_ENTER, (1828, 496)),
    ])
    def test_forbidden_points_are_refused(self, env, texts, point):
        switcher = self._switcher(env, texts)
        for right_strip in (False, True):
            with pytest.raises(sw.StepFailed):
                switcher._click(sw.Hit(*point), "x", right_strip=right_strip)
        assert env.auto.calls == []

    def test_right_strip_needs_explicit_permission(self, env):
        switcher = self._switcher(env, RIGHT_STRIP_BOX)
        with pytest.raises(sw.StepFailed):
            switcher._click(sw.Hit(1840, 650), "x")  # 退出下面的空白，不在文字上也不行
        assert env.auto.calls == []

    def test_delete_icon_side_is_refused_for_list_rows(self, env):
        switcher = self._switcher(env, dropdown_texts(MASK_A, [MASK_A, MASK_B]))
        with pytest.raises(sw.StepFailed):
            switcher._click(sw.Hit(1198, 651), "行", x_max=sw.LIST_CLICK_X_MAX)
        assert env.auto.calls == []

    def test_logout_click_is_not_near_the_quit_button(self, env):
        switcher = self._switcher(env, LOGIN_PAGE)
        page = sw.find_login_page(switcher.obs)
        assert page.logout is not None and abs(page.logout.y - 401) > 100
        switcher._click(page.logout, "登出", right_strip=True)
        assert len(env.auto.calls) == 1

    def test_ocr_returns_right_strip_quit_but_dialog_quit_is_missing(self, tmp_path):
        # 退出登录弹窗里的「退出」没认出来，只有登录页右侧的「退出」：认不出弹窗，不会去点右侧的退出
        world = World()
        env = Env(tmp_path, world)
        original = world.observation

        def without_dialog_exit():
            obs = original()
            if world.state == "logout_dialog":
                obs.elements = [e for e in obs.elements if not (e.text == "退出" and e.center[0] < 1500)]
            return obs

        env.tools._observer = lambda grid: without_dialog_exit()
        out = env.tools.switch_account("账号乙")
        assert world.names() == ["power", "back.confirm", "login.logout"]
        assert "quit_game" not in world.names() and "没等到" in out.text and "弹窗「退出登录」" in out.text

    def test_dialog_exit_region_excludes_right_strip(self):
        obs = build_obs(LOGOUT_DIALOG)
        dialog = sw.find_logout_dialog(obs)
        assert dialog.exit.point == (1088, 678) and dialog.cancel.point == (812, 679)
        only_strip = build_obs([item for item in LOGOUT_DIALOG if not (item[0] == "退出" and item[1] < 1500)])
        assert sw.find_logout_dialog(only_strip) is None
        # 就算右侧的「退出」恰好和弹窗按钮在同一高度，也不能当成弹窗里的按钮
        same_height = build_obs([item for item in LOGOUT_DIALOG if not (item[0] == "退出" and item[1] < 1500)] + [("退出", 1828, 678)])
        assert sw.find_logout_dialog(same_height) is None

    def test_ensure_front_failure_stops_before_click(self, env):
        env.tools._game = SimpleNamespace(is_front=lambda: False, bring_front=lambda: False, is_running=lambda: True)
        switcher = self._switcher(env, LOGIN_PAGE)
        with pytest.raises(sw.StepFailed):
            switcher._click(sw.Hit(969, 838), "开始游戏")
        assert env.auto.calls == []


# ---------------------------------------------------------------- 识别函数 ----

class TestRecognizers:
    def test_classify_each_screen(self):
        cases = {
            sw.KIND_PHONE_MENU: build_obs([("跃迁", 1601, 667)], menu_image()),
            sw.KIND_BACK_DIALOG: build_obs(BACK_DIALOG),
            sw.KIND_LOGIN_PAGE: build_obs(LOGIN_PAGE),
            sw.KIND_LOGOUT_DIALOG: build_obs(LOGOUT_DIALOG),
            sw.KIND_LOGIN_BOX: build_obs(box_texts(MASK_A)),
            sw.KIND_DROPDOWN: build_obs(dropdown_texts(MASK_A, [MASK_A, MASK_B])),
            sw.KIND_CLICK_ENTER: build_obs(CLICK_ENTER),
            sw.KIND_WARNING: build_obs([("警告：游戏前详阅", 960, 426)]),
            sw.KIND_LOADING: build_obs([("数值加载中（43.64%）", 348, 947)]),
            sw.KIND_UNKNOWN: build_obs([("任务", 1601, 926)]),
        }
        for kind, obs in cases.items():
            assert sw.classify(obs) == kind, kind

    def test_power_icon_found_only_in_phone_menu_corner(self):
        assert sw.find_phone_menu(build_obs([], menu_image())).point == (1868, 978)
        assert sw.find_phone_menu(build_obs([])) is None
        moved = Image.new("RGB", (1920, 1080), (20, 24, 40))
        moved.paste(Image.open(TEMPLATE).convert("RGB"), (400, 300))  # 别处出现同样的图标不算
        assert sw.find_phone_menu(build_obs([], moved)) is None

    def test_login_page_not_reported_under_dialogs(self):
        assert sw.find_login_page(build_obs(LOGIN_PAGE)).logout.point == (1829, 280)
        assert sw.find_login_page(build_obs(LOGOUT_DIALOG)) is None
        assert sw.find_login_page(build_obs(LOGIN_PAGE + BACK_DIALOG)) is None

    def test_login_page_during_loading_has_no_logout(self):
        page = sw.find_login_page(build_obs([item for item in LOGIN_PAGE if item[0] != "登出"]))
        assert page is not None and page.logout is None

    def test_click_enter_ignores_login_page(self):
        assert sw.find_click_enter(build_obs(LOGIN_PAGE)) is None
        assert sw.find_click_enter(build_obs(CLICK_ENTER)).point == (960, 1014)

    def test_login_box_and_dropdown_are_exclusive(self):
        box = sw.find_login_box(build_obs(box_texts(MASK_A)))
        assert box.mask.prefix == "111" and box.mask.suffix == "11" and box.arrow.x == int(1920 * sw.DROPDOWN_ARROW_X)
        assert sw.find_dropdown(build_obs(box_texts(MASK_A))) is None
        dropdown = sw.find_dropdown(build_obs(dropdown_texts(MASK_A, [MASK_A, MASK_B])))
        assert [str(row.mask) for row in dropdown.rows] == ["111…11", "222…22"]
        assert sw.find_login_box(build_obs(dropdown_texts(MASK_A, [MASK_A, MASK_B]))) is None

    def test_toast_with_number_is_not_a_login_box(self):
        toast = [("222******22，欢迎进入游戏", 1002, 186), ("开始游戏", 969, 838), ("登出", 1829, 304)]
        assert sw.find_login_box(build_obs(toast)) is None and sw.find_dropdown(build_obs(toast)) is None


class TestMasks:
    @pytest.mark.parametrize("a,b,expected", [
        ("111******11", "111******11", True),
        ("111******11", "111*****11", True),  # 星号个数认错
        ("111******11", "11*****11", True),  # 丢了开头一位
        ("111******11", "222******22", False),
        ("111******11", "111******22", False),
        ("111******11", "222******11", False),
    ])
    def test_masks_match(self, a, b, expected):
        assert sw.masks_match(sw.parse_mask(a), sw.parse_mask(b)) is expected

    @pytest.mark.parametrize("text", ["账号", "", "1234", "12*34", None])
    def test_not_a_mask(self, text):
        assert sw.parse_mask(text) is None


class TestAccounts:
    def _accounts(self, tmp_path, data):
        path = tmp_path / "accounts.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return sw.load_accounts(str(path))

    def test_lookup(self, tmp_path):
        accounts = self._accounts(tmp_path, {UID_A: {"name": "账号甲", "login": MASK_A}, UID_B: {"name": "账号乙乙"}, UID_C: "账号丙"})
        assert sw.find_account(accounts, UID_A)[0].login == MASK_A
        assert sw.find_account(accounts, "账号乙乙")[0].uid == UID_B and sw.find_account(accounts, "账号乙乙")[0].login is None
        assert sw.find_account(accounts, "乙")[0].uid == UID_B  # 唯一的包含匹配
        assert sw.find_account(accounts, "账号")[0] is None and "多个账号" in sw.find_account(accounts, "账号")[1]
        assert sw.find_account(accounts, "丙")[0].uid == UID_C
        assert sw.find_account(accounts, "没有")[0] is None

    def test_bad_files_are_empty(self, tmp_path):
        assert sw.load_accounts(str(tmp_path / "none.json")) == []
        bad = tmp_path / "bad.json"
        bad.write_text("[1, 2", encoding="utf-8")
        assert sw.load_accounts(str(bad)) == []


class TestNoPersonalData:
    def test_code_and_assets_contain_no_real_identifiers(self):
        here = os.path.dirname(__file__)
        files = [os.path.join(here, "test_llm_agent_account_switch.py"),
                 os.path.join(sw.ROOT, "module", "llm_agent", "account_switch.py")]
        import re
        for path in files:
            text = open(path, encoding="utf-8").read()
            for uid in re.findall(r"(?<!\d)\d{9,10}(?!\d)", text):
                assert len(set(uid)) == 1, f"{path} 里有疑似真实 UID：{uid}"
            for mask in re.findall(r"\d{3}\*{5,}\d{2}", text):
                assert mask[:3] in ("111", "222", "333") and mask[-2:] in ("11", "22", "33")
