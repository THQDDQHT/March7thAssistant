"""动作：模型输出的 JSON → 规范化的 Action → 执行。

模型每步只能输出一个动作：click / show_grid / click_grid / press_key / wait / done / give_up。
参数规范化（类型、范围）在这里做；是否允许执行由 guard.py 判定；
点击和按键复用 auto 的输入层（auto.click_element_with_pos / auto.press_key），所以暂停和停止指令照常生效。
"""
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from module.llm_agent import scancode_input
from module.llm_agent.observe import GRID_CELL_RE, cell_rect

ACTION_NAMES = ("click", "show_grid", "click_grid", "press_key", "wait", "done", "give_up")
WAIT_MIN = 1.0
WAIT_MAX = 5.0


@dataclass
class Action:
    name: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"action": self.name, **self.params}

    def describe(self) -> str:
        """写进历史的一句话，例如 click(id=3)。"""
        if not self.params:
            return self.name
        return f"{self.name}(" + ", ".join(f"{k}={v}" for k, v in self.params.items()) + ")"


@dataclass
class ActionResult:
    ok: bool
    message: str = ""


def _to_int(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def normalize_action(data) -> Tuple[Optional[Action], str]:
    """模型输出的 dict → (Action, "")；不合法时返回 (None, 原因)，原因会反馈给模型。"""
    if not isinstance(data, dict):
        return None, "输出必须是一个 JSON 对象"
    name = data.get("action")
    if not isinstance(name, str) or name.strip().lower() not in ACTION_NAMES:
        return None, f"未知动作：{name!r}，只能用 {', '.join(ACTION_NAMES)}"
    name = name.strip().lower()

    if name == "click":
        element_id = _to_int(data.get("id"))
        if element_id is None:
            return None, "click 需要整数参数 id"
        return Action(name, {"id": element_id}), ""
    if name == "click_grid":
        cell = data.get("cell")
        if not isinstance(cell, str) or not GRID_CELL_RE.match(cell.strip().upper()):
            return None, "click_grid 需要参数 cell，格式为列 A-P 加行 1-9，例如 H5"
        return Action(name, {"cell": cell.strip().upper()}), ""
    if name == "press_key":
        key = data.get("key")
        if not isinstance(key, str) or not key.strip():
            return None, "press_key 需要字符串参数 key"
        return Action(name, {"key": key.strip().lower()}), ""
    if name == "wait":
        try:
            seconds = float(data.get("seconds"))
        except (TypeError, ValueError):
            return None, "wait 需要数字参数 seconds（1-5）"
        if seconds != seconds:  # NaN
            return None, "wait 需要数字参数 seconds（1-5）"
        return Action(name, {"seconds": min(max(seconds, WAIT_MIN), WAIT_MAX)}), ""
    if name in ("done", "give_up"):
        return Action(name, {"reason": str(data.get("reason", ""))[:200]}), ""
    return Action(name), ""  # show_grid


class ActionExecutor:
    """执行已通过护栏的动作。auto 默认用 M7A 的全局单例（延迟导入），测试里可以注入。"""

    def __init__(self, auto=None, sleep=time.sleep):
        self._auto = auto
        self._sleep = sleep

    @property
    def auto(self):
        if self._auto is None:
            from module.automation import auto
            self._auto = auto
        return self._auto

    def execute(self, action: Action, observation) -> ActionResult:
        """执行一个动作。show_grid / done / give_up 不产生输入，由接管循环自己处理。"""
        if action.name == "click":
            element = observation.get(action.params["id"])
            if element is None:
                return ActionResult(False, f"没有编号为 {action.params['id']} 的元素")
            if not self._click_rect(element.box, observation):
                return ActionResult(False, f"点击 [{element.id}] {element.text} 没有完成（窗口不可用或鼠标被占用）")
            return ActionResult(True, f"点击了 [{element.id}] {element.text}")
        if action.name == "click_grid":
            rect = cell_rect(action.params["cell"], observation.size)
            if rect is None:
                return ActionResult(False, f"格子 {action.params['cell']} 不合法")
            left, top, right, bottom = rect
            if not self._click_rect([[left, top], [right, top], [right, bottom], [left, bottom]], observation):
                return ActionResult(False, f"点击格子 {action.params['cell']} 没有完成（窗口不可用或鼠标被占用）")
            return ActionResult(True, f"点击了格子 {action.params['cell']}")
        if action.name == "press_key":
            if scancode_input.use_for(self.auto):
                # 前台输入：pyautogui 的扫描码是 0，战斗快捷键收不到，改用硬件扫描码发键（后台输入、云游戏不走这里）
                from utils.pause import pause_ctl
                pause_ctl.checkpoint()
                done = scancode_input.press(action.params["key"])
            else:
                done = self.auto.press_key(action.params["key"])
            if done is False:
                return ActionResult(False, f"按 {action.params['key']} 键没有完成（窗口不可用）")
            return ActionResult(True, f"按了 {action.params['key']} 键")
        if action.name == "wait":
            self._sleep(action.params["seconds"])
            return ActionResult(True, f"等待了 {action.params['seconds']:g} 秒")
        return ActionResult(True, "")

    def _click_rect(self, box, observation=None) -> bool:
        """截图坐标的框 → 屏幕绝对坐标 → 点击中心。

        优先用这次观察时记下的 screenshot_pos / scale 换算（公式同 auto.calculate_text_position：
        先除以 scale，再加 pos），不依赖「观察之后 auto 没有再截过图」；两者缺失（离线观察）时退回 auto 当前的值。
        点击走 auto 的输入层，所以上游的后台输入模式自动生效；返回 False 表示点击没有完成
        （后台输入遇到窗口不可用、鼠标被占用等可恢复情况）。InputInterrupted / GameWindowLost 会原样抛出。
        """
        pos = getattr(observation, "screenshot_pos", None)
        scale = getattr(observation, "scale", None)
        if pos is not None and scale:
            absolute = ((int(box[0][0] / scale) + pos[0], int(box[0][1] / scale) + pos[1]),
                        (int(box[2][0] / scale) + pos[0], int(box[2][1] / scale) + pos[1]))
        else:
            absolute = self.auto.calculate_text_position(box, relative=False)
        return self.auto.click_element_with_pos(absolute) is not False
