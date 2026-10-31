"""护栏：写在代码里，不靠 prompt。

规则（设计文档第 8 节）：
1. 危险词表（常量，配置 llm_takeover_extra_danger_words 可以追加）。
2. click 的目标元素文字包含危险词 → 拒绝。
3. 目标是确认类按钮（文字为 确认/确定/是/同意，或者是确认图标），同时画面上任意文字包含危险词 → 拒绝。
4. 画面上任意文字包含危险词时，禁止 click_grid（网格点击看不到目标是什么，所以从严处理）；
   另外 click_grid 只能在网格模式下用。
5. 按键只允许白名单里的几个；画面上有危险词时，只允许按 esc（enter / space / f 可能等于点了确认）。
上限（步数、时长、次数）和暂停响应在接管循环里（takeover.py）。
被拒绝的动作会反馈给模型，同时计入步数。
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from module.llm_agent.actions import Action

DANGER_WORDS = (
    "购买", "支付", "充值", "跃迁", "星琼", "古老梦华", "专票", "通票",
    "兑换", "分解", "弃置", "出售", "删除", "消耗", "确认支付",
)
CONFIRM_TEXTS = ("确认", "确定", "是", "同意")
ALLOWED_KEYS = ("esc", "enter", "space", "f")
# 画面上有危险词时唯一允许的按键
SAFE_KEY_ON_DANGER = "esc"

# 规则编号（写进记录，方便复盘）
RULE_INVALID_ACTION = "invalid_action"
RULE_INVALID_TARGET = "invalid_target"
RULE_DANGER_TARGET = "danger_target"  # 规则 2
RULE_DANGER_CONFIRM = "danger_confirm"  # 规则 3
RULE_DANGER_GRID = "danger_grid"  # 规则 4
RULE_GRID_INACTIVE = "grid_inactive"
RULE_KEY_NOT_ALLOWED = "key_not_allowed"  # 规则 5
RULE_DANGER_KEY = "danger_key"  # 规则 5：画面上有危险词时只允许 esc

_STRIP_CHARS = " \t\r\n。.!！?？:：、,，()（）[]【】"


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    rule: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "rule": self.rule, "reason": self.reason}


ALLOW = Verdict(True)


def _compact(text: str) -> str:
    """去掉所有空白，避免 OCR 把「购 买」拆开后漏过。"""
    return "".join((text or "").split())


class Guard:
    def __init__(self, extra_danger_words: Iterable[str] = (), allowed_keys: Iterable[str] = ALLOWED_KEYS):
        # 按键白名单做成参数：接管用默认的 ALLOWED_KEYS，第二期 MCP 工具传入扩充后的白名单
        self.allowed_keys = tuple(str(key).strip().lower() for key in allowed_keys)
        extra = tuple(_compact(str(word)) for word in extra_danger_words if _compact(str(word)))
        # 去重并保持顺序
        self.danger_words = tuple(dict.fromkeys(DANGER_WORDS + extra))

    def find_danger_word(self, text: str) -> Optional[str]:
        compact = _compact(text)
        for word in self.danger_words:
            if word in compact:
                return word
        return None

    def screen_danger_word(self, texts: Sequence[str]) -> Optional[str]:
        """画面上任意文字包含危险词时，返回命中的危险词。

        先逐条检查，再把所有元素文字拼接起来检查一次：OCR 可能把一个词拆进相邻的两个框
        （例如「购」和「买」），只逐条检查会漏掉。拼接检查宁可误拦也不漏拦。
        """
        for text in texts:
            word = self.find_danger_word(text)
            if word:
                return word
        return self.find_danger_word("".join(texts))

    @staticmethod
    def is_confirm_element(element) -> bool:
        if element.kind == "icon_confirm":
            return True
        return _compact(element.text).strip(_STRIP_CHARS) in CONFIRM_TEXTS

    def check(self, action: Action, observation) -> Verdict:
        """判定一个动作能不能执行。observation 提供 elements / get() 和 grid 标志。"""
        name = action.name
        if name == "click":
            return self._check_click(action, observation)
        if name == "click_grid":
            return self._check_click_grid(observation)
        if name == "press_key":
            key = str(action.params.get("key", "")).strip().lower()
            if key not in self.allowed_keys:
                return Verdict(False, RULE_KEY_NOT_ALLOWED, f"按键 {key!r} 不在白名单内，只允许 {', '.join(self.allowed_keys)}")
            if key != SAFE_KEY_ON_DANGER:
                word = self.screen_danger_word(observation.all_texts())
                if word:
                    return Verdict(False, RULE_DANGER_KEY, f"画面上出现危险词「{word}」，只允许按 {SAFE_KEY_ON_DANGER}")
            return ALLOW
        if name in ("show_grid", "wait", "done", "give_up"):
            return ALLOW
        return Verdict(False, RULE_INVALID_ACTION, f"未知动作 {name!r}")

    def _check_click(self, action: Action, observation) -> Verdict:
        element = observation.get(action.params.get("id"))
        if element is None:
            return Verdict(False, RULE_INVALID_TARGET, f"没有编号为 {action.params.get('id')} 的元素")
        word = self.find_danger_word(element.text)
        if word:
            return Verdict(False, RULE_DANGER_TARGET, f"目标文字「{element.text}」包含危险词「{word}」")
        if self.is_confirm_element(element):
            word = self.screen_danger_word(observation.all_texts())
            if word:
                return Verdict(False, RULE_DANGER_CONFIRM, f"画面上出现危险词「{word}」，不能点确认类按钮")
        return ALLOW

    def _check_click_grid(self, observation) -> Verdict:
        if not observation.grid:
            return Verdict(False, RULE_GRID_INACTIVE, "当前不是网格模式，请先用 show_grid")
        word = self.screen_danger_word(observation.all_texts())
        if word:
            return Verdict(False, RULE_DANGER_GRID, f"画面上出现危险词「{word}」，禁止网格点击")
        return ALLOW
