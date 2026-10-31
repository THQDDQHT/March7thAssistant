# coding:utf-8
"""大模型接管：护栏（设计文档第 8 节的每条规则）和动作参数规范化。"""
import pytest
from PIL import Image

from module.llm_agent import guard as gd
from module.llm_agent.actions import Action, normalize_action
from module.llm_agent.observe import Element, Observation


def _obs(texts, grid=False, kinds=None):
    kinds = kinds or {}
    elements = [Element(i + 1, text, (0, i * 50, 100, i * 50 + 40), kinds.get(i + 1, "text")) for i, text in enumerate(texts)]
    return Observation(image=Image.new("RGB", (1920, 1080)), elements=elements, grid=grid)


def _click(element_id):
    return Action("click", {"id": element_id})


class TestDangerWordsConstant:
    def test_table_contents(self):
        expected = ["购买", "支付", "充值", "跃迁", "星琼", "古老梦华", "专票", "通票",
                    "兑换", "分解", "弃置", "出售", "删除", "确认支付"]
        assert list(gd.DANGER_WORDS) == expected

    def test_extra_words_appended(self):
        guard = gd.Guard(extra_danger_words=["礼包", " ", ""])
        assert guard.danger_words[:len(gd.DANGER_WORDS)] == gd.DANGER_WORDS
        assert "礼包" in guard.danger_words
        assert "" not in guard.danger_words

    def test_extra_words_take_effect(self):
        obs = _obs(["领取礼包"])
        assert gd.Guard().check(_click(1), obs).allowed
        assert not gd.Guard(["礼包"]).check(_click(1), obs).allowed


class TestRule2DangerTarget:
    @pytest.mark.parametrize("word", gd.DANGER_WORDS)
    def test_every_danger_word_blocks_click_target(self, word):
        verdict = gd.Guard().check(_click(1), _obs([f"立即{word}"]))
        assert not verdict.allowed
        assert verdict.rule in (gd.RULE_DANGER_TARGET,)

    def test_spaced_text_still_blocked(self):
        assert not gd.Guard().check(_click(1), _obs(["购 买"])).allowed

    def test_safe_target_allowed_even_if_other_text_dangerous(self):
        # 目标是「关闭」，旁边有「购买」：规则 2/3 都不拦（关闭不是确认类）
        assert gd.Guard().check(_click(1), _obs(["关闭", "购买"])).allowed


class TestRule3ConfirmWithDangerOnScreen:
    @pytest.mark.parametrize("text", ["确认", "确定", "是", "同意", " 确 认 ", "确认。"])
    def test_confirm_text_blocked_when_danger_on_screen(self, text):
        verdict = gd.Guard().check(_click(1), _obs([text, "是否消耗 60 星琼"]))
        assert not verdict.allowed
        assert verdict.rule == gd.RULE_DANGER_CONFIRM

    def test_confirm_icon_blocked_when_danger_on_screen(self):
        obs = _obs(["[确认图标]", "星琼不足"], kinds={1: "icon_confirm"})
        verdict = gd.Guard().check(_click(1), obs)
        assert not verdict.allowed
        assert verdict.rule == gd.RULE_DANGER_CONFIRM

    @pytest.mark.parametrize("text", ["确认", "确定", "是"])
    def test_confirm_allowed_when_only_spending_stamina(self, text):
        # 开拓力随便花：「消耗」不再是危险词，「是否消耗 60 开拓力」的确认框可以点确认
        assert gd.Guard().check(_click(1), _obs([text, "是否消耗 60 开拓力"])).allowed
        assert gd.Guard().check(Action("press_key", {"key": "enter"}), _obs([text, "是否消耗 60 开拓力"])).allowed

    def test_consuming_premium_currency_still_blocked(self):
        assert not gd.Guard().check(_click(1), _obs(["确认", "消耗星琼"])).allowed
        assert not gd.Guard().check(_click(1), _obs(["消耗古老梦华"])).allowed

    def test_consume_is_not_a_danger_word(self):
        assert "消耗" not in gd.DANGER_WORDS

    def test_confirm_allowed_without_danger(self):
        assert gd.Guard().check(_click(1), _obs(["确认", "网络连接已断开"])).allowed

    def test_non_confirm_button_not_blocked_by_rule3(self):
        assert gd.Guard().check(_click(1), _obs(["稍后再看", "星琼不足"])).allowed

    def test_text_merely_containing_confirm_is_not_confirm_class(self):
        # 「确认收到」不是确认类按钮的固定写法，不按规则 3 处理
        assert gd.Guard().check(_click(1), _obs(["确认收到", "星琼不足"])).allowed


class TestRule4Grid:
    def test_grid_click_blocked_when_danger_on_screen(self):
        verdict = gd.Guard().check(Action("click_grid", {"cell": "H5"}), _obs(["跃迁"], grid=True))
        assert not verdict.allowed
        assert verdict.rule == gd.RULE_DANGER_GRID

    def test_grid_click_allowed_without_danger(self):
        assert gd.Guard().check(Action("click_grid", {"cell": "H5"}), _obs(["关闭"], grid=True)).allowed

    def test_grid_click_requires_grid_mode(self):
        verdict = gd.Guard().check(Action("click_grid", {"cell": "H5"}), _obs(["关闭"], grid=False))
        assert not verdict.allowed
        assert verdict.rule == gd.RULE_GRID_INACTIVE


class TestRule5Keys:
    @pytest.mark.parametrize("key", ["esc", "enter", "space", "f", "ESC"])
    def test_whitelisted_keys(self, key):
        assert gd.Guard().check(Action("press_key", {"key": key}), _obs([])).allowed

    @pytest.mark.parametrize("key", ["alt+f4", "f4", "delete", "w", "ctrl", "tab", ""])
    def test_other_keys_blocked(self, key):
        verdict = gd.Guard().check(Action("press_key", {"key": key}), _obs([]))
        assert not verdict.allowed
        assert verdict.rule == gd.RULE_KEY_NOT_ALLOWED

    def test_whitelist_constant(self):
        assert gd.ALLOWED_KEYS == ("esc", "enter", "space", "f")


class TestOtherActions:
    def test_unknown_element_id_rejected(self):
        verdict = gd.Guard().check(_click(7), _obs(["关闭"]))
        assert not verdict.allowed
        assert verdict.rule == gd.RULE_INVALID_TARGET

    @pytest.mark.parametrize("name,params", [
        ("show_grid", {}), ("wait", {"seconds": 2}), ("done", {"reason": ""}), ("give_up", {"reason": ""}),
    ])
    def test_control_actions_allowed(self, name, params):
        assert gd.Guard().check(Action(name, params), _obs(["购买"])).allowed

    def test_unknown_action_rejected(self):
        assert not gd.Guard().check(Action("drag"), _obs([])).allowed


class TestNormalizeAction:
    def test_click_id_variants(self):
        assert normalize_action({"action": "click", "id": 3})[0].params == {"id": 3}
        assert normalize_action({"action": "click", "id": "3"})[0].params == {"id": 3}
        assert normalize_action({"action": "click", "id": 3.0})[0].params == {"id": 3}

    @pytest.mark.parametrize("data", [
        {"action": "click"}, {"action": "click", "id": "abc"}, {"action": "click", "id": True},
        {"action": "click_grid", "cell": "Z9"}, {"action": "click_grid"},
        {"action": "press_key"}, {"action": "wait", "seconds": "x"},
        {"action": "fly"}, {"foo": 1}, [], "click",
    ])
    def test_invalid_actions(self, data):
        action, error = normalize_action(data)
        assert action is None
        assert error

    def test_wait_clamped_to_1_5(self):
        assert normalize_action({"action": "wait", "seconds": 99})[0].params["seconds"] == 5.0
        assert normalize_action({"action": "wait", "seconds": 0})[0].params["seconds"] == 1.0

    def test_case_insensitive_and_cell_upper(self):
        action = normalize_action({"action": "CLICK_GRID", "cell": "h5"})[0]
        assert action.name == "click_grid"
        assert action.params == {"cell": "H5"}

    def test_describe(self):
        assert Action("click", {"id": 3}).describe() == "click(id=3)"
        assert Action("show_grid").describe() == "show_grid"


class TestDangerKeyRule:
    """画面上有危险词时，enter/space/f 可能等于点了确认，只允许按 esc。"""

    @pytest.mark.parametrize("key", ["enter", "space", "f"])
    def test_other_keys_blocked_when_danger_on_screen(self, key):
        verdict = gd.Guard().check(Action("press_key", {"key": key}), _obs(["购买", "确认"]))
        assert not verdict.allowed
        assert verdict.rule == gd.RULE_DANGER_KEY

    def test_esc_still_allowed_when_danger_on_screen(self):
        assert gd.Guard().check(Action("press_key", {"key": "esc"}), _obs(["购买", "确认"])).allowed

    @pytest.mark.parametrize("key", ["enter", "space", "f", "esc"])
    def test_all_whitelisted_keys_allowed_without_danger(self, key):
        assert gd.Guard().check(Action("press_key", {"key": key}), _obs(["关闭"])).allowed

    def test_non_whitelisted_key_keeps_original_rule(self):
        verdict = gd.Guard().check(Action("press_key", {"key": "tab"}), _obs(["购买"]))
        assert verdict.rule == gd.RULE_KEY_NOT_ALLOWED

    def test_split_danger_word_also_blocks_key(self):
        assert gd.Guard().check(Action("press_key", {"key": "enter"}), _obs(["购", "买"])).rule == gd.RULE_DANGER_KEY

    def test_extra_danger_words_apply(self):
        assert not gd.Guard(["礼包"]).check(Action("press_key", {"key": "enter"}), _obs(["礼包"])).allowed


class TestDangerWordSplitAcrossBoxes:
    """OCR 把一个危险词拆进相邻两个框时，画面级检查（规则 3、4、按键）也要拦住。"""

    def test_screen_danger_word_on_joined_text(self):
        guard = gd.Guard()
        assert guard.screen_danger_word(["购", "买"]) == "购买"
        assert guard.screen_danger_word(["确认", "支", "付"]) == "支付"
        assert guard.screen_danger_word(["关闭", "返回"]) is None

    def test_rule3_confirm_blocked_when_word_split(self):
        verdict = gd.Guard().check(_click(1), _obs(["确认", "购", "买"]))
        assert not verdict.allowed
        assert verdict.rule == gd.RULE_DANGER_CONFIRM

    def test_rule4_grid_blocked_when_word_split(self):
        verdict = gd.Guard().check(Action("click_grid", {"cell": "H5"}), _obs(["跃", "迁"], grid=True))
        assert not verdict.allowed
        assert verdict.rule == gd.RULE_DANGER_GRID

    def test_split_extra_word(self):
        assert not gd.Guard(["礼包"]).check(Action("click_grid", {"cell": "H5"}), _obs(["礼", "包"], grid=True)).allowed

    def test_rule2_target_alone_not_affected(self):
        # 规则 2 只看目标自己：目标「关闭」旁边的「购」「买」不会拦住点击关闭
        assert gd.Guard().check(_click(1), _obs(["关闭", "购", "买"])).allowed
