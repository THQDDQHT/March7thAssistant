"""单次出手状态机：真实输入的替身可吞键、变阶段、报错；不用游戏或 MCP。"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from module.llm_agent import battle_hud as bh
from module.llm_agent import errors
from module.llm_agent import mcp_tools as mt
from module.llm_agent.actions import ActionResult
from module.llm_agent.guard import Guard
from module.llm_agent.observe import Element, Observation


def hud(phase=bh.PHASE_COMMAND, selected=bh.SELECTED_BASIC, sp=3, ready=True, auto=False):
    return bh.BattleHud(in_battle=True, phase=phase, selected=selected, skill_points=sp,
                        team_size=4, slots=[bh.SlotHud(i, ready) for i in range(1, 5)], auto_on=auto)


class Arena:
    def __init__(self, tmp_path, monkeypatch):
        self.now = 0.0
        self.current = hud()
        self.texts = []
        self.keys = []
        self.looks = 0
        self.after_frames = []
        self.on_press = self.default_press
        self.on_observe = None
        self.active = None
        self.front_problem = None
        self.observer_error = False
        self.observer_calls = []
        self.full_looks = 0
        self.light_looks = 0
        self.full_latency = 0
        self.light_latency = 0
        monkeypatch.setattr(bh, "render_hud_panel", lambda image: b"new-hud-image")

        def capture(full):
            self.looks += 1
            self.full_looks += int(full)
            self.light_looks += int(not full)
            self.now += self.full_latency if full else self.light_latency
            self.observer_calls.append(self.current)
            if self.observer_error:
                raise RuntimeError("截图失败")
            if self.on_observe:
                self.on_observe()
            if self.after_frames:
                self.current = self.after_frames.pop(0)
            observation = Observation(Image.new("RGB", (320, 180)),
                                      [Element(i + 1, text, (1, 1, 10, 10)) for i, text in enumerate(self.texts)])
            observation.hud = self.current
            return observation

        def press(key, hold):
            self.keys.append(key)
            return self.on_press(key)

        self.tools = mt.McpTools(
            root=str(tmp_path), record_base=str(tmp_path / "records"), observer=lambda grid: capture(True),
            auto=SimpleNamespace(input_handler=SimpleNamespace(is_background=False)),
            screen=SimpleNamespace(get_current_screen=lambda **kw: False),
            game=SimpleNamespace(is_front=lambda: True),
            executor=SimpleNamespace(press=press), guard=Guard((), mt.MCP_ALLOWED_KEYS),
            runs=SimpleNamespace(read_active_run=lambda: None),
            config_get=lambda key, default=None: default,
            sleep=lambda seconds: setattr(self, "now", self.now + seconds), clock=lambda: self.now,
            hud_reader=lambda image: self.current,
        )
        monkeypatch.setattr(self.tools, "_active_task", lambda: self.active)
        monkeypatch.setattr(self.tools, "_ensure_front", lambda for_input: self.front_problem)
        monkeypatch.setattr(self.tools, "_screenshot", lambda: capture(False).image)

    def default_press(self, key):
        if key in ("q", "e"):
            self.current = hud(selected="basic" if key == "q" else "skill", sp=self.current.skill_points)
        elif key in ("1", "2", "3", "4"):
            self.current = hud(bh.PHASE_ULT_PENDING)
        elif key == "space":
            self.after_frames = [hud(bh.PHASE_OTHER), hud(), hud()]
        return ActionResult(True, "已发键")

    def act(self, action="basic", **kwargs):
        return self.tools.battle_act(action, timeout_seconds=kwargs.pop("timeout_seconds", 5), **kwargs)

    def record(self):
        paths = sorted(Path(self.tools.recorder.base_dir).glob("*/*-battle_act.json"))
        return json.loads(paths[-1].read_text())


@pytest.fixture
def arena(tmp_path, monkeypatch):
    return Arena(tmp_path, monkeypatch)


@pytest.mark.parametrize("action,key", [("basic", "q"), ("skill", "e")])
def test_select_verify_confirm_wait_and_log(arena, action, key):
    out = arena.act(action)
    assert arena.keys == [key, "space"]
    assert "已观察完成" in out.text and out.image == b"new-hud-image"
    assert arena.tools._last is None  # HUD 轻观察不能复用 OCR 元素 id
    data = arena.record()
    assert data["confirm_attempted"] and data["result"]["ok"]
    assert data["hud_before"]["phase"] == "command" and data["hud_after"]["phase"] == "command"
    assert [item["key"] for item in data["trace"] if item["event"] == "key_attempt"] == [key, "space"]


def test_swallowed_selection_must_never_confirm(arena):
    arena.on_press = lambda key: ActionResult(True, "吞键但报告成功")
    out = arena.act("skill", timeout_seconds=12)
    assert arena.keys == ["e"] and "未确认出手" in out.text
    assert "未获核实" in out.text


def test_selection_lost_to_animation_cannot_confirm_next_turn(arena):
    def press(key):
        arena.after_frames = [hud(bh.PHASE_OTHER), hud(selected="skill"), hud(selected="skill")]
        return ActionResult(True, "已发键")
    arena.on_press = press
    assert "离开普通指令" in arena.act("skill").text
    assert arena.keys == ["e"]


def test_selection_lost_to_hidden_hud_cannot_confirm_next_turn(arena):
    def press(key):
        arena.after_frames = [bh.BattleHud(in_battle=False), hud(selected="skill"), hud(selected="skill")]
        return ActionResult(True, "已发键")
    arena.on_press = press
    assert "离开普通指令" in arena.act("skill").text
    assert arena.keys == ["e"]


def test_swallowed_confirm_does_not_succeed_on_same_command_or_repeat(arena):
    def press(key):
        return ActionResult(True, "吞空格") if key == "space" else arena.default_press(key)
    arena.on_press = press
    out = arena.act()
    assert arena.keys == ["q", "space"]
    assert "执行结果不确定" in out.text and "禁止盲目重试" in out.text
    assert out.image and not arena.record()["result"]["ok"]


def test_sp_change_is_evidence_without_assuming_skill_cost(arena):
    arena.current = hud(sp=0)
    def press(key):
        if key == "space":
            arena.after_frames = [hud(selected="skill", sp=2), hud(selected="skill", sp=2)]
            return ActionResult(True, "零技能点也可使用特殊战技")
        return arena.default_press(key)
    arena.on_press = press
    assert "已观察完成" in arena.act("skill").text
    assert arena.keys == ["e", "space"]


@pytest.mark.parametrize("auto", [True, None])
def test_auto_on_or_unknown_refused(arena, auto):
    arena.current = hud(auto=auto)
    assert "自动战斗" in arena.act().text and arena.keys == []


def test_existing_ultimate_not_cancelled_by_qe(arena):
    arena.current = hud(bh.PHASE_ULT_PENDING)
    assert "confirm_pending" in arena.act("skill").text
    assert arena.keys == []
    assert "已观察完成" in arena.act("confirm_pending").text
    assert arena.keys == ["space"]


@pytest.mark.parametrize("ready", [True, None])
def test_ultimate_ready_or_unknown_enters_pending_then_confirms(arena, ready):
    arena.current = hud(ready=ready)
    assert "已观察完成" in arena.act("ultimate", slot=2).text
    assert arena.keys == ["2", "space"]


def test_ultimate_known_not_full_has_no_input_and_no_guessed_energy_message(arena):
    arena.current = hud(ready=False)
    text = arena.act("ultimate", slot=2, timeout_seconds=12).text
    assert arena.keys == [] and "圆盘识别为未满" in text and "能量不足" not in text


@pytest.mark.parametrize("toast", [True, False])
def test_ultimate_no_pending_distinguishes_observed_energy_message_from_unknown(arena, toast):
    arena.current = hud(ready=None)
    def press(key):
        arena.texts = ["能量不足"] if toast else []
        return ActionResult(True, "输入无变化")
    arena.on_press = press
    text = arena.act("ultimate", slot=2, timeout_seconds=12).text
    assert arena.keys == ["2"] and "未确认出手" in text
    assert ("能量不足" in text) is toast
    if not toast:
        assert "输入未生效" in text


@pytest.mark.parametrize("phase", [bh.PHASE_OTHER, None])
def test_special_phase_or_hidden_hud_never_gets_space(arena, phase):
    arena.current = hud(phase)
    assert "未执行" in arena.act().text and arena.keys == []
    assert arena.now <= 5


def test_initial_hidden_animation_waits_for_stable_command(arena):
    arena.after_frames = [bh.BattleHud(in_battle=False), hud(bh.PHASE_OTHER), hud(), hud()]
    assert "已观察完成" in arena.act().text
    assert arena.keys == ["q", "space"]


def test_hidden_hud_after_confirm_is_not_battle_end(arena):
    def press(key):
        if key == "space":
            arena.after_frames = [bh.BattleHud(in_battle=False), bh.BattleHud(in_battle=False), hud(), hud()]
            return ActionResult(True, "已发键")
        return arena.default_press(key)
    arena.on_press = press
    assert "执行结果不确定" in arena.act().text
    assert arena.keys.count("space") == 1


def test_result_after_confirm_is_success(arena):
    def press(key):
        if key == "space":
            arena.after_frames = [bh.BattleHud(in_battle=False, phase=bh.PHASE_RESULT, result_text="挑战结束")]
            return ActionResult(True, "已发键")
        return arena.default_press(key)
    arena.on_press = press
    out = arena.act()
    assert "已观察完成" in out.text and "挑战结束" in out.text
    assert arena.keys == ["q", "space"]


def test_guard_rechecks_new_ocr_before_confirm(arena):
    def observe():
        if arena.keys == ["q"]:
            arena.texts = ["购买"]
    arena.on_observe = observe
    assert "护栏拒绝" in arena.act().text and arena.keys == ["q"]


def test_guard_rechecks_new_ocr_before_selection(arena):
    arena.texts = ["星琼"]
    assert "护栏拒绝" in arena.act().text and arena.keys == []


@pytest.mark.parametrize("failure", ["task", "window"])
def test_task_and_window_refuse_without_input(arena, failure):
    if failure == "task":
        arena.active = "任务运行"
    else:
        arena.front_problem = "窗口丢失"
    assert "未执行" in arena.act().text and arena.keys == []


def test_task_appearing_before_confirmation_stops(arena):
    def observe():
        if arena.keys == ["q"]:
            arena.active = "新任务运行"
    arena.on_observe = observe
    assert "任务占用" in arena.act().text and arena.keys == ["q"]


def test_focus_lost_during_full_ocr_prevents_input(arena):
    def observe():
        if arena.full_looks:
            arena.tools.game.is_front = lambda: False
    arena.on_observe = observe
    assert "前台焦点" in arena.act().text and arena.keys == []


def test_selection_changes_during_final_guard_snapshot_prevents_space(arena):
    def observe():
        if arena.full_looks == 2:
            arena.current = hud(selected="basic")
    arena.on_observe = observe
    assert "输入前状态已变化" in arena.act("skill").text
    assert arena.keys == ["e"]


@pytest.mark.parametrize("key", ["q", "space"])
def test_passthrough_interrupt_is_reraised_and_logged_without_final_capture(arena, monkeypatch, key):
    class InputInterrupted(RuntimeError):
        pass
    monkeypatch.setattr(errors, "_passthrough", (InputInterrupted,))
    before_error = []
    def press(actual):
        if actual == key:
            before_error.append(arena.looks)
            raise InputInterrupted("必须由用户处理")
        return arena.default_press(actual)
    arena.on_press = press
    with pytest.raises(InputInterrupted):
        arena.act()
    assert arena.looks == before_error[0]
    assert arena.tools._last is None
    assert arena.record()["confirm_attempted"] is (key == "space")
    assert not arena.record()["result"]["ok"]


def test_confirm_input_error_is_uncertain_and_gets_new_picture(arena):
    def press(key):
        if key == "space":
            raise RuntimeError("输入中断")
        return arena.default_press(key)
    arena.on_press = press
    out = arena.act()
    assert "执行结果不确定" in out.text and out.image
    assert arena.keys == ["q", "space"] and arena.tools._last is None


def test_screenshot_error_after_confirm_clears_old_element_ids(arena):
    def press(key):
        if key == "space":
            arena.observer_error = True
            return ActionResult(True, "已发键")
        return arena.default_press(key)
    arena.on_press = press
    out = arena.act()
    assert "执行结果不确定" in out.text and arena.tools._last is None
    assert "旧元素编号已清除" in out.text
    assert arena.keys.count("space") == 1


def test_animation_timeout_is_uncertain_and_does_not_confirm_again(arena):
    def press(key):
        if key == "space":
            arena.current = hud(bh.PHASE_OTHER)
            return ActionResult(True, "已发键")
        return arena.default_press(key)
    arena.on_press = press
    assert "执行结果不确定" in arena.act().text
    assert arena.keys == ["q", "space"] and arena.now <= 5


def test_slow_full_observer_can_take_two_stable_samples(arena):
    arena.full_latency = 2.7
    arena.light_latency = 0.1
    out = arena.act("skill", timeout_seconds=25)
    assert "已观察完成" in out.text and arena.keys == ["e", "space"]
    assert arena.full_looks == 2 and arena.light_looks >= 7
    assert 5 < arena.now < 15


def test_slow_observer_respects_total_timeout_before_input(arena):
    arena.on_observe = lambda: setattr(arena, "now", arena.now + 6)
    assert "未执行" in arena.act(timeout_seconds=5).text
    assert arena.keys == []


def test_mutex_busy_does_not_start(arena, monkeypatch):
    from contextlib import contextmanager
    @contextmanager
    def busy(**kwargs):
        raise mt.ToolBusy("另一个操作执行中")
        yield
    monkeypatch.setattr(arena.tools, "_exclusive", busy)
    assert "未执行" in arena.act().text and arena.keys == [] and arena.looks == 0


@pytest.mark.parametrize("action,slot,timeout", [
    ("other", None, 5), ("basic", 1, 5), ("ultimate", None, 5), ("ultimate", True, 5),
    ("ultimate", 5, 5), ("basic", None, float("nan")), ("basic", None, True), ("skill", None, 46),
])
def test_arguments_invalid_without_observation(arena, action, slot, timeout):
    assert "未执行" in arena.act(action, slot=slot, timeout_seconds=timeout).text
    assert arena.keys == [] and arena.looks == 0
