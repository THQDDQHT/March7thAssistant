# coding:utf-8
"""前台输入的硬件扫描码发键：键码映射、SendInput 参数、按住时的 down/up 顺序、失败返回、前台/后台/离线的分流。

mock 掉 ctypes.windll.user32，测试时不会向系统发任何按键。
"""
import ctypes
import sys
from types import SimpleNamespace

import pytest

from module.llm_agent import mcp_tools as mt
from module.llm_agent import scancode_input as si
from module.llm_agent.actions import Action, ActionExecutor
from module.automation.local_input import LocalInput

# 标准 US 键盘的 set-1 扫描码（MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) 的真实值）
SCAN = {"e": 0x12, "2": 0x03, "space": 0x39, "enter": 0x1C, "esc": 0x01, "a": 0x1E, "1": 0x02, "4": 0x05,
        "q": 0x10, "w": 0x11, "s": 0x1F, "d": 0x20, "f": 0x21, "0": 0x0B, "z": 0x2C}
VK_BY_SCAN = {}


class FakeUser32:
    def __init__(self, sendinput_result=1, fail_on=None):
        self.events = []        # (顺序号, wVk, wScan, dwFlags)
        self.map_calls = []
        self.result = sendinput_result
        self.fail_on = fail_on  # 第几次 SendInput 返回 0
        self.count = 0

    def MapVirtualKeyW(self, vk, map_type):
        self.map_calls.append((vk, map_type))
        for key, scan in SCAN.items():
            if si.virtual_key(key) == vk:
                return scan
        return 0x7F

    def SendInput(self, count, pointer, size):
        self.count += 1
        event = pointer._obj
        assert count == 1 and size == ctypes.sizeof(si.INPUT)
        assert event.type == si.INPUT_KEYBOARD
        ki = event.u.ki
        self.events.append((self.count, ki.wVk, ki.wScan, ki.dwFlags))
        return 0 if self.fail_on == self.count else self.result


class TestKeyMapping:
    @pytest.mark.parametrize("key,vk", [("a", 0x41), ("e", 0x45), ("z", 0x5A), ("A", 0x41), ("0", 0x30), ("2", 0x32),
                                        ("9", 0x39), ("space", 0x20), ("enter", 0x0D), ("esc", 0x1B), (" E ", 0x45)])
    def test_virtual_key(self, key, vk):
        assert si.virtual_key(key) == vk

    @pytest.mark.parametrize("key", ["", "ab", "tab", "f1", "ctrl+e", "é", "中", None, 5, "-", "shift"])
    def test_unsupported_keys(self, key):
        assert si.virtual_key(key) is None

    @pytest.mark.parametrize("key", ["e", "2", "space", "enter", "esc", "a", "1", "z"])
    def test_scan_code_is_map_virtual_key_of_vk(self, key):
        user32 = FakeUser32()
        assert si.scan_code(key, user32) == SCAN[key]
        assert user32.map_calls == [(si.virtual_key(key), si.MAPVK_VK_TO_VSC)]  # MapVirtualKeyW(vk, 0)

    def test_key_codes_named_in_the_task(self):
        user32 = FakeUser32()
        assert (si.scan_code("e", user32), si.scan_code("2", user32), si.scan_code("space", user32)) == (0x12, 0x03, 0x39)

    def test_unsupported_key_has_no_scan_code(self):
        assert si.scan_code("tab", FakeUser32()) is None

    def test_zero_scan_code_means_unmappable(self):
        class Zero(FakeUser32):
            def MapVirtualKeyW(self, vk, map_type):
                return 0
        assert si.scan_code("e", Zero()) is None

    @pytest.mark.skipif(sys.platform != "win32", reason="wintypes 的 DWORD 宽度只在 Windows 上是 4 字节")
    def test_struct_layout(self):
        assert ctypes.sizeof(si.KEYBDINPUT) == (24 if ctypes.sizeof(ctypes.c_size_t) == 8 else 16)
        assert ctypes.sizeof(si.INPUT) == (40 if ctypes.sizeof(ctypes.c_size_t) == 8 else 28)


class TestSend:
    def test_press_sends_scancode_only_events(self):
        user32 = FakeUser32()
        slept = []
        assert si.press("e", 0.4, sleep=slept.append, user32=user32) is True
        # 只带扫描码：wVk=0，flags=KEYEVENTF_SCANCODE，松开多一个 KEYEVENTF_KEYUP
        assert user32.events == [(1, 0, 0x12, si.KEYEVENTF_SCANCODE), (2, 0, 0x12, si.KEYEVENTF_SCANCODE | si.KEYEVENTF_KEYUP)]
        assert slept == [0.4]

    def test_default_hold_is_0_15(self):
        slept = []
        si.press("space", sleep=slept.append, user32=FakeUser32())
        assert slept == [0.15]

    def test_down_and_up_separately(self):
        user32 = FakeUser32()
        assert si.key_down("2", user32) is True and si.key_up("2", user32) is True
        assert [event[1:] for event in user32.events] == [(0, 0x03, si.KEYEVENTF_SCANCODE),
                                                          (0, 0x03, si.KEYEVENTF_SCANCODE | si.KEYEVENTF_KEYUP)]

    def test_down_up_order_with_sleep_between(self):
        user32 = FakeUser32()
        order = []
        original = user32.SendInput

        def traced(count, pointer, size):
            order.append(("send", pointer._obj.u.ki.dwFlags & si.KEYEVENTF_KEYUP))
            return original(count, pointer, size)

        user32.SendInput = traced
        si.press("w", 2.0, sleep=lambda s: order.append(("sleep", s)), user32=user32)
        assert order == [("send", 0), ("sleep", 2.0), ("send", si.KEYEVENTF_KEYUP)]

    def test_key_is_released_even_if_sleep_is_interrupted(self):
        user32 = FakeUser32()

        def boom(seconds):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            si.press("w", 1, sleep=boom, user32=user32)
        assert len(user32.events) == 2 and user32.events[1][3] & si.KEYEVENTF_KEYUP

    def test_sendinput_failure_on_down_returns_false_and_sends_no_up(self):
        user32 = FakeUser32(fail_on=1)
        assert si.press("e", user32=user32, sleep=lambda s: None) is False
        assert len(user32.events) == 1

    def test_sendinput_failure_on_up_returns_false(self):
        assert si.press("e", user32=FakeUser32(fail_on=2), sleep=lambda s: None) is False

    @pytest.mark.parametrize("result", [0, 2, -1])
    def test_only_one_event_counts_as_success(self, result):
        assert si.key_down("e", FakeUser32(sendinput_result=result)) is False

    def test_unsupported_key_returns_false_without_sending(self):
        user32 = FakeUser32()
        assert si.press("tab", user32=user32) is False and user32.events == []

    @pytest.mark.skipif(sys.platform == "win32", reason="非 Windows 环境才验证没有 user32")
    def test_importable_and_safe_without_windll(self):
        assert si.press("e") is False and si.key_down("e") is False and si.key_up("e") is False


class FakeAuto:
    def __init__(self, handler):
        self.input_handler = handler
        self.calls = []

    def press_key(self, key, wait_time=0.2):
        self.calls.append((key, wait_time))
        return True


def foreground_handler():
    return LocalInput(logger=SimpleNamespace(debug=lambda m: None, error=lambda m: None))


def BackgroundLike():
    return SimpleNamespace(is_background=True)  # 不是 LocalInput 实例


def CloudLike():
    return SimpleNamespace(is_background=False)  # 云游戏的 CdpInput 也不是 LocalInput 实例


class TestRouting:
    @pytest.fixture(autouse=True)
    def _win32(self, monkeypatch):
        monkeypatch.setattr(si.sys, "platform", "win32")
        self.sent = []
        monkeypatch.setattr(si, "press", lambda key, hold=si.DEFAULT_HOLD, **kw: self.sent.append((key, hold)) or True)

    def test_use_for(self, monkeypatch):
        assert si.use_for(FakeAuto(foreground_handler())) is True
        assert si.use_for(FakeAuto(BackgroundLike())) is False
        assert si.use_for(FakeAuto(CloudLike())) is False
        assert si.use_for(FakeAuto(None)) is False
        assert si.use_for(mt.OfflineAuto()) is False
        assert si.use_for(SimpleNamespace()) is False
        monkeypatch.setattr(si.sys, "platform", "darwin")
        assert si.use_for(FakeAuto(foreground_handler())) is False

    def test_mcp_foreground_uses_scancode(self):
        auto = FakeAuto(foreground_handler())
        result = mt.McpExecutor(auto=auto).press("e", 0)
        assert result.ok and self.sent == [("e", si.DEFAULT_HOLD)] and auto.calls == []

    def test_mcp_foreground_hold(self):
        auto = FakeAuto(foreground_handler())
        result = mt.McpExecutor(auto=auto).press("w", 2.5)
        assert result.ok and "按住 2.5 秒" in result.message and self.sent == [("w", 2.5)]

    def test_mcp_background_and_cloud_use_auto(self):
        for handler in (BackgroundLike(), CloudLike()):
            auto = FakeAuto(handler)
            assert mt.McpExecutor(auto=auto).press("e", 0).ok
            assert mt.McpExecutor(auto=auto).press("w", 1.5).ok
            assert auto.calls == [("e", 0.2), ("w", 1.5)]
        assert self.sent == []

    def test_mcp_offline_uses_noop_auto(self):
        auto = mt.OfflineAuto()
        assert mt.McpExecutor(auto=auto).press("e", 0).ok
        assert auto.calls and auto.calls[0][0] == "press_key" and self.sent == []

    def test_mcp_scancode_failure_reports_not_done(self, monkeypatch):
        monkeypatch.setattr(si, "press", lambda key, hold=0.15, **kw: False)
        result = mt.McpExecutor(auto=FakeAuto(foreground_handler())).press("e", 0)
        assert result.ok is False and "没有完成" in result.message

    def test_takeover_executor_foreground_uses_scancode(self):
        auto = FakeAuto(foreground_handler())
        result = ActionExecutor(auto=auto).execute(Action("press_key", {"key": "esc"}), None)
        assert result.ok and self.sent == [("esc", si.DEFAULT_HOLD)] and auto.calls == []

    def test_takeover_executor_background_unchanged(self):
        auto = FakeAuto(BackgroundLike())
        result = ActionExecutor(auto=auto).execute(Action("press_key", {"key": "esc"}), None)
        assert result.ok and auto.calls == [("esc", 0.2)] and self.sent == []

    def test_takeover_executor_failure(self, monkeypatch):
        monkeypatch.setattr(si, "press", lambda key, hold=0.15, **kw: False)
        result = ActionExecutor(auto=FakeAuto(foreground_handler())).execute(Action("press_key", {"key": "f"}), None)
        assert result.ok is False and "没有完成" in result.message

    def test_mcp_tool_end_to_end_foreground_after_ensure_front(self, tmp_path, monkeypatch):
        """press_key 工具：先 _ensure_front（切前台），再用扫描码发键。"""
        order = []
        auto = FakeAuto(foreground_handler())
        game = SimpleNamespace(is_front=lambda: order.append("is_front") or False,
                               bring_front=lambda: order.append("bring_front") or True)
        monkeypatch.setattr(si, "press", lambda key, hold=si.DEFAULT_HOLD, **kw: order.append(("scan", key, hold)) or True)
        from module.llm_agent.observe import Observation, Element
        from PIL import Image
        tools = mt.McpTools(root=str(tmp_path), record_base=str(tmp_path / "rec"), auto=auto, screen=SimpleNamespace(
            get_current_screen=lambda autotry=False: False, current_screen=None), game=game,
            observer=lambda grid: Observation(image=Image.new("RGB", (1920, 1080)), elements=[Element(1, "关闭", (0, 0, 10, 10))]),
            guard=mt.Guard((), mt.MCP_ALLOWED_KEYS), config_get=lambda k, d=None: d, sleep=lambda s: None,
            runs=SimpleNamespace(read_active_run=lambda: None, describe_run=str), settle_seconds=0)
        out = tools.press_key("e")
        assert "按了 e 键" in out.text
        assert order.index("bring_front") < order.index(("scan", "e", si.DEFAULT_HOLD))
        assert auto.calls == []
