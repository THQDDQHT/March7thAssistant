# coding:utf-8
"""大模型接管：screen.py 的两个钩子。接管关闭或没有 key 时，行为必须和原来完全一致。"""
import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from module.llm_agent import config as lc
from module.llm_agent import takeover as tk

ROOT = Path(__file__).resolve().parents[2]
KEY = "sk-test-SECRET-1234567890"


def _stub(name, **members):
    module = ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    return module


class _InputInterrupted(RuntimeError):
    pass


class _GameWindowLost(RuntimeError):
    pass


class _FakeAuto:
    """永远认不出任何界面。"""

    def take_screenshot(self):
        pass

    def perform_ocr(self):
        self.ocr_result = []

    def find_element(self, *args, **kwargs):
        return None


def _build_screen(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *args, **kwargs: None)
    cfg = MagicMock()
    cfg.get_value.side_effect = lambda key, default=None: default
    stubs = {
        "module.automation": _stub("module.automation", auto=_FakeAuto()),
        "module.automation.background_input": _stub(
            "module.automation.background_input",
            InputInterrupted=_InputInterrupted, GameWindowLost=_GameWindowLost),
        "module.config": _stub("module.config", cfg=cfg),
        "module.logger": _stub("module.logger", log=MagicMock()),
    }
    spec = importlib.util.spec_from_file_location("screen_hooks_under_test", ROOT / "module" / "screen" / "screen.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    screen = module.Screen.__new__(module.Screen)
    screen.screen_map = {"main": {"name": "主界面", "image_path": "main.png", "actions": []}}
    screen.current_screen = None
    screen.current_screen_threshold = 0
    screen.wait_screen_change_time = 0.5
    screen.lock = threading.Lock()
    screen.logger = MagicMock()
    screen.autotry_calls = 0

    def fake_autotry():
        screen.autotry_calls += 1
    screen._handle_autotry = fake_autotry
    return screen, cfg


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    tk.reset_state()
    monkeypatch.delenv(lc.ENV_API_KEY, raising=False)
    # 默认关闭，不读真实的 config.yaml，测试结果不受本机配置影响
    monkeypatch.setattr(lc, "load_config", lambda get_value=None: lc.TakeoverConfig())
    yield
    tk.reset_state()


def _enable(monkeypatch, **overrides):
    """让真实的 try_takeover 读到「已开启」的配置。"""
    values = dict(enable=True)
    values.update(overrides)
    monkeypatch.setattr(lc, "load_config", lambda get_value=None: lc.TakeoverConfig(**values))


class TestGetCurrentScreenHook:
    def test_disabled_behaves_exactly_as_before(self, monkeypatch):
        screen, _ = _build_screen(monkeypatch)
        built = []
        monkeypatch.setattr(tk, "_build_takeover", lambda cfg, key: built.append(1))
        assert screen.get_current_screen(max_retries=5) is False
        assert screen.autotry_calls == 5  # 每次失败都走原来的 _handle_autotry
        assert built == []

    def test_enabled_without_key_behaves_exactly_as_before(self, monkeypatch):
        screen, _ = _build_screen(monkeypatch)
        _enable(monkeypatch)
        built = []
        monkeypatch.setattr(tk, "_build_takeover", lambda cfg, key: built.append(1))
        assert screen.get_current_screen(max_retries=5) is False
        assert screen.autotry_calls == 5
        assert built == []

    def test_triggers_on_nth_failure_and_returns_true_on_success(self, monkeypatch):
        screen, cfg = _build_screen(monkeypatch)
        calls = []
        monkeypatch.setattr(tk, "try_takeover", lambda trigger, goal=tk.ANY_KNOWN, **kw: calls.append((trigger, goal)) or True)
        assert screen.get_current_screen(max_retries=10) is True
        assert calls == [("get_current_screen", tk.ANY_KNOWN)]
        assert screen.autotry_calls == 2  # 第 3 次失败后接管成功，不再走 _handle_autotry

    def test_failed_takeover_continues_original_retry_logic(self, monkeypatch):
        screen, cfg = _build_screen(monkeypatch)
        calls = []
        monkeypatch.setattr(tk, "try_takeover", lambda trigger, goal=tk.ANY_KNOWN, **kw: calls.append(trigger) or False)
        assert screen.get_current_screen(max_retries=6) is False
        assert calls == ["get_current_screen"]  # 一次调用只接管一次
        assert screen.autotry_calls == 6

    def test_threshold_follows_config(self, monkeypatch):
        screen, cfg = _build_screen(monkeypatch)
        cfg.get_value.side_effect = lambda key, default=None: 5 if key == "llm_takeover_after_retries" else default
        monkeypatch.setattr(tk, "try_takeover", lambda *a, **kw: True)
        assert screen.get_current_screen(max_retries=10) is True
        assert screen.autotry_calls == 4

    def test_not_triggered_without_autotry(self, monkeypatch):
        screen, _ = _build_screen(monkeypatch)
        monkeypatch.setattr(tk, "try_takeover", lambda *a, **kw: pytest.fail("autotry=False 不应触发接管"))
        assert screen.get_current_screen(autotry=False) is False

    def test_hook_swallows_errors(self, monkeypatch):
        screen, _ = _build_screen(monkeypatch)

        def boom(*args, **kwargs):
            raise RuntimeError("接管模块出错")
        monkeypatch.setattr(tk, "try_takeover", boom)
        assert screen.get_current_screen(max_retries=4) is False
        assert screen.autotry_calls == 4


PASSTHROUGH_ERRORS = [_InputInterrupted("游戏窗口已最小化"), _GameWindowLost("游戏进程已退出")]


class TestPassthroughExceptions:
    """上游后台输入的 InputInterrupted / GameWindowLost 必须原样向上抛，不能被钩子的 except Exception 吞掉。"""

    @pytest.mark.parametrize("error", PASSTHROUGH_ERRORS, ids=lambda e: type(e).__name__)
    def test_get_current_screen_hook_reraises(self, monkeypatch, error):
        screen, _ = _build_screen(monkeypatch)

        def boom(*args, **kwargs):
            raise error
        monkeypatch.setattr(tk, "try_takeover", boom)
        with pytest.raises(type(error)):
            screen.get_current_screen(max_retries=5)
        assert screen.autotry_calls == 2  # 在第 3 次失败的钩子里抛出，没有继续重试

    @pytest.mark.parametrize("error", PASSTHROUGH_ERRORS, ids=lambda e: type(e).__name__)
    def test_wait_for_screen_change_hook_reraises(self, monkeypatch, error):
        screen, _ = _build_screen(monkeypatch)

        def boom(*args, **kwargs):
            raise error
        monkeypatch.setattr(tk, "try_takeover", boom)
        screen.change_to = MagicMock()
        with pytest.raises(type(error)):
            screen.wait_for_screen_change("main", max_recursion=0)
        screen.change_to.assert_not_called()

    @pytest.mark.parametrize("error", PASSTHROUGH_ERRORS, ids=lambda e: type(e).__name__)
    def test_error_from_retry_change_to_propagates_and_resets_flag(self, monkeypatch, error):
        screen, _ = _build_screen(monkeypatch)
        monkeypatch.setattr(tk, "try_takeover", lambda *a, **kw: True)
        screen.change_to = MagicMock(side_effect=error)
        with pytest.raises(type(error)):
            screen.wait_for_screen_change("main", max_recursion=0)
        assert screen._llm_retrying is False

    def test_stop_request_not_swallowed(self, monkeypatch):
        from utils.pause import RunStopRequested
        screen, _ = _build_screen(monkeypatch)

        def stop(*args, **kwargs):
            raise RunStopRequested()
        monkeypatch.setattr(tk, "try_takeover", stop)
        with pytest.raises(RunStopRequested):
            screen.get_current_screen(max_retries=5)


class TestWaitForScreenChangeHook:
    def test_disabled_raises_original_error(self, monkeypatch):
        screen, _ = _build_screen(monkeypatch)
        screen.change_to = MagicMock()
        with pytest.raises(RuntimeError, match="无法切换到指定游戏界面"):
            screen.wait_for_screen_change("main", max_recursion=0)
        screen.change_to.assert_not_called()

    def test_no_key_raises_original_error(self, monkeypatch):
        screen, _ = _build_screen(monkeypatch)
        _enable(monkeypatch)
        screen.change_to = MagicMock()
        with pytest.raises(RuntimeError, match="无法切换到指定游戏界面"):
            screen.wait_for_screen_change("main", max_recursion=0)

    def test_takeover_success_retries_change_to_once(self, monkeypatch):
        screen, _ = _build_screen(monkeypatch)
        calls = []
        monkeypatch.setattr(tk, "try_takeover", lambda trigger, goal=tk.ANY_KNOWN, **kw: calls.append((trigger, goal)) or True)
        screen.change_to = MagicMock()
        screen.wait_for_screen_change("main", max_recursion=0)
        # 切换超时要以目标界面为接管目标：卡住时当前界面往往认得出来，用「任意已知界面」会直接判定成功
        assert calls == [("wait_for_screen_change", "main")]
        screen.change_to.assert_called_once_with("main", max_recursion=0)

    def test_takeover_failure_raises_original_error(self, monkeypatch):
        screen, _ = _build_screen(monkeypatch)
        monkeypatch.setattr(tk, "try_takeover", lambda *a, **kw: False)
        screen.change_to = MagicMock()
        with pytest.raises(RuntimeError, match="无法切换到指定游戏界面"):
            screen.wait_for_screen_change("main", max_recursion=0)
        screen.change_to.assert_not_called()

    def test_retry_failing_again_does_not_trigger_second_takeover(self, monkeypatch):
        screen, _ = _build_screen(monkeypatch)
        calls = []
        monkeypatch.setattr(tk, "try_takeover", lambda *a, **kw: calls.append(1) or True)
        # 重试的 change_to 再次等待超时：直接报错，不再接管
        screen.change_to = MagicMock(side_effect=lambda target, max_recursion=2: screen.wait_for_screen_change(target, max_recursion=0))
        with pytest.raises(RuntimeError, match="无法切换到指定游戏界面"):
            screen.wait_for_screen_change("main", max_recursion=0)
        assert calls == [1]
        assert screen._llm_retrying is False  # 标志已复位

    def test_not_triggered_while_retries_remain(self, monkeypatch):
        screen, _ = _build_screen(monkeypatch)
        monkeypatch.setattr(tk, "try_takeover", lambda *a, **kw: pytest.fail("还有重试次数时不应接管"))
        screen.change_to = MagicMock()
        screen.wait_for_screen_change("main", max_recursion=2)
        screen.change_to.assert_called_once_with("main", max_recursion=1)

    def test_real_per_run_limit_applies_to_hooks(self, monkeypatch):
        """真实的 try_takeover：每次运行最多接管 max_per_run 次，超过后走原来的报错流程。"""
        monkeypatch.setenv(lc.ENV_API_KEY, KEY)
        _enable(monkeypatch, max_per_run=1)
        built = []

        class _Fake:
            def run(self, trigger, goal):
                return tk.TakeoverResult(True, "ok")
        monkeypatch.setattr(tk, "_build_takeover", lambda cfg, key: built.append(1) or _Fake())
        screen, _ = _build_screen(monkeypatch)
        screen.change_to = MagicMock()
        screen.wait_for_screen_change("main", max_recursion=0)  # 第 1 次：接管成功
        with pytest.raises(RuntimeError):
            screen.wait_for_screen_change("main", max_recursion=0)  # 第 2 次：超过上限
        assert len(built) == 1
