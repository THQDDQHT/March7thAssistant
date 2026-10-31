# coding:utf-8
"""大模型接管：接管循环、步数/时间/次数上限、防重入、没有 key 时降级、记录、key 不泄露、暂停响应。"""
import json
import os
from unittest.mock import MagicMock

import pytest
from PIL import Image

from module.llm_agent import config as lc
from module.llm_agent import takeover as tk
from module.llm_agent.actions import ActionResult
from module.llm_agent.client import Decision, LLMError, Usage
from module.llm_agent.guard import Guard
from module.llm_agent.observe import Element, Observation
from utils.pause import RunStopRequested

KEY = "sk-test-SECRET-1234567890"


class _LogRecorder:
    def __init__(self):
        self.messages = []

    def __getattr__(self, name):
        return lambda message: self.messages.append((name, str(message)))


class _Pause:
    def __init__(self, stop_at=None):
        self.calls = 0
        self.stop_at = stop_at

    def checkpoint(self):
        self.calls += 1
        if self.stop_at is not None and self.calls >= self.stop_at:
            raise RunStopRequested()


class _FakeScreen:
    """get_current_screen 按脚本依次返回；脚本用完后一直返回 False。"""

    def __init__(self, script=(), screen_id="main"):
        self.script = list(script)
        self.current_screen = None
        self.screen_id = screen_id
        self.checks = 0

    def get_current_screen(self, autotry=True, max_retries=10):
        assert autotry is False
        self.checks += 1
        recognized = self.script.pop(0) if self.script else False
        self.current_screen = self.screen_id if recognized else None
        return recognized

    def get_name(self, screen_id):
        return screen_id


class _FakeClient:
    """按脚本依次给出决策：dict 是动作，Decision 原样返回，异常会抛出。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.usage = Usage()

    def decide(self, messages):
        self.calls.append(messages)
        item = self.script.pop(0)
        self.usage.add({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        if isinstance(item, Exception):
            raise item
        if isinstance(item, Decision):
            return item
        return Decision(item, json.dumps(item, ensure_ascii=False), raw_replies=[json.dumps(item, ensure_ascii=False)])


class _FakeExecutor:
    def __init__(self):
        self.executed = []

    def execute(self, action, observation):
        self.executed.append(action.to_dict())
        return ActionResult(True, f"执行了 {action.describe()}")


class _Clock:
    def __init__(self, step=0.0):
        self.now = 0.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


def _config(**overrides):
    values = dict(enable=True, max_steps=12)
    values.update(overrides)
    return lc.TakeoverConfig(**values)


def _make_observer(texts=("关闭", "稍后再看"), record=None):
    def observer(grid=False):
        if record is not None:
            record.append(grid)
        elements = [Element(i + 1, text, (100, 100 + i * 60, 300, 140 + i * 60)) for i, text in enumerate(texts)]
        return Observation(image=Image.new("RGB", (1920, 1080), (20, 20, 20)), elements=elements, grid=grid)
    return observer


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    tk.reset_state()
    monkeypatch.setattr(tk, "log", _LogRecorder())
    monkeypatch.setattr(tk, "pause_ctl", _Pause())
    monkeypatch.delenv(lc.ENV_API_KEY, raising=False)
    yield
    tk.reset_state()


def _build(tmp_path, script, screen_script=(), cfg=None, texts=("关闭", "稍后再看"), grid_record=None,
           clock=None, pause=None):
    client = _FakeClient(script)
    screen = _FakeScreen(screen_script)
    executor = _FakeExecutor()
    sleeps = []
    takeover = tk.Takeover(
        cfg or _config(), client, guard=Guard(), executor=executor,
        observer=_make_observer(texts, grid_record), screen=screen, screen_names=["主界面", "手机菜单"],
        recorder_factory=lambda: tk.Recorder(root=str(tmp_path / "rec"), api_key=KEY),
        clock=clock or _Clock(), sleep=sleeps.append,
    )
    takeover.sleeps = sleeps
    return takeover, client, screen, executor


def _user_text(messages):
    return messages[1]["content"][0]["text"]


class TestLoop:
    def test_click_then_recognized_is_success(self, tmp_path):
        takeover, client, screen, executor = _build(
            tmp_path, [{"thought": "关弹窗", "action": "click", "id": 1}], screen_script=[False, True])
        result = takeover.run("get_current_screen")
        assert result.success
        assert result.steps == 1
        assert executor.executed == [{"action": "click", "id": 1}]
        assert takeover.sleeps == [tk.SETTLE_SECONDS]  # 动作后等 1.5 秒再检测
        assert screen.checks == 2  # 开始前预检一次 + 动作后一次

    def test_already_recognized_costs_no_model_call(self, tmp_path):
        takeover, client, screen, executor = _build(tmp_path, [], screen_script=[True])
        result = takeover.run("wait_for_screen_change")
        assert result.success and result.steps == 0
        assert client.calls == []

    def test_done_without_recognition_is_failed_step_and_loop_continues(self, tmp_path):
        takeover, client, screen, executor = _build(
            tmp_path,
            [{"action": "done", "reason": "好了"}, {"action": "click", "id": 1}],
            screen_script=[False, False, True])
        result = takeover.run("t")
        assert result.success and result.steps == 2
        assert executor.executed == [{"action": "click", "id": 1}]
        assert "done：失败" in _user_text(client.calls[1])

    def test_done_with_recognition_is_success(self, tmp_path):
        takeover, client, screen, executor = _build(tmp_path, [{"action": "done", "reason": "好了"}], screen_script=[False, True])
        assert takeover.run("t").success
        assert executor.executed == []

    def test_give_up_ends_with_failure(self, tmp_path):
        takeover, client, screen, executor = _build(tmp_path, [{"action": "give_up", "reason": "不会"}])
        result = takeover.run("t")
        assert not result.success and result.steps == 1
        assert "不会" in result.reason

    def test_llm_error_ends_with_failure(self, tmp_path):
        takeover, client, screen, executor = _build(tmp_path, [LLMError("HTTP 500")])
        result = takeover.run("t")
        assert not result.success
        assert "模型调用失败" in result.reason

    def test_unparseable_decision_is_rejected_step(self, tmp_path):
        takeover, client, screen, executor = _build(
            tmp_path, [Decision(None, "乱", error="重答后仍然无法解析"), {"action": "click", "id": 1}], screen_script=[False, True])
        result = takeover.run("t")
        assert result.success and result.steps == 2
        assert "被拒绝" in _user_text(client.calls[1])

    def test_observer_failure_ends(self, tmp_path):
        takeover, client, screen, executor = _build(tmp_path, [])

        def boom(grid=False):
            raise RuntimeError("截图失败")
        takeover.observer = boom
        result = takeover.run("t")
        assert not result.success and "观察失败" in result.reason


class TestGuardInLoop:
    def test_rejected_action_not_executed_counted_and_fed_back(self, tmp_path):
        takeover, client, screen, executor = _build(
            tmp_path,
            [{"action": "click", "id": 1}, {"action": "click", "id": 2}],
            screen_script=[False, True],
            texts=("立即购买", "稍后再看"))
        result = takeover.run("t")
        assert executor.executed == [{"action": "click", "id": 2}]
        assert result.steps == 2  # 被拒绝的动作也计入步数
        assert "被拒绝" in _user_text(client.calls[1]) and "购买" in _user_text(client.calls[1])
        step1 = json.load(open(os.path.join(result.record_dir, "01.json"), encoding="utf-8"))
        assert step1["verdict"]["allowed"] is False
        assert step1["verdict"]["rule"] == "danger_target"
        assert "executed" not in step1

    def test_disallowed_key_not_pressed(self, tmp_path):
        takeover, client, screen, executor = _build(
            tmp_path, [{"action": "press_key", "key": "alt+f4"}, {"action": "give_up", "reason": "x"}])
        takeover.run("t")
        assert executor.executed == []

    def test_grid_flow(self, tmp_path):
        grids = []
        takeover, client, screen, executor = _build(
            tmp_path,
            [{"action": "click_grid", "cell": "H5"}, {"action": "show_grid"}, {"action": "click_grid", "cell": "H5"}],
            screen_script=[False, True], grid_record=grids)
        result = takeover.run("t")
        # 第 1 步：非网格模式点格子被拒；第 2 步 show_grid；第 3 步在网格图上点击成功
        assert grids == [False, False, True]
        assert executor.executed == [{"action": "click_grid", "cell": "H5"}]
        assert result.success and result.steps == 3
        assert "网格模式" in _user_text(client.calls[2])


class TestLimits:
    def test_max_steps(self, tmp_path):
        takeover, client, screen, executor = _build(
            tmp_path, [{"action": "wait", "seconds": 1}] * 10, cfg=_config(max_steps=3))
        result = takeover.run("t")
        assert not result.success
        assert result.steps == 3
        assert len(client.calls) == 3
        assert "步数上限" in result.reason

    def test_max_seconds(self, tmp_path):
        # 每次读时钟前进 100 秒：第二次循环就超过 180 秒
        takeover, client, screen, executor = _build(
            tmp_path, [{"action": "wait", "seconds": 1}] * 10, clock=_Clock(step=100))
        result = takeover.run("t")
        assert not result.success
        assert "时间上限" in result.reason
        assert len(client.calls) < 10

    def test_pause_checkpoint_called_every_step(self, tmp_path, monkeypatch):
        pause = _Pause()
        monkeypatch.setattr(tk, "pause_ctl", pause)
        takeover, client, screen, executor = _build(tmp_path, [{"action": "wait", "seconds": 1}] * 4, cfg=_config(max_steps=4))
        takeover.run("t")
        assert pause.calls == 4

    def test_stop_request_propagates_and_summary_written(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tk, "pause_ctl", _Pause(stop_at=2))
        takeover, client, screen, executor = _build(tmp_path, [{"action": "wait", "seconds": 1}] * 4)
        with pytest.raises(RunStopRequested):
            takeover.run("t")
        assert len(client.calls) == 1
        summaries = [os.path.join(root, "summary.json") for root, _, files in os.walk(tmp_path / "rec") if "summary.json" in files]
        assert len(summaries) == 1
        assert "interrupted" in json.load(open(summaries[0], encoding="utf-8"))["reason"]


class TestMessages:
    def test_each_request_has_system_text_and_exactly_one_image_in_user_message(self, tmp_path):
        takeover, client, screen, executor = _build(
            tmp_path, [{"action": "wait", "seconds": 1}] * 3, cfg=_config(max_steps=3))
        takeover.run("t")
        assert len(client.calls) == 3
        for messages in client.calls:
            assert messages[0]["role"] == "system" and isinstance(messages[0]["content"], str)
            images = [part for message in messages for part in (message["content"] if isinstance(message["content"], list) else [])
                      if part["type"] == "image_url"]
            assert len(images) == 1 and messages[1]["role"] == "user"  # 历史只有文字，只有当前这一步带图
        assert "第 3/3 步" in _user_text(client.calls[2])
        assert "第 1 步 wait" in _user_text(client.calls[2]) and "第 2 步 wait" in _user_text(client.calls[2])

    def test_screen_change_note_added_to_history(self, tmp_path):
        takeover, client, screen, executor = _build(tmp_path, [{"action": "click", "id": 1}, {"action": "give_up", "reason": "x"}])
        observations = iter([("A",), ("B",)])
        takeover.observer = lambda grid=False: _make_observer(next(observations))(grid)
        takeover.run("t")
        assert "画面有变化" in _user_text(client.calls[1])


class TestRecords:
    def test_files_and_content(self, tmp_path):
        takeover, client, screen, executor = _build(
            tmp_path, [{"thought": "关弹窗", "action": "click", "id": 1}], screen_script=[False, True])
        result = takeover.run("wait_for_screen_change")
        rec = result.record_dir
        assert sorted(os.listdir(rec)) == ["01.jpg", "01.json", "summary.json"]
        assert Image.open(os.path.join(rec, "01.jpg")).size == (1280, 720)
        step = json.load(open(os.path.join(rec, "01.json"), encoding="utf-8"))
        for field in ("elements", "model_replies", "model_action", "verdict", "executed", "screen_after"):
            assert field in step
        assert step["elements"][0]["text"] == "关闭"
        assert step["screen_after"] == {"recognized": True, "screen": "main"}
        summary = json.load(open(os.path.join(rec, "summary.json"), encoding="utf-8"))
        assert summary["trigger"] == "wait_for_screen_change"
        assert summary["goal"] == tk.ANY_KNOWN
        assert summary["success"] is True and summary["steps"] == 1
        assert summary["usage"]["total_tokens"] == 15
        assert "elapsed_seconds" in summary

    def test_key_not_in_records_or_logs(self, tmp_path, monkeypatch):
        monkeypatch.setenv(lc.ENV_API_KEY, KEY)
        # 模型回复和错误信息里都夹带了 key，落盘和日志前必须被脱敏
        leaked = Decision({"action": "click", "id": 1, "thought": f"my key {KEY}"}, f"raw {KEY}", raw_replies=[f"raw {KEY}"])
        takeover, client, screen, executor = _build(tmp_path, [leaked, LLMError(f"HTTP 401 {KEY}")], screen_script=[False, False])
        result = takeover.run("t")
        assert "模型调用失败" in result.reason and KEY not in result.reason
        for root, _, files in os.walk(tmp_path):
            for name in files:
                if name.endswith(".json"):
                    assert KEY not in open(os.path.join(root, name), encoding="utf-8").read()
        assert all(KEY not in message for _, message in tk.log.messages)

    def test_record_dir_not_reused(self, tmp_path):
        a = tk.Recorder(root=str(tmp_path))
        b = tk.Recorder(root=str(tmp_path))
        assert a.dir != b.dir


class TestTryTakeover:
    class _Built:
        def __init__(self, success=True, reenter=None, error=None):
            self.success, self.reenter, self.error = success, reenter, error

        def run(self, trigger, goal):
            if self.reenter:
                self.reenter()
            if self.error:
                raise self.error
            return tk.TakeoverResult(self.success, "ok")

    def _factory(self, **kwargs):
        built = []

        def build(cfg, key):
            built.append(key)
            return self._Built(**kwargs)
        return build, built

    def test_disabled_does_nothing(self, monkeypatch):
        monkeypatch.setenv(lc.ENV_API_KEY, KEY)
        build, built = self._factory()
        assert tk.try_takeover("t", cfg=_config(enable=False), build=build) is False
        assert built == []

    def test_no_key_degrades_with_single_warning(self):
        build, built = self._factory()
        for _ in range(3):
            assert tk.try_takeover("t", cfg=_config(), build=build) is False
        assert built == []
        warnings = [m for level, m in tk.log.messages if level == "warning"]
        assert len(warnings) == 1 and lc.ENV_API_KEY in warnings[0]

    def test_empty_key_counts_as_no_key(self, monkeypatch):
        monkeypatch.setenv(lc.ENV_API_KEY, "   ")
        build, built = self._factory()
        assert tk.try_takeover("t", cfg=_config(), build=build) is False
        assert built == []

    def test_success_returns_true_and_key_passed_to_build(self, monkeypatch):
        monkeypatch.setenv(lc.ENV_API_KEY, KEY)
        build, built = self._factory()
        assert tk.try_takeover("t", cfg=_config(), build=build) is True
        assert built == [KEY]
        assert tk.is_active() is False

    def test_failed_takeover_returns_false(self, monkeypatch):
        monkeypatch.setenv(lc.ENV_API_KEY, KEY)
        build, _ = self._factory(success=False)
        assert tk.try_takeover("t", cfg=_config(), build=build) is False

    def test_per_run_limit(self, monkeypatch):
        monkeypatch.setenv(lc.ENV_API_KEY, KEY)
        build, built = self._factory()
        results = [tk.try_takeover("t", cfg=_config(max_per_run=3), build=build) for _ in range(5)]
        assert results == [True, True, True, False, False]
        assert len(built) == 3

    def test_failed_attempts_also_count(self, monkeypatch):
        monkeypatch.setenv(lc.ENV_API_KEY, KEY)
        build, built = self._factory(success=False)
        for _ in range(4):
            tk.try_takeover("t", cfg=_config(max_per_run=2), build=build)
        assert len(built) == 2

    def test_no_reentry_while_active(self, monkeypatch):
        monkeypatch.setenv(lc.ENV_API_KEY, KEY)
        inner = []
        built = []

        def build(cfg, key):
            built.append(1)
            return self._Built(reenter=lambda: inner.append(tk.try_takeover("inner", cfg=_config(), build=build)))
        assert tk.try_takeover("outer", cfg=_config(), build=build) is True
        assert inner == [False]
        assert len(built) == 1
        assert tk.is_active() is False

    def test_exception_resets_active_and_returns_false(self, monkeypatch):
        monkeypatch.setenv(lc.ENV_API_KEY, KEY)
        build, _ = self._factory(error=RuntimeError(f"boom {KEY}"))
        assert tk.try_takeover("t", cfg=_config(), build=build) is False
        assert tk.is_active() is False
        assert all(KEY not in message for _, message in tk.log.messages)

    def test_stop_request_propagates_and_resets_active(self, monkeypatch):
        monkeypatch.setenv(lc.ENV_API_KEY, KEY)
        build, _ = self._factory(error=RunStopRequested())
        with pytest.raises(RunStopRequested):
            tk.try_takeover("t", cfg=_config(), build=build)
        assert tk.is_active() is False

    def test_default_build_creates_real_client(self, monkeypatch):
        monkeypatch.setenv(lc.ENV_API_KEY, KEY)
        takeover = tk._build_takeover(_config(base_url="https://x.test", model="m"), KEY)
        assert takeover.client.base_url == "https://x.test" and takeover.client.model == "m"


class TestPassthroughExceptions:
    """InputInterrupted / GameWindowLost（上游后台输入）和停止指令必须原样抛出，不能被接管吞掉。"""

    @pytest.fixture(params=["InputInterrupted", "GameWindowLost"])
    def error(self, request):
        from module.automation import background_input
        return getattr(background_input, request.param)("后台输入出错")

    def test_helper(self, error):
        from module.llm_agent.errors import passthrough_exceptions, reraise_passthrough
        assert type(error) in passthrough_exceptions()
        with pytest.raises(type(error)):
            reraise_passthrough(error)
        reraise_passthrough(RuntimeError("普通错误"))  # 普通异常不受影响
        reraise_passthrough(LLMError("x"))

    def test_error_in_screen_check_propagates(self, tmp_path, error):
        takeover, client, screen, executor = _build(tmp_path, [{"action": "click", "id": 1}], screen_script=[False])
        calls = {"n": 0}
        real = screen.get_current_screen

        def check(autotry=True, max_retries=10):
            calls["n"] += 1
            if calls["n"] == 2:  # 动作之后的检测
                raise error
            return real(autotry=autotry)
        screen.get_current_screen = check
        with pytest.raises(type(error)):
            takeover.run("t")
        assert executor.executed == [{"action": "click", "id": 1}]

    def test_error_in_first_screen_check_propagates(self, tmp_path, error):
        takeover, client, screen, executor = _build(tmp_path, [])
        screen.get_current_screen = MagicMock(side_effect=error)
        with pytest.raises(type(error)):
            takeover.run("t")
        assert client.calls == []

    def test_error_in_observer_propagates(self, tmp_path, error):
        takeover, client, screen, executor = _build(tmp_path, [])

        def boom(grid=False):
            raise error
        takeover.observer = boom
        with pytest.raises(type(error)):
            takeover.run("t")
        assert client.calls == []

    def test_error_in_executor_propagates(self, tmp_path, error):
        takeover, client, screen, executor = _build(tmp_path, [{"action": "click", "id": 1}] * 3)
        executor.execute = MagicMock(side_effect=error)
        with pytest.raises(type(error)):
            takeover.run("t")
        assert executor.execute.call_count == 1

    def test_ordinary_executor_error_is_failed_step_not_crash(self, tmp_path):
        takeover, client, screen, executor = _build(
            tmp_path, [{"action": "click", "id": 1}, {"action": "give_up", "reason": "x"}])
        executor.execute = MagicMock(side_effect=ValueError("后台输入不支持此按键"))
        result = takeover.run("t")
        assert not result.success and result.steps == 2
        assert "执行失败" in _user_text(client.calls[1])

    def test_summary_written_when_interrupted(self, tmp_path, error):
        takeover, client, screen, executor = _build(tmp_path, [])
        screen.get_current_screen = MagicMock(side_effect=error)
        with pytest.raises(type(error)):
            takeover.run("t")
        summaries = [os.path.join(root, "summary.json") for root, _, files in os.walk(tmp_path / "rec") if "summary.json" in files]
        assert "interrupted" in json.load(open(summaries[0], encoding="utf-8"))["reason"]

    def test_try_takeover_reraises_and_resets_active(self, monkeypatch, error):
        monkeypatch.setenv(lc.ENV_API_KEY, KEY)

        class _Boom:
            def run(self, trigger, goal):
                raise error
        with pytest.raises(type(error)):
            tk.try_takeover("t", cfg=_config(), build=lambda cfg, key: _Boom())
        assert tk.is_active() is False
        # 次数照常计入：重新来一次仍然受 max_per_run 约束
        assert tk._count == 1
