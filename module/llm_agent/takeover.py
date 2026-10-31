"""接管循环、上限控制和记录。

流程：观察 → 问模型 → 解析动作 → 护栏 → 执行 → 等 1.5 秒 → get_current_screen(autotry=False)。
只要识别到 screens.json 里任何一个已知界面就判定成功（goal 为 ANY_KNOWN 时），
不需要等模型说 done；模型说了 done 但没有识别到已知界面，算一次失败的步骤，循环继续。

上限：每次接管最多 llm_takeover_max_steps 步、最长 180 秒；每次运行（进程）最多接管 llm_takeover_max_per_run 次；
接管过程中不会再触发新的接管（防重入）。每一步都调用一次 pause_ctl.checkpoint()，响应暂停和停止。
"""
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from module.llm_agent import config as llm_config
from module.llm_agent.actions import ActionExecutor, normalize_action
from module.llm_agent.client import LLMClient, LLMError, build_step_message, build_system_prompt
from module.llm_agent.guard import RULE_INVALID_ACTION, Guard, Verdict
from module.llm_agent.observe import observe
from module.logger import log
from utils.pause import pause_ctl

ANY_KNOWN = "ANY_KNOWN"

SCREENS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "config", "screens.json")
RECORD_ROOT = os.path.join("logs", "llm_takeover")
# 每个动作执行后等待界面变化的时间（秒）
SETTLE_SECONDS = 1.5
# 写进历史的最多步数（历史只保留文字，控制 token 消耗）
HISTORY_MAX = 12


def load_screen_names(path: str = SCREENS_PATH) -> List[str]:
    """读出 screens.json 里全部已知界面的名称（用于 system 提示词）。"""
    with open(path, "r", encoding="utf-8") as file:
        return [item["name"] for item in json.load(file)]


# ---------------------------------------------------------------- 记录 ----

class Recorder:
    """把每次接管写到 logs/llm_takeover/<时间戳>/：NN.jpg、NN.json、summary.json。

    写记录失败不能影响接管本身；落盘前会把文本里的 API key 替换成 ***。
    """

    def __init__(self, root: str = RECORD_ROOT, api_key: Optional[str] = None):
        self._api_key = api_key
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(root, stamp)
        suffix = 1
        while os.path.exists(path):
            suffix += 1
            path = os.path.join(root, f"{stamp}-{suffix}")
        self.dir = path
        try:
            os.makedirs(self.dir, exist_ok=True)
        except OSError as e:
            _warn(f"大模型接管：无法创建记录目录 {self.dir}：{e}")
            self.dir = None

    def _dump(self, name: str, data: dict):
        if not self.dir:
            return
        try:
            text = llm_config.redact(json.dumps(data, ensure_ascii=False, indent=2), self._api_key)
            with open(os.path.join(self.dir, name), "w", encoding="utf-8") as file:
                file.write(text)
        except Exception as e:
            _warn(f"大模型接管：写记录失败：{e}")

    def write_step(self, step: int, image_jpeg: Optional[bytes], data: dict):
        if image_jpeg and self.dir:
            try:
                with open(os.path.join(self.dir, f"{step:02d}.jpg"), "wb") as file:
                    file.write(image_jpeg)
            except OSError as e:
                _warn(f"大模型接管：写标注图失败：{e}")
        self._dump(f"{step:02d}.json", data)

    def write_summary(self, data: dict):
        self._dump("summary.json", data)


# ---------------------------------------------------------------- 接管循环 ----

@dataclass
class TakeoverResult:
    success: bool
    reason: str
    steps: int = 0
    elapsed: float = 0.0
    record_dir: Optional[str] = None


@dataclass
class _Run:
    """一次接管过程中的可变状态。"""
    history: List[str] = field(default_factory=list)
    grid_next: bool = False
    steps: int = 0
    prev_texts: Optional[list] = None
    note_change: bool = False  # 下一步开头要不要把「画面有没有变化」补进上一条历史
    success: bool = False
    reason: str = ""
    screen_id: Optional[str] = None


def _safe(text) -> str:
    return llm_config.redact(str(text))


# 所有日志都先脱敏：模型回复、异常信息里万一带着 API key，也不会写进日志
def _debug(message):
    log.debug(_safe(message))


def _info(message):
    log.info(_safe(message))


def _warn(message):
    log.warning(_safe(message))


def _error(message):
    log.error(_safe(message))


class Takeover:
    def __init__(self, cfg: llm_config.TakeoverConfig, client: LLMClient, guard: Optional[Guard] = None,
                 executor: Optional[ActionExecutor] = None, observer: Callable = observe,
                 screen=None, screen_names: Optional[List[str]] = None, recorder_factory: Callable = None,
                 clock: Callable = time.monotonic, sleep: Callable = time.sleep):
        self.cfg = cfg
        self.client = client
        self.guard = guard or Guard(cfg.extra_danger_words)
        self.executor = executor or ActionExecutor()
        self.observer = observer
        self._screen = screen
        self._screen_names = screen_names
        self._recorder_factory = recorder_factory or (lambda: Recorder(api_key=llm_config.get_api_key()))
        self._clock = clock
        self._sleep = sleep

    @property
    def screen(self):
        if self._screen is None:
            from module.screen import screen
            self._screen = screen
        return self._screen

    def screen_names(self) -> List[str]:
        if self._screen_names is None:
            self._screen_names = load_screen_names()
        return self._screen_names

    def _goal_text(self, goal: str) -> str:
        if goal == ANY_KNOWN:
            return "回到脚本认识的任何一个界面（见 system 消息里的界面清单）"
        try:
            return f"回到界面「{self.screen.get_name(goal)}」"
        except Exception:
            return f"回到界面 {goal}"

    def _check_screen(self, goal: str):
        """用 get_current_screen(autotry=False) 检测当前界面，返回 (是否达成目标, 界面 id)。"""
        try:
            screen = self.screen
            recognized = bool(screen.get_current_screen(autotry=False))
            screen_id = screen.current_screen if recognized else None
        except Exception as e:
            _warn(f"大模型接管：检测界面失败：{_safe(e)}")
            return False, None
        if not recognized:
            return False, None
        return (goal == ANY_KNOWN or screen_id == goal), screen_id

    def run(self, trigger: str, goal: str = ANY_KNOWN) -> TakeoverResult:
        recorder = self._recorder_factory()
        started = self._clock()
        started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        run = _Run()
        _warn(f"大模型接管开始：触发点={trigger} 目标={goal} 模型={self.cfg.model}")
        try:
            self._loop(run, goal, recorder, started)
        except BaseException as e:
            # 暂停/停止指令（RunStopRequested）等：记录后原样抛出
            run.reason = f"interrupted: {type(e).__name__}"
            raise
        finally:
            elapsed = self._clock() - started
            recorder.write_summary({
                "trigger": trigger, "goal": goal, "started_at": started_at,
                "success": run.success, "reason": run.reason, "screen": run.screen_id,
                "steps": run.steps, "elapsed_seconds": round(elapsed, 2),
                "model": self.cfg.model, "usage": self.client.usage.to_dict(),
            })
        if run.success:
            _info(f"大模型接管成功：{run.steps} 步，当前界面 {run.screen_id}，记录目录 {recorder.dir}")
        else:
            _warn(f"大模型接管失败：{run.reason}（{run.steps} 步），记录目录 {recorder.dir}")
        return TakeoverResult(run.success, run.reason, run.steps, elapsed, recorder.dir)

    def _loop(self, run: _Run, goal: str, recorder: Recorder, started: float):
        # 先检测一次：可能在触发到接管之间画面已经恢复（比如只是加载慢），省一次模型调用
        reached, screen_id = self._check_screen(goal)
        if reached:
            run.success, run.reason, run.screen_id = True, "已经是已知界面，无需操作", screen_id
            return
        system = {"role": "system", "content": build_system_prompt(self.screen_names())}
        while True:
            if run.steps >= self.cfg.max_steps:
                run.reason = f"达到步数上限 {self.cfg.max_steps}"
                return
            if self._clock() - started >= self.cfg.max_seconds:
                run.reason = f"达到时间上限 {self.cfg.max_seconds} 秒"
                return
            pause_ctl.checkpoint()
            run.steps += 1
            if not self._step(run, goal, recorder, system, started):
                return

    def _step(self, run: _Run, goal: str, recorder: Recorder, system: dict, started: float) -> bool:
        """执行一步。返回 True 继续循环，False 结束（run.success / run.reason 已设置）。"""
        step = run.steps
        step_start = self._clock()
        record = {"step": step}
        image_jpeg = None
        try:
            grid = run.grid_next
            run.grid_next = False
            try:
                observation = self.observer(grid=grid)
                image_jpeg = observation.annotated_jpeg()
            except Exception as e:
                run.reason = f"观察失败：{_safe(e)}"
                record["error"] = run.reason
                return False
            record["grid"] = grid
            record["elements"] = [element.to_dict() for element in observation.elements]

            texts = observation.all_texts()
            if run.note_change and run.history:
                run.history[-1] += "；画面有变化" if texts != run.prev_texts else "；画面没有明显变化"
            run.note_change = False
            run.prev_texts = texts

            message = build_step_message(step, self.cfg.max_steps, self._goal_text(goal), observation.table_text(),
                                         run.history[-HISTORY_MAX:], image_jpeg, grid)
            try:
                decision = self.client.decide([system, message])
            except LLMError as e:
                run.reason = f"模型调用失败：{_safe(e)}"
                record["error"] = run.reason
                return False
            record["model_replies"] = decision.raw_replies
            record["model_action"] = decision.action
            record["usage_total"] = self.client.usage.to_dict()

            if decision.action is None:
                action, error = None, decision.error
            else:
                action, error = normalize_action(decision.action)
            if action is None:
                verdict = Verdict(False, RULE_INVALID_ACTION, error)
                described = "(无法解析的动作)"
            else:
                verdict = self.guard.check(action, observation)
                described = action.describe()
                record["action"] = action.to_dict()
            record["verdict"] = verdict.to_dict()

            if not verdict.allowed:
                # 被拒绝的动作反馈给模型，同时计入步数
                _warn(f"大模型接管：第 {step} 步 {described} 被护栏拒绝：{verdict.reason}")
                run.history.append(f"第 {step} 步 {described}：被拒绝（{verdict.reason}），不要重复")
                return True

            thought = str((decision.action or {}).get("thought", ""))[:80]
            _info(f"大模型接管：第 {step} 步 {described} {thought}")

            if action.name == "give_up":
                run.reason = f"模型放弃：{action.params.get('reason', '')}"
                return False
            if action.name == "show_grid":
                run.grid_next = True
                run.history.append(f"第 {step} 步 show_grid：下一张图叠加了网格")
                return True
            if action.name == "done":
                reached, screen_id = self._check_screen(goal)
                record["screen_after"] = {"recognized": reached, "screen": screen_id}
                if reached:
                    run.success, run.reason, run.screen_id = True, "模型 done 且已识别到已知界面", screen_id
                    return False
                run.history.append(f"第 {step} 步 done：失败，画面仍然不是已知界面，继续处理")
                return True

            result = self.executor.execute(action, observation)
            record["executed"] = {"ok": result.ok, "message": result.message}
            self._sleep(SETTLE_SECONDS)
            reached, screen_id = self._check_screen(goal)
            record["screen_after"] = {"recognized": reached, "screen": screen_id}
            if reached:
                run.success, run.reason, run.screen_id = True, "已识别到已知界面", screen_id
                return False
            run.history.append(f"第 {step} 步 {described}：{result.message or '已执行'}，仍未识别到已知界面")
            run.note_change = True
            return True
        finally:
            record["elapsed_seconds"] = round(self._clock() - step_start, 2)
            recorder.write_step(step, image_jpeg, record)


# ---------------------------------------------------------------- 入口 ----

_state_lock = threading.Lock()
_active = False  # 防重入：接管过程中不会再触发新的接管
_count = 0  # 本次运行（进程）已发起的接管次数
_warned_no_key = False


def is_active() -> bool:
    return _active


def reset_state():
    """清空防重入标志和计数（测试用）。"""
    global _active, _count, _warned_no_key
    with _state_lock:
        _active, _count, _warned_no_key = False, 0, False


def _build_takeover(cfg: llm_config.TakeoverConfig, api_key: str) -> Takeover:
    client = LLMClient(cfg.base_url, cfg.model, api_key)
    return Takeover(cfg, client)


def try_takeover(trigger: str, goal: str = ANY_KNOWN, cfg: Optional[llm_config.TakeoverConfig] = None,
                 build: Callable = None) -> bool:
    """screen.py 钩子调用的入口：能接管并成功返回 True，其余情况（关闭、没有 key、重入、超次数、失败）返回 False。

    接管关闭或没有 key 时什么都不做，调用方走原来的流程。
    """
    global _active, _count, _warned_no_key
    if cfg is None:
        cfg = llm_config.load_config()
    if not cfg.enable:
        return False
    api_key = llm_config.get_api_key()
    if not api_key:
        if not _warned_no_key:
            _warned_no_key = True
            _warn(f"大模型接管已开启，但没有设置环境变量 {llm_config.ENV_API_KEY}，本次运行不会接管")
        return False
    with _state_lock:
        if _active:
            _debug("大模型接管：正在接管中，不再重复触发")
            return False
        if _count >= cfg.max_per_run:
            _warn(f"大模型接管：本次运行已接管 {_count} 次，达到上限 {cfg.max_per_run}，不再接管")
            return False
        _active = True
        _count += 1
    try:
        takeover = (build or _build_takeover)(cfg, api_key)
        return takeover.run(trigger, goal).success
    except Exception as e:
        _error(f"大模型接管出错：{_safe(e)}")
        return False
    finally:
        with _state_lock:
            _active = False
