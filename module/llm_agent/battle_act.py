"""有界的单次战斗动作：选中、核实、确认一次，再只观察。

不识别行动者或目标；所有动作作用于游戏当前目标。OTHER 包含动画和特殊技能模式，
因此绝不在 OTHER 发空格；确认后看不到行动证据时，不能把同一指令帧误报成完成。
依赖 McpTools 的输入、观察、互斥和护栏，不依赖 mcp 包。
"""
import math

from module.llm_agent import battle_hud as bh
from module.llm_agent.actions import Action, ActionResult
from module.llm_agent.errors import reraise_passthrough
from module.llm_agent.observe import Observation

POLL_SECONDS = 0.35
INITIAL_WAIT = 8.0
SELECT_WAIT = 8.0
STABLE_LOOKS = 2
MAX_LOOKS = 160  # 注入的时钟即使不前进，也不能无限循环
TIMEOUT_MIN, TIMEOUT_MAX = 5.0, 45.0
ACTIONS = ("basic", "skill", "ultimate", "confirm_pending")


class StopAction(Exception):
    pass


def run(tools, action, slot=None, timeout_seconds=25):
    """返回 ToolOutput；最多一个选择键和一个确认键，不调用通用连发工具。"""
    from module.llm_agent import mcp_tools as mt

    args = {"action": action, "slot": slot, "timeout_seconds": timeout_seconds}
    if not isinstance(action, str) or action not in ACTIONS:
        return tools._invalid("battle_act", args, "action 只能是 basic、skill、ultimate 或 confirm_pending。")
    if action == "ultimate":
        if type(slot) is not int or not 1 <= slot <= 4:
            return tools._invalid("battle_act", args, "ultimate 必须指定整数 slot（1–4）。")
    elif slot is not None:
        return tools._invalid("battle_act", args, "只有 ultimate 可以指定 slot；其余动作作用于当前选中技能/目标。")
    if isinstance(timeout_seconds, bool):
        return tools._invalid("battle_act", args, "timeout_seconds 必须是 5–45 秒的有限数字。")
    try:
        timeout_seconds = float(timeout_seconds)
    except (TypeError, ValueError):
        return tools._invalid("battle_act", args, "timeout_seconds 必须是 5–45 秒的有限数字。")
    if not math.isfinite(timeout_seconds) or not TIMEOUT_MIN <= timeout_seconds <= TIMEOUT_MAX:
        return tools._invalid("battle_act", args, "timeout_seconds 必须是 5–45 秒的有限数字。")

    started = tools._clock()
    try:
        with tools._exclusive(timeout=timeout_seconds):
            controller = _Action(tools, action, slot, started, timeout_seconds)
            return controller.execute(args)
    except mt.ToolBusy as error:
        return mt.ToolOutput(f"【未执行】{error}")


class _Action:
    def __init__(self, tools, action, slot, started, timeout):
        self.tools = tools
        self.action = action
        self.slot = slot
        self.started = started
        self.deadline = started + timeout
        self.trace = []
        self.before = None
        self.after = None
        self.confirm_attempted = False
        self.selection_attempted = False
        self.looks = 0
        self.evidence = []
        self.before_screen = tools._last_screen

    def note(self, event, **data):
        self.trace.append(dict(seconds=round(self.tools._clock() - self.started, 2), event=event, **data))

    def check_time(self):
        if self.tools._clock() >= self.deadline or self.looks >= MAX_LOOKS:
            raise StopAction("等待超时")

    def observe(self, full=False):
        self.check_time()
        problem = self.tools._ensure_front(False)
        if problem:
            raise StopAction(problem)
        # 出错后不能让上一次元素 id 仍被误用。
        self.tools._last = None
        self.after = None
        if full:
            observation = self.tools._take_observation(False)
        else:
            image = self.tools._screenshot()
            observation = Observation(image=image, hud=self.tools._hud_reader(image))
            # 不能把没有整屏 OCR 的图当成普通输入的护栏依据，也不能复用旧元素 id。
            self.tools._last = None
        self.looks += 1
        self.after = observation
        if self.before is None:
            self.before = observation
            self.before_screen = self.tools._last_screen
        hud = observation.hud
        self.note("observe", mode="full" if full else "light", hud=hud.to_dict() if hud is not None else None)
        return observation

    def pause(self, until=None):
        self.check_time()
        remaining = min(self.deadline, until or self.deadline) - self.tools._clock()
        if remaining > 0:
            self.tools._sleep(min(POLL_SECONDS, remaining))

    @staticmethod
    def phase(observation):
        hud = observation.hud
        if hud is not None and hud.phase == bh.PHASE_RESULT:
            return bh.PHASE_RESULT
        return hud.phase if hud is not None and hud.in_battle is True else None

    def eligible(self, observation, phase, selected=None):
        hud = observation.hud
        return (self.phase(observation) == phase and hud.auto_on is False
                and (selected is None or hud.selected == selected))

    def stable(self, phase, selected=None, timeout=SELECT_WAIT, selection=False):
        """相隔一次轮询的两张新图必须匹配；任何不匹配都不会确认。"""
        until = min(self.deadline, self.tools._clock() + timeout)
        count = 0
        checked_energy_message = False
        while True:
            if self.tools._clock() >= until:
                if selection:
                    raise StopAction("选择键未获核实（可能输入未生效、正在动画或状态未知）；没有发送空格确认")
                raise StopAction("未等到稳定且自动战斗明确关闭的可输入阶段；可能正在动画、特殊技能阶段或不在战斗中")
            observation = self.observe()
            if selection and self.action == "ultimate" and self.phase(observation) != bh.PHASE_ULT_PENDING and not checked_energy_message:
                # 只在序号未获响应时做一次整屏 OCR 查真实提示，正常轮询不跑 OCR。
                checked_energy_message = True
                observation = self.observe(full=True)
                if self.phase(observation) != bh.PHASE_ULT_PENDING:
                    text = "".join("".join(observation.all_texts()).split())
                    if "能量不足" in text:
                        self.note("energy_message", text="能量不足")
                        raise StopAction("画面明确提示「能量不足」；未确认出手")
            hud = observation.hud
            current = self.phase(observation)
            if current == bh.PHASE_RESULT:
                raise StopAction("已到战斗结算页，没有发起本次动作")
            if hud is not None and hud.auto_on is not False and current in (bh.PHASE_COMMAND, bh.PHASE_ULT_PENDING):
                raise StopAction("自动战斗已开启或状态不确定；必须明确关闭后才能手动出手")
            if phase == bh.PHASE_COMMAND and current == bh.PHASE_ULT_PENDING:
                raise StopAction("已有终结技待释放；不会用 Q/E 取消，请核实当前目标后显式调用 confirm_pending")
            if selection and self.action in ("basic", "skill") and current != bh.PHASE_COMMAND:
                raise StopAction("选中核验期间已离开普通指令阶段；不会等待下一回合再补发空格")
            count = count + 1 if self.eligible(observation, phase, selected) else 0
            if count >= STABLE_LOOKS:
                return observation
            if self.tools._clock() >= until:
                if selection:
                    raise StopAction("选择键未获核实（可能输入未生效、正在动画或状态未知）；没有发送空格确认")
                raise StopAction("未等到稳定且自动战斗明确关闭的可输入阶段；可能正在动画、特殊技能阶段或不在战斗中")
            self.pause(until)

    def press(self, key, phase, selected=None):
        """前台检查后再取新图和 guard；不沿用切前台之前的图。"""
        self.check_time()
        task = self.tools._active_task()
        if task:
            raise StopAction(task + "；任务占用期间不发送战斗输入")
        problem = self.tools._ensure_front(True)
        if problem:
            raise StopAction(problem)
        observation = self.observe(full=True)
        if not self.eligible(observation, phase, selected):
            raise StopAction("输入前状态已变化或自动状态不确定；停止发送按键")
        desired = {"q": bh.SELECTED_BASIC, "e": bh.SELECTED_SKILL}.get(key)
        if desired is not None and observation.hud.selected == desired:
            # 实机：对已经选中的 Q/E 再按一次会直接出手，不能把它当作幂等的选择键。
            self.note("selection_skipped", key=key, reason="输入前新图确认已选中")
            return observation
        if self.action == "ultimate" and key != "space":
            hud = observation.hud
            if next((item.ult_ready for item in hud.slots if item.index == self.slot), None) is False:
                raise StopAction("输入前该位置终结技圆盘识别为未满；没有按序号尝试")
        verdict = self.tools.guard.check(Action("press_key", {"key": key}), observation)
        self.note("guard", key=key, verdict=verdict.to_dict())
        if not verdict.allowed:
            raise StopAction("护栏拒绝：" + verdict.reason)
        self.check_time()
        task = self.tools._active_task()
        if task:
            raise StopAction(task + "；观察期间出现任务占用，停止发送输入")
        handler = getattr(self.tools.auto, "input_handler", None)
        if getattr(handler, "is_background", False) is not True and not self.tools.game.is_front():
            # 整屏 OCR 可能花几秒，期间焦点可能被用户切走。此处不抢回焦点并复用旧图。
            raise StopAction("观察期间游戏已失去前台焦点；没有发送按键，请重新观察")
        # 即使底层返回失败或抛异常，空格也可能已送达；必须先标记，之后绝不补发。
        if key == "space":
            if self.confirm_attempted:
                raise StopAction("已经尝试过一次确认，不会重复")
            self.confirm_attempted = True
        else:
            self.selection_attempted = True
        self.note("key_attempt", key=key)
        self.tools._last = None
        self.after = None
        result = self.tools.executor.press(key, 0)
        self.note("key_result", key=key, ok=result.ok, message=result.message)
        if not result.ok:
            raise StopAction("按键报告失败：" + result.message)
        return observation

    def wait_after_confirm(self, confirmed):
        baseline = confirmed.hud
        command_count = 0
        while True:
            self.pause()
            observation = self.observe()
            hud = observation.hud
            phase = self.phase(observation)
            if phase == bh.PHASE_RESULT:
                self.evidence.append("结算页")
                return "已观察到战斗结算：" + (hud.result_text or "结算页")
            # OTHER 只作为「离开指令」证据，从不在这里发送任何按键。
            if phase == bh.PHASE_OTHER and "阶段离开指令" not in self.evidence:
                self.evidence.append("阶段离开指令")
            if (hud is not None and baseline.skill_points is not None and hud.skill_points is not None
                    and hud.skill_points != baseline.skill_points and "技能点变化" not in self.evidence):
                self.evidence.append("技能点变化")
            # HUD 暂时隐藏不等于离开战斗，也不能单凭这一帧证明动作发生，继续有界观察。
            command_count = command_count + 1 if self.eligible(observation, bh.PHASE_COMMAND) else 0
            if command_count >= STABLE_LOOKS and self.evidence:
                return "已观察到状态变化并回到稳定可输入阶段（证据：" + "、".join(self.evidence) + "）"

    def execute(self, args):
        from module.llm_agent import mcp_tools as mt

        complete = False
        try:
            task = self.tools._active_task()
            if task:
                raise StopAction(task + "；任务占用期间不发送战斗输入")
            if self.tools.offline:
                raise StopAction("离线模式不能验证真实出手，请使用 battle_state 只读观察")
            phase = bh.PHASE_ULT_PENDING if self.action == "confirm_pending" else bh.PHASE_COMMAND
            initial = self.stable(phase, timeout=INITIAL_WAIT)
            if self.action == "ultimate":
                hud = initial.hud
                if hud.team_size is not None and self.slot > hud.team_size:
                    raise StopAction("指定位置不在识别到的队伍中")
                ready = next((item.ult_ready for item in hud.slots if item.index == self.slot), None)
                if ready is False:
                    raise StopAction("该位置终结技圆盘识别为未满；没有按序号尝试（不推断失败原因）")
                self.press(str(self.slot), bh.PHASE_COMMAND)
                self.stable(bh.PHASE_ULT_PENDING, selection=True)
                confirmed = self.press("space", bh.PHASE_ULT_PENDING)
            elif self.action == "confirm_pending":
                confirmed = self.press("space", bh.PHASE_ULT_PENDING)
            else:
                selected = bh.SELECTED_BASIC if self.action == "basic" else bh.SELECTED_SKILL
                if initial.hud.selected != selected:
                    self.press("q" if self.action == "basic" else "e", bh.PHASE_COMMAND)
                else:
                    self.note("selection_skipped", key="q" if self.action == "basic" else "e", reason="初始稳定画面已选中")
                self.stable(bh.PHASE_COMMAND, selected, selection=True)
                confirmed = self.press("space", bh.PHASE_COMMAND, selected)
            message = self.wait_after_confirm(confirmed)
            complete = True
            summary = "【已观察完成】" + message + "。仅作用于游戏当前目标；未识别行动者或目标身份。"
        except Exception as error:
            # 上游窗口/输入中断必须原样上抛，不再截图；finally 留下确认可能已送达的证据。
            self.note("stop", reason=self.tools._safe(error), confirm_attempted=self.confirm_attempted)
            reraise_passthrough(error)
            reason = str(error) if isinstance(error, StopAction) else self.tools._safe(error)
            if self.confirm_attempted:
                summary = "【执行结果不确定】已经尝试发送一次空格确认；" + reason + "。动作可能已执行，禁止盲目重试；先用 battle_state/observe 核实。"
            elif self.selection_attempted:
                summary = "【未发空格】" + reason + "。选择键也可能已触发动作；禁止盲目重试，先用 battle_state/observe 核实当前画面。"
            else:
                summary = "【未执行】" + reason + "。"
            if self.after is None:
                # 输入函数报错也可能已送达。只补一张只读新图，不再发键，不重启闭环。
                try:
                    if self.tools._ensure_front(False) is None:
                        image = self.tools._screenshot()
                        self.after = Observation(image=image, hud=self.tools._hud_reader(image))
                        self.tools._last = None
                        self.note("final_observe", hud=self.after.hud.to_dict() if self.after.hud is not None else None)
                except Exception as final_error:
                    self.note("final_observe_failed", reason=self.tools._safe(final_error))
                    self.after = None
                    reraise_passthrough(final_error)
        finally:
            # BaseException（停止指令等）也要留证据并清除旧 id；不在中断后继续操作。
            if self.after is None or "summary" not in locals():
                self.tools._last = None
            self.tools._write_record(
                "battle_act", args, self.started, result=ActionResult(complete, locals().get("summary", "流程被中断；确认可能已送达，禁止盲目重试")),
                before=self.before, before_screen=self.before_screen,
                extra={"trace": self.trace, "confirm_attempted": self.confirm_attempted,
                       "selection_attempted": self.selection_attempted, "evidence": self.evidence,
                       "hud_before": self.before.hud.to_dict() if self.before is not None and self.before.hud is not None else None,
                       "hud_after": self.after.hud.to_dict() if self.after is not None and self.after.hud is not None else None})
        image = None
        if self.after is not None:
            hud = self.after.hud
            if hud is not None:
                summary += "\n" + (bh.describe_hud(hud) or "战斗状态：不确定") + "\n" + bh.format_hud_details(hud)
            try:
                image = bh.render_hud_panel(self.after.image)
            except Exception as error:
                self.tools._warn("生成战斗动作结果图失败：" + self.tools._safe(error))
                image = self.after.annotated_jpeg()
        else:
            summary += "\n没有取得输入后的新图，旧元素编号已清除；请先 observe。"
        keys = [item["key"] for item in self.trace if item["event"] == "key_attempt"]
        summary += "\n按键轨迹：" + (" → ".join(keys) if keys else "无") + "；确认最多一次。完整 HUD/护栏/按键轨迹已记录到 battle_act 日志。"
        if self.tools._last is None:
            summary += "\n本图为 HUD 观察，旧元素编号已清除；需要点击元素时先 observe。"
        return mt.ToolOutput(summary, image)
