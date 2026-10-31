"""DeepSeek 调用与 JSON 解析。

接口是 OpenAI 兼容的 `POST {base_url}/chat/completions`，图片用
`{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`，
**图片只能放在 user 消息里**（system 消息是纯文字）。
API key 只从环境变量读取，只出现在 Authorization 请求头里，任何日志和异常信息里都不带它。
"""
import base64
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests

from module.llm_agent.config import NETWORK_RETRIES, REQUEST_TIMEOUT, redact
from module.logger import log

RETRY_STATUS = (429, 500, 502, 503, 504)
RETRY_DELAY = 1.0
# 错误信息里最多带回的响应正文长度
ERROR_BODY_MAX = 200

JSON_RETRY_PROMPT = (
    "你上一条回复不是合法的 JSON。请只输出一个 JSON 对象，不要带任何其它文字或代码块标记，"
    '例如：{"thought": "...", "action": "click", "id": 3}'
)


class LLMError(Exception):
    """模型调用失败（网络、HTTP 状态、响应格式）。"""


class ParseError(ValueError):
    """模型回复里找不到合法的动作 JSON。"""


# ------------------------------------------------------------- JSON 解析 ----

def _balanced_object_end(text: str, start: int) -> int:
    """从 text[start] == '{' 开始找配对的 '}'（跳过字符串里的括号），找不到返回 -1。"""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def parse_action_json(text: str) -> dict:
    """从模型回复里取第一个能解析、且带 action 字段的 `{...}`。

    模型可能在 JSON 前后加说明文字、```json 代码块标记，这里都能容忍。
    :raises ParseError: 找不到合法的动作 JSON。
    """
    if not isinstance(text, str) or not text.strip():
        raise ParseError("回复为空")
    position = text.find("{")
    while position != -1:
        end = _balanced_object_end(text, position)
        if end != -1:
            try:
                data = json.loads(text[position:end + 1])
            except ValueError:
                data = None
            if isinstance(data, dict) and "action" in data:
                return data
        position = text.find("{", position + 1)
    raise ParseError("回复里没有带 action 字段的 JSON 对象")


# ------------------------------------------------------------- 提示词 ----

def build_system_prompt(screen_names: List[str]) -> str:
    """system 消息（纯文字）：任务、已知界面、动作、护栏、常见处理方式。"""
    names = "、".join(screen_names)
    return (
        "你在操作《崩坏：星穹铁道》PC 版。自动化脚本（三月七小助手）遇到了它认不出的画面，"
        "你的目标是关掉意外弹窗、跳过多余的界面，把画面拉回脚本能识别的界面，然后交还给脚本。\n\n"
        f"脚本认识的界面共 {len(screen_names)} 个：{names}。\n"
        "只要画面变成其中任何一个，接管就算成功（系统会自动检测，不需要等你说完成）。\n\n"
        "每一步你会收到：带编号方框的游戏截图、元素表（格式 [编号] 文字 @ (横向%, 纵向%)）、步数、上一步动作和结果。"
        "你每一步只能输出一个 JSON 对象，不要输出其它文字：\n"
        '{"thought": "简短的判断", "action": "<动作名>", ...参数}\n\n'
        "可用动作：\n"
        '- click：点击元素，参数 id（元素编号）。例：{"thought": "关闭弹窗", "action": "click", "id": 3}\n'
        "- show_grid：下一张截图改为叠加 16×9 网格（列 A-P、行 1-9），用来点没有文字的图标按钮。\n"
        "- click_grid：点击格子中心，参数 cell（如 H5），只能在网格模式下使用。\n"
        f"- press_key：按键，参数 key，只允许 esc、enter、space、f。\n"
        "- wait：等待，参数 seconds（1-5），用于等加载或动画。\n"
        "- done：你认为已经回到正常界面，参数 reason。\n"
        "- give_up：你判断自己处理不了，参数 reason。\n\n"
        "护栏（由程序强制执行，不靠你自觉）：绝不购买、抽卡、充值、兑换、分解、出售、删除或消耗任何资源；"
        "点击危险文字、画面上有危险文字时点确认类按钮、画面上有危险文字时用网格点击，都会被程序拒绝。"
        "被拒绝的动作不要重复，换一个更安全的办法，或者 give_up。\n\n"
        "常见处理方式：优先点「关闭」「返回」「稍后再看」「跳过」，或者按 esc；"
        "遇到对话就点「点击继续」或「点击空白处关闭」；加载中就 wait；"
        "不要点任何会花费资源、进入商店或抽卡的按钮。"
    )


def build_step_message(step: int, max_steps: int, goal_text: str, table_text: str,
                       history_lines: List[str], image_jpeg: bytes, grid: bool) -> dict:
    """每一步的 user 消息：当前标注图 + 元素表 + 步数 + 历史（历史只有文字）。"""
    parts = [f"第 {step}/{max_steps} 步。目标：{goal_text}"]
    if history_lines:
        parts.append("之前的步骤（只有文字）：\n" + "\n".join(history_lines))
    else:
        parts.append("这是第一步，还没有历史步骤。")
    if grid:
        parts.append("当前是网格模式：图上叠加了 16×9 网格（列 A-P、行 1-9），可以用 click_grid 点格子中心。")
    parts.append("当前元素表：\n" + table_text)
    parts.append("请输出下一步动作的 JSON。")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "\n\n".join(parts)},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")}},
        ],
    }


# ------------------------------------------------------------- 客户端 ----

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0

    def add(self, usage: Optional[dict]):
        self.requests += 1
        if isinstance(usage, dict):
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            self.total_tokens += int(usage.get("total_tokens") or 0)

    def to_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens, "requests": self.requests}


@dataclass
class Decision:
    """一次决策的结果：解析出的动作 dict 和模型原始回复；解析失败时 action 为 None。"""
    action: Optional[dict]
    raw: str
    error: str = ""
    raw_replies: List[str] = field(default_factory=list)


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = REQUEST_TIMEOUT,
                 retries: int = NETWORK_RETRIES, http=None, sleep=time.sleep):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self._http = http if http is not None else requests
        self._sleep = sleep
        self.usage = Usage()

    def __repr__(self):
        return f"LLMClient(base_url={self.base_url!r}, model={self.model!r})"

    def _safe(self, text) -> str:
        return redact(str(text), self._api_key)

    def chat(self, messages: List[dict]) -> str:
        """调用 chat/completions，返回模型回复文字。网络错误和 429/5xx 重试 retries 次。

        :raises LLMError: 重试用尽，或者遇到不该重试的 HTTP 错误、响应格式不对。
        """
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 512}
        last_error = "未知错误"
        for attempt in range(self.retries + 1):
            if attempt:
                self._sleep(RETRY_DELAY * attempt)
            try:
                response = self._http.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last_error = f"网络错误：{type(e).__name__}"
                log.warning(f"大模型接管：请求失败（第 {attempt + 1} 次）：{last_error}")
                continue
            status = response.status_code
            if status in RETRY_STATUS:
                last_error = f"HTTP {status}"
                log.warning(f"大模型接管：请求失败（第 {attempt + 1} 次）：{last_error}")
                continue
            if status != 200:
                body = self._safe((response.text or "")[:ERROR_BODY_MAX])
                raise LLMError(f"HTTP {status}：{body}")
            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as e:
                raise LLMError(f"响应格式不对：{type(e).__name__}") from None
            self.usage.add(data.get("usage"))
            return content if isinstance(content, str) else ""
        raise LLMError(f"重试 {self.retries} 次后仍然失败：{last_error}")

    def decide(self, messages: List[dict]) -> Decision:
        """让模型选下一步动作。回复解析失败就让它重答一次，再失败返回 action=None 的 Decision。"""
        raw = self.chat(messages)
        replies = [raw]
        try:
            return Decision(parse_action_json(raw), raw, raw_replies=replies)
        except ParseError as first_error:
            log.warning(f"大模型接管：回复解析失败（{first_error}），让模型重答一次")
        retry_messages = messages + [
            {"role": "assistant", "content": raw or "(空回复)"},
            {"role": "user", "content": JSON_RETRY_PROMPT},
        ]
        raw2 = self.chat(retry_messages)
        replies.append(raw2)
        try:
            return Decision(parse_action_json(raw2), raw2, raw_replies=replies)
        except ParseError as second_error:
            return Decision(None, raw2, error=f"重答后仍然无法解析：{second_error}", raw_replies=replies)
