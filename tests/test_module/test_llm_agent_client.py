# coding:utf-8
"""大模型接管：HTTP 调用（mock）、JSON 解析容错、重试、key 不泄露。"""
import json

import pytest
import requests

from module.llm_agent import client as cl

KEY = "sk-test-SECRET-1234567890"


class _Resp:
    def __init__(self, status=200, content=None, text=None, usage=None, body=None):
        self.status_code = status
        if body is None:
            body = {"choices": [{"message": {"content": content}}], "usage": usage or {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}
        self._body = body
        self.text = text if text is not None else json.dumps(body, ensure_ascii=False)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _FakeHttp:
    """按脚本依次返回响应（或抛异常），并记录每次请求。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def logs(monkeypatch):
    messages = []

    class _Log:
        def __getattr__(self, name):
            return lambda message: messages.append((name, str(message)))

    monkeypatch.setattr(cl, "log", _Log())
    return messages


def _client(script, **kwargs):
    http = _FakeHttp(script)
    return cl.LLMClient("https://api.example.com/", "deepseek-flash", KEY, http=http, sleep=lambda s: None, **kwargs), http


class TestParseActionJson:
    def test_plain(self):
        assert cl.parse_action_json('{"thought": "x", "action": "click", "id": 3}') == {"thought": "x", "action": "click", "id": 3}

    def test_code_fence_and_prose(self):
        text = '好的，我来点击：\n```json\n{"thought": "关弹窗", "action": "click", "id": 2}\n```\n以上。'
        assert cl.parse_action_json(text)["id"] == 2

    def test_takes_first_object(self):
        text = '{"action": "wait", "seconds": 2} 或者 {"action": "click", "id": 1}'
        assert cl.parse_action_json(text)["action"] == "wait"

    def test_braces_inside_string_do_not_break(self):
        text = '{"thought": "文字里有 } 和 { 括号", "action": "press_key", "key": "esc"}'
        assert cl.parse_action_json(text)["key"] == "esc"

    def test_skips_junk_braces_before_real_json(self):
        assert cl.parse_action_json('{} {"action": "done", "reason": "ok"}')["action"] == "done"

    def test_skips_invalid_first_candidate(self):
        assert cl.parse_action_json('{not json} {"action": "show_grid"}')["action"] == "show_grid"

    @pytest.mark.parametrize("text", ["", "   ", None, "没有 JSON", '{"thought": "缺 action"}', '{"action": "click", "id": 3', "[1, 2]"])
    def test_failures(self, text):
        with pytest.raises(cl.ParseError):
            cl.parse_action_json(text)


class TestMessages:
    def test_system_prompt_is_plain_text_and_has_required_parts(self):
        names = [f"界面{i}" for i in range(56)]
        prompt = cl.build_system_prompt(names)
        assert isinstance(prompt, str)
        assert "界面0" in prompt and "界面55" in prompt and "56" in prompt
        for action in ("click", "show_grid", "click_grid", "press_key", "wait", "done", "give_up"):
            assert action in prompt
        assert "esc" in prompt and "购买" in prompt

    def test_system_prompt_describes_key_screens(self):
        prompt = cl.build_system_prompt(["主界面", "手机菜单"])
        assert "主界面：就是大世界的探索画面" in prompt
        for part in ("小地图", "任务追踪", "队伍头像", "疾跑"):
            assert part in prompt
        assert "手机菜单：在主界面按 esc 打开" in prompt
        assert "直接输出 done" in prompt
        assert len(prompt) < 1500  # 控制 token

    def test_step_message_has_image_table_step_and_history(self):
        message = cl.build_step_message(2, 12, "目标", "[1] 关闭 @ (50%, 50%)", ["第 1 步 click(id=3)：已执行"], b"\xff\xd8jpeg", False)
        assert message["role"] == "user"
        text = message["content"][0]["text"]
        assert "第 2/12 步" in text and "[1] 关闭" in text and "第 1 步 click(id=3)" in text
        image = message["content"][1]
        assert image["type"] == "image_url"
        assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_grid_hint(self):
        message = cl.build_step_message(1, 12, "目标", "", [], b"x", True)
        assert "网格模式" in message["content"][0]["text"]


class TestChat:
    def test_request_shape_and_key_only_in_header(self):
        client, http = _client([_Resp(content='{"action": "done"}')])
        messages = [{"role": "system", "content": "sys"}, cl.build_step_message(1, 12, "g", "t", [], b"jpg", False)]
        assert client.chat(messages) == '{"action": "done"}'
        call = http.calls[0]
        assert call["url"] == "https://api.example.com/chat/completions"
        assert call["headers"]["Authorization"] == f"Bearer {KEY}"
        assert call["timeout"] == 60
        assert call["json"]["model"] == "deepseek-flash"
        assert KEY not in json.dumps(call["json"])

    def test_images_only_in_user_messages(self):
        client, http = _client([_Resp(content="乱码"), _Resp(content='{"action": "done"}')])
        messages = [{"role": "system", "content": "sys"}, cl.build_step_message(1, 12, "g", "t", [], b"jpg", False)]
        client.decide(messages)
        for call in http.calls:
            for message in call["json"]["messages"]:
                if message["role"] != "user":
                    assert isinstance(message["content"], str)

    def test_usage_accumulates(self):
        client, _ = _client([_Resp(content="{}"), _Resp(content="{}")])
        client.chat([])
        client.chat([])
        assert client.usage.to_dict() == {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220, "requests": 2}

    def test_network_error_retried_twice_then_succeeds(self, logs):
        client, http = _client([requests.ConnectionError("boom " + KEY), requests.Timeout(), _Resp(content="ok")])
        assert client.chat([]) == "ok"
        assert len(http.calls) == 3

    def test_network_error_exhausted(self, logs):
        client, http = _client([requests.ConnectionError()] * 3)
        with pytest.raises(cl.LLMError):
            client.chat([])
        assert len(http.calls) == 3  # 首次 + 重试 2 次

    def test_server_error_retried_but_client_error_not(self, logs):
        client, http = _client([_Resp(status=503, body={}), _Resp(content="ok")])
        assert client.chat([]) == "ok"
        client, http = _client([_Resp(status=401, text="unauthorized")])
        with pytest.raises(cl.LLMError):
            client.chat([])
        assert len(http.calls) == 1

    def test_malformed_response(self):
        client, _ = _client([_Resp(body={"unexpected": 1})])
        with pytest.raises(cl.LLMError):
            client.chat([])

    def test_key_never_in_error_or_logs_or_repr(self, logs):
        # 服务端把请求头回显在错误正文里，也不能带出 key
        client, _ = _client([_Resp(status=400, text=f"bad request, Authorization: Bearer {KEY}")])
        with pytest.raises(cl.LLMError) as info:
            client.chat([])
        assert KEY not in str(info.value)
        client, _ = _client([requests.ConnectionError(f"proxy error {KEY}")] * 3)
        with pytest.raises(cl.LLMError) as info:
            client.chat([])
        assert KEY not in str(info.value)
        assert all(KEY not in message for _, message in logs)
        assert KEY not in repr(client)


class TestDecide:
    MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    def test_first_reply_ok(self):
        client, http = _client([_Resp(content='{"thought": "t", "action": "click", "id": 3}')])
        decision = client.decide(self.MESSAGES)
        assert decision.action == {"thought": "t", "action": "click", "id": 3}
        assert len(http.calls) == 1

    def test_unparseable_then_retry_succeeds(self, logs):
        client, http = _client([_Resp(content="我觉得应该点关闭"), _Resp(content='{"action": "press_key", "key": "esc"}')])
        decision = client.decide(self.MESSAGES)
        assert decision.action["key"] == "esc"
        assert len(http.calls) == 2
        retry_messages = http.calls[1]["json"]["messages"]
        assert retry_messages[-2] == {"role": "assistant", "content": "我觉得应该点关闭"}
        assert retry_messages[-1]["role"] == "user" and "JSON" in retry_messages[-1]["content"]
        assert decision.raw_replies == ["我觉得应该点关闭", '{"action": "press_key", "key": "esc"}']

    def test_unparseable_twice_gives_none(self, logs):
        client, http = _client([_Resp(content="乱"), _Resp(content="还是乱")])
        decision = client.decide(self.MESSAGES)
        assert decision.action is None
        assert decision.error
        assert len(http.calls) == 2  # 只重答一次

    def test_empty_reply_retried(self, logs):
        client, http = _client([_Resp(content=""), _Resp(content='{"action": "show_grid"}')])
        assert client.decide(self.MESSAGES).action == {"action": "show_grid"}
