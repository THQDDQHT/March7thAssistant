# coding:utf-8
"""第二期 MCP server 接线：token 中间件、工具注册、content 列表、启动参数。依赖 mcp 包，没装就跳过。"""
import asyncio
import base64
import os

import pytest
from PIL import Image

pytest.importorskip("mcp")

from mcp.types import ImageContent, TextContent  # noqa: E402

from module.llm_agent import mcp_server as ms  # noqa: E402
from module.llm_agent.mcp_tools import ToolOutput  # noqa: E402

TOKEN = "unit-test-token-0123456789"
JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
EXPECTED_TOOLS = sorted("""observe click click_grid press_key press_keys scroll drag wait list_screens goto_screen list_m7a_tasks
    run_m7a_task stop_m7a_task get_status web_search fetch_page leave_for_user resolve_todo list_todo_for_user list_characters get_m7a_config set_m7a_config""".split())


class FakeTools:
    """每个工具方法记录调用并返回 ToolOutput；observe / click 带图。"""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def method(*args):
            self.calls.append((name, args))
            image = JPEG if name in ("observe", "click") else None
            return ToolOutput(f"{name} 的结果", image)
        return method


async def _drive(app, scope_extra=None, headers=()):
    scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": list(headers)}
    scope.update(scope_extra or {})
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return sent


class TestBearerAuthMiddleware:
    def _app(self):
        calls = []

        async def inner(scope, receive, send):
            calls.append(scope["type"])
            if scope["type"] == "http":
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": b"ok"})

        return ms.BearerAuthMiddleware(inner, TOKEN), calls

    def test_missing_token_is_401(self):
        app, calls = self._app()
        sent = asyncio.run(_drive(app))
        assert sent[0]["status"] == 401 and calls == []
        assert (b"www-authenticate", b'Bearer realm="m7a-mcp"') in sent[0]["headers"]

    @pytest.mark.parametrize("value", [b"Bearer wrong", b"Bearer ", b"Basic " + TOKEN.encode(), TOKEN.encode(), b"bearer"])
    def test_bad_token_is_401(self, value):
        app, calls = self._app()
        sent = asyncio.run(_drive(app, headers=[(b"authorization", value)]))
        assert sent[0]["status"] == 401 and calls == []

    def test_good_token_passes(self):
        app, calls = self._app()
        sent = asyncio.run(_drive(app, headers=[(b"authorization", b"Bearer " + TOKEN.encode())]))
        assert sent[0]["status"] == 200 and calls == ["http"]

    def test_scheme_is_case_insensitive(self):
        app, calls = self._app()
        sent = asyncio.run(_drive(app, headers=[(b"Authorization", b"bearer " + TOKEN.encode())]))
        assert sent[0]["status"] == 200

    def test_lifespan_passes_through_and_websocket_rejected(self):
        app, calls = self._app()
        asyncio.run(_drive(app, {"type": "lifespan"}))
        assert calls == ["lifespan"]
        sent = asyncio.run(_drive(app, {"type": "websocket"}))
        assert sent == [{"type": "websocket.close", "code": 1008}]


class TestContent:
    def test_text_plus_jpeg(self):
        blocks = ms.to_content(ToolOutput("你好", JPEG))
        assert isinstance(blocks[0], TextContent) and blocks[0].text == "你好"
        assert isinstance(blocks[1], ImageContent) and blocks[1].mimeType == "image/jpeg"
        assert base64.b64decode(blocks[1].data) == JPEG

    def test_text_only(self):
        assert len(ms.to_content(ToolOutput("x"))) == 1


class TestRegistration:
    def test_all_tools_registered_with_schemas(self):
        mcp = ms.create_mcp(FakeTools())
        tools = asyncio.run(mcp.list_tools())
        assert sorted(tool.name for tool in tools) == EXPECTED_TOOLS
        by_name = {tool.name: tool for tool in tools}
        assert by_name["click"].inputSchema["required"] == ["id"]
        assert by_name["click"].inputSchema["properties"]["id"]["type"] == "integer"
        assert by_name["press_key"].inputSchema["properties"]["hold_seconds"]["default"] == 0
        assert by_name["press_keys"].inputSchema["required"] == ["keys"]
        assert by_name["press_keys"].inputSchema["properties"]["keys"]["type"] == "array"
        assert by_name["press_keys"].inputSchema["properties"]["gap_seconds"]["default"] == 0.3
        assert by_name["scroll"].inputSchema["required"] == ["direction"]
        assert by_name["fetch_page"].inputSchema["properties"]["max_chars"]["default"] == 8000
        assert all(tool.description for tool in tools)

    def test_observation_tool_returns_text_and_jpeg(self):
        tools = FakeTools()
        mcp = ms.create_mcp(tools)
        result = asyncio.run(mcp.call_tool("observe", {"grid": True}))
        blocks = result[0] if isinstance(result, tuple) else result
        assert [block.type for block in blocks] == ["text", "image"]
        assert blocks[1].mimeType == "image/jpeg"
        assert tools.calls == [("observe", (True,))]

    def test_arguments_are_forwarded(self):
        tools = FakeTools()
        mcp = ms.create_mcp(tools)

        async def run():
            await mcp.call_tool("press_key", {"key": "w", "hold_seconds": 2})
            await mcp.call_tool("press_keys", {"keys": ["space", "2"]})
            await mcp.call_tool("press_keys", {"keys": ["e"], "gap_seconds": 0.5})
            await mcp.call_tool("scroll", {"direction": "up", "amount": 2, "cell": "H5"})
            await mcp.call_tool("drag", {"from_cell": "A1", "to_cell": "B2"})
            await mcp.call_tool("leave_for_user", {"item": "活动", "reason": "难"})
            await mcp.call_tool("web_search", {"query": "攻略"})
            await mcp.call_tool("resolve_todo", {"item": "活动"})
            await mcp.call_tool("get_m7a_config", {"keys": ["forgottenhall_level"]})
            await mcp.call_tool("get_m7a_config", {})
            await mcp.call_tool("set_m7a_config", {"key": "forgottenhall_level", "value": [1, 12]})
            await mcp.call_tool("set_m7a_config", {"key": "forgottenhall_enable", "value": True})
            await mcp.call_tool("list_characters", {})

        asyncio.run(run())
        assert tools.calls == [
            ("press_key", ("w", 2.0)), ("press_keys", (["space", "2"], 0.3)), ("press_keys", (["e"], 0.5)), ("scroll", ("up", 2, None, "H5")), ("drag", ("A1", "B2")),
            ("leave_for_user", ("活动", "难", "")), ("web_search", ("攻略", 8)),
            ("resolve_todo", ("活动", "")), ("get_m7a_config", (["forgottenhall_level"],)), ("get_m7a_config", ([],)),
            ("set_m7a_config", ("forgottenhall_level", [1, 12])), ("set_m7a_config", ("forgottenhall_enable", True)),
            ("list_characters", ())]

    def test_text_only_tool(self):
        mcp = ms.create_mcp(FakeTools())
        result = asyncio.run(mcp.call_tool("get_status", {}))
        blocks = result[0] if isinstance(result, tuple) else result
        assert [block.type for block in blocks] == ["text"]


class TestHttpEndToEnd:
    def _client(self, token):
        from starlette.testclient import TestClient
        return TestClient(ms.build_app(FakeTools(), "127.0.0.1", 8770, token), base_url="http://127.0.0.1:8770")

    BODY = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
    HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}

    def test_401_without_or_with_wrong_token(self):
        with self._client(TOKEN) as client:
            assert client.post("/mcp", json=self.BODY, headers=self.HEADERS).status_code == 401
            assert client.post("/mcp", json=self.BODY, headers={**self.HEADERS, "Authorization": "Bearer nope"}).status_code == 401

    def test_initialize_with_token(self):
        with self._client(TOKEN) as client:
            response = client.post("/mcp", json=self.BODY, headers={**self.HEADERS, "Authorization": f"Bearer {TOKEN}"})
            assert response.status_code == 200
            assert "m7a" in response.text

    def test_no_auth_mode_has_no_middleware(self):
        with self._client(None) as client:
            assert client.post("/mcp", json=self.BODY, headers=self.HEADERS).status_code == 200


class TestMain:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ms, "setup_logging", lambda: None)
        cwd = os.getcwd()
        monkeypatch.delenv("M7A_MCP_TOKEN", raising=False)
        # main() 会往 os.environ 里写这两个变量：先 setenv 一次，让 monkeypatch 在测试结束时还原，不泄露给别的测试
        monkeypatch.setenv("MARCH7TH_CONFIG_READONLY", "")
        monkeypatch.setenv("M7A_LLM_TAKEOVER_DISABLE", "")
        monkeypatch.setattr(ms, "read_user_env", lambda name: os.environ.get(name, "").strip() or None)
        self.runs = []
        monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: self.runs.append((app, kwargs)))
        self.image = tmp_path / "shot.png"
        Image.new("RGB", (64, 64)).save(self.image)
        yield
        os.chdir(cwd)

    def test_refuses_to_start_without_token(self):
        assert ms.main(["--offline-image", str(self.image)]) == 2
        assert self.runs == []

    def test_no_auth_only_with_loopback(self):
        assert ms.main(["--no-auth", "--host", "0.0.0.0", "--offline-image", str(self.image)]) == 2
        assert ms.main(["--no-auth", "--host", "192.168.1.5", "--offline-image", str(self.image)]) == 2
        assert self.runs == []

    def test_server_process_disables_first_phase_takeover(self, monkeypatch):
        assert os.environ["M7A_LLM_TAKEOVER_DISABLE"] == ""
        assert ms.main(["--no-auth", "--offline-image", str(self.image)]) == 0
        assert os.environ["M7A_LLM_TAKEOVER_DISABLE"] == "1"
        from module.llm_agent import config as lc
        assert lc.takeover_disabled()

    def test_no_auth_loopback_starts_with_defaults(self):
        assert ms.main(["--no-auth", "--offline-image", str(self.image)]) == 0
        app, kwargs = self.runs[0]
        assert kwargs["host"] == "127.0.0.1" and kwargs["port"] == 8770
        assert not isinstance(app, ms.BearerAuthMiddleware)
        assert os.environ["MARCH7TH_CONFIG_READONLY"] == "1"  # 离线模式不允许写 config.yaml

    def test_token_from_environment_wraps_app(self, monkeypatch):
        monkeypatch.setenv("M7A_MCP_TOKEN", TOKEN)
        assert ms.main(["--host", "0.0.0.0", "--port", "9999", "--offline-image", str(self.image)]) == 0
        app, kwargs = self.runs[0]
        assert isinstance(app, ms.BearerAuthMiddleware)
        assert kwargs["host"] == "0.0.0.0" and kwargs["port"] == 9999

    def test_parse_args_defaults(self):
        args = ms.parse_args([])
        assert (args.host, args.port, args.offline_image, args.no_auth, args.stateful) == ("127.0.0.1", 8770, None, False, False)

    def test_read_user_env_prefers_process_environment(self, monkeypatch):
        monkeypatch.undo()
        monkeypatch.setenv("M7A_MCP_TOKEN", "  abc  ")
        assert ms.read_user_env("M7A_MCP_TOKEN") == "abc"
