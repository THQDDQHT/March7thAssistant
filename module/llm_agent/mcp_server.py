"""第二期：M7A 的 MCP server（streamable HTTP）。

启动：uv run --with "mcp>=1.20,<2" python -m module.llm_agent.mcp_server
      [--host 127.0.0.1] [--port 8770] [--offline-image 截图.png] [--no-auth] [--stateful]

- 工具逻辑全部在 mcp_tools.py（不依赖 mcp 包）；这里只做 MCP 接线、token 校验和启动参数。
- 传输：streamable HTTP，默认 127.0.0.1:8770，路径 /mcp。
- 鉴权：ASGI 中间件校验 `Authorization: Bearer $M7A_MCP_TOKEN`，不对返回 401；
  没有设置 token 时拒绝启动，除非显式加 --no-auth，而且 --no-auth 只允许和 127.0.0.1 搭配。
- 注意 mcp 2.x 改了 API，这里必须用 mcp 1.x（mcp>=1.20,<2）。
- 默认无状态（stateless_http）：服务重启后 Codex 不需要重新建立会话；--stateful 可以改回有状态。
- 观察类工具返回「文字 + 图片」的 content 列表，图片是 JPEG。
"""
import argparse
import asyncio
import hmac
import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from typing import Annotated, Any, Callable, Optional, Union

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ENV_TOKEN = "M7A_MCP_TOKEN"
ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_TAKEOVER_DISABLE = "M7A_LLM_TAKEOVER_DISABLE"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770
MCP_PATH = "/mcp"
MIN_TOKEN_LENGTH = 16

INSTRUCTIONS = (
    "操作《崩坏：星穹铁道》PC 版（三月七小助手 M7A）。先 get_status；能 goto_screen / run_m7a_task 的就别一步步点；"
    "每步看完新观察再决定下一步；护栏拒绝的操作不要绕开；绝不购买、抽卡、花星琼；搞不定的用 leave_for_user 留给用户。"
)


# ---------------------------------------------------------------- token ----

def read_user_env(name: str) -> Optional[str]:
    """先读进程环境变量，Windows 上读不到时再读注册表里的用户环境变量（HKCU\\Environment）。

    计划任务启动的进程不一定拿到登录之后才设置的环境变量；注册表里的值是安装脚本写进去的那份。
    """
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                registry_value, _ = winreg.QueryValueEx(key, name)
            return str(registry_value).strip() or None
        except OSError:
            return None
    return None


class BearerAuthMiddleware:
    """ASGI 中间件：校验 Authorization: Bearer <token>，不对返回 401。lifespan 原样放行。"""

    def __init__(self, app, token: str):
        self.app = app
        self._token = token.encode("utf-8")

    def _authorized(self, scope) -> bool:
        for name, value in scope.get("headers", []):
            if name.lower() == b"authorization":
                scheme, _, credential = value.partition(b" ")
                if scheme.lower() == b"bearer":
                    return hmac.compare_digest(credential.strip(), self._token)
                return False
        return False

    async def __call__(self, scope, receive, send):
        kind = scope["type"]
        if kind == "lifespan":
            await self.app(scope, receive, send)
            return
        if kind == "http" and self._authorized(scope):
            await self.app(scope, receive, send)
            return
        if kind == "http":
            body = b'{"error": "unauthorized"}'
            await send({"type": "http.response.start", "status": 401, "headers": [
                (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b'Bearer realm="m7a-mcp"')]})
            await send({"type": "http.response.body", "body": body})
        elif kind == "websocket":
            await send({"type": "websocket.close", "code": 1008})


# ---------------------------------------------------------------- 工具注册 ----

def to_content(output):
    """ToolOutput → MCP content 列表：先文字，有图再加一张 JPEG。"""
    import base64
    from mcp.types import ImageContent, TextContent
    blocks = [TextContent(type="text", text=output.text)]
    if output.image:
        blocks.append(ImageContent(type="image", data=base64.b64encode(output.image).decode("ascii"),
                                   mimeType="image/jpeg"))
    return blocks


def create_mcp(tools, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, stateless: bool = True):
    """用 FastMCP 注册全部工具。同步的工具逻辑放到线程里跑，事件循环不会被一个小时的 run_m7a_task 堵住。"""
    from mcp.server.fastmcp import FastMCP
    from pydantic import Field

    mcp = FastMCP("m7a", instructions=INSTRUCTIONS, host=host, port=port, streamable_http_path=MCP_PATH,
                  stateless_http=stateless, log_level="WARNING")

    async def call(func: Callable, *args):
        return to_content(await asyncio.to_thread(func, *args))

    def tool(description: str):
        return mcp.tool(description=description, structured_output=False)

    @tool("看当前游戏画面：返回当前界面、元素表（编号 + 文字 + 位置）和标注图。grid=true 时标注图改成 16×9 网格，用来点识别不出文字的图标。")
    async def observe(grid: Annotated[bool, Field(description="是否叠加网格（click_grid 之前需要）")] = False):
        return await call(tools.observe, grid)

    @tool("点击最近一次观察里编号为 id 的元素。会过护栏（危险词、确认类按钮）。执行后自动返回新画面。")
    async def click(id: Annotated[int, Field(description="元素表里的编号")]):
        return await call(tools.click, id)

    @tool("点击网格里的某个格子中心（如 H5）。上一次观察必须是 observe(grid=true)；画面有危险词时禁止。执行后自动返回新画面。")
    async def click_grid(cell: Annotated[str, Field(description="列 A-P + 行 1-9，例如 H5")]):
        return await call(tools.click_grid, cell)

    @tool("按键。白名单：esc enter space f e q w a s d 1 2 3 4；画面有危险词时只允许 esc。hold_seconds>0 为按住（最多 5 秒，只用于走几步的短距离移动）。执行后自动返回新画面。")
    async def press_key(key: Annotated[str, Field(description="按键名")],
                        hold_seconds: Annotated[float, Field(description="按住秒数，0 表示轻按；最多 5")] = 0):
        return await call(tools.press_key, key, hold_seconds)

    @tool("滚动鼠标滚轮。位置用元素 id 或格子 cell 指定，都不给就在画面中央。执行后自动返回新画面。")
    async def scroll(direction: Annotated[str, Field(description="up 或 down")],
                     amount: Annotated[int, Field(description="滚动格数 1-10")] = 3,
                     id: Annotated[Optional[int], Field(description="在这个元素中心滚动（元素表编号）")] = None,
                     cell: Annotated[Optional[str], Field(description="在这个格子中心滚动，例如 H5")] = None):
        return await call(tools.scroll, direction, amount, id, cell)

    @tool("按住鼠标从一个格子拖到另一个格子。画面有危险词时拒绝。执行后自动返回新画面。")
    async def drag(from_cell: Annotated[str, Field(description="起点格子，例如 C5")],
                   to_cell: Annotated[str, Field(description="终点格子，例如 L5")]):
        return await call(tools.drag, from_cell, to_cell)

    @tool("等待几秒（加载、动画），然后返回新画面。")
    async def wait(seconds: Annotated[float, Field(description="1-10 秒")] = 2):
        return await call(tools.wait, seconds)

    @tool("列出 M7A 认识的全部界面（ID 和名称），供 goto_screen 使用。")
    async def list_screens():
        return await call(tools.list_screens)

    @tool("用 M7A 的界面跳转图直接跳到某个界面（比一步步点可靠）。执行后自动返回新画面。")
    async def goto_screen(screen_id: Annotated[str, Field(description="list_screens 里的界面 ID，例如 main")]):
        return await call(tools.goto_screen, screen_id)

    @tool("列出 M7A 现成的任务（ID 和名称），供 run_m7a_task 使用。")
    async def list_m7a_tasks():
        return await call(tools.list_m7a_tasks)

    @tool("运行 M7A 的现成任务并等它结束，返回运行记录摘要和日志末尾。任务运行期间所有操作类工具都会被拒绝。")
    async def run_m7a_task(task_id: Annotated[str, Field(description="list_m7a_tasks 里的任务 ID，例如 daily")],
                           timeout_minutes: Annotated[int, Field(description="最长等待分钟数，1-60")] = 60):
        return await call(tools.run_m7a_task, task_id, timeout_minutes)

    @tool("向正在运行的 M7A 任务发停止指令。")
    async def stop_m7a_task():
        return await call(tools.stop_m7a_task)

    @tool("查看状态：游戏窗口在不在、当前界面、正在跑的任务、最近 5 次运行、输入模式（前台或后台）、留给用户的未处理事项数。")
    async def get_status():
        return await call(tools.get_status)

    @tool("联网搜索（DeepSeek Search）：返回标题、链接、摘要、日期。")
    async def web_search(query: Annotated[str, Field(description="搜索词，1-128 字符，可用 site: 等运算符")],
                         count: Annotated[int, Field(description="结果条数 1-20")] = 8):
        return await call(tools.web_search, query, count)

    @tool("抓取网页正文（Markdown）。长网页按 offset / max_chars 分段读取。")
    async def fetch_page(url: Annotated[str, Field(description="网页地址")],
                         offset: Annotated[int, Field(description="从第几个字符开始")] = 0,
                         max_chars: Annotated[int, Field(description="本段最多字符数 500-20000")] = 8000):
        return await call(tools.fetch_page, url, offset, max_chars)

    @tool("把搞不定或不该做的事留给用户：记到 todo_for_user.md 并通过 M7A 通知用户，返回当前未处理的全部事项。之后回到主界面继续做别的。")
    async def leave_for_user(item: Annotated[str, Field(description="事项名称，例如活动名")],
                             reason: Annotated[str, Field(description="为什么留给用户")],
                             how_to_reach: Annotated[str, Field(description="怎么进到那个界面")] = ""):
        return await call(tools.leave_for_user, item, reason, how_to_reach)

    @tool("把留给用户的某条事项标为已处理（在 todo_for_user.md 里勾掉并记下说明和时间）。只有这件事真正完成了才调用。"
          "名称先按完全相同匹配，再按包含关系匹配；匹配不到或匹配到多条会报错并列出未处理清单。返回更新后的未处理清单。")
    async def resolve_todo(item: Annotated[str, Field(description="事项名称（leave_for_user 时写的 item）")],
                           note: Annotated[str, Field(description="处理说明，例如怎么完成的")] = ""):
        return await call(tools.resolve_todo, item, note)

    @tool("列出队伍配置里能用的全部角色 ID（英文，区分大小写）和中文名。")
    async def list_characters():
        return await call(tools.list_characters)

    @tool("读取 M7A 配置（只能读白名单内的键：混沌回忆 forgottenhall_*、虚构叙事 purefiction_*、末日幻影 apocalyptic_* 的 "
          "enable/level/team*，以及体力计划 power_enable、power_plan、power_plan_keep、instance_type、instance_names、"
          "instance_names_challenge_count）。keys 为空列表表示读全部白名单键；白名单外的键会被拒绝。")
    async def get_m7a_config(keys: Annotated[Union[list[str], str], Field(description="配置键名列表，例如 [\"forgottenhall_team1\"]")] = []):
        return await call(tools.get_m7a_config, keys)

    @tool("修改 M7A 配置（只能改白名单内的键，写入前严格校验，返回「旧值 → 新值」）。"
          "enable 类是 true/false；level 是 [最小, 最大]（混沌回忆 1–12，虚构叙事和末日幻影 1–4）；"
          "team 是 1–4 个 [角色ID, 秘技次数]（角色 ID 见 list_characters，秘技次数 -1 到 2，-1 表示最后一个放秘技和普攻的，每队最多一个）；"
          "power_plan 是 [[副本类型, 副本名称, 次数], ...]；instance_type 是副本类型；"
          "instance_names / instance_names_challenge_count 是 {副本类型: 值}，可只给要改的类型。"
          "有任务在跑时也能写，但下次运行任务时才生效。")
    async def set_m7a_config(key: Annotated[str, Field(description="配置键名，例如 forgottenhall_team1")],
                             value: Annotated[Any, Field(description="新值（JSON 值，类型见工具说明）")]):
        return await call(tools.set_m7a_config, key, value)

    @tool("读出留给用户、还没处理的事项清单。")
    async def list_todo_for_user():
        return await call(tools.list_todo_for_user)

    return mcp


def build_app(tools, host: str, port: int, token: Optional[str], stateless: bool = True):
    """返回最终的 ASGI 应用：FastMCP 的 streamable HTTP 应用，外面套 token 中间件（token 为 None 即不鉴权）。"""
    app = create_mcp(tools, host, port, stateless).streamable_http_app()
    return BearerAuthMiddleware(app, token) if token else app


# ---------------------------------------------------------------- 启动 ----

def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m module.llm_agent.mcp_server",
                                     description="M7A 的 MCP server（streamable HTTP，路径 /mcp）")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"监听地址，默认 {DEFAULT_HOST}；0.0.0.0 开放给局域网（必须有 token）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"监听端口，默认 {DEFAULT_PORT}")
    parser.add_argument("--offline-image", metavar="PNG", help="离线模式：用这张固定截图代替实时截图，所有输入改成空操作")
    parser.add_argument("--no-auth", action="store_true", help=f"不校验 token（仅允许和 --host {DEFAULT_HOST} 搭配，只用于本机开发）")
    parser.add_argument("--stateful", action="store_true", help="使用有状态会话（默认无状态：服务重启后客户端不用重连）")
    return parser.parse_args(argv)


def setup_logging() -> None:
    """日志写到 logs/mcp_agent/server.log（pythonw 没有控制台，stdout/stderr 可能是 None）；token 和 key 从不打印。"""
    handlers = []
    log_dir = os.path.join(ROOT, "logs", "mcp_agent")
    try:
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(RotatingFileHandler(os.path.join(log_dir, "server.log"), maxBytes=1024 * 1024, backupCount=3,
                                            encoding="utf-8"))
    except OSError:
        pass
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    else:
        sys.stdout = sys.stderr = open(os.devnull, "w")
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main(argv=None) -> int:
    args = parse_args(argv)
    os.chdir(ROOT)  # M7A 的相对路径（assets、logs、config.yaml）都以仓库根目录为准
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    if args.offline_image:
        os.environ["MARCH7TH_CONFIG_READONLY"] = "1"  # 离线测试不允许写 config.yaml
    # server 里由 Codex 看图决策，screen.change_to 等失败时直接返回错误，不再嵌套第一期的 DeepSeek 接管
    os.environ[ENV_TAKEOVER_DISABLE] = "1"
    setup_logging()
    log = logging.getLogger("m7a.mcp")

    if args.no_auth and args.host != DEFAULT_HOST:
        log.error("--no-auth 只允许和 --host %s 搭配", DEFAULT_HOST)
        sys.stderr.write(f"--no-auth 只允许和 --host {DEFAULT_HOST} 搭配\n") if sys.stderr else None
        return 2
    token = None
    if not args.no_auth:
        token = read_user_env(ENV_TOKEN)
        if not token:
            message = f"没有设置环境变量 {ENV_TOKEN}，拒绝启动。本机开发可以加 --no-auth（仅限 {DEFAULT_HOST}）。"
            log.error(message)
            sys.stderr.write(message + "\n") if sys.stderr else None
            return 2
        if len(token) < MIN_TOKEN_LENGTH:
            log.warning("%s 太短（少于 %d 个字符），建议换成随机长串", ENV_TOKEN, MIN_TOKEN_LENGTH)
    if not os.environ.get(ENV_API_KEY, "").strip():
        key = read_user_env(ENV_API_KEY)
        if key:
            os.environ[ENV_API_KEY] = key

    try:
        import uvicorn
        import mcp  # noqa: F401
    except ImportError as e:
        message = f"缺少依赖（{e}）。请用：uv run --with \"mcp>=1.20,<2\" python -m module.llm_agent.mcp_server"
        log.error(message)
        sys.stderr.write(message + "\n") if sys.stderr else None
        return 3

    from module.llm_agent.mcp_tools import McpTools
    tools = McpTools(offline_image=args.offline_image)
    app = build_app(tools, args.host, args.port, token, stateless=not args.stateful)
    log.info("M7A MCP server 启动：http://%s:%d%s 模式=%s 鉴权=%s 会话=%s", args.host, args.port, MCP_PATH,
             "离线" if args.offline_image else "实机", "关闭" if args.no_auth else "token",
             "有状态" if args.stateful else "无状态")
    if not args.offline_image:
        threading.Thread(target=tools.warm_up, name="m7a-mcp-warmup", daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_config=None, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
