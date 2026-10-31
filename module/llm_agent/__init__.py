"""大模型异常接管（第一期）。

M7A 认不出当前游戏界面、或者切换界面超时的时候，不再直接报错，
而是交给视觉大模型看截图、点按钮，把画面拉回 M7A 认识的界面，再交还给原来的脚本。

- observe.py：截图 → OCR → 模板 → 元素编号 → 标注图
- guard.py：护栏（写在代码里，不靠 prompt）
- actions.py：动作解析与执行
- client.py：DeepSeek 调用、JSON 解析
- takeover.py：接管循环、上限控制、记录
- cli.py：开发用命令行（observe / decide / takeover）

第二期（MCP server，把 M7A 的能力包装成工具，让 Codex 直接操作游戏）：
- mcp_tools.py：全部工具逻辑（不依赖 mcp 包，可单测）
- search.py：DeepSeek Search / Fetch 客户端
- mcp_server.py：FastMCP 接线、token 中间件、启动参数（需要 mcp>=1.20,<2，用 uv run --with 叠加）

工具层（observe / guard / actions）不依赖 DeepSeek。默认关闭，见 config.example.yaml 的 llm_takeover_* 配置。
"""
