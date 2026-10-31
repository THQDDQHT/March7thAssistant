"""开发用命令行：python -m module.llm_agent.cli <子命令>

- observe --image x.png --out 目录：离线跑，输出元素表 JSON 和标注图，不需要游戏窗口
- decide --image x.png：离线调用模型，只打印模型选的动作和护栏判定，不点击（需要 DEEPSEEK_API_KEY）
- takeover：实机接管，必须在能看到游戏窗口的会话里跑（会真的点击）
"""
import argparse
import json
import os
import sys

from PIL import Image


def _print(text=""):
    sys.stdout.write(text + "\n")


def cmd_observe(args) -> int:
    from module.llm_agent.observe import observe_image
    image = Image.open(args.image)
    observation = observe_image(image)
    os.makedirs(args.out, exist_ok=True)
    elements_path = os.path.join(args.out, "elements.json")
    with open(elements_path, "w", encoding="utf-8") as file:
        json.dump({"image": os.path.abspath(args.image), "size": list(observation.size),
                   "elements": [element.to_dict() for element in observation.elements]},
                  file, ensure_ascii=False, indent=2)
    annotated_path = os.path.join(args.out, "annotated.jpg")
    with open(annotated_path, "wb") as file:
        file.write(observation.annotated_jpeg())
    observation.grid = True
    grid_path = os.path.join(args.out, "grid.jpg")
    with open(grid_path, "wb") as file:
        file.write(observation.annotated_jpeg())
    _print(observation.table_text())
    _print(f"\n共 {len(observation.elements)} 个元素")
    _print(f"元素表：{elements_path}")
    _print(f"标注图：{annotated_path}")
    _print(f"网格图：{grid_path}")
    return 0


def cmd_decide(args) -> int:
    from module.llm_agent import config as llm_config
    from module.llm_agent.actions import normalize_action
    from module.llm_agent.client import LLMClient, LLMError, build_step_message, build_system_prompt
    from module.llm_agent.guard import Guard
    from module.llm_agent.observe import observe_image
    from module.llm_agent.takeover import ANY_KNOWN, load_screen_names

    api_key = llm_config.get_api_key()
    if not api_key:
        _print(f"没有设置环境变量 {llm_config.ENV_API_KEY}，无法调用模型")
        return 2
    cfg = llm_config.load_config()
    observation = observe_image(Image.open(args.image), grid=args.grid)
    client = LLMClient(cfg.base_url, cfg.model, api_key)
    messages = [
        {"role": "system", "content": build_system_prompt(load_screen_names())},
        build_step_message(1, cfg.max_steps, "回到脚本认识的任何一个界面" if args.goal == ANY_KNOWN else args.goal,
                           observation.table_text(), [], observation.annotated_jpeg(), args.grid),
    ]
    try:
        decision = client.decide(messages)
    except LLMError as e:
        _print(llm_config.redact(f"模型调用失败：{e}", api_key))
        return 1
    _print("模型原始回复：" + llm_config.redact(decision.raw, api_key))
    if decision.action is None:
        _print("解析失败：" + decision.error)
        return 1
    action, error = normalize_action(decision.action)
    if action is None:
        _print("动作不合法：" + error)
        return 1
    verdict = Guard(cfg.extra_danger_words).check(action, observation)
    _print("动作：" + json.dumps(action.to_dict(), ensure_ascii=False))
    _print("护栏：" + json.dumps(verdict.to_dict(), ensure_ascii=False))
    _print("用量：" + json.dumps(client.usage.to_dict()))
    _print("（decide 只打印动作，不会点击）")
    return 0


def cmd_takeover(args) -> int:
    from module.llm_agent import config as llm_config
    from module.llm_agent.takeover import ANY_KNOWN, Takeover
    from module.llm_agent.client import LLMClient

    api_key = llm_config.get_api_key()
    if not api_key:
        _print(f"没有设置环境变量 {llm_config.ENV_API_KEY}，无法接管")
        return 2
    cfg = llm_config.load_config()
    result = Takeover(cfg, LLMClient(cfg.base_url, cfg.model, api_key)).run(args.trigger, ANY_KNOWN)
    _print(f"结果：{'成功' if result.success else '失败'}，{result.reason}，{result.steps} 步，{result.elapsed:.1f} 秒")
    _print(f"记录目录：{result.record_dir}")
    return 0 if result.success else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m module.llm_agent.cli", description="大模型异常接管开发工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_observe = sub.add_parser("observe", help="离线：输出元素表 JSON 和标注图（不需要游戏窗口）")
    p_observe.add_argument("--image", required=True, help="截图 png 路径")
    p_observe.add_argument("--out", required=True, help="输出目录")
    p_observe.set_defaults(func=cmd_observe)

    p_decide = sub.add_parser("decide", help="离线：调用模型，只打印动作，不点击")
    p_decide.add_argument("--image", required=True, help="截图 png 路径")
    p_decide.add_argument("--grid", action="store_true", help="用网格模式的标注图")
    p_decide.add_argument("--goal", default="ANY_KNOWN", help="目标，默认 ANY_KNOWN")
    p_decide.set_defaults(func=cmd_decide)

    p_takeover = sub.add_parser("takeover", help="实机接管（会真的点击，必须在能看到游戏窗口的会话里跑）")
    p_takeover.add_argument("--trigger", default="cli", help="写进记录的触发点名称")
    p_takeover.set_defaults(func=cmd_takeover)
    return parser


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
