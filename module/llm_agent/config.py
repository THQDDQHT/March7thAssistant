"""接管相关的配置读取。

配置项来自 config.yaml（`llm_takeover_*`，默认值见 assets/config/config.example.yaml）；
API key 只从环境变量 DEEPSEEK_API_KEY 读取，不写进配置文件，也不出现在任何日志和记录里。
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

ENV_API_KEY = "DEEPSEEK_API_KEY"
# 设为 1/true 时整个进程不做第一期的接管（MCP server 里由外部 agent 看图决策，不再嵌套一层）
ENV_TAKEOVER_DISABLE = "M7A_LLM_TAKEOVER_DISABLE"

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_AFTER_RETRIES = 3
DEFAULT_MAX_STEPS = 12
DEFAULT_MAX_PER_RUN = 3

# 每次接管的最长时间（秒），写在代码里，不开放配置
MAX_SECONDS = 180
# 单次 HTTP 请求超时（秒）和网络错误重试次数
REQUEST_TIMEOUT = 60
NETWORK_RETRIES = 2


@dataclass(frozen=True)
class TakeoverConfig:
    enable: bool = False
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    after_retries: int = DEFAULT_AFTER_RETRIES
    max_steps: int = DEFAULT_MAX_STEPS
    max_per_run: int = DEFAULT_MAX_PER_RUN
    extra_danger_words: Tuple[str, ...] = ()
    max_seconds: int = MAX_SECONDS


def _as_int(value, default, minimum=1):
    """配置值转整数，非法或小于下限时回退默认值。"""
    try:
        if isinstance(value, bool):
            return default
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= minimum else default


def _as_bool(value) -> bool:
    """开关只认布尔 true（或字符串 true/1），其余一律视为关闭。"""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1")
    return value is True


def _as_words(value) -> Tuple[str, ...]:
    """配置里的追加危险词：接受字符串列表，忽略空项和非字符串。"""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if isinstance(item, (str, int)) and str(item).strip())


def load_config(get_value=None) -> TakeoverConfig:
    """读取接管配置。

    :param get_value: 取配置值的函数 (key, default) -> value；默认用 M7A 的 cfg.get_value，测试里可以注入。
    """
    if get_value is None:
        from module.config import cfg
        get_value = cfg.get_value
    base_url = str(get_value("llm_takeover_base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).strip().rstrip("/")
    model = str(get_value("llm_takeover_model", DEFAULT_MODEL) or DEFAULT_MODEL).strip()
    return TakeoverConfig(
        enable=_as_bool(get_value("llm_takeover_enable", False)),
        base_url=base_url or DEFAULT_BASE_URL,
        model=model or DEFAULT_MODEL,
        after_retries=_as_int(get_value("llm_takeover_after_retries", DEFAULT_AFTER_RETRIES), DEFAULT_AFTER_RETRIES),
        max_steps=_as_int(get_value("llm_takeover_max_steps", DEFAULT_MAX_STEPS), DEFAULT_MAX_STEPS),
        max_per_run=_as_int(get_value("llm_takeover_max_per_run", DEFAULT_MAX_PER_RUN), DEFAULT_MAX_PER_RUN, minimum=0),
        extra_danger_words=_as_words(get_value("llm_takeover_extra_danger_words", [])),
    )


def takeover_disabled() -> bool:
    """环境变量 M7A_LLM_TAKEOVER_DISABLE 为 1/true 时，本进程禁用接管（优先于 llm_takeover_enable）。"""
    return os.environ.get(ENV_TAKEOVER_DISABLE, "").strip().lower() in ("1", "true")


def get_api_key() -> Optional[str]:
    """只从环境变量读取 API key；没有或为空返回 None。"""
    key = os.environ.get(ENV_API_KEY, "").strip()
    return key or None


def redact(text, api_key: Optional[str] = None):
    """把文本里出现的 API key 替换成 ***（日志和记录文件落盘前的最后一道保险）。"""
    if not isinstance(text, str):
        return text
    key = api_key if api_key is not None else os.environ.get(ENV_API_KEY, "").strip()
    if key:
        text = text.replace(key, "***")
    return text
