# coding:utf-8
"""大模型接管：配置读取、环境变量 key、config.example.yaml 默认值与旧配置自动补默认值。"""
import re
from pathlib import Path

import pytest

from module.llm_agent import config as lc

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "assets" / "config" / "config.example.yaml"

KEYS = [
    "llm_takeover_enable", "llm_takeover_base_url", "llm_takeover_model", "llm_takeover_after_retries",
    "llm_takeover_max_steps", "llm_takeover_max_per_run", "llm_takeover_extra_danger_words",
]


def _getter(values):
    return lambda key, default=None: values.get(key, default)


class TestLoadConfig:
    def test_defaults_when_nothing_configured(self):
        cfg = lc.load_config(_getter({}))
        assert cfg.enable is False
        assert cfg.base_url == "https://api.deepseek.com"
        assert cfg.model == "deepseek-flash"
        assert (cfg.after_retries, cfg.max_steps, cfg.max_per_run) == (3, 12, 3)
        assert cfg.extra_danger_words == ()
        assert cfg.max_seconds == 180

    def test_values_read(self):
        cfg = lc.load_config(_getter({
            "llm_takeover_enable": True, "llm_takeover_base_url": "https://x.test/v1/", "llm_takeover_model": "m",
            "llm_takeover_after_retries": 5, "llm_takeover_max_steps": 20, "llm_takeover_max_per_run": 1,
            "llm_takeover_extra_danger_words": ["礼包", " 商店 ", ""],
        }))
        assert cfg.enable is True
        assert cfg.base_url == "https://x.test/v1"  # 去掉末尾斜杠
        assert (cfg.after_retries, cfg.max_steps, cfg.max_per_run) == (5, 20, 1)
        assert cfg.extra_danger_words == ("礼包", "商店")

    @pytest.mark.parametrize("value,expected", [(True, True), ("true", True), ("True", True), ("1", True),
                                                (False, False), ("false", False), (None, False), (1, False), ("yes", False)])
    def test_enable_is_strict(self, value, expected):
        assert lc.load_config(_getter({"llm_takeover_enable": value})).enable is expected

    @pytest.mark.parametrize("bad", ["abc", None, -1, 0, True, "3.5"])
    def test_bad_numbers_fall_back_to_default(self, bad):
        cfg = lc.load_config(_getter({"llm_takeover_max_steps": bad, "llm_takeover_after_retries": bad}))
        assert cfg.max_steps == 12 and cfg.after_retries == 3

    def test_max_per_run_zero_allowed(self):
        assert lc.load_config(_getter({"llm_takeover_max_per_run": 0})).max_per_run == 0

    def test_extra_words_accepts_string_and_ignores_garbage(self):
        assert lc.load_config(_getter({"llm_takeover_extra_danger_words": "礼包"})).extra_danger_words == ("礼包",)
        assert lc.load_config(_getter({"llm_takeover_extra_danger_words": {"a": 1}})).extra_danger_words == ()


class TestApiKey:
    def test_from_env_only(self, monkeypatch):
        monkeypatch.delenv(lc.ENV_API_KEY, raising=False)
        assert lc.get_api_key() is None
        monkeypatch.setenv(lc.ENV_API_KEY, "  sk-abc  ")
        assert lc.get_api_key() == "sk-abc"
        monkeypatch.setenv(lc.ENV_API_KEY, "")
        assert lc.get_api_key() is None

    def test_env_name(self):
        assert lc.ENV_API_KEY == "DEEPSEEK_API_KEY"

    def test_config_getter_never_consulted_for_key(self):
        asked = []

        def getter(key, default=None):
            asked.append(key)
            return default
        lc.load_config(getter)
        assert not any("key" in key.lower() for key in asked)

    def test_redact(self, monkeypatch):
        assert lc.redact("a sk-123 b", "sk-123") == "a *** b"
        monkeypatch.setenv(lc.ENV_API_KEY, "sk-env")
        assert lc.redact("token sk-env") == "token ***"
        assert lc.redact(5) == 5


class TestExampleYaml:
    def test_every_key_has_chinese_comment_and_expected_default(self):
        text = EXAMPLE.read_text(encoding="utf-8")
        for key in KEYS:
            match = re.search(rf"^{key}: .*#\s*(.+)$", text, re.M)
            assert match, key
            assert re.search(r"[一-鿿]", match.group(1)), key
        assert re.search(r"^llm_takeover_enable: false\b", text, re.M)

    def test_api_key_not_in_example(self):
        text = EXAMPLE.read_text(encoding="utf-8")
        assert not re.search(r"^llm_takeover_\w*key", text, re.M)


class TestOldUserConfigGetsDefaults:
    def test_missing_keys_filled_from_example(self, tmp_path, monkeypatch):
        """用户已有的 config.yaml 里没有 llm_takeover_* 时，Config 加载后能拿到默认值，并写回文件。"""
        from module.config.config import Config
        monkeypatch.delenv("MARCH7TH_CONFIG_READONLY", raising=False)
        monkeypatch.setattr(Config, "_readonly", Config._readonly)  # Config.__init__ 会改这个类属性，让 monkeypatch 负责还原
        version = tmp_path / "version.txt"
        version.write_text("1.0.0", encoding="utf-8")
        user = tmp_path / "config.yaml"
        user.write_text("log_level: DEBUG\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        config = Config(str(version), str(EXAMPLE), str(user))

        assert config.get_value("log_level") == "DEBUG"  # 用户自己的值保留
        assert config.get_value("llm_takeover_enable") is False
        assert config.get_value("llm_takeover_base_url") == "https://api.deepseek.com"
        assert config.get_value("llm_takeover_model") == "deepseek-flash"
        assert config.get_value("llm_takeover_after_retries") == 3
        assert config.get_value("llm_takeover_max_steps") == 12
        assert config.get_value("llm_takeover_max_per_run") == 3
        assert config.get_value("llm_takeover_extra_danger_words") == []
        saved = user.read_text(encoding="utf-8")
        assert all(f"{key}:" in saved for key in KEYS)
        # 同一份配置经 load_config 读出来也是关闭状态
        assert lc.load_config(config.get_value).enable is False
