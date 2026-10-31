# coding:utf-8
"""MCP 配置工具：list_characters / get_m7a_config / set_m7a_config。

白名单、各类校验、写入后保留注释和格式、不覆盖外部改动、任务运行中的提示、离线模式不碰真实文件。
大部分用内存里的假配置；注释保留、外部改动、离线副本用 M7A 真正的 Config 对象在临时目录里测。
"""
import copy
import json
import os
import shutil

import pytest
from PIL import Image

from module.llm_agent import mcp_config as mc
from module.llm_agent import mcp_tools as mt

ROOT = mc.ROOT
EXAMPLE = os.path.join(ROOT, "assets", "config", "config.example.yaml")
VERSION = os.path.join(ROOT, "assets", "config", "version.txt")

WHITELIST = [
    "forgottenhall_enable", "forgottenhall_level", "forgottenhall_team1", "forgottenhall_team2",
    "purefiction_enable", "purefiction_stellar_enable", "purefiction_level", "purefiction_team1", "purefiction_team2",
    "purefiction_team3", "apocalyptic_enable", "apocalyptic_level", "apocalyptic_team1", "apocalyptic_team2",
    "power_enable", "power_plan", "power_plan_keep", "instance_type", "instance_names", "instance_names_challenge_count"]

FORBIDDEN = ["after_finish", "game_path", "game_path_launcher", "notify_telegram_token", "notification_enable",
             "auto_update", "local_control_mode", "forgottenhall_timestamp", "loop_mode", "DEEPSEEK_API_KEY",
             "M7A_MCP_TOKEN", "notify_smtp_password", "cloud_game_enable", "instance_teams", "use_fuel"]


class FakeCfg:
    """内存里的假 Config：config 字典 + _load_config / save_config / get_value / _read_file_config。"""

    def __init__(self, data=None, readonly=False):
        self.config = copy.deepcopy(data if data is not None else {
            "forgottenhall_enable": False, "forgottenhall_level": [10, 12],
            "forgottenhall_team1": [["Asta", -1], ["JingYuan", 1]], "forgottenhall_team2": [["FuXuan", 1]],
            "forgottenhall_timestamp": 123.0,
            "purefiction_enable": False, "purefiction_stellar_enable": False, "purefiction_level": [3, 4],
            "purefiction_team1": [["Asta", -1]], "purefiction_team2": [["FuXuan", 1]],
            "purefiction_team3": [["DanHeng", -1]], "purefiction_timestamp": 456.0,
            "apocalyptic_enable": False, "apocalyptic_level": [3, 4], "apocalyptic_team1": [["Asta", -1]],
            "apocalyptic_team2": [["FuXuan", 1]], "apocalyptic_timestamp": 0,
            "power_enable": True, "power_plan": [], "power_plan_keep": False, "instance_type": "拟造花萼（金）",
            "instance_names": {"拟造花萼（金）": "回忆之蕾", "侵蚀隧洞": "睿治之径"},
            "instance_names_challenge_count": {"拟造花萼（金）": 24, "侵蚀隧洞": 6},
            "after_finish": "None", "notify_telegram_token": "SECRET-TOKEN"})
        self.file = copy.deepcopy(self.config)
        self.readonly = readonly
        self.saves = []

    @property
    def data(self):
        return self.config

    def _load_config(self, path=None, save=True):
        self.config.update(copy.deepcopy(self.file))

    def get_value(self, key, default=None):
        return copy.deepcopy(self.config.get(key, default))

    def save_config(self):
        if not self.readonly:
            self.saves.append({k: v for k, v in self.config.items() if self.file.get(k) != v})
            self.file = copy.deepcopy(self.config)

    def _read_file_config(self, path=None):
        return copy.deepcopy(self.file)


CHARS = ["Asta", "JingYuan", "Tingyun", "Huohuo", "FuXuan", "Bronya", "Pela", "Qingque", "DanHeng", "Arlan"]


def make_service(cfg=None):
    cfg = cfg or FakeCfg()
    service = mc.ConfigService(lambda: mc.M7aConfig(cfg), character_ids=lambda: CHARS)
    return service, cfg


def make_tools(tmp_path, cfg=None, running=None):
    service, cfg = make_service(cfg)
    tools = mt.McpTools(root=str(tmp_path), record_base=str(tmp_path / "rec"), config_service=service,
                        character_ids=lambda: CHARS, character_names=lambda: {"Asta": "艾丝妲"})
    tools._active_task = lambda: running
    return tools, cfg


# ---------------------------------------------------------------- 白名单 ----

class TestWhitelist:
    def test_exact_key_list(self):
        assert sorted(mc.CONFIG_KEYS) == sorted(WHITELIST)

    def test_all_keys_exist_in_example_config(self):
        text = open(EXAMPLE, encoding="utf-8").read()
        for key in WHITELIST:
            assert f"\n{key}:" in "\n" + text, key

    @pytest.mark.parametrize("key", FORBIDDEN)
    def test_get_and_set_reject_other_keys(self, tmp_path, key):
        tools, cfg = make_tools(tmp_path)
        out = tools.get_m7a_config([key])
        assert "不允许读取" in out.text and "SECRET-TOKEN" not in out.text
        out = tools.set_m7a_config(key, "x")
        assert "不允许修改" in out.text
        assert cfg.saves == []

    def test_get_never_returns_sensitive_values(self, tmp_path):
        tools, _ = make_tools(tmp_path)
        out = tools.get_m7a_config([])
        assert "SECRET-TOKEN" not in out.text and "notify_" not in out.text and "after_finish" not in out.text

    def test_mixed_get_is_rejected_entirely(self, tmp_path):
        tools, _ = make_tools(tmp_path)
        out = tools.get_m7a_config(["forgottenhall_level", "after_finish"])
        assert "不允许读取" in out.text and "forgottenhall_level =" not in out.text


# ---------------------------------------------------------------- list / get ----

class TestListAndGet:
    def test_list_characters_real_directory(self):
        ids = mc.list_character_ids()
        assert 100 <= len(ids) <= 120 and "Asta" in ids and "DanHeng" in ids
        text = mc.format_characters(ids, mc.load_character_names())
        assert f"共 {len(ids)} 个角色" in text and "Asta  艾丝妲" in text

    def test_list_characters_tool_and_missing_name(self, tmp_path):
        tools, _ = make_tools(tmp_path)
        text = tools.list_characters().text
        assert "共 10 个角色" in text and "Asta  艾丝妲" in text and "\nJingYuan\n" in text + "\n"

    def test_get_returns_json_values(self, tmp_path):
        tools, _ = make_tools(tmp_path)
        out = tools.get_m7a_config(["forgottenhall_level", "forgottenhall_team1", "power_plan"]).text.splitlines()
        assert out == ['forgottenhall_level = [10, 12]', 'forgottenhall_team1 = [["Asta", -1], ["JingYuan", 1]]',
                       'power_plan = []']

    @pytest.mark.parametrize("keys", ['["forgottenhall_level"]', "forgottenhall_level", "forgottenhall_level,"])
    def test_keys_accepts_text_forms(self, tmp_path, keys):
        tools, _ = make_tools(tmp_path)
        assert tools.get_m7a_config(keys).text == "forgottenhall_level = [10, 12]"

    def test_empty_keys_returns_all_whitelisted(self, tmp_path):
        tools, _ = make_tools(tmp_path)
        lines = tools.get_m7a_config([]).text.splitlines()
        assert sorted(line.split(" = ")[0] for line in lines) == sorted(WHITELIST)

    def test_get_reloads_from_file_first(self, tmp_path):
        tools, cfg = make_tools(tmp_path)
        cfg.file["forgottenhall_level"] = [1, 2]  # 图形界面刚改的
        assert tools.get_m7a_config(["forgottenhall_level"]).text == "forgottenhall_level = [1, 2]"


# ---------------------------------------------------------------- 校验 ----

class TestValidation:
    def set(self, tmp_path, key, value):
        tools, cfg = make_tools(tmp_path)
        return tools.set_m7a_config(key, value).text, cfg

    @pytest.mark.parametrize("value", ["true", 1, 0, None, "yes", [True]])
    def test_bool_must_be_boolean(self, tmp_path, value):
        text, cfg = self.set(tmp_path, "forgottenhall_enable", value)
        if value == "true":
            assert "→ true" in text  # 字符串形式的 JSON 布尔值可以解析
        else:
            assert "必须是布尔值" in text and cfg.saves == []

    def test_bool_ok(self, tmp_path):
        text, cfg = self.set(tmp_path, "purefiction_stellar_enable", True)
        assert "purefiction_stellar_enable：false → true" in text
        assert cfg.file["purefiction_stellar_enable"] is True

    @pytest.mark.parametrize("key,value,ok", [
        ("forgottenhall_level", [1, 12], True), ("forgottenhall_level", [12, 12], True), ("forgottenhall_level", [7, 13], False),
        ("forgottenhall_level", [0, 5], False), ("forgottenhall_level", [8, 7], False), ("forgottenhall_level", [1], False),
        ("forgottenhall_level", [1, 2, 3], False), ("forgottenhall_level", [1.5, 2], False), ("forgottenhall_level", [True, 2], False),
        ("forgottenhall_level", "7", False), ("forgottenhall_level", ["1", "2"], False),
        ("purefiction_level", [1, 4], True), ("purefiction_level", [1, 5], False),
        ("apocalyptic_level", [3, 4], True), ("apocalyptic_level", [3, 5], False), ("apocalyptic_level", [-1, 4], False)])
    def test_level(self, tmp_path, key, value, ok):
        text, cfg = self.set(tmp_path, key, value)
        if ok:
            assert "→" in text or "没有变化" in text
        else:
            assert text.startswith("【未执行】") and cfg.saves == []

    @pytest.mark.parametrize("team,ok", [
        ([["Asta", -1]], True),
        ([["Asta", -1], ["JingYuan", 1], ["Tingyun", 2], ["Huohuo", 0]], True),
        ([["Asta", 0], ["JingYuan", 1]], True),                       # 可以没有 -1
        ([], False),                                                  # 至少 1 人
        ([["Asta", 0]] * 2, False),                                   # 角色重复
        ([["Asta", 1], ["JingYuan", 1], ["Tingyun", 1], ["Huohuo", 1], ["Pela", 1]], False),  # 超过 4 人
        ([["NoSuchChar", 1]], False),                                 # 角色不存在
        ([["asta", 1]], False),                                       # 区分大小写
        ([["Asta", -2]], False), ([["Asta", 3]], False), ([["Asta", 1.5]], False), ([["Asta", True]], False),
        ([["Asta", "1"]], False),
        ([["Asta", -1], ["JingYuan", -1]], False),                    # 只能有一个 -1
        ([["Asta"]], False), ([["Asta", 1, 2]], False), (["Asta"], False), ("Asta", False), ([[1, 1]], False)])
    def test_team(self, tmp_path, team, ok):
        text, cfg = self.set(tmp_path, "apocalyptic_team1", team)
        if ok:
            assert "apocalyptic_team1" in text and cfg.file["apocalyptic_team1"] == team
        else:
            assert text.startswith("【未执行】") and cfg.saves == []

    def test_team_given_as_json_text(self, tmp_path):
        text, cfg = self.set(tmp_path, "forgottenhall_team1", '[["Asta", -1], ["Pela", 1]]')
        assert cfg.file["forgottenhall_team1"] == [["Asta", -1], ["Pela", 1]]

    def test_team_cross_team_duplicate_is_a_warning(self, tmp_path):
        text, cfg = self.set(tmp_path, "forgottenhall_team1", [["FuXuan", 1]])  # 队伍 2 里已经有符玄
        assert "FuXuan 同时出现在 forgottenhall_team2" in text
        assert cfg.file["forgottenhall_team1"] == [["FuXuan", 1]]

    def test_team3_only_checked_in_stellar_mode(self, tmp_path):
        text, _ = self.set(tmp_path, "purefiction_team1", [["DanHeng", -1]])
        assert "同时出现" not in text  # 星启模式没开，队伍三不算
        cfg = FakeCfg()
        cfg.data["purefiction_stellar_enable"] = cfg.file["purefiction_stellar_enable"] = True
        tools, _ = make_tools(tmp_path, cfg)
        assert "purefiction_team3" in tools.set_m7a_config("purefiction_team1", [["DanHeng", -1]]).text

    def test_instance_type(self, tmp_path):
        text, cfg = self.set(tmp_path, "instance_type", "侵蚀隧洞")
        assert cfg.file["instance_type"] == "侵蚀隧洞"
        for bad in ("历战余响", "随便", 5, None):
            text, _ = self.set(tmp_path, "instance_type", bad)
            assert text.startswith("【未执行】")

    def test_instance_names_partial_merge_and_validation(self, tmp_path):
        text, cfg = self.set(tmp_path, "instance_names", {"饰品提取": "永恒笑剧"})
        assert cfg.file["instance_names"] == {"拟造花萼（金）": "回忆之蕾", "侵蚀隧洞": "睿治之径", "饰品提取": "永恒笑剧"}
        for bad in ({"饰品提取": "不存在"}, {"不存在的类型": "x"}, {}, "x", {"侵蚀隧洞": 5}):
            assert self.set(tmp_path, "instance_names", bad)[0].startswith("【未执行】")
        assert self.set(tmp_path, "instance_names", {"凝滞虚影": "无"})[0].count("→") == 1  # 「无」表示跳过，允许

    def test_challenge_count_ranges(self, tmp_path):
        text, cfg = self.set(tmp_path, "instance_names_challenge_count", {"侵蚀隧洞": 3, "凝滞虚影": 8})
        assert cfg.file["instance_names_challenge_count"]["侵蚀隧洞"] == 3
        for bad in ({"侵蚀隧洞": 7}, {"侵蚀隧洞": 0}, {"拟造花萼（金）": 25}, {"历战余响": 4}, {"侵蚀隧洞": "6"}, {"侵蚀隧洞": True}, {"x": 1}):
            assert self.set(tmp_path, "instance_names_challenge_count", bad)[0].startswith("【未执行】"), bad

    def test_power_plan(self, tmp_path):
        plan = [["拟造花萼（金）", "回忆之蕾", 10], ["侵蚀隧洞", "睿治之径", 5]]
        text, cfg = self.set(tmp_path, "power_plan", plan)
        assert cfg.file["power_plan"] == plan
        bad_plans = [[["拟造花萼（金）", "回忆之蕾"]], [["随便", "回忆之蕾", 1]], [["拟造花萼（金）", "不存在", 1]],
                     [["拟造花萼（金）", "无", 1]], [["拟造花萼（金）", "回忆之蕾", 0]], [["拟造花萼（金）", "回忆之蕾", 1000]],
                     [["拟造花萼（金）", "回忆之蕾", "3"]], "x", [["拟造花萼（金）", "回忆之蕾", True]],
                     [["拟造花萼（金）", "回忆之蕾", 1]] * 21]
        for bad in bad_plans:
            assert self.set(tmp_path, "power_plan", bad)[0].startswith("【未执行】"), bad

    def test_power_plan_can_be_cleared(self, tmp_path):
        cfg = FakeCfg()
        cfg.data["power_plan"] = cfg.file["power_plan"] = [["侵蚀隧洞", "睿治之径", 5]]
        tools, _ = make_tools(tmp_path, cfg)
        assert "→ []" in tools.set_m7a_config("power_plan", []).text


# ---------------------------------------------------------------- 写入行为 ----

class TestWriteBehaviour:
    def test_returns_old_to_new_and_records(self, tmp_path):
        tools, cfg = make_tools(tmp_path)
        out = tools.set_m7a_config("forgottenhall_level", [1, 12]).text
        assert "forgottenhall_level：[10, 12] → [1, 12]" in out
        records = [json.load(open(p, encoding="utf-8")) for p in sorted((tmp_path / "rec").rglob("*set_m7a_config.json"))]
        assert records[0]["args"] == {"key": "forgottenhall_level", "value": [1, 12]}
        assert records[0]["old"] == [10, 12] and records[0]["new"] == [1, 12]

    def test_rejections_are_recorded_too(self, tmp_path):
        tools, _ = make_tools(tmp_path)
        tools.set_m7a_config("after_finish", "Shutdown")
        records = [json.load(open(p, encoding="utf-8")) for p in sorted((tmp_path / "rec").rglob("*set_m7a_config.json"))]
        assert records[0]["verdict"]["rule"] == "config_rejected"

    def test_unchanged_value_does_not_write(self, tmp_path):
        tools, cfg = make_tools(tmp_path)
        assert "没有变化" in tools.set_m7a_config("forgottenhall_level", [10, 12]).text
        assert cfg.saves == []

    def test_level_change_resets_timestamp_like_gui(self, tmp_path):
        tools, cfg = make_tools(tmp_path)
        out = tools.set_m7a_config("forgottenhall_level", [1, 12]).text
        assert cfg.file["forgottenhall_timestamp"] == 0 and "forgottenhall_timestamp 重置为 0" in out
        assert len(cfg.saves) == 1  # 一次落盘
        tools.set_m7a_config("purefiction_stellar_enable", True)
        assert cfg.file["purefiction_timestamp"] == 0
        # 队伍、enable 变化不动时间戳
        cfg.file["apocalyptic_timestamp"] = cfg.data["apocalyptic_timestamp"] = 99
        tools.set_m7a_config("apocalyptic_team1", [["Asta", 1]])
        tools.set_m7a_config("apocalyptic_enable", True)
        assert cfg.file["apocalyptic_timestamp"] == 99

    def test_other_keys_are_not_touched(self, tmp_path):
        tools, cfg = make_tools(tmp_path)
        tools.set_m7a_config("forgottenhall_enable", True)
        assert cfg.file["after_finish"] == "None" and cfg.file["notify_telegram_token"] == "SECRET-TOKEN"

    def test_task_running_says_takes_effect_next_run(self, tmp_path):
        tools, cfg = make_tools(tmp_path, running="有 M7A 任务正在运行：任务 daily")
        out = tools.set_m7a_config("forgottenhall_enable", True).text
        assert cfg.file["forgottenhall_enable"] is True
        assert "下次运行任务时才生效" in out and "任务 daily" in out
        out = tools.set_m7a_config("power_plan", [["侵蚀隧洞", "睿治之径", 2]]).text
        assert "写回" in out

    def test_no_task_says_written(self, tmp_path):
        tools, _ = make_tools(tmp_path)
        assert "已写入 config.yaml" in tools.set_m7a_config("forgottenhall_enable", True).text

    def test_readback_mismatch_is_reported(self, tmp_path):
        cfg = FakeCfg(readonly=True)  # 假装保存被跳过（只读进程）
        tools, _ = make_tools(tmp_path, cfg)
        out = tools.set_m7a_config("forgottenhall_enable", True).text
        assert "读回" in out and "没有写成功" in out

    def test_external_change_is_not_overwritten(self, tmp_path):
        tools, cfg = make_tools(tmp_path)
        cfg.file["forgottenhall_level"] = [2, 3]          # 图形界面刚改了关卡
        tools.set_m7a_config("forgottenhall_enable", True)  # 我们改另一个键
        assert cfg.file["forgottenhall_level"] == [2, 3] and cfg.file["forgottenhall_enable"] is True

    def test_set_works_without_game_or_lock(self, tmp_path):
        tools, _ = make_tools(tmp_path)
        tools._lock.acquire()  # 操作锁被占着也能改配置（不是游戏操作）
        try:
            assert "→" in tools.set_m7a_config("forgottenhall_enable", True).text
        finally:
            tools._lock.release()


# ---------------------------------------------------------------- 真实 Config：注释、外部改动、离线 ----

@pytest.fixture
def real_cfg(tmp_path, monkeypatch):
    from module.config.config import Config
    monkeypatch.delenv("MARCH7TH_CONFIG_READONLY", raising=False)
    previous = Config._readonly
    path = tmp_path / "config.yaml"
    shutil.copyfile(EXAMPLE, path)
    cfg = object.__new__(Config)  # 绕开单例，不碰真实的 cfg
    Config.__init__(cfg, VERSION, EXAMPLE, str(path))
    yield cfg, path
    Config._readonly = previous


class TestRealConfig:
    def _service(self, cfg):
        return mc.ConfigService(lambda: mc.M7aConfig(cfg), character_ids=mc.list_character_ids)

    def test_comments_and_format_are_preserved(self, real_cfg):
        cfg, path = real_cfg
        before = path.read_text(encoding="utf-8").splitlines()
        service = self._service(cfg)
        service.set("forgottenhall_level", [1, 12])
        service.set("forgottenhall_team1", [["Asta", -1], ["Pela", 1]])
        service.set("forgottenhall_enable", True)
        service.set("power_plan", [["拟造花萼（金）", "回忆之蕾", 10]])
        after = path.read_text(encoding="utf-8").splitlines()
        # 注释行一个不少
        comments_before = [line for line in before if line.strip().startswith("#")]
        comments_after = [line for line in after if line.strip().startswith("#")]
        assert comments_before == comments_after
        # 行尾注释还在
        assert any(line.startswith("forgottenhall_enable: true") and "# 是否启用混沌回忆功能" in line for line in after)
        assert any(line.startswith("forgottenhall_level:") and "混沌回忆关卡范围" in line for line in after)
        # 其它键的行原样不动
        for prefix in ("purefiction_enable", "apocalyptic_level", "auto_battle_detect_enable", "use_fuel"):
            assert [l for l in after if l.startswith(prefix)] == [l for l in before if l.startswith(prefix)], prefix
        # 值确实改了，读回来是普通类型
        from ruamel.yaml import YAML
        data = YAML().load(path.read_text(encoding="utf-8"))
        assert mc.plain(data["forgottenhall_level"]) == [1, 12]
        assert mc.plain(data["forgottenhall_team1"]) == [["Asta", -1], ["Pela", 1]]
        assert mc.plain(data["power_plan"]) == [["拟造花萼（金）", "回忆之蕾", 10]]

    def test_power_plan_set_then_cleared_restores_original_text(self, real_cfg):
        cfg, path = real_cfg
        original = path.read_text(encoding="utf-8")
        service = self._service(cfg)
        service.set("power_plan", [["拟造花萼（金）", "回忆之蕾", 10], ["侵蚀隧洞", "睿治之径", 5]])
        mid = path.read_text(encoding="utf-8")
        assert "power_plan:\n- - 拟造花萼（金）\n  - 回忆之蕾\n  - 10\n- - 侵蚀隧洞" in mid
        service.set("power_plan", [])
        assert path.read_text(encoding="utf-8") == original

    def test_team_shrink_keeps_following_comment(self, real_cfg):
        cfg, path = real_cfg
        service = self._service(cfg)
        service.set("forgottenhall_level", [1, 12])
        service.set("forgottenhall_team1", [["Asta", -1]])
        text = path.read_text(encoding="utf-8")
        assert "forgottenhall_team1: # 队伍1配置。\n- - Asta\n  - -1\nforgottenhall_team2: # 队伍2配置。" in text
        assert "# 混沌回忆队伍配置，详细配置每个队伍成员及其秘技使用次数。" in text

    def test_diff_is_limited_to_changed_keys(self, real_cfg):
        import difflib
        cfg, path = real_cfg
        before = path.read_text(encoding="utf-8").splitlines()
        self._service(cfg).set("apocalyptic_level", [1, 4])
        after = path.read_text(encoding="utf-8").splitlines()
        changed = [l for l in difflib.unified_diff(before, after, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        assert changed and len(changed) <= 6, changed

    def test_external_edit_made_after_load_survives(self, real_cfg):
        cfg, path = real_cfg
        service = self._service(cfg)
        service.get(["forgottenhall_level"])  # 内存里已经是旧状态
        text = path.read_text(encoding="utf-8")
        assert "use_fuel: false" in text
        path.write_text(text.replace("use_fuel: false", "use_fuel: true"), encoding="utf-8")  # 图形界面刚改的别的键
        service.set("purefiction_enable", True)
        final = path.read_text(encoding="utf-8")
        assert "use_fuel: true" in final and "purefiction_enable: true" in final

    def test_get_sees_external_changes(self, real_cfg):
        cfg, path = real_cfg
        service = self._service(cfg)
        text = path.read_text(encoding="utf-8").replace("forgottenhall_enable: false", "forgottenhall_enable: true")
        path.write_text(text, encoding="utf-8")
        assert service.get(["forgottenhall_enable"]) == "forgottenhall_enable = true"

    def test_readonly_process_cannot_write_and_is_told_so(self, real_cfg):
        from module.config.config import Config
        cfg, path = real_cfg
        before = path.read_bytes()
        Config._readonly = True
        with pytest.raises(mc.ConfigError, match="没有写成功"):
            self._service(cfg).set("forgottenhall_enable", True)
        assert path.read_bytes() == before


class TestOfflineMode:
    @pytest.fixture
    def fake_root(self, tmp_path):
        root = tmp_path / "root"
        (root / "assets" / "config").mkdir(parents=True)
        shutil.copyfile(EXAMPLE, root / "assets" / "config" / "config.example.yaml")
        shutil.copyfile(VERSION, root / "assets" / "config" / "version.txt")
        text = open(EXAMPLE, encoding="utf-8").read().replace("use_fuel: False", "use_fuel: true")  # 真实配置里有个和默认值不同的键
        (root / "config.yaml").write_text(text, encoding="utf-8")
        image = tmp_path / "shot.png"
        Image.new("RGB", (64, 64)).save(image)
        return root, image

    def test_offline_writes_only_to_temp_copy(self, fake_root, monkeypatch):
        from module.config.config import Config
        root, image = fake_root
        real_file = root / "config.yaml"
        before = real_file.read_bytes()
        monkeypatch.setenv("MARCH7TH_CONFIG_READONLY", "1")  # 离线 server 就是这个环境
        previous = Config._readonly
        tools = mt.McpTools(offline_image=str(image), root=str(root), observer=lambda grid: None)
        out = tools.set_m7a_config("forgottenhall_enable", True).text
        assert "forgottenhall_enable：false → true" in out and "离线模式" in out
        assert real_file.read_bytes() == before                                # 真实文件一个字节都没动
        copy_path = root / "logs" / "mcp_agent" / "offline" / "config_copy" / "config.yaml"
        assert "forgottenhall_enable: true" in copy_path.read_text(encoding="utf-8")
        copy_text = copy_path.read_text(encoding="utf-8")
        assert "use_fuel: true" in copy_text                                   # 副本是真实配置的拷贝
        assert "# 是否启用混沌回忆功能" in copy_text                              # 副本保留注释
        assert Config._readonly == previous                                    # 只读标志已还原
        assert tools.get_m7a_config(["forgottenhall_enable"]).text == "forgottenhall_enable = true"

    def test_offline_copy_is_refreshed_from_real_file_on_start(self, fake_root, monkeypatch):
        root, image = fake_root
        monkeypatch.setenv("MARCH7TH_CONFIG_READONLY", "1")
        tools = mt.McpTools(offline_image=str(image), root=str(root), observer=lambda grid: None)
        tools.set_m7a_config("forgottenhall_enable", True)
        tools2 = mt.McpTools(offline_image=str(image), root=str(root), observer=lambda grid: None)
        assert tools2.get_m7a_config(["forgottenhall_enable"]).text == "forgottenhall_enable = false"
