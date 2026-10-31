"""MCP 工具 list_characters / get_m7a_config / set_m7a_config 的逻辑（不依赖 mcp 包，可单测）。

只允许读写白名单内的配置键：逐光捡金三个模式（混沌回忆 forgottenhall、虚构叙事 purefiction、末日幻影 apocalyptic）
和体力计划相关的键。其它键（token、key、账号、密码、通知、更新、after_finish、游戏路径、定时任务……）一律拒绝，
get 也不例外，所以敏感值不可能被读出来。

写入走 M7A 自己的配置接口（module/config/config.py 的 Config.set_values）：
- 它先从文件重新加载再改再整份写回（原子替换，ruamel 保留注释和格式），所以不会覆盖图形界面或别的进程刚写入的其它键；
- 图形界面有 config-watch，会发现文件被外部修改并自动重新加载（app/main_window.py 的 is_config_changed / _load_config）；
- 写完再从文件读回来核对一次。
离线模式写到临时副本，不碰真实的 config.yaml。
"""
import copy
import json
import os
import shutil
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

from module.llm_agent.observe import ROOT

CHARACTER_DIR = os.path.join(ROOT, "assets", "images", "share", "character")
CHARACTER_NAMES_PATH = os.path.join(ROOT, "assets", "config", "character_names.json")
INSTANCE_NAMES_PATH = os.path.join(ROOT, "assets", "config", "instance_names.json")

# 各模式的最高关卡（图形界面的关卡范围控件上限：混沌回忆 12 层，虚构叙事和末日幻影 4 层）
LEVEL_MAX = {"forgottenhall": 12, "purefiction": 4, "apocalyptic": 4}
TEAM_SIZE_MAX = 4
# 秘技次数：-1 表示「最后一个放秘技和普攻的角色」（每队最多一个）；图形界面可选 -1/0/1/2
TECHNIQUE_MIN, TECHNIQUE_MAX = -1, 2
# 副本类型 → 连续挑战次数上限（tasks/power/power.py 的 attempts_per_run_max）
CHALLENGE_COUNT_MAX = {"拟造花萼（金）": 24, "拟造花萼（赤）": 24, "凝滞虚影": 8, "侵蚀隧洞": 6, "饰品提取": 6, "历战余响": 3}
# 配置文件里 instance_type 可选的副本类型（config.example.yaml 的注释）
INSTANCE_TYPES = ("拟造花萼（金）", "拟造花萼（赤）", "凝滞虚影", "侵蚀隧洞", "饰品提取")
POWER_PLAN_MAX_ITEMS = 20
POWER_PLAN_MAX_COUNT = 999

# 白名单：键 → (校验类型, 参数)
CONFIG_KEYS: Dict[str, Tuple[str, Any]] = {}
for _prefix, _max in LEVEL_MAX.items():
    CONFIG_KEYS[f"{_prefix}_enable"] = ("bool", None)
    CONFIG_KEYS[f"{_prefix}_level"] = ("level", _prefix)
    CONFIG_KEYS[f"{_prefix}_team1"] = ("team", _prefix)
    CONFIG_KEYS[f"{_prefix}_team2"] = ("team", _prefix)
CONFIG_KEYS["purefiction_stellar_enable"] = ("bool", None)
CONFIG_KEYS["purefiction_team3"] = ("team", "purefiction")
CONFIG_KEYS.update({
    "power_enable": ("bool", None),
    "power_plan": ("power_plan", None),
    "power_plan_keep": ("bool", None),
    "instance_type": ("instance_type", None),
    "instance_names": ("instance_names", None),
    "instance_names_challenge_count": ("challenge_count", None),
})

# 这些键变化时，图形界面会同时把对应的 *_timestamp 重置为 0（否则本周已经跑过的话，日常会因「尚未刷新」跳过）
_TIMESTAMP_RESET_KEYS = {"forgottenhall_level": "forgottenhall_timestamp", "purefiction_level": "purefiction_timestamp",
                         "apocalyptic_level": "apocalyptic_timestamp", "purefiction_stellar_enable": "purefiction_timestamp"}


class ConfigError(ValueError):
    """参数不合法；消息是给模型看的中文说明。"""


# ---------------------------------------------------------------- 角色 ----

def list_character_ids(directory: str = CHARACTER_DIR) -> List[str]:
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(name[:-4] for name in names if name.lower().endswith(".png"))


def load_character_names(path: str = CHARACTER_NAMES_PATH) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return {str(key): str(value) for key, value in data.items()}
    except (OSError, ValueError, AttributeError):
        return {}


def format_characters(ids: List[str], names: Dict[str, str]) -> str:
    if not ids:
        return "没有找到角色图片目录（assets/images/share/character）。"
    lines = [f"共 {len(ids)} 个角色（队伍配置里用左边的英文 ID，区分大小写）："]
    lines += [f"{identifier}  {names[identifier]}" if identifier in names else identifier for identifier in ids]
    return "\n".join(lines)


# ---------------------------------------------------------------- 值的整理 ----

def plain(value):
    """ruamel 的 CommentedSeq / CommentedMap / 标量 → 普通的 list / dict / int / str / bool / float。"""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    return value


def to_json(value) -> str:
    return json.dumps(plain(value), ensure_ascii=False, default=str)


def _loads_if_text(value, kind: str):
    """模型有时会把列表、布尔值当成 JSON 字符串传进来：这几类值允许先解析一次。"""
    if isinstance(value, str) and kind in ("bool", "level", "team", "power_plan", "instance_names", "challenge_count"):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------- 校验 ----

def validate_bool(key: str, value) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{key} 必须是布尔值 true 或 false。")
    return value


def validate_level(key: str, value, mode: str) -> List[int]:
    highest = LEVEL_MAX[mode]
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_is_int(item) for item in value)):
        raise ConfigError(f"{key} 必须是 [最小, 最大] 两个整数，例如 [1, {highest}]。")
    low, high = value
    if not (1 <= low <= highest and 1 <= high <= highest):
        raise ConfigError(f"{key} 的关卡必须在 1–{highest} 之间。")
    if low > high:
        raise ConfigError(f"{key} 的最小关卡不能大于最大关卡。")
    return [low, high]


def validate_team(key: str, value, character_ids) -> List[list]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= TEAM_SIZE_MAX:
        raise ConfigError(f"{key} 必须是 1–{TEAM_SIZE_MAX} 个 [角色ID, 秘技次数]，例如 [[\"Asta\", -1], [\"JingYuan\", 1]]。")
    known = set(character_ids)
    team, seen, leaders = [], set(), 0
    for index, member in enumerate(value, 1):
        if not isinstance(member, (list, tuple)) or len(member) != 2:
            raise ConfigError(f"{key} 的第 {index} 位必须是 [角色ID, 秘技次数]。")
        character, technique = member
        if not isinstance(character, str) or character not in known:
            raise ConfigError(f"{key} 的第 {index} 位角色 ID {character!r} 不存在，请用 list_characters 查看（区分大小写）。")
        if not _is_int(technique) or not TECHNIQUE_MIN <= technique <= TECHNIQUE_MAX:
            raise ConfigError(f"{key} 的第 {index} 位秘技次数必须是 {TECHNIQUE_MIN}–{TECHNIQUE_MAX} 的整数"
                              f"（-1 表示最后一个放秘技和普攻的角色）。")
        if character in seen:
            raise ConfigError(f"{key} 里角色 {character} 重复了。")
        seen.add(character)
        leaders += technique == -1
        team.append([character, technique])
    if leaders > 1:
        raise ConfigError(f"{key} 里只能有一个角色的秘技次数是 -1（最后一个放秘技和普攻的角色）。")
    return team


def load_instance_names(path: str = INSTANCE_NAMES_PATH) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_instance_type(key: str, value) -> str:
    if not isinstance(value, str) or value not in INSTANCE_TYPES:
        raise ConfigError(f"{key} 必须是以下之一：{'、'.join(INSTANCE_TYPES)}。")
    return value


def validate_instance_names(key: str, value, current: dict, instances: dict) -> dict:
    """副本名称：{副本类型: 副本名称}，可以只给要改的类型，其余沿用当前值。"""
    if not isinstance(value, dict) or not value:
        raise ConfigError(f"{key} 必须是 {{副本类型: 副本名称}}，例如 {{\"侵蚀隧洞\": \"睿治之径\"}}（可只给要改的类型）。")
    merged = dict(current)
    for instance_type, name in value.items():
        if instance_type not in instances:
            raise ConfigError(f"{key} 里没有副本类型「{instance_type}」，可选：{'、'.join(instances)}。")
        if not isinstance(name, str) or name not in instances[instance_type]:
            raise ConfigError(f"「{instance_type}」没有副本「{name}」，可选：{'、'.join(instances[instance_type])}。")
        merged[instance_type] = name
    return merged


def validate_challenge_count(key: str, value, current: dict) -> dict:
    if not isinstance(value, dict) or not value:
        raise ConfigError(f"{key} 必须是 {{副本类型: 连续挑战次数}}，例如 {{\"侵蚀隧洞\": 6}}（可只给要改的类型）。")
    merged = dict(current)
    for instance_type, count in value.items():
        if instance_type not in CHALLENGE_COUNT_MAX:
            raise ConfigError(f"{key} 里没有副本类型「{instance_type}」，可选：{'、'.join(CHALLENGE_COUNT_MAX)}。")
        highest = CHALLENGE_COUNT_MAX[instance_type]
        if not _is_int(count) or not 1 <= count <= highest:
            raise ConfigError(f"「{instance_type}」的连续挑战次数必须是 1–{highest} 的整数。")
        merged[instance_type] = count
    return merged


def validate_power_plan(key: str, value, instances: dict) -> List[list]:
    """体力计划：[[副本类型, 副本名称, 次数], ...]，例如 [["拟造花萼（金）", "回忆之蕾", 10]]。空列表表示清空计划。"""
    if not isinstance(value, (list, tuple)) or len(value) > POWER_PLAN_MAX_ITEMS:
        raise ConfigError(f"{key} 必须是最多 {POWER_PLAN_MAX_ITEMS} 项的列表，每项 [副本类型, 副本名称, 次数]。")
    plan = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ConfigError(f"{key} 的第 {index} 项必须是 [副本类型, 副本名称, 次数]。")
        instance_type, name, count = item
        if instance_type not in instances:
            raise ConfigError(f"{key} 的第 {index} 项副本类型「{instance_type}」不存在，可选：{'、'.join(instances)}。")
        if not isinstance(name, str) or name not in instances[instance_type] or name == "无":
            valid = [n for n in instances[instance_type] if n != "无"]
            raise ConfigError(f"{key} 的第 {index} 项：「{instance_type}」没有可用的副本「{name}」，可选：{'、'.join(valid)}。")
        if not _is_int(count) or not 1 <= count <= POWER_PLAN_MAX_COUNT:
            raise ConfigError(f"{key} 的第 {index} 项次数必须是 1–{POWER_PLAN_MAX_COUNT} 的整数。")
        plan.append([instance_type, name, count])
    return plan


# ---------------------------------------------------------------- 配置读写 ----

class M7aConfig:
    """对 M7A Config 对象的薄封装：读之前从文件重新加载，写用 Config.set_values，写完从文件读回。"""

    def __init__(self, cfg, force_writable: bool = False):
        self._cfg = cfg
        self._force_writable = force_writable  # 离线模式下的临时副本：进程里设了只读标志也要能写副本

    def reload(self):
        self._cfg._load_config(save=False)

    def get(self, key, default=None):
        return plain(self._cfg.get_value(key, default))

    def file_value(self, key):
        """直接从文件读（不经过内存），读不到返回 (False, None)。"""
        data = self._cfg._read_file_config()
        if not data or key not in data:
            return False, None
        return True, plain(data[key])

    @contextmanager
    def _writable(self):
        if not self._force_writable:
            yield
            return
        config_class = type(self._cfg)
        previous = config_class._readonly
        config_class._readonly = False
        try:
            yield
        finally:
            config_class._readonly = previous

    def set_values(self, values: dict):
        """等价于 Config.set_values（先从文件重新加载，再改，再一次性原子落盘），
        区别只是列表和字典原地更新，保留键行尾的注释和下一段前面的注释
        （Config.set_values 直接替换成普通列表，ruamel 会把这些注释丢掉）。"""
        with self._writable():
            self._cfg._load_config(save=False)
            for key, value in values.items():
                assign_preserving(self._cfg.config, key, value)
            self._cfg.save_config()


def _is_seq(node) -> bool:
    return node.__class__.__name__ == "CommentedSeq"


def _is_map(node) -> bool:
    return node.__class__.__name__ == "CommentedMap"


def _replace_seq(old, new: list):
    """原地替换一个 CommentedSeq 的内容。

    保留：键行尾的注释（old.ca.comment）、最后一个元素后面的整段注释（下一个配置块前面的说明，ruamel 挂在 ca.items 上）、
    以及空列表 [] 与块状列表之间的写法切换。
    """
    trailing = old.ca.items.get(len(old) - 1) if len(old) else None
    was_empty = len(old) == 0
    del old[:]
    old.ca.items.clear()
    for item in new:
        old.append(_build_node(item))
    if was_empty and new:
        old.fa.set_block_style()
    elif not new:
        old.fa.set_flow_style()
    if trailing and new:
        old.ca.items[len(new) - 1] = trailing
    elif trailing:  # 清空时把那段注释挂回序列本身，免得丢
        old.ca.items[0] = trailing


def _build_node(value):
    if isinstance(value, (list, tuple)):
        from ruamel.yaml.comments import CommentedSeq
        node = CommentedSeq()
        for item in value:
            node.append(_build_node(item))
        return node
    return copy.deepcopy(value)


def assign_preserving(container, key, value):
    """container[key] = value，但如果原来是列表或字典就原地更新，保留 ruamel 挂在上面的注释。"""
    old = container.get(key)
    if _is_seq(old) and isinstance(value, (list, tuple)):
        _replace_seq(old, list(value))
    elif _is_map(old) and isinstance(value, dict):
        for sub_key in list(old.keys()):
            if sub_key not in value:
                del old[sub_key]
        for sub_key, sub_value in value.items():
            assign_preserving(old, sub_key, sub_value)
    elif isinstance(value, (list, tuple, dict)):
        container[key] = copy.deepcopy(value)
    else:
        container[key] = value


def make_config_copy(root: str, work_dir: str):
    """离线模式：把真实 config.yaml 复制到 work_dir，再用一个独立的 Config 对象管理这份副本。"""
    from module.config.config import Config
    os.makedirs(work_dir, exist_ok=True)
    target = os.path.join(work_dir, "config.yaml")
    source = os.path.join(root, "config.yaml")
    if os.path.exists(source):
        shutil.copyfile(source, target)
    elif os.path.exists(target):
        os.remove(target)
    previous = Config._readonly
    config = object.__new__(Config)  # 绕开单例：不能动真实的 cfg
    try:
        Config._readonly = False
        Config.__init__(config, os.path.join(root, "assets", "config", "version.txt"),
                        os.path.join(root, "assets", "config", "config.example.yaml"), target)
    finally:
        Config._readonly = previous
    return config, target


class ConfigService:
    """get_m7a_config / set_m7a_config 的实现。cfg_provider 返回 M7aConfig；其它依赖可注入以便测试。"""

    def __init__(self, cfg_provider: Callable[[], M7aConfig], character_ids: Callable[[], List[str]] = list_character_ids,
                 instances: Callable[[], dict] = load_instance_names):
        self._provider = cfg_provider
        self._character_ids = character_ids
        self._instances = instances
        self._lock = threading.Lock()

    # ---- 读 ----

    @staticmethod
    def _parse_keys(keys) -> List[str]:
        if keys is None:
            return []
        if isinstance(keys, str):
            text = keys.strip()
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed]
                except ValueError:
                    pass
            return [part.strip() for part in text.replace("，", ",").split(",") if part.strip()]
        if isinstance(keys, (list, tuple)):
            return [str(item).strip() for item in keys]
        raise ConfigError("keys 必须是配置键名的列表。")

    def get(self, keys) -> str:
        wanted = self._parse_keys(keys) or list(CONFIG_KEYS)
        denied = [key for key in wanted if key not in CONFIG_KEYS]
        if denied:
            raise ConfigError(f"不允许读取这些配置键：{'、'.join(denied)}。可读的键：{'、'.join(CONFIG_KEYS)}。")
        config = self._provider()
        config.reload()
        lines = []
        for key in dict.fromkeys(wanted):
            lines.append(f"{key} = {to_json(config.get(key))}")
        return "\n".join(lines)

    # ---- 写 ----

    def _validate(self, key: str, value, config: M7aConfig):
        """返回 (规范化后的新值, 警告列表)。"""
        kind, param = CONFIG_KEYS[key]
        value = _loads_if_text(value, kind)
        warnings: List[str] = []
        if kind == "bool":
            return validate_bool(key, value), warnings
        if kind == "level":
            return validate_level(key, value, param), warnings
        if kind == "team":
            team = validate_team(key, value, self._character_ids())
            warnings += self._cross_team_warnings(key, team, param, config)
            return team, warnings
        if kind == "instance_type":
            return validate_instance_type(key, value), warnings
        if kind == "instance_names":
            return validate_instance_names(key, value, config.get(key, {}) or {}, self._instances()), warnings
        if kind == "challenge_count":
            return validate_challenge_count(key, value, config.get(key, {}) or {}), warnings
        if kind == "power_plan":
            return validate_power_plan(key, value, self._instances()), warnings
        raise ConfigError(f"内部错误：未知校验类型 {kind}")

    @staticmethod
    def _cross_team_warnings(key: str, team: list, mode: str, config: M7aConfig) -> List[str]:
        """同一模式里，不同队伍不能用同一个角色（图形界面会拦）；这里只提醒，不拦，方便分步调整。"""
        mine = {member[0] for member in team}
        warnings = []
        for number in (1, 2, 3):
            other_key = f"{mode}_team{number}"
            if other_key == key or other_key not in CONFIG_KEYS:
                continue
            if other_key == "purefiction_team3" and not config.get("purefiction_stellar_enable", False):
                continue
            other = config.get(other_key, []) or []
            shared = sorted(mine & {member[0] for member in other if isinstance(member, list) and member})
            if shared:
                warnings.append(f"注意：{'、'.join(shared)} 同时出现在 {other_key} 里，游戏不允许不同队伍用同一角色，请接着改 {other_key}。")
        return warnings

    def set(self, key: str, value, task_running: Optional[str] = None) -> Tuple[str, dict]:
        """校验并写入。返回 (给模型的文字, 记录用的字典)。校验失败抛 ConfigError。"""
        if key not in CONFIG_KEYS:
            raise ConfigError(f"不允许修改配置键「{key}」。可改的键：{'、'.join(CONFIG_KEYS)}。")
        with self._lock:
            config = self._provider()
            config.reload()
            old = config.get(key)
            new, warnings = self._validate(key, value, config)
            updates = {key: new}
            notes = []
            reset_key = _TIMESTAMP_RESET_KEYS.get(key)
            if new == old:
                text = f"{key} 没有变化：{to_json(old)}"
                return text, {"key": key, "old": old, "new": new, "changed": False}
            if reset_key and config.get(reset_key, 0):
                updates[reset_key] = 0
                notes.append(f"同时把 {reset_key} 重置为 0（和图形界面一致，否则本周已跑过的话日常会因「尚未刷新」跳过）。")
            config.set_values(updates)
            # 写完从文件读回来核对
            found, written = config.file_value(key)
            if not found or written != plain(new):
                raise ConfigError(f"写入后读回的值和预期不一致（读到 {to_json(written)}），配置可能没有写成功，请检查 config.yaml 是否被占用或只读。")
        lines = [f"{key}：{to_json(old)} → {to_json(new)}"]
        lines += notes + warnings
        if task_running:
            lines.append(f"当前有 M7A 任务在运行（{task_running}）：下次运行任务时才生效。")
            if key == "power_plan":
                lines.append("体力计划任务结束时会把它自己的剩余计划写回，可能覆盖刚才的修改；任务结束后用 get_m7a_config 再确认一次。")
        else:
            lines.append("已写入 config.yaml，下次运行任务时生效；图形界面开着的话会自动重新加载。")
        return "\n".join(lines), {"key": key, "old": old, "new": new, "changed": True,
                                  "extra_updates": {k: v for k, v in updates.items() if k != key}, "warnings": warnings}

