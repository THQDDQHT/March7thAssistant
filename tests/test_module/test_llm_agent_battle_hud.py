# coding:utf-8
"""战斗 HUD 识别：夹具是 HUD 局部拼成的整图 / 涂黑 UID 区域的缩小整图（fixtures/battle_hud/，JPEG）。

像素判断（阶段、选中、血条、终结技圆盘、倍速自动）直接对夹具跑；OCR 一律替身（技能点数字、波次、剩余轮次、结算标题）。
另有一个真 OCR 的端到端用例，OCR 模型跑不起来时跳过。
"""
import os

import numpy as np
import pytest
from PIL import Image

from module.llm_agent import battle_hud as bh

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "battle_hud")


def load(name) -> Image.Image:
    return Image.open(os.path.join(FIXTURES, name + ".jpg")).convert("RGB")


def fake_ocr(sp=None, wave=None, rounds=None, title=None, conf=0.99):
    """替身 OCR：对 _ocr_texts 要读的几块裁剪图，按名字给定文字（不给的当没读到）。"""
    values = {"sp": sp, "wave": wave, "rounds": rounds, "title": title}

    def ocr_texts(ocr, crops):
        return {name: (values[name], conf) for name, _ in crops if values.get(name) is not None}
    return ocr_texts


@pytest.fixture
def read(monkeypatch):
    """read(夹具名, sp=..., wave=..., rounds=..., title=...) → BattleHud；OCR 用替身。"""
    def _read(name, **texts):
        monkeypatch.setattr(bh, "_ocr_texts", fake_ocr(**texts))
        return bh.read_battle_hud(load(name))
    return _read


# ---------------------------------------------------------------- 夹具上的整体判断 ----

class TestCommandPhase:
    @pytest.mark.parametrize("name, selected, sp, team", [
        ("cmd_basic_sp3_3p", "basic", 3, 3),
        ("cmd_skill_sp3_3p", "skill", 3, 3),
        ("cmd_basic_sp2_3p_preview_lit", "basic", 2, 3),  # 数字 2，但「将获得」的预览星被扫光特效照白，亮星 3 颗
        ("cmd_basic_sp5_4p", "basic", 5, 4),
        ("cmd_skill_sp5_4p", "skill", 5, 4),
        ("cmd_basic_sp4_4p", "basic", 4, 4),
    ])
    def test_selected_and_skill_points(self, read, name, selected, sp, team):
        hud = read(name, sp=str(sp), wave="1/1", rounds="2")
        assert hud.in_battle is True
        assert hud.phase == "command"
        assert hud.selected == selected
        assert hud.skill_points == sp
        assert hud.team_size == team
        assert hud.rounds_left == 2

    def test_auto_speed_and_wave(self, read):
        hud = read("cmd_basic_sp3_3p", sp="3", wave="1/ 1")
        # 试镜不能开自动：自动图标是灰掉的，框不亮
        assert (hud.auto_on, hud.speed_on, hud.wave) == (False, True, (1, 1))

    def test_digit_that_contradicts_the_stars_is_unknown(self, read):
        # 普攻选中、亮 3 颗星：数字只能是 3 或 2（亮星 = 数字 或 数字+1）；读成 5 或 4 都对不上
        for wrong in ("5", "4", "1"):
            hud = read("cmd_basic_sp3_3p", sp=wrong)
            assert hud.phase == "command" and hud.selected == "basic" and hud.skill_points is None

    def test_skill_selected_expects_one_star_deducted(self, read):
        assert read("cmd_skill_sp3_3p", sp="3").skill_points == 3
        assert read("cmd_skill_sp3_3p", sp="4").skill_points is None  # 战技选中亮 2 颗：数字只能是 3（或 2）

    def test_low_confidence_or_missing_or_multi_digit_is_unknown(self, monkeypatch):
        monkeypatch.setattr(bh, "_ocr_texts", fake_ocr(sp="3", conf=0.5))
        assert bh.read_battle_hud(load("cmd_basic_sp3_3p")).skill_points is None
        monkeypatch.setattr(bh, "_ocr_texts", fake_ocr())
        hud = bh.read_battle_hud(load("cmd_basic_sp3_3p"))
        assert hud.skill_points is None and hud.selected == "basic"
        monkeypatch.setattr(bh, "_ocr_texts", fake_ocr(sp="35"))
        assert bh.read_battle_hud(load("cmd_basic_sp3_3p")).skill_points is None

    def test_ocr_failure_does_not_break_the_pixel_fields(self):
        def boom(image):
            raise RuntimeError("ocr 挂了")
        hud = bh.read_battle_hud(load("cmd_basic_sp3_3p"), ocr=boom)
        assert (hud.phase, hud.selected, hud.team_size) == ("command", "basic", 3)
        assert hud.skill_points is None and hud.wave is None and "RuntimeError" in hud.debug["ocr_error"]
        assert bh.read_battle_hud(load("result_trial"), ocr=boom).phase is None


class TestUltimate:
    def test_ult_pending(self, read):
        hud = read("ult_pending_3p", sp="2", wave="1/1", rounds="2")
        assert hud.phase == "ult_pending" and hud.selected is None
        assert hud.skill_points == 2
        assert hud.ult_ready_slots() == [1]

    def test_slot_ready_flags(self, read):
        # 4 人队：1 号主演徽记发光（计数 ≥ 6）、2 号圆盘满；3 号液面不满
        hud = read("cmd_skill_sp5_4p", sp="5")
        flags = {slot.index: slot.ult_ready for slot in hud.slots}
        assert flags[1] is True and flags[2] is True and flags[3] is False

    def test_dim_emblem_and_partial_discs_are_not_ready(self, read):
        hud = read("cmd_basic_sp3_3p", sp="3")
        assert [slot.ult_ready for slot in hud.slots] == [False, False, False]


class TestWhiteDisc:
    """白 / 银色满盘（试用角色的终结技圆盘）：实机上曾经把满盘报成「未满」。"""

    def test_white_full_disc_is_ready_in_command_and_pending(self, read):
        hud = read("b2_cmd_white_full_disc", sp="5", wave="1/2", rounds="0")
        assert hud.team_size == 2 and hud.skill_points == 5
        assert hud.slots[1].ult_ready is True
        hud = read("b2_ult_pending_white_disc", sp="5")
        assert hud.phase == "ult_pending" and hud.slots[1].ult_ready is True

    def test_gray_partial_and_empty_discs_are_not_ready(self, read):
        assert [slot.ult_ready for slot in read("b2_cmd_gray_partial_disc", sp="1").slots] == [False, False]
        hud = read("b2_cmd_empty_after_ult", sp="5")  # 2 号放完终结技，圆盘空了；1 号淡蓝色盘满了没发光：说不准
        assert [slot.ult_ready for slot in hud.slots] == [None, False]

    def test_pale_full_blue_disc_without_glow_is_unknown(self, read):
        # 大黑塔（1 号主演）的淡蓝色液面已经满了但徽记还没发光：说不准，不能报「未满」
        hud = read("b2_cmd_white_full_disc", sp="5")
        assert hud.slots[0].ult_ready is None


class TestSwitchesOnlyWhenSettled:
    def test_animation_without_buttons_does_not_claim_off(self, read):
        # 动画里倍速按钮被盖住 / 变暗：倍速其实开着，不能报「关」（模型会去按 b 把它关掉）
        hud = read("b2_anim_no_speed_box")
        assert hud.phase == "other" and hud.speed_on is None and hud.auto_on is None

    def test_animation_with_visible_frame_says_on(self, read):
        hud = read("b2_anim_speed_box_on")
        assert hud.phase == "other" and hud.speed_on is True and hud.auto_on is None

    def test_command_phase_reports_off(self, read):
        hud = read("cmd_basic_sp3_3p", sp="3")
        assert hud.speed_on is True and hud.auto_on is False

    def test_unsettled_phase_never_reports_off(self, read):
        for name in ("enemy_turn", "attack_anim", "special_mode", "hud_hidden"):
            hud = read(name)
            assert hud.speed_on in (None, True) and hud.auto_on in (None, True), name

    def test_switch_state_function(self):
        assert bh._switch_state(80, False) is True and bh._switch_state(80, True) is True
        assert bh._switch_state(-30, True) is False
        assert bh._switch_state(-30, False) is None and bh._switch_state(30, True) is None


class TestOtherPhases:
    @pytest.mark.parametrize("name", ["enemy_turn", "attack_anim", "special_mode", "auto_on_wave1_4p", "auto_on_wave2_4p"])
    def test_animations_and_special_modes_are_other(self, read, name):
        hud = read(name, wave="1/1", rounds="2")
        assert hud.in_battle is True and hud.phase == "other"
        assert hud.selected is None and hud.skill_points is None

    def test_special_mode_looks_like_command_buttons_but_has_no_skill_point_panel(self, read):
        # 飞霄终结技连段：E 大 Q 小的按钮、没有技能点面板；只有一个角色
        hud = read("special_mode")
        assert hud.phase == "other" and hud.team_size == 1

    def test_auto_and_speed_in_auto_battle(self, read):
        hud = read("auto_on_wave2_4p", wave="2/2")
        assert hud.auto_on is True and hud.speed_on is True and hud.wave == (2, 2)

    def test_hidden_hud_is_not_battle(self, read):
        hud = read("hud_hidden")
        assert hud.in_battle is False and hud.phase is None and hud.selected is None


class TestNonBattle:
    def test_team_select_page(self, read):
        hud = read("team_select")
        assert hud.in_battle is False and hud.phase is None

    @pytest.mark.parametrize("name", ["result_trial", "result_s_rank"])
    def test_result_page(self, read, name):
        hud = read(name, title="挑战结束")
        assert hud.in_battle is False and hud.phase == "result" and hud.result_text == "挑战结束"

    def test_defeat_page_with_red_title(self, read):
        hud = read("b2_result_defeat", title="战斗失败")
        assert hud.in_battle is False and hud.phase == "result" and hud.result_text == "战斗失败"
        assert hud.debug["result_gate"]["red"] >= bh.RESULT_MIN_RED  # 红色标题：米黄色文字几乎没有
        assert hud.debug["result_gate"]["cream"] < bh.RESULT_MIN_CREAM

    def test_result_page_with_other_title_is_not_result(self, read):
        assert read("result_trial", title="下一章").phase is None

    def test_result_page_does_not_ocr_when_it_is_not_dark_text_page(self, monkeypatch):
        calls = []
        monkeypatch.setattr(bh, "_ocr_texts", lambda ocr, crops: calls.append(crops) or {})
        bh.read_battle_hud(load("team_select"))  # 亮色界面：不会去 OCR
        bh.read_battle_hud(Image.new("RGB", (1920, 1080)))  # 全黑：没有米黄色文字，不 OCR
        assert calls == []

    def test_not_16x9_or_tiny_image(self):
        hud = bh.read_battle_hud(Image.new("RGB", (1000, 1000)))
        assert hud.in_battle is None and hud.debug["error"] == "not_16x9"
        assert bh.read_battle_hud(Image.new("RGB", (64, 36))).in_battle is None

    def test_noise_image_is_not_battle(self):
        noise = np.random.default_rng(1).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
        hud = bh.read_battle_hud(Image.fromarray(noise))
        assert hud.in_battle is False and hud.phase is None


class TestResolution:
    @pytest.mark.parametrize("size", [(1280, 720), (2560, 1440)])
    def test_other_16x9_sizes(self, monkeypatch, size):
        monkeypatch.setattr(bh, "_ocr_texts", fake_ocr(sp="3"))
        image = load("cmd_skill_sp3_3p").resize(size, Image.Resampling.LANCZOS)
        hud = bh.read_battle_hud(image)
        assert (hud.phase, hud.selected, hud.skill_points, hud.team_size) == ("command", "skill", 3, 3)


# ---------------------------------------------------------------- 内部判断 ----

class TestSkillPointCrossCheck:
    @pytest.mark.parametrize("digit, lit, selected, ok", [
        (3, 3, "basic", True), (3, 4, "basic", True), (3, 2, "basic", False), (3, 5, "basic", False),
        (5, 5, "basic", True), (5, 4, "basic", False),
        (3, 2, "skill", True), (3, 3, "skill", True), (3, 1, "skill", False), (3, 4, "skill", False),
        (2, 1, None, True), (2, 2, None, True), (2, 3, None, True), (2, 4, None, False),
        (0, 0, "basic", True), (6, 5, None, False), (-1, 0, None, False),
    ])
    def test_consistency(self, digit, lit, selected, ok):
        assert bh._sp_consistent(digit, lit, selected) is ok

    def test_lit_prefix_counts_leading_bright_stars_only(self):
        cores = [(246, 0.0), (245, 0.0), (120, 0.4), (246, 0.0), (110, 0.1)]
        assert bh._lit_prefix(cores) == 2


class TestOcrStacking:
    def test_texts_are_mapped_back_to_their_crops(self):
        crops = [("sp", np.zeros((50, 45, 3), np.uint8)), ("wave", np.zeros((32, 68, 3), np.uint8)),
                 ("rounds", np.zeros((30, 36, 3), np.uint8))]
        canvas, bands = bh._stack_crops(crops)
        assert canvas.shape[0] >= sum(h for h in (50, 32, 30)) * bh.OCR_SCALE
        assert bands["sp"][1] <= bands["wave"][0] and bands["wave"][1] <= bands["rounds"][0]

        def ocr(image):
            def entry(name, text, x, conf=0.97):
                top, bottom = bands[name]
                return [[[x, top], [x + 40, top], [x + 40, bottom], [x, bottom]], (text, conf)]
            # 同一块里分成两段识别出来：按横坐标拼起来；顺序故意打乱
            return [entry("rounds", "2", 30), entry("wave", "2", 120), entry("wave", "1/", 30), entry("sp", "4", 30, 0.8)]

        texts = bh._ocr_texts(ocr, crops)
        assert texts["wave"] == ("1/2", 0.97)
        assert texts["rounds"] == ("2", 0.97)
        assert texts["sp"] == ("4", 0.8)

    def test_empty_and_malformed_results(self):
        assert bh._ocr_texts(lambda image: [], []) == {}
        crops = [("sp", np.zeros((50, 45, 3), np.uint8))]
        assert bh._ocr_texts(lambda image: None, crops) == {}
        assert bh._ocr_texts(lambda image: [["坏数据"]], crops) == {}

    @pytest.mark.parametrize("text, expected", [("1/2", (1, 2)), ("1/ 1", (1, 1)), ("2／2", (2, 2)), ("3/2", None), ("12", None)])
    def test_parse_wave(self, text, expected):
        assert bh._parse_wave(text) == expected

    @pytest.mark.parametrize("text, expected", [("2", 2), ("12", 12), ("·2", 2), ("", None), ("a", None), ("123", None)])
    def test_parse_rounds(self, text, expected):
        assert bh._parse_rounds(text) == expected


class TestUltReadyRules:
    def test_rules(self):
        plain = {"white": 0.0, "white_mid": 0.0, "white_outside": 0.0, "top": 120, "bottom": 125}
        glow = dict(plain, bright=0.4, hue_frac=0.8, hue=185, disc=5500, compact=0.78)
        dim_emblem = dict(plain, bright=0.02, hue_frac=0.75, hue=185, disc=5300, compact=0.81)
        full_disc = dict(plain, bright=0.05, hue_frac=0.66, hue=235, disc=5200, compact=0.94)
        polluted = dict(full_disc, compact=0.77)  # 背景有大片颜色：不是填满的圆盘
        half = dict(plain, bright=0.18, hue_frac=0.36, hue=115, disc=3600, compact=0.88)
        in_between = dict(plain, bright=0.1, hue_frac=0.5, hue=5, disc=4500, compact=0.9)
        assert bh._ult_ready(glow) is True
        assert bh._ult_ready(dim_emblem) is False
        assert bh._ult_ready(full_disc) is True
        assert bh._ult_ready(polluted) is None
        assert bh._ult_ready(half) is False
        assert bh._ult_ready(in_between) is None
        # 白 / 银色盘：中段一整片纯白才是满；只有下沿一块白液面（含扫过的反光）、或背景发白，都不能报满
        white_full = dict(plain, bright=0.04, hue_frac=0.07, hue=5, disc=600, compact=0.95, white=0.2, white_mid=0.25,
                          white_outside=0.05)
        white_partial = dict(white_full, white=0.11, white_mid=0.0)
        white_flash_bg = dict(white_full, white_outside=0.4)
        gray_partial = dict(plain, bright=0.04, hue_frac=0.07, hue=5, disc=650, compact=0.96, top=140, bottom=200,
                            white=0.03)
        x_overlay = dict(plain, bright=0.12, hue_frac=0.19, hue=235, disc=1346, compact=0.58, white=0.05, white_mid=0.07)
        pale_full_blue = dict(plain, bright=0.08, hue_frac=0.74, hue=205, disc=5270, compact=0.89)
        assert bh._ult_ready(white_full) is True
        assert bh._ult_ready(white_partial) is None
        assert bh._ult_ready(white_flash_bg) is None
        assert bh._ult_ready(gray_partial) is False
        assert bh._ult_ready(x_overlay) is None       # 「禁用」叠加层 + 蓝光：说不准
        assert bh._ult_ready(pale_full_blue) is None  # 淡蓝色盘满了但没发光：说不准
        # 没见过的风格（大片彩色、不集中在圆盘里）一律不报「未满」
        unseen = dict(plain, bright=0.1, hue_frac=0.4, hue=300, disc=3000, compact=0.6)
        assert bh._ult_ready(unseen) is None


# ---------------------------------------------------------------- 文字和图 ----

class TestDescribe:
    def test_command_line(self):
        hud = bh.BattleHud(in_battle=True, phase="command", selected="skill", skill_points=3, team_size=3,
                           slots=[bh.SlotHud(1, True), bh.SlotHud(2, False), bh.SlotHud(3, True)],
                           rounds_left=2, wave=(1, 2), auto_on=False, speed_on=True)
        assert bh.describe_hud(hud) == ("战斗状态：轮到我方出手｜选中：战技｜技能点 3｜终结技就绪：1号、3号｜剩余轮次 2｜波次 1/2｜"
                                        "自动：关｜倍速：开")

    def test_unknown_fields_say_unknown(self):
        hud = bh.BattleHud(in_battle=True, phase="command", slots=[bh.SlotHud(1), bh.SlotHud(2)])
        assert bh.describe_hud(hud) == ("战斗状态：轮到我方出手｜选中：不确定｜技能点 不确定｜终结技就绪：不确定｜剩余轮次 不确定｜波次 不确定｜"
                                        "自动：不确定｜倍速：不确定")

    def test_phase_unknown_and_ult_pending_and_other(self):
        assert bh.describe_hud(bh.BattleHud(in_battle=True)).startswith("战斗状态：不确定｜")
        pending = bh.describe_hud(bh.BattleHud(in_battle=True, phase="ult_pending", skill_points=2))
        assert pending.startswith("战斗状态：终结技待释放（按空格释放）｜技能点 2｜")
        other = bh.describe_hud(bh.BattleHud(in_battle=True, phase="other"))
        assert "选中" not in other and "技能点" not in other

    def test_no_ready_slot_and_mixed(self):
        none_ready = bh.BattleHud(in_battle=True, phase="other", slots=[bh.SlotHud(1, False), bh.SlotHud(2, False)])
        assert "终结技就绪：无" in bh.describe_hud(none_ready)
        mixed = bh.BattleHud(in_battle=True, phase="other", slots=[bh.SlotHud(1, True), bh.SlotHud(2, None)])
        assert "终结技就绪：1号（其余不确定）" in bh.describe_hud(mixed)

    def test_result_and_non_battle(self):
        assert bh.describe_hud(bh.BattleHud(in_battle=False, phase="result", result_text="挑战结束")) == "战斗状态：结算页（挑战结束）"
        assert bh.describe_hud(bh.BattleHud(in_battle=False)) is None
        assert bh.describe_hud(bh.BattleHud(in_battle=None)) is None

    def test_details_for_command_with_unreadable_fields(self):
        hud = bh.BattleHud(in_battle=True, phase="command", slots=[bh.SlotHud(1, True), bh.SlotHud(2, None)])
        details = bh.format_hud_details(hud)
        assert "1号 可放；2号 不确定" in details and "技能点没读准" in details and "选中的是普攻还是战技没认出来" in details


class TestPanel:
    def test_panel_is_clean_720p_plus_strip(self):
        data = bh.render_hud_panel(load("cmd_basic_sp3_3p"))
        assert data[:3] == b"\xff\xd8\xff"
        import io
        image = Image.open(io.BytesIO(data))
        assert image.size[0] == 1280 and image.size[1] > 720 + 300
        # 上面 1280×720 是整张画面缩小：夹具上除了 HUD 局部都是黑的，随便取一块非 HUD 的区域应是黑的
        assert np.asarray(image.crop((500, 300, 700, 400))).mean() < 10

    def test_panel_accepts_other_sizes(self):
        assert bh.render_hud_panel(Image.new("RGB", (1280, 720))) [:3] == b"\xff\xd8\xff"


# ---------------------------------------------------------------- 真 OCR 端到端 ----

def _real_ocr_or_skip():
    try:
        from module.ocr import ocr as engine
        engine.recognize_multi_lines(np.zeros((64, 64, 3), np.uint8))
    except Exception as e:  # OCR 模型没装 / 平台不支持
        pytest.skip(f"真 OCR 不可用：{e}")
    return engine.recognize_multi_lines


class TestRealOcr:
    @pytest.mark.parametrize("name, selected, sp, wave, rounds", [
        ("cmd_basic_sp3_3p", "basic", 3, (1, 1), 2),
        ("cmd_skill_sp5_4p", "skill", 5, (1, 2), 2),
        ("cmd_basic_sp4_4p", "basic", 4, (1, 2), 2),
    ])
    def test_end_to_end(self, name, selected, sp, wave, rounds):
        ocr = _real_ocr_or_skip()
        hud = bh.read_battle_hud(load(name), ocr=ocr)
        assert (hud.phase, hud.selected, hud.skill_points, hud.wave, hud.rounds_left) == ("command", selected, sp, wave, rounds)

    def test_result_title(self):
        ocr = _real_ocr_or_skip()
        hud = bh.read_battle_hud(load("result_trial"), ocr=ocr)
        assert hud.phase == "result" and hud.result_text == "挑战结束"
