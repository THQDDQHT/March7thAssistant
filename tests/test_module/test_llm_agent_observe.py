# coding:utf-8
"""大模型接管：观察层（元素编号与排序、坐标换算、元素表、标注图、网格）。全部离线，mock 掉 OCR 和 auto。"""
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from module.llm_agent import observe as ob
from module.llm_agent.actions import Action, ActionExecutor


def _box(left, top, right, bottom):
    """OCR 的四点框格式。"""
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _ocr(text, left, top, right, bottom, conf=0.95):
    return [_box(left, top, right, bottom), (text, conf)]


class TestOrdering:
    def test_numbered_from_one_top_to_bottom_left_to_right(self):
        ocr_result = [
            _ocr("下", 100, 500, 200, 530),
            _ocr("右上", 900, 50, 1000, 80),
            _ocr("左上", 100, 50, 200, 80),
            _ocr("中", 500, 300, 600, 330),
        ]
        elements = ob.build_elements(ocr_result)
        assert [e.id for e in elements] == [1, 2, 3, 4]
        assert [e.text for e in elements] == ["左上", "右上", "中", "下"]

    def test_same_row_with_jitter_sorted_by_left(self):
        # 同一行的文字框上沿相差几个像素，不能因此乱序
        ocr_result = [
            _ocr("C", 700, 103, 760, 137),
            _ocr("A", 100, 98, 160, 132),
            _ocr("B", 400, 100, 460, 135),
            _ocr("下一行", 100, 200, 200, 235),
        ]
        elements = ob.build_elements(ocr_result)
        assert [e.text for e in elements] == ["A", "B", "C", "下一行"]

    def test_different_rows_not_merged(self):
        ocr_result = [_ocr("上", 500, 100, 560, 130), _ocr("下", 100, 160, 160, 190)]
        assert [e.text for e in ob.build_elements(ocr_result)] == ["上", "下"]

    def test_low_confidence_and_blank_text_dropped(self):
        ocr_result = [
            _ocr("清楚", 10, 10, 100, 40, conf=0.9),
            _ocr("噪声", 10, 60, 100, 90, conf=0.2),
            _ocr("   ", 10, 110, 100, 140, conf=0.9),
        ]
        assert [e.text for e in ob.build_elements(ocr_result)] == ["清楚"]

    @pytest.mark.parametrize("empty", [False, None, []])
    def test_empty_ocr_result(self, empty):
        assert ob.build_elements(empty) == []

    def test_icons_are_elements_and_sorted_with_text(self):
        ocr_result = [_ocr("标题", 100, 20, 300, 60)]
        icons = [("[关闭图标]", "icon_close", (1700, 30, 1760, 60))]
        elements = ob.build_elements(ocr_result, icons)
        assert [e.text for e in elements] == ["标题", "[关闭图标]"]
        assert elements[1].kind == "icon_close"

    def test_rotated_box_uses_bounding_rect(self):
        box = [[10, 20], [110, 25], [108, 60], [8, 55]]
        element = ob.build_elements([[box, ("斜", 0.9)]])[0]
        assert element.rect == (8, 20, 110, 60)

    def test_center_and_box(self):
        element = ob.Element(1, "x", (100, 200, 300, 240))
        assert element.center == (200, 220)
        assert element.box == [[100, 200], [300, 200], [300, 240], [100, 240]]


class TestTable:
    def test_percent_coordinates(self):
        elements = [ob.Element(1, "关闭", (1810, 60, 1910, 120)), ob.Element(2, "中", (864, 486, 1056, 594))]
        table = ob.format_element_table(elements, (1920, 1080))
        assert table.splitlines() == ["[1] 关闭 @ (97%, 8%)", "[2] 中 @ (50%, 50%)"]

    def test_empty_table(self):
        assert "没有识别到" in ob.format_element_table([], (1920, 1080))

    def test_long_text_truncated(self):
        table = ob.format_element_table([ob.Element(1, "字" * 100, (0, 0, 10, 10))], (1920, 1080))
        assert len(table) < 100


class TestGrid:
    def test_cell_rect(self):
        # 1920x1080 → 每格 120x120；H5 = 第 8 列第 5 行
        assert ob.cell_rect("H5", (1920, 1080)) == (840, 480, 960, 600)
        assert ob.cell_rect("A1", (1920, 1080)) == (0, 0, 120, 120)
        assert ob.cell_rect("p9", (1920, 1080)) == (1800, 960, 1920, 1080)

    @pytest.mark.parametrize("cell", ["Q1", "A0", "A10", "", "11", None])
    def test_invalid_cell(self, cell):
        assert ob.cell_rect(cell, (1920, 1080)) is None

    def test_cell_name(self):
        assert ob.cell_name(7, 4) == "H5"
        assert ob.cell_name(0, 0) == "A1"
        assert ob.cell_name(15, 8) == "P9"


class TestAnnotate:
    def test_annotated_is_1280x720_jpeg(self):
        image = Image.new("RGB", (1920, 1080), (30, 30, 30))
        elements = [ob.Element(1, "a", (100, 100, 300, 140)), ob.Element(2, "[关闭图标]", (1700, 30, 1860, 60), "icon_close")]
        data = ob.annotate(image, elements)
        result = Image.open(io.BytesIO(data))
        assert result.format == "JPEG"
        assert result.size == (1280, 720)

    def test_grid_image_same_size_and_differs(self):
        image = Image.new("RGB", (1920, 1080), (30, 30, 30))
        plain = ob.annotate(image, [])
        grid = ob.annotate(image, [], grid=True)
        assert Image.open(io.BytesIO(grid)).size == (1280, 720)
        assert plain != grid

    def test_box_is_drawn_at_scaled_position(self):
        image = Image.new("RGB", (1920, 1080), (0, 0, 0))
        canvas = ob.draw_annotations(image, [ob.Element(1, "a", (600, 300, 900, 450), "icon_close")])
        # 缩放系数 2/3：框的左边线在 x=400、上边线在 y=200（图标框是黄色）
        assert canvas.getpixel((400, 250))[0] > 200
        assert canvas.getpixel((600, 200))[0] > 200
        assert canvas.getpixel((10, 700)) == (0, 0, 0)

    def test_non_16_9_image_keeps_aspect(self):
        image = Image.new("RGB", (1000, 1000), (0, 0, 0))
        assert Image.open(io.BytesIO(ob.annotate(image, []))).size == (720, 720)


class TestObserveImage:
    def test_injected_ocr_and_icons(self):
        image = Image.new("RGB", (1920, 1080))
        observation = ob.observe_image(
            image,
            ocr_fn=lambda img: [_ocr("确认", 800, 700, 900, 740)],
            icon_fn=lambda img: [("[关闭图标]", "icon_close", (10, 10, 60, 40))],
        )
        assert [e.text for e in observation.elements] == ["[关闭图标]", "确认"]
        assert observation.get(2).text == "确认"
        assert observation.get(99) is None
        assert observation.all_texts() == ["[关闭图标]", "确认"]
        assert observation.grid is False

    def test_live_observe_uses_take_screenshot_and_records_pos_scale(self, monkeypatch):
        image = Image.new("RGB", (1920, 1080))
        auto = SimpleNamespace(take_screenshot=lambda: (image, (10, 20, 1920, 1080), 1.0))
        monkeypatch.setattr(ob, "run_ocr", lambda img: [_ocr("关闭", 1, 1, 50, 30)])
        monkeypatch.setattr(ob, "find_icons", lambda img: [])
        observation = ob.observe(grid=True, auto=auto)
        assert observation.grid is True
        assert observation.screenshot_pos == (10, 20, 1920, 1080)
        assert observation.scale == 1.0
        assert [e.text for e in observation.elements] == ["关闭"]


class TestFindIcons:
    def test_template_pasted_on_image_is_found(self):
        import cv2
        import numpy as np
        template = cv2.imread(ob.os.path.join(ob.ROOT, "assets/images/zh_CN/base/confirm.png"), cv2.IMREAD_COLOR)
        canvas = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        h, w = template.shape[:2]
        # 模板的透明背景读成灰度后接近黑色，直接贴上去即可模拟游戏里的按钮
        canvas[600:600 + h, 900:900 + w] = template
        hits = ob.find_icons(Image.fromarray(canvas))
        confirm = [hit for hit in hits if hit[1] == "icon_confirm"]
        assert confirm, hits
        left, top, right, bottom = confirm[0][2]
        assert abs(left - 900) <= 2 and abs(top - 600) <= 2
        assert (right - left, bottom - top) == (w, h)

    def test_blank_image_has_no_hits(self):
        assert ob.find_icons(Image.new("RGB", (1920, 1080), (40, 40, 40))) == []


class _FakeAuto:
    """用 Automation 的真实换算函数搭一个替身，验证「先除以 scale，再加 screenshot_pos」。"""

    def __init__(self, scale, pos):
        from module.automation.automation import Automation
        self._cls = Automation
        self.screenshot_scale_factor = scale
        self.screenshot_pos = pos
        self.clicks = []
        self.keys = []

    def calculate_text_position(self, box, relative):
        return self._cls.calculate_text_position(self, box, relative)

    def click_element_with_pos(self, coordinates):
        self.clicks.append(self._cls.calculate_click_position(self, coordinates))
        return True

    def press_key(self, key):
        self.keys.append(key)


class TestActionCoordinates:
    def _observation(self, elements, size=(2400, 1350)):
        return ob.Observation(image=Image.new("RGB", size), elements=elements)

    def test_click_converts_screenshot_to_absolute(self):
        auto = _FakeAuto(scale=1.25, pos=(100, 50, 2400, 1350))
        element = ob.Element(1, "关闭", (200, 100, 300, 140))
        ActionExecutor(auto=auto).execute(Action("click", {"id": 1}), self._observation([element]))
        # 左上 (int(200/1.25)+100, int(100/1.25)+50) = (260, 130)，右下 (340, 162)，中心 (300, 146)
        assert auto.clicks == [(300, 146)]

    def test_click_scale_one_only_offsets(self):
        auto = _FakeAuto(scale=1.0, pos=(0, 0, 1920, 1080))
        element = ob.Element(1, "x", (100, 200, 300, 240))
        ActionExecutor(auto=auto).execute(Action("click", {"id": 1}), self._observation([element], (1920, 1080)))
        assert auto.clicks == [(200, 220)]

    def test_click_grid_clicks_cell_center(self):
        auto = _FakeAuto(scale=1.0, pos=(10, 20, 1920, 1080))
        ActionExecutor(auto=auto).execute(Action("click_grid", {"cell": "H5"}), self._observation([], (1920, 1080)))
        # H5 = (840,480)-(960,600)，中心 (900,540)，加上窗口偏移
        assert auto.clicks == [(910, 560)]

    def test_press_key_and_wait(self):
        auto = _FakeAuto(1.0, (0, 0, 1920, 1080))
        slept = []
        executor = ActionExecutor(auto=auto, sleep=slept.append)
        executor.execute(Action("press_key", {"key": "esc"}), self._observation([]))
        executor.execute(Action("wait", {"seconds": 2.5}), self._observation([]))
        assert auto.keys == ["esc"]
        assert slept == [2.5]

    def test_click_unknown_id_does_nothing(self):
        auto = _FakeAuto(1.0, (0, 0, 1920, 1080))
        result = ActionExecutor(auto=auto).execute(Action("click", {"id": 9}), self._observation([]))
        assert result.ok is False
        assert auto.clicks == []
