"""观察：截图 → OCR → 模板 → 元素编号 → 标注图。

元素坐标一律是「截图坐标」（截图左上角为原点，单位像素）；换算成屏幕绝对坐标由 actions.py
复用 auto.calculate_text_position 完成（先除以 scale，再加上 screenshot_pos）。
本模块的核心函数 observe_image 只依赖一张 PIL 图片，不需要游戏窗口，cli 的离线命令直接用它。
"""
import io
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from module.llm_agent.battle_hud import BattleHud

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# OCR 置信度低于该值的文字当作噪声丢弃
OCR_MIN_CONF = 0.6

# 模板匹配的图标：(模板路径, 元素文字, 种类, 匹配阈值)
ICON_TEMPLATES = (
    ("assets/images/zh_CN/base/click_close.png", "[关闭图标]", "icon_close", 0.85),
    ("assets/images/zh_CN/base/confirm.png", "[确认图标]", "icon_confirm", 0.85),
)

# 给模型看的标注图：最大尺寸和 JPEG 质量
ANNOTATED_SIZE = (1280, 720)
# 50 时文字和编号仍清晰；Codex 每轮都会把历史里的截图整体重发，图太大会撞上 DeepSeek 48 MiB 的请求上限（413）
JPEG_QUALITY = 50

# 网格：16 列（A-P）× 9 行（1-9）
GRID_COLS = 16
GRID_ROWS = 9
GRID_CELL_RE = re.compile(r"^([A-P])([1-9])$")

# 元素表里单个元素文字的最长长度
TABLE_TEXT_MAX = 40

_BOX_COLORS = (
    (220, 40, 40), (0, 150, 60), (40, 90, 220),
    (230, 120, 0), (180, 30, 160), (0, 140, 160),
)
_ICON_COLOR = (255, 215, 0)


@dataclass
class Element:
    """画面上的一个可点击元素（文字框或图标）。"""
    id: int
    text: str
    rect: Tuple[int, int, int, int]  # (左, 上, 右, 下)，截图坐标
    kind: str = "text"  # text / icon_close / icon_confirm
    conf: float = 1.0

    @property
    def center(self) -> Tuple[int, int]:
        left, top, right, bottom = self.rect
        return (left + right) // 2, (top + bottom) // 2

    @property
    def box(self) -> List[List[int]]:
        """四点框（左上、右上、右下、左下），和 OCR 结果里的 box 格式一致。"""
        left, top, right, bottom = self.rect
        return [[left, top], [right, top], [right, bottom], [left, bottom]]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "conf": round(float(self.conf), 3),
            "rect": list(self.rect),
            "center": list(self.center),
        }


@dataclass
class Observation:
    """一次观察的结果。"""
    image: Image.Image
    elements: List[Element] = field(default_factory=list)
    grid: bool = False  # 标注图是否叠加网格
    screenshot_pos: Optional[tuple] = None
    scale: Optional[float] = None
    hud: Optional["BattleHud"] = None  # 战斗状态（mcp_tools 观察时顺带识别）；在战斗中时标注图不画编号框

    @property
    def in_battle(self) -> bool:
        return getattr(self.hud, "in_battle", None) is True

    @property
    def size(self) -> Tuple[int, int]:
        return self.image.size

    def get(self, element_id) -> Optional[Element]:
        for element in self.elements:
            if element.id == element_id:
                return element
        return None

    def all_texts(self) -> List[str]:
        return [element.text for element in self.elements]

    def table_text(self) -> str:
        return format_element_table(self.elements, self.size)

    def annotated_jpeg(self) -> bytes:
        # 战斗画面里元素表基本是噪声，编号框还会压住 Q/E 按钮，所以战斗中只给干净的缩放图
        return annotate(self.image, [] if self.in_battle else self.elements, grid=self.grid)


# ---------------------------------------------------------------- 元素整理 ----

def _rect_from_box(box) -> Tuple[int, int, int, int]:
    """OCR 的四点框 → 外接矩形。"""
    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    return int(min(xs)), int(min(ys)), int(round(max(xs))), int(round(max(ys)))


def order_elements(items) -> List[Element]:
    """按「从上到下、从左到右」排序并从 1 开始编号。

    纯按左上角坐标排序时，同一行文字因为框高度不同、上沿会差几个像素，顺序会乱；
    这里先按中心点纵坐标分行（与本行第一个元素的纵向距离小于较矮者高度的一半算同一行），
    行内再按左边界排序。

    :param items: [(text, rect, kind, conf), ...]
    """
    entries = []
    for text, rect, kind, conf in items:
        left, top, right, bottom = rect
        entries.append({
            "text": text, "rect": rect, "kind": kind, "conf": conf,
            "cy": (top + bottom) / 2, "h": max(1, bottom - top), "left": left,
        })
    entries.sort(key=lambda e: (e["cy"], e["left"]))

    rows: List[list] = []
    for entry in entries:
        if rows:
            anchor = rows[-1][0]
            if entry["cy"] - anchor["cy"] <= 0.5 * min(anchor["h"], entry["h"]):
                rows[-1].append(entry)
                continue
        rows.append([entry])

    elements = []
    for row in rows:
        for entry in sorted(row, key=lambda e: e["left"]):
            elements.append(Element(len(elements) + 1, entry["text"], entry["rect"], entry["kind"], entry["conf"]))
    return elements


def build_elements(ocr_result, icon_hits=(), min_conf: float = OCR_MIN_CONF) -> List[Element]:
    """OCR 结果和图标命中 → 编号后的元素列表。

    :param ocr_result: [[box(4 点), (text, conf)], ...]（auto.ocr_result 的格式），可以是 False/None。
    :param icon_hits: [(text, kind, rect), ...]
    """
    items = []
    for entry in ocr_result or []:
        try:
            box, (text, conf) = entry
        except (TypeError, ValueError):
            continue
        text = (text or "").strip()
        if not text or conf is None or float(conf) < min_conf:
            continue
        items.append((text, _rect_from_box(box), "text", float(conf)))
    for text, kind, rect in icon_hits:
        items.append((text, tuple(int(v) for v in rect), kind, 1.0))
    return order_elements(items)


# ------------------------------------------------------------- 识别（OCR/模板） ----

def run_ocr(image: Image.Image):
    """对整张图跑 OCR，返回 [[box, (text, conf)], ...]；失败返回空列表。"""
    import numpy as np
    from module.ocr import ocr
    result = ocr.recognize_multi_lines(np.array(image))
    return result or []


def find_icons(image: Image.Image, templates=ICON_TEMPLATES) -> List[Tuple[str, str, Tuple[int, int, int, int]]]:
    """多目标模板匹配，返回 [(文字, 种类, 矩形), ...]（截图坐标）。

    匹配方式与 auto.find_image_with_multiple_targets 一致（灰度 + TM_CCOEFF_NORMED），
    只是不依赖 auto 的当前截图，所以离线也能用。
    """
    import cv2
    import numpy as np
    from utils.image_utils import ImageUtils

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    hits = []
    for path, text, kind, threshold in templates:
        full_path = path if os.path.isabs(path) else os.path.join(ROOT, path)
        template = ImageUtils.read_image(full_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue
        height, width = template.shape[:2]
        if gray.shape[0] < height or gray.shape[1] < width:
            continue
        for x, y in ImageUtils.scale_and_match_template_with_multiple_targets(gray, template, threshold, None):
            hits.append((text, kind, (int(x), int(y), int(x) + width, int(y) + height)))
    return hits


def observe_image(image: Image.Image, grid: bool = False,
                  ocr_fn: Callable = None, icon_fn: Callable = None) -> Observation:
    """对一张图片做观察（不需要游戏窗口）。ocr_fn / icon_fn 可注入，测试时用来替身。"""
    image = image.convert("RGB")
    ocr_result = (ocr_fn or run_ocr)(image)
    icon_hits = (icon_fn or find_icons)(image)
    return Observation(image=image, elements=build_elements(ocr_result, icon_hits), grid=grid)


def observe(grid: bool = False, auto=None) -> Observation:
    """实机观察：用 auto.take_screenshot() 截图，再走 observe_image。"""
    if auto is None:
        from module.automation import auto
    image, screenshot_pos, scale = auto.take_screenshot()
    observation = observe_image(image, grid=grid)
    observation.screenshot_pos = screenshot_pos
    observation.scale = scale
    return observation


# ---------------------------------------------------------------- 文字表 ----

def format_element_table(elements, size) -> str:
    """给模型看的元素表：`[编号] 文字 @ (x%, y%)`，百分比是元素中心相对整张截图的位置。"""
    if not elements:
        return "（没有识别到任何文字或图标）"
    width, height = size
    lines = []
    for element in elements:
        cx, cy = element.center
        text = element.text if len(element.text) <= TABLE_TEXT_MAX else element.text[:TABLE_TEXT_MAX] + "…"
        lines.append(f"[{element.id}] {text} @ ({round(cx * 100 / width)}%, {round(cy * 100 / height)}%)")
    return "\n".join(lines)


# ---------------------------------------------------------------- 网格 ----

def cell_name(col: int, row: int) -> str:
    """列号、行号（都从 0 开始）→ 格子名，例如 (7, 4) → H5。"""
    return f"{chr(ord('A') + col)}{row + 1}"


def cell_rect(cell: str, size) -> Optional[Tuple[int, int, int, int]]:
    """格子名 → 截图坐标矩形；格子名不合法返回 None。"""
    match = GRID_CELL_RE.match((cell or "").strip().upper())
    if not match:
        return None
    col = ord(match.group(1)) - ord("A")
    row = int(match.group(2)) - 1
    width, height = size
    return (round(col * width / GRID_COLS), round(row * height / GRID_ROWS),
            round((col + 1) * width / GRID_COLS), round((row + 1) * height / GRID_ROWS))


# ---------------------------------------------------------------- 标注图 ----

def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _scaled_size(size) -> Tuple[int, int]:
    """缩放到不超过 ANNOTATED_SIZE，保持宽高比（1920×1080 正好是 1280×720）。"""
    width, height = size
    ratio = min(ANNOTATED_SIZE[0] / width, ANNOTATED_SIZE[1] / height)
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def draw_annotations(image: Image.Image, elements, grid: bool = False) -> Image.Image:
    """缩放后画上元素框和编号（grid=True 时改画 16×9 网格）。返回 RGB 图。"""
    target = _scaled_size(image.size)
    canvas = image.convert("RGB").resize(target, Image.Resampling.LANCZOS)
    sx = target[0] / image.size[0]
    sy = target[1] / image.size[1]

    if grid:
        return _draw_grid(canvas)

    draw = ImageDraw.Draw(canvas)
    font = _font(15)
    placed = []
    for index, element in enumerate(elements):
        left, top, right, bottom = element.rect
        l, t, r, b = left * sx, top * sy, right * sx, bottom * sy
        is_icon = element.kind.startswith("icon")
        color = _ICON_COLOR if is_icon else _BOX_COLORS[index % len(_BOX_COLORS)]
        draw.rectangle([l, t, r, b], outline=color, width=2)

        label = str(element.id)
        x0, y0, x1, y1 = draw.textbbox((0, 0), label, font=font)
        w, h = x1 - x0 + 6, y1 - y0 + 4
        # 编号标签的候选位置：框上方、框内左上、框下方、框上方靠右；选第一个不压住其它标签的
        candidates = [(l, t - h), (l, t), (l, b), (r - w, t - h), (r - w, b)]
        spot = None
        for cx, cy in candidates:
            cx = min(max(0, cx), target[0] - w)
            cy = min(max(0, cy), target[1] - h)
            rect = (cx, cy, cx + w, cy + h)
            if not any(_overlaps(rect, other) for other in placed):
                spot = rect
                break
        if spot is None:
            cx = min(max(0, candidates[0][0]), target[0] - w)
            cy = min(max(0, candidates[0][1]), target[1] - h)
            spot = (cx, cy, cx + w, cy + h)
        placed.append(spot)
        draw.rectangle(spot, fill=color)
        draw.text((spot[0] + 3 - x0, spot[1] + 2 - y0), label, fill=(0, 0, 0) if is_icon else (255, 255, 255), font=font)
    return canvas


def _draw_grid(canvas: Image.Image) -> Image.Image:
    """在缩放后的图上叠加 16×9 网格，每个格子左上角标格子名（列 A-P + 行 1-9）。"""
    width, height = canvas.size
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(13)
    for col in range(GRID_COLS + 1):
        x = round(col * width / GRID_COLS)
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255, 140), width=1)
    for row in range(GRID_ROWS + 1):
        y = round(row * height / GRID_ROWS)
        draw.line([(0, y), (width, y)], fill=(255, 255, 255, 140), width=1)
    for col in range(GRID_COLS):
        for row in range(GRID_ROWS):
            x = round(col * width / GRID_COLS)
            y = round(row * height / GRID_ROWS)
            draw.text((x + 3, y + 1), cell_name(col, row), fill=(255, 235, 59, 255), font=font,
                      stroke_width=1, stroke_fill=(0, 0, 0, 255))
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def annotate(image: Image.Image, elements, grid: bool = False) -> bytes:
    """生成给模型看的标注图（JPEG 字节）。"""
    buffer = io.BytesIO()
    draw_annotations(image, elements, grid=grid).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()
