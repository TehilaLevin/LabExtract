"""Yellow-highlight crops: zoomed views of highlighted regions, sent with the full page to the extractor.

Yellow highlighter is detected with an HSV mask plus morphology and a minimum area. Each region is cut
from the 300 DPI page (original pixels, deskew geometry only) with padding and upscaled, so small and
handwritten text gets more pixels than in the downscaled full page. When the region holds blue pen ink, a
cleaned "blue ink only" version is added: a blue hue/saturation mask removes the yellow highlight, black
print and barcodes, and small connected components (specks) are dropped.

Additive by design: a page without yellow regions yields no crops and is extracted exactly as before.
Crops live in memory only. HIGHLIGHT_DEBUG=true (dev only) writes them to data/temp/highlight_debug/,
which is git-ignored and emptied at the start of every run.
"""
import logging
import shutil
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src import config

logger = logging.getLogger(__name__)

# Detection runs on a copy downscaled by this factor (speed); boxes are mapped back to full resolution.
_WORK_SCALE = 0.5
# OpenCV HSV (H 0-179): light highlighter colours. Yellow, and also the green / cyan markers seen in the
# samples (H ~85-90 on the ID fields). Highlighter is bright (V >= 170); blue pen ink is darker, so it
# does not match even though its hue overlaps cyan.
YELLOW_LOW = (18, 50, 170)
YELLOW_HIGH = (100, 255, 255)
# Blue pen ink, RGB rule (see blue_ink_mask).
BLUE_MAX_RED = 170
BLUE_MIN_B_MINUS_R = 35
BLUE_MAX_MEAN = 170
MIN_REGION_AREA = 0.0004     # fraction of the page: smaller yellow blobs are specks / stray marks
MIN_REGION_HEIGHT_PX = 20    # at 300 DPI (~1.7 mm)
MAX_REGION_AREA = 0.15       # a huge yellow area is not a highlighted field (e.g. a coloured page)
MERGE_GAP_PX = 25            # highlight strokes this close (300 DPI) belong to one region
PAD_X_PX = 140               # horizontal padding, 300 DPI pixels (~1.2 cm): highlighter often covers only
PAD_Y_PX = 45                # part of a handwritten value, and the label sits beside it
UPSCALE = 2.0                # 2-3x per the design; capped by MAX_CROP_SIDE
MAX_CROP_SIDE = 1800         # stays under the provider's per-image limits even with many images
CROP_IMAGE_MAX_SIDE = MAX_CROP_SIDE  # passed to image_part when the crops are encoded
MAX_CROPS_PER_FORM = 10      # colour crops + the blue-only ones + pages: kept within 20 images per request
MIN_BLUE_FRACTION = 0.004    # below this share of blue pixels the blue-only version would be empty
MIN_INK_COMPONENT_PX = 30    # connected components smaller than this (300 DPI) are noise: pen strokes are
                             # long, while colour fringes on barcodes and print are small specks


@dataclass
class HighlightCrop:
    page: int                         # 1-based within the form
    box: tuple                        # (left, top, right, bottom) in page pixels, padding included
    position: str                     # e.g. "upper right" - helps the model match crop and page
    color: Image.Image                # upscaled colour crop
    blue_only: Optional[Image.Image]  # upscaled blue-ink-only crop, None when there is little blue ink


def _yellow_mask(image: Image.Image) -> np.ndarray:
    small = image.convert("RGB").resize((max(1, int(image.width * _WORK_SCALE)),
                                         max(1, int(image.height * _WORK_SCALE))), Image.BILINEAR)
    hsv = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(YELLOW_LOW), np.array(YELLOW_HIGH))
    gap = max(3, int(MERGE_GAP_PX * _WORK_SCALE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))           # specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((gap, gap * 2), np.uint8))  # join strokes
    return mask


def find_yellow_regions(image: Image.Image) -> list:
    """(left, top, right, bottom) boxes of yellow-highlighted regions at full resolution, reading order."""
    mask = _yellow_mask(image)
    page_area = mask.shape[0] * mask.shape[1]
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_REGION_AREA * page_area or area > MAX_REGION_AREA * page_area:
            continue
        if h < MIN_REGION_HEIGHT_PX * _WORK_SCALE:
            continue
        s = 1 / _WORK_SCALE
        boxes.append((int(x * s), int(y * s), int((x + w) * s), int((y + h) * s)))
    # Top to bottom, then right to left (Hebrew reading order).
    return sorted(boxes, key=lambda b: (b[1] // 150, -b[2]))


def _position(box: tuple, size: tuple) -> str:
    cx, cy = (box[0] + box[2]) / 2 / size[0], (box[1] + box[3]) / 2 / size[1]
    vertical = "top" if cy < 1 / 3 else ("middle" if cy < 2 / 3 else "bottom")
    horizontal = "left" if cx < 1 / 3 else ("center" if cx < 2 / 3 else "right")
    return f"{vertical} {horizontal}"


def _upscale(img: Image.Image) -> Image.Image:
    factor = min(UPSCALE, MAX_CROP_SIDE / max(img.size))
    if factor <= 1.0:
        return img
    return img.resize((round(img.width * factor), round(img.height * factor)), Image.LANCZOS)


def blue_ink_mask(crop: Image.Image) -> np.ndarray:
    """uint8 mask (255 = blue pen ink). An HSV hue band alone misses ink drawn over yellow highlighter,
    which turns green-cyan and light, so the rule is in RGB: red clearly below blue (black print and grey
    barcodes have no such tint), red low (yellow highlighter has high red), and the pixel darker than
    highlighter (a light cyan marker is bright)."""
    rgb = np.asarray(crop.convert("RGB")).astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ink = (r < BLUE_MAX_RED) & (b - r > BLUE_MIN_B_MINUS_R) & (b > 70) & ((r + g + b) // 3 < BLUE_MAX_MEAN)
    return ink.astype(np.uint8) * 255


def blue_ink_only(crop: Image.Image) -> Optional[Image.Image]:
    """Black-on-white image of the blue pen strokes in crop, or None when there is too little blue ink."""
    mask = blue_ink_mask(crop)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_INK_COMPONENT_PX
    clean = keep[labels]
    if clean.mean() < MIN_BLUE_FRACTION:
        return None
    return Image.fromarray(np.where(clean, 0, 255).astype(np.uint8), mode="L")


def highlight_crops(pages: list) -> list:
    """HighlightCrops for a form's 300 DPI page images, in reading order. With more than MAX_CROPS_PER_FORM
    regions, those with blue pen ink (handwriting - where the zoom helps most) are kept first, then the
    largest. Never raises: a page that cannot be analysed simply yields no crops."""
    found = []
    for page_no, page in enumerate(pages, start=1):
        try:
            boxes = find_yellow_regions(page)
            rgb = page.convert("RGB")
            for x0, y0, x1, y1 in boxes:
                box = (max(0, x0 - PAD_X_PX), max(0, y0 - PAD_Y_PX),
                       min(page.width, x1 + PAD_X_PX), min(page.height, y1 + PAD_Y_PX))
                region = rgb.crop(box)
                found.append((page_no, box, _position(box, page.size), region, blue_ink_only(region)))
        except Exception as exc:  # analysis must never break extraction
            logger.warning("Highlight detection failed on page %d: %s", page_no, type(exc).__name__)
    if len(found) > MAX_CROPS_PER_FORM:
        ranked = sorted(found, key=lambda f: (f[4] is None, -(f[1][2] - f[1][0]) * (f[1][3] - f[1][1])))
        keep = {id(f) for f in ranked[:MAX_CROPS_PER_FORM]}
        found = [f for f in found if id(f) in keep]
    return [HighlightCrop(page_no, box, position, _upscale(region), _upscale(blue) if blue is not None else None)
            for page_no, box, position, region, blue in found]


def crop_contents(crops: list, n_pages: int, image_part, max_side: int) -> list:
    """Labelled content items (text + images) for the extraction request; [] when there are no crops."""
    if not crops:
        return []
    items = [
        f"ZOOMED CROPS: below are {len(crops)} crop(s) of yellow-highlighted regions of the page(s) above, cut "
        f"from the 300 DPI scan and enlarged about {UPSCALE:g}x. They show the same ink as the page: use them to "
        "read small or handwritten text, and use the full page to decide which field each crop belongs to (the "
        "highlighting itself does not tell you the field). A 'blue ink only' version keeps only blue pen strokes "
        "(highlighter, black print and barcodes removed) - printed text is NOT in it, read print from the colour "
        "crop. If the text is still not legible, return null. Do not guess."
    ]
    for i, c in enumerate(crops, start=1):
        where = f"page {c.page}, {c.position} area" if n_pages > 1 else f"{c.position} area of the page"
        items += [f"Crop {i} of {len(crops)} ({where}) - colour:", image_part(c.color, max_side)]
        if c.blue_only is not None:
            items += [f"Crop {i} of {len(crops)} - blue ink only:", image_part(c.blue_only, max_side)]
    return items


def save_debug_crops(crops: list, form_label: str) -> None:
    """DEV ONLY (HIGHLIGHT_DEBUG): write the crops as PNG to data/temp/highlight_debug/ for inspection."""
    if not config.HIGHLIGHT_DEBUG or not crops:
        return
    folder = config.HIGHLIGHT_DEBUG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() else "_" for ch in form_label)[:40]
    for i, c in enumerate(crops, start=1):
        c.color.save(folder / f"{safe}_crop{i}_color.png")
        if c.blue_only is not None:
            c.blue_only.save(folder / f"{safe}_crop{i}_blue.png")


def clear_debug_crops() -> int:
    """Delete data/temp/highlight_debug/ (called at the start of every run). Returns files removed."""
    folder = config.HIGHLIGHT_DEBUG_DIR
    if not folder.exists():
        return 0
    count = sum(1 for p in folder.rglob("*") if p.is_file())
    shutil.rmtree(folder, ignore_errors=True)
    return count
