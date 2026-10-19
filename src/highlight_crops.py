"""Highlight crops: zoomed views of highlighted regions, sent with the full page to the extractor.

Highlighter (yellow, green, cyan) is detected with an HSV mask plus morphology and a minimum area. Regions
close to each other are merged, so a highlighted label and its highlighted value end up in one crop. Each
region is cut from the 300 DPI page (original pixels, deskew geometry only) with padding and upscaled, so
small and handwritten text gets more pixels than in the downscaled full page. A cleaned "ink only" version
is added: highlighter colours removed, any dark ink kept (blue pen, black pen, print), specks dropped. When
almost no ink is left the cleaned version is not sent.

Highlighting is only a hint of where to zoom: in the samples it also covers decoy values (visit, intake and
form dates), so the prompt says it never tells which field a value belongs to.

Additive by design: a page without highlighting yields no crops and is extracted exactly as before.
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
# OpenCV HSV (H 0-179): light highlighter colours - yellow, and the green / cyan markers seen in the
# samples (H ~85-90 on the ID fields). Highlighter is bright (V >= 170); pen ink is darker.
HIGHLIGHT_LOW = (18, 50, 170)
HIGHLIGHT_HIGH = (100, 255, 255)
MIN_REGION_AREA = 0.0004     # fraction of the page: smaller blobs are specks / stray marks
MIN_REGION_HEIGHT_PX = 20    # at 300 DPI (~1.7 mm)
MAX_REGION_AREA = 0.15       # a huge highlighted area is not a field (e.g. a coloured page)
STROKE_GAP_PX = 25           # highlighter strokes this close (300 DPI) are one region
# Regions this close are merged into one crop (label + value, stamp heading + date): horizontal gap up to
# ~1.3 cm on the same line, vertical gap up to ~0.6 cm. A merge never makes a crop larger than the limits.
MERGE_GAP_X_PX = 150
MERGE_GAP_Y_PX = 70
MERGED_MAX_WIDTH = 0.6       # fraction of the page width
MERGED_MAX_HEIGHT = 0.2      # fraction of the page height
PAD_X_PX = 140               # horizontal padding, 300 DPI pixels (~1.2 cm): highlighter often covers only
PAD_Y_PX = 45                # part of a handwritten value, and the label sits beside it
UPSCALE = 2.0                # 2-3x per the design; capped by MAX_CROP_SIDE
MAX_CROP_SIDE = 1800         # stays under the per-image limits even with many images
CROP_IMAGE_MAX_SIDE = MAX_CROP_SIDE  # passed to image_part when the crops are encoded
MAX_CROPS_PER_FORM = 9       # page + 9 colour + 9 ink-only = 19 images, within 20 per request
# Ink: dark pixels (black pen, print, dark blue) or blue-tinted pixels (blue pen, also where it crosses
# yellow highlighter and turns green-cyan). Highlighter is bright, so it is never ink.
INK_MAX_LUMA = 160           # 0-255: grey stamp ink still counts; highlighter (luma > 200) never does
BLUE_MAX_RED = 170
BLUE_MIN_B_MINUS_R = 35
BLUE_MAX_MEAN = 170
MIN_INK_COMPONENT_PX = 30    # connected components smaller than this (300 DPI) are specks
MIN_INK_FRACTION = 0.003     # below this share of ink pixels the cleaned version is not sent


@dataclass
class HighlightCrop:
    page: int                         # 1-based within the form
    box: tuple                        # (left, top, right, bottom) in page pixels, padding included
    position: str                     # e.g. "top right" - helps the model match crop and page
    color: Image.Image                # upscaled colour crop
    ink_only: Optional[Image.Image]   # upscaled cleaned crop (black ink on white), None when almost no ink
    has_blue_ink: bool = False        # blue pen strokes present (handwriting) - preferred under the cap


def _highlight_mask(image: Image.Image) -> np.ndarray:
    small = image.convert("RGB").resize((max(1, int(image.width * _WORK_SCALE)),
                                         max(1, int(image.height * _WORK_SCALE))), Image.BILINEAR)
    hsv = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(HIGHLIGHT_LOW), np.array(HIGHLIGHT_HIGH))
    gap = max(3, int(STROKE_GAP_PX * _WORK_SCALE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))           # specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((gap, gap * 2), np.uint8))  # join strokes
    return mask


def find_highlight_regions(image: Image.Image) -> list:
    """(left, top, right, bottom) boxes of highlighted regions at full resolution, before merging."""
    mask = _highlight_mask(image)
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
    return _reading_order(boxes)


def _reading_order(boxes: list) -> list:
    """Top to bottom, then right to left (Hebrew reading order)."""
    return sorted(boxes, key=lambda b: (b[1] // 150, -b[2]))


def _close(a: tuple, b: tuple) -> bool:
    gap_x = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    gap_y = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return gap_x <= MERGE_GAP_X_PX and gap_y <= MERGE_GAP_Y_PX


def merge_regions(boxes: list, page_size: tuple) -> list:
    """Merge regions that are close together (see MERGE_GAP_*) unless the result would exceed the size
    limits. Repeats until nothing merges; returns boxes in reading order."""
    w, h = page_size
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                union = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                if (_close(a, b) and union[2] - union[0] <= MERGED_MAX_WIDTH * w
                        and union[3] - union[1] <= MERGED_MAX_HEIGHT * h):
                    boxes[i] = union
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return _reading_order(boxes)


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


def _channels(crop: Image.Image) -> tuple:
    rgb = np.asarray(crop.convert("RGB")).astype(np.int32)  # int32: the luma sums overflow int16
    return rgb[..., 0], rgb[..., 1], rgb[..., 2]


def blue_ink_mask(crop: Image.Image) -> np.ndarray:
    """Boolean mask of blue pen ink. In RGB, because blue ink over yellow highlighter turns green-cyan and
    light: red clearly below blue (black print and grey barcodes have no such tint), red low (yellow
    highlighter has high red), and darker than highlighter (a light cyan marker is bright)."""
    r, g, b = _channels(crop)
    return (r < BLUE_MAX_RED) & (b - r > BLUE_MIN_B_MINUS_R) & (b > 70) & ((r + g + b) // 3 < BLUE_MAX_MEAN)


def ink_mask(crop: Image.Image) -> np.ndarray:
    """Boolean mask of dark ink of any colour (blue or black pen, print). Highlighter is bright: never ink."""
    r, g, b = _channels(crop)
    luma = (299 * r + 587 * g + 114 * b) // 1000
    return (luma < INK_MAX_LUMA) | blue_ink_mask(crop)


def _drop_specks(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_INK_COMPONENT_PX
    return keep[labels]


def ink_only(crop: Image.Image) -> Optional[Image.Image]:
    """Black-on-white image of the ink in crop (highlighter removed, specks dropped), or None when almost no
    ink is left."""
    clean = _drop_specks(ink_mask(crop))
    if clean.mean() < MIN_INK_FRACTION:
        return None
    return Image.fromarray(np.where(clean, 0, 255).astype(np.uint8), mode="L")


def highlight_crops(pages: list) -> list:
    """HighlightCrops for a form's 300 DPI page images, in reading order. Close regions are merged first.
    With more than MAX_CROPS_PER_FORM regions, those with blue pen ink (handwriting - where the zoom helps
    most) are kept first, then the largest. Never raises: a page that cannot be analysed yields no crops."""
    found = []
    for page_no, page in enumerate(pages, start=1):
        try:
            rgb = page.convert("RGB")
            for x0, y0, x1, y1 in merge_regions(find_highlight_regions(page), page.size):
                box = (max(0, x0 - PAD_X_PX), max(0, y0 - PAD_Y_PX),
                       min(page.width, x1 + PAD_X_PX), min(page.height, y1 + PAD_Y_PX))
                region = rgb.crop(box)
                blue = bool(_drop_specks(blue_ink_mask(region)).mean() >= MIN_INK_FRACTION)
                found.append((page_no, box, _position(box, page.size), region, ink_only(region), blue))
        except Exception as exc:  # analysis must never break extraction
            logger.warning("Highlight detection failed on page %d: %s", page_no, type(exc).__name__)
    if len(found) > MAX_CROPS_PER_FORM:
        ranked = sorted(found, key=lambda f: (not f[5], -(f[1][2] - f[1][0]) * (f[1][3] - f[1][1])))
        keep = {id(f) for f in ranked[:MAX_CROPS_PER_FORM]}
        found = [f for f in found if id(f) in keep]
    return [HighlightCrop(page_no, box, position, _upscale(region), _upscale(ink) if ink is not None else None, blue)
            for page_no, box, position, region, ink, blue in found]


def crop_contents(crops: list, n_pages: int, image_part, max_side: int) -> list:
    """Labelled content items (text + images) for the extraction request; [] when there are no crops."""
    if not crops:
        return []
    items = [
        f"ZOOMED CROPS: below are {len(crops)} crop(s) of highlighted regions of the page(s) above, cut from the "
        f"300 DPI scan and enlarged about {UPSCALE:g}x. They are only magnified views to help legibility: the "
        "highlighting does NOT tell you which field a value belongs to, and some highlighted values are decoys. "
        "Identify every field from its printed label and position on the full page. An 'ink only' version "
        "shows the same crop with the highlighter colour removed (dark ink of any colour kept). If the text is "
        "still not legible, return null. Do not guess."
    ]
    for i, c in enumerate(crops, start=1):
        where = f"page {c.page}, {c.position} area" if n_pages > 1 else f"{c.position} area of the page"
        items += [f"Crop {i} of {len(crops)} ({where}) - colour:", image_part(c.color, max_side)]
        if c.ink_only is not None:
            items += [f"Crop {i} of {len(crops)} - ink only:", image_part(c.ink_only, max_side)]
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
        if c.ink_only is not None:
            c.ink_only.save(folder / f"{safe}_crop{i}_ink.png")


def clear_debug_crops() -> int:
    """Delete data/temp/highlight_debug/ (called at the start of every run). Returns files removed."""
    folder = config.HIGHLIGHT_DEBUG_DIR
    if not folder.exists():
        return 0
    count = sum(1 for p in folder.rglob("*") if p.is_file())
    shutil.rmtree(folder, ignore_errors=True)
    return count
