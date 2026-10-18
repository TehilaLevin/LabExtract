"""Source crops: for every field, the image area its value was read from, for the review screen.

The full-page pass reports where each field is (field_locations; name_locations for the names). Each box
is cropped from the full-resolution page (config.CROP_SOURCE), encoded as JPEG and kept in memory only -
never written to disk, logged or exported. A box is the model's claim of where it read the value, so the
review screen shows it as "where the value was read", not as proof that the value is right.
"""
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from src.constants import FIELD_KEYS
from src.image_preprocessor import resize_for_model

NAME_KEYS = ("first_name", "last_name")

PAD_X = 0.25            # same padding as the targeted re-read: the label next to the value stays visible
PAD_Y = 0.5
MIN_PAD_PX = 60
NAME_PAD_X = 0.35       # same padding as the name check (model name boxes can be too narrow)
NAME_PAD_Y = 0.6
NAME_MIN_PAD_PX = 40
MAX_SIDE = 1200         # display size: readable on screen, small in memory
JPEG_QUALITY = 85


@dataclass(frozen=True)
class SourceCrop:
    page: int           # 1-based page within the form
    box: tuple          # (ymin, xmin, ymax, xmax), normalized 0-1000, as reported by the model
    jpeg: bytes


def _encode(image) -> bytes:
    buf = BytesIO()
    resize_for_model(image.convert("RGB"), MAX_SIDE).save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _crop(images: list, page: int, box: tuple, pad_x: float, pad_y: float, min_pad: int) -> Optional[SourceCrop]:
    from src.extraction_engine import padded_pixel_box

    if page > len(images):
        return None
    img = images[page - 1]
    pixel_box = padded_pixel_box(img.size, box, pad_x, pad_y, min_pad)
    if pixel_box is None:
        return None
    return SourceCrop(page, tuple(box), _encode(img.crop(pixel_box)))


def source_crops(images: list, extraction) -> dict:
    """{field key: [SourceCrop]} for every field the model located; fields without a usable box are absent.

    `images` are the form's full-resolution pages (same geometry as the pages the model saw). Both
    names share the name-location crops: one box covers the whole name and its labels.
    """
    if extraction.error:
        return {}
    out: dict = {}
    for loc in extraction.field_locations:
        crop = _crop(images, loc["page"], loc["box"], PAD_X, PAD_Y, MIN_PAD_PX)
        if crop is not None:
            out.setdefault(loc["field"], []).append(crop)
    name_crops = [c for c in (_crop(images, loc["page"], loc["box"], NAME_PAD_X, NAME_PAD_Y, NAME_MIN_PAD_PX)
                              for loc in extraction.name_locations) if c is not None]
    if name_crops:
        for key in NAME_KEYS:
            out[key] = list(name_crops)
    return {k: out[k] for k in FIELD_KEYS if k in out}
