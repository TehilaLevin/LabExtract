"""Light Pillow preprocessing: auto-contrast, brightness fix for dark scans, basic deskew.

The original image is always kept next to the processed one, in case preprocessing
hurts readability. Colour is preserved on purpose: blue handwriting and coloured
stamps are easier to read in colour.
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# Mean luminance (0-255) below which a scan is treated as dark.
DARK_MEAN_THRESHOLD = 150
TARGET_MEAN = 200
MAX_BRIGHTNESS_FACTOR = 1.6

# Deskew search: small angles only (large rotations are handled by fix_rotation).
DESKEW_MAX_ANGLE = 5.0
DESKEW_STEP = 0.25
DESKEW_MIN_ANGLE = 0.3       # below this, rotating would only blur the image
DESKEW_MIN_GAIN = 1.15       # best angle must beat 0 deg by this ratio
_DESKEW_WORK_SIDE = 1000

VALID_ROTATIONS = (0, 90, 180, 270)


@dataclass
class PreprocessResult:
    original: Image.Image
    processed: Image.Image
    steps: list = field(default_factory=list)  # e.g. ["autocontrast", "brighten x1.30", "deskew -1.50"]


def mean_luminance(image: Image.Image) -> float:
    gray = image.convert("L")
    gray.thumbnail((_DESKEW_WORK_SIDE, _DESKEW_WORK_SIDE))
    return float(np.asarray(gray, dtype=np.float32).mean())


def _projection_score(binary: Image.Image, angle: float) -> float:
    rotated = binary.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=0)
    rows = np.asarray(rotated, dtype=np.float32).sum(axis=1)
    return float(np.var(rows))


def estimate_skew(image: Image.Image) -> float:
    """Return the angle (degrees, PIL counter-clockwise convention) that straightens text.

    Projection-profile method: text lines produce the sharpest row-sum profile when
    horizontal. Returns 0.0 when no clear improvement is found.
    """
    gray = image.convert("L")
    gray.thumbnail((_DESKEW_WORK_SIDE, _DESKEW_WORK_SIDE))
    arr = np.asarray(gray, dtype=np.uint8)
    threshold = min(int(arr.mean()) - 30, 160)
    binary = Image.fromarray(((arr < threshold) * 255).astype(np.uint8))  # ink = white

    base = _projection_score(binary, 0.0)
    if base <= 0:
        return 0.0
    angles = np.arange(-DESKEW_MAX_ANGLE, DESKEW_MAX_ANGLE + 1e-9, DESKEW_STEP)
    scores = [(_projection_score(binary, float(a)), float(a)) for a in angles]
    best_score, best_angle = max(scores)
    if abs(best_angle) < DESKEW_MIN_ANGLE or best_score < base * DESKEW_MIN_GAIN:
        return 0.0
    return best_angle


def preprocess(image: Image.Image) -> PreprocessResult:
    """Return the original plus a lightly enhanced copy (never modifies the input)."""
    original = image
    img = image.convert("RGB")
    steps = []

    img = ImageOps.autocontrast(img, cutoff=0.5, preserve_tone=True)
    steps.append("autocontrast")

    mean = mean_luminance(img)
    if mean < DARK_MEAN_THRESHOLD:
        factor = min(MAX_BRIGHTNESS_FACTOR, TARGET_MEAN / max(mean, 1.0))
        img = ImageEnhance.Brightness(img).enhance(factor)
        steps.append(f"brighten x{factor:.2f}")

    angle = estimate_skew(img)
    if angle:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
        steps.append(f"deskew {angle:+.2f}")

    return PreprocessResult(original=original, processed=img, steps=steps)


def fix_rotation(image: Image.Image, rotate_clockwise: int) -> Image.Image:
    """Rotate by a multiple of 90 degrees *clockwise* (as reported by the model)."""
    if rotate_clockwise not in VALID_ROTATIONS:
        raise ValueError(f"Rotation must be one of {VALID_ROTATIONS}, got {rotate_clockwise}")
    if rotate_clockwise == 0:
        return image
    return image.rotate(-rotate_clockwise, expand=True)  # PIL rotates counter-clockwise


def resize_for_model(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale (never upscale) so the longest side is at most max_side."""
    if max(image.size) <= max_side:
        return image
    copy = image.copy()
    copy.thumbnail((max_side, max_side), Image.LANCZOS)
    return copy
