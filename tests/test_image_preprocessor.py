import pytest
from PIL import Image, ImageDraw

from src import image_preprocessor as ip


def _text_like_page(color=(0, 0, 0), background=(255, 255, 255)) -> Image.Image:
    """A page with horizontal 'text lines' (thick dashed bars)."""
    img = Image.new("RGB", (800, 1000), background)
    draw = ImageDraw.Draw(img)
    for y in range(80, 920, 40):
        for x in range(80, 720, 60):
            draw.rectangle([x, y, x + 45, y + 12], fill=color)
    return img


def test_original_is_kept_unchanged():
    page = _text_like_page()
    before = page.tobytes()
    result = ip.preprocess(page)
    assert result.original is page
    assert page.tobytes() == before
    assert result.processed is not page


def test_dark_scan_is_brightened():
    dark = _text_like_page(color=(0, 0, 0), background=(90, 90, 90))
    result = ip.preprocess(dark)
    assert any(s.startswith("brighten") for s in result.steps) or ip.mean_luminance(result.processed) > 150
    assert ip.mean_luminance(result.processed) > ip.mean_luminance(dark)


def test_bright_scan_is_not_brightened():
    result = ip.preprocess(_text_like_page())
    assert not any(s.startswith("brighten") for s in result.steps)


@pytest.mark.parametrize("skew", [3.0, -2.0])
def test_skew_is_detected_and_corrected(skew):
    skewed = _text_like_page().rotate(skew, expand=True, fillcolor="white")
    angle = ip.estimate_skew(skewed)
    assert angle == pytest.approx(-skew, abs=0.5)


def test_straight_page_is_not_deskewed():
    assert ip.estimate_skew(_text_like_page()) == 0.0


def test_fix_rotation_is_clockwise():
    img = Image.new("RGB", (100, 50), "white")
    img.putpixel((0, 0), (255, 0, 0))  # top-left marker
    rotated = ip.fix_rotation(img, 90)
    assert rotated.size == (50, 100)
    assert rotated.getpixel((49, 0)) == (255, 0, 0)  # clockwise: top-left -> top-right


def test_fix_rotation_rejects_odd_angles():
    with pytest.raises(ValueError):
        ip.fix_rotation(Image.new("RGB", (10, 10)), 45)


def test_resize_for_model_never_upscales():
    small = Image.new("RGB", (100, 80))
    assert ip.resize_for_model(small, 500) is small
    assert max(ip.resize_for_model(Image.new("RGB", (3000, 1000)), 1500).size) == 1500
