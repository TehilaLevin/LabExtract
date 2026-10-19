"""Yellow-highlight crops: detection, blue-ink isolation, and the additive extraction flow. No API calls."""
import numpy as np
from PIL import Image, ImageDraw

from src import config
from src import extraction_engine as ee
from src import highlight_crops as hc

PAGE = (2480, 3508)  # A4 at 300 DPI
YELLOW = (250, 245, 120)
BLUE_INK = (40, 80, 200)


def _page(highlights=(), blue_text=False, black_text=False) -> Image.Image:
    img = Image.new("RGB", PAGE, "white")
    d = ImageDraw.Draw(img)
    for box in highlights:
        d.rectangle(box, fill=YELLOW)
    if blue_text:
        for i in range(8):  # pen-like strokes over the first highlight
            x = 520 + i * 60
            d.line([(x, 1010), (x + 30, 1070)], fill=BLUE_INK, width=7)
    if black_text:
        for i in range(8):
            x = 1520 + i * 60
            d.rectangle([x, 1015, x + 12, 1065], fill=(20, 20, 20))
    return img


def test_page_without_highlight_has_no_crops():
    assert hc.highlight_crops([_page()]) == []


def test_small_specks_are_ignored():
    assert hc.find_yellow_regions(_page([(100, 100, 115, 112)])) == []


def test_highlight_is_found_cropped_with_padding_and_upscaled():
    crops = hc.highlight_crops([_page([(500, 1000, 1000, 1080)], blue_text=True)])
    assert len(crops) == 1
    c = crops[0]
    x0, y0, x1, y1 = c.box
    assert x0 <= 500 - hc.PAD_X_PX + 5 and x1 >= 1000 + hc.PAD_X_PX - 5 and y0 < 1000 and y1 > 1080
    assert c.color.width > (x1 - x0) * 1.5                          # enlarged ~2x
    assert c.position == "middle right" or c.position.startswith(("top", "middle"))
    assert c.blue_only is not None                                   # blue pen strokes found


def test_blue_only_keeps_pen_and_drops_highlight_and_black_print():
    crops = hc.highlight_crops([_page([(500, 1000, 1000, 1080), (1500, 1000, 2000, 1080)],
                                      blue_text=True, black_text=True)])
    by_x = sorted(crops, key=lambda c: c.box[0])
    pen, printed = by_x[0], by_x[1]
    assert pen.blue_only is not None and printed.blue_only is None
    ink = pen.blue_only.convert("L")
    assert ink.getextrema()[0] == 0                                  # black strokes on ...
    assert (np.asarray(ink) == 255).mean() > 0.8                     # ... a white page


def test_crop_contents_are_labelled():
    crops = hc.highlight_crops([_page([(500, 1000, 1000, 1080)], blue_text=True)])
    items = hc.crop_contents(crops, 1, ee.image_part, hc.CROP_IMAGE_MAX_SIDE)
    texts = [i for i in items if isinstance(i, str)]
    assert "Do not guess" in texts[0]
    assert texts[1].startswith("Crop 1 of 1 (") and "colour" in texts[1]
    assert texts[2] == "Crop 1 of 1 - blue ink only:"
    assert hc.crop_contents([], 1, ee.image_part, 100) == []


def test_debug_crops_saved_only_with_flag_and_cleared(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "HIGHLIGHT_DEBUG_DIR", tmp_path / "highlight_debug")
    crops = hc.highlight_crops([_page([(500, 1000, 1000, 1080)], blue_text=True)])
    monkeypatch.setattr(config, "HIGHLIGHT_DEBUG", False)
    hc.save_debug_crops(crops, "form1")
    assert not (tmp_path / "highlight_debug").exists()
    monkeypatch.setattr(config, "HIGHLIGHT_DEBUG", True)
    hc.save_debug_crops(crops, "form1")
    assert len(list((tmp_path / "highlight_debug").glob("*.png"))) == 2
    assert hc.clear_debug_crops() == 2 and not (tmp_path / "highlight_debug").exists()


# ---- extraction flow --------------------------------------------------------------------------

def _capture(monkeypatch):
    seen = {}

    def fake_generate(contents, schema, **kwargs):
        seen["contents"] = contents
        return {}

    monkeypatch.setattr(ee, "generate_json", fake_generate)
    return seen


def test_extraction_adds_crops_for_claude_only(monkeypatch):
    seen = _capture(monkeypatch)
    page = _page([(500, 1000, 1000, 1080)], blue_text=True)
    monkeypatch.setattr(config, "LLM_PROVIDER", "claude")
    monkeypatch.setattr(config, "HIGHLIGHT_CROPS_ENABLED", True)
    res = ee.extract_form([page], ["כללית"], crop_images=[page])
    assert res.highlight_crops == 1
    assert sum(isinstance(c, ee.ImagePart) for c in seen["contents"]) == 3   # page + colour + blue
    assert seen["contents"][-1] == "Extract the fields from this form now."

    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    assert ee.extract_form([page], ["כללית"], crop_images=[page]).highlight_crops == 0
    assert sum(isinstance(c, ee.ImagePart) for c in seen["contents"]) == 1   # Gemini request unchanged


def test_crops_reach_the_gemini_request_when_enabled(monkeypatch, tmp_path):
    """End to end through the Gemini provider with a fake genai client: the crops are image parts of the
    request that leaves the app (no network)."""
    from src import llm_provider

    sent = {}

    class _Models:
        def generate_content(self, **kwargs):
            sent["contents"] = kwargs["contents"]
            return type("R", (), {"text": "{}", "usage_metadata": None})()

    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "DEV_CACHE", False)
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setattr(llm_provider.get_provider_named("gemini"), "client",
                        lambda: type("C", (), {"models": _Models()})())
    page = _page([(500, 1000, 1000, 1080)], blue_text=True)

    monkeypatch.setattr(config, "HIGHLIGHT_CROPS_PROVIDERS", ("claude",))      # default: Gemini unchanged
    ee.extract_form([page], ["כללית"], crop_images=[page])
    assert sum(not isinstance(p, str) for p in sent["contents"]) == 1

    monkeypatch.setattr(config, "HIGHLIGHT_CROPS_PROVIDERS", ("claude", "gemini"))
    res = ee.extract_form([page], ["כללית"], crop_images=[page])
    parts = [p for p in sent["contents"] if not isinstance(p, str)]
    assert res.highlight_crops == 1 and len(parts) == 3                         # page + colour + blue
    assert all(p.inline_data.mime_type == "image/jpeg" for p in parts)
    assert any(isinstance(p, str) and p.startswith("ZOOMED CROPS") for p in sent["contents"])


def test_extraction_without_highlight_is_unchanged(monkeypatch):
    seen = _capture(monkeypatch)
    monkeypatch.setattr(config, "LLM_PROVIDER", "claude")
    page = _page()
    ee.extract_form([page], ["כללית"], crop_images=[page])
    with_crops = list(seen["contents"])
    monkeypatch.setattr(config, "HIGHLIGHT_CROPS_ENABLED", False)
    ee.extract_form([page], ["כללית"], crop_images=[page])
    assert len(with_crops) == len(seen["contents"]) == 2                      # page + instruction
