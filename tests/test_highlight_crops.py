"""Highlight crops: detection, merging, ink isolation, and the additive extraction flow. No API calls."""
import numpy as np
from PIL import Image, ImageDraw

from src import config
from src import extraction_engine as ee
from src import highlight_crops as hc

PAGE = (2480, 3508)  # A4 at 300 DPI
YELLOW = (250, 245, 120)
CYAN = (170, 240, 235)
BLUE_INK = (40, 80, 200)
BLACK_INK = (25, 25, 25)


def _page(highlights=(), ink=None, ink_box=(520, 1010), color=YELLOW) -> Image.Image:
    img = Image.new("RGB", PAGE, "white")
    d = ImageDraw.Draw(img)
    for box in highlights:
        d.rectangle(box, fill=color)
    if ink:
        x0, y0 = ink_box
        for i in range(8):  # pen-like strokes over the first highlight
            d.line([(x0 + i * 60, y0), (x0 + i * 60 + 30, y0 + 60)], fill=ink, width=7)
    return img


def test_page_without_highlight_has_no_crops():
    assert hc.highlight_crops([_page()]) == []


def test_small_specks_are_ignored():
    assert hc.find_highlight_regions(_page([(100, 100, 115, 112)])) == []


def test_highlight_is_found_cropped_with_padding_and_upscaled():
    crops = hc.highlight_crops([_page([(500, 1000, 1000, 1080)], ink=BLUE_INK)])
    assert len(crops) == 1
    c = crops[0]
    x0, y0, x1, y1 = c.box
    assert x0 <= 500 - hc.PAD_X_PX + 5 and x1 >= 1000 + hc.PAD_X_PX - 5 and y0 < 1000 and y1 > 1080
    assert c.color.width > (x1 - x0) * 1.5                          # enlarged ~2x
    assert c.ink_only is not None and c.has_blue_ink


def test_green_and_cyan_markers_are_found_too():
    assert len(hc.find_highlight_regions(_page([(500, 1000, 1000, 1080)], color=CYAN))) == 1


def test_black_handwriting_is_kept_in_the_ink_only_version():
    # The earlier blue-only filter returned nothing here (black pen, e.g. a handwritten central number).
    crops = hc.highlight_crops([_page([(500, 1000, 1000, 1080)], ink=BLACK_INK)])
    ink = np.asarray(crops[0].ink_only.convert("L"))
    assert crops[0].ink_only is not None and not crops[0].has_blue_ink
    assert (ink == 0).any() and (ink == 255).mean() > 0.8              # black strokes on a white page


def test_highlighter_colour_is_removed_from_the_ink_only_version():
    crop = Image.new("RGB", (400, 120), YELLOW)
    ImageDraw.Draw(crop).rectangle([0, 60, 400, 120], fill=CYAN)
    assert hc.ink_only(crop) is None                                   # only highlighter: nothing to send
    ImageDraw.Draw(crop).line([(20, 20), (380, 100)], fill=BLUE_INK, width=6)
    ink = np.asarray(hc.ink_only(crop).convert("L"))
    assert 0.003 < (ink == 0).mean() < 0.2                              # the stroke only, no highlighter


def test_nearby_regions_merge_into_one_crop():
    # A highlighted label and its highlighted value, 100 px apart on one line -> one crop.
    label_and_value = [(1500, 1000, 1700, 1060), (1000, 1000, 1400, 1060)]
    assert len(hc.merge_regions(label_and_value, PAGE)) == 1
    # A stamp heading above its date (60 px gap) -> one crop.
    assert len(hc.merge_regions([(500, 2000, 900, 2080), (520, 2140, 880, 2200)], PAGE)) == 1
    # Far apart -> separate crops.
    assert len(hc.merge_regions([(100, 100, 300, 160), (1800, 3000, 2100, 3060)], PAGE)) == 2
    crops = hc.highlight_crops([_page(label_and_value)])
    assert len(crops) == 1


def test_merging_never_builds_a_huge_crop():
    column = [(1000, 100 + i * 120, 1300, 160 + i * 120) for i in range(20)]   # 20 lines, 60 px apart
    merged = hc.merge_regions(column, PAGE)
    assert all(b[3] - b[1] <= hc.MERGED_MAX_HEIGHT * PAGE[1] for b in merged) and len(merged) > 1


def test_crop_contents_are_labelled_and_warn_about_decoys():
    crops = hc.highlight_crops([_page([(500, 1000, 1000, 1080)], ink=BLUE_INK)])
    items = hc.crop_contents(crops, 1, ee.image_part, hc.CROP_IMAGE_MAX_SIDE)
    texts = [i for i in items if isinstance(i, str)]
    assert "does NOT tell you which field" in texts[0] and "Do not guess" in texts[0]
    assert texts[1].startswith("Crop 1 of 1 (") and "colour" in texts[1]
    assert texts[2] == "Crop 1 of 1 - ink only:"
    assert hc.crop_contents([], 1, ee.image_part, 100) == []


def test_debug_crops_saved_only_with_flag_and_cleared(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "HIGHLIGHT_DEBUG_DIR", tmp_path / "highlight_debug")
    crops = hc.highlight_crops([_page([(500, 1000, 1000, 1080)], ink=BLUE_INK)])
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


def test_extraction_adds_the_crops(monkeypatch):
    seen = _capture(monkeypatch)
    page = _page([(500, 1000, 1000, 1080)], ink=BLUE_INK)
    monkeypatch.setattr(config, "HIGHLIGHT_CROPS_ENABLED", True)
    res = ee.extract_form([page], ["כללית"], crop_images=[page])
    assert res.highlight_crops == 1
    assert sum(isinstance(c, ee.ImagePart) for c in seen["contents"]) == 3   # page + colour + ink only
    assert seen["contents"][-1] == "Extract the fields from this form now."
    monkeypatch.setattr(config, "HIGHLIGHT_CROPS_ENABLED", False)
    assert ee.extract_form([page], ["כללית"], crop_images=[page]).highlight_crops == 0


def test_crops_reach_the_claude_request(fake_claude, monkeypatch):
    page = _page([(500, 1000, 1000, 1080)], ink=BLUE_INK)
    fake_claude.reply = __import__("tests.conftest", fromlist=["fake_reply"]).fake_reply(text="{}")
    ee.extract_form([page], ["כללית"], crop_images=[page])
    content = fake_claude.kwargs["messages"][0]["content"]
    assert sum(b["type"] == "image" for b in content) == 3
    assert any(b["type"] == "text" and b["text"].startswith("ZOOMED CROPS") for b in content)


def test_extraction_without_highlight_is_unchanged(monkeypatch):
    seen = _capture(monkeypatch)
    page = _page()
    ee.extract_form([page], ["כללית"], crop_images=[page])
    with_crops = list(seen["contents"])
    monkeypatch.setattr(config, "HIGHLIGHT_CROPS_ENABLED", False)
    ee.extract_form([page], ["כללית"], crop_images=[page])
    assert len(with_crops) == len(seen["contents"]) == 2                      # page + instruction
