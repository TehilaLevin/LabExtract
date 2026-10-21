"""Dictionary check of handwritten-name readings: look-alike letter corrections, suggestions only."""
import json

import pytest

from src import config
from src import extraction_engine as ee
from src import name_dictionary as nd
from src import record_builder as rb
from src import review_session as rs
from tests.test_record_builder import TODAY, _form, _hand

NAMES = frozenset(nd.lookup_key(w) for w in ("אזולאי", "טובה", "גרין", "כהן", "דהן", "מזרחי", "ישראל", "ישראלי",
                                             "לוי", "לביא"))


@pytest.mark.parametrize("word, expected", [
    ("אצולאי", "אזולאי"),   # ז read as צ (seen on a form)
    ("אצלאי", "אזולאי"),    # ז/צ and a dropped ו: two edits
    ("טבה", "טובה"),        # dropped ו
    ("כהו", "כהן"),         # ו / ן, final letter restored
    ("מצרחי", "מזרחי"),
    ("גרין", None),         # already a name
    ("ניקסוביל", None),     # no name within reach
    ("א", None),            # too short
])
def test_correct_word(word, expected):
    assert nd.correct_word(word, NAMES) == expected


def test_ambiguous_correction_is_not_made():
    names = frozenset({"חנ", "הנ"})                 # lookup keys of חן and הן
    assert nd.correct_word("תן", names) is None     # ת looks like both ח and ה: no single answer


def test_correct_name_corrects_each_word():
    assert nd.correct_name("מאור אצולאי", NAMES) == "מאור אזולאי"
    assert nd.correct_name("טובה גרין", NAMES) is None


def test_names_file_is_public_word_lists():
    data = json.loads(config.NAMES_PATH.read_text(encoding="utf-8"))
    assert data["first_names"] and data["last_names"] and "CBS" in data["_note"]
    assert all(" " not in w for w in data["first_names"] + data["last_names"])


def _crop(first, last, conf=0.6):
    return ee.NameReading("labeled_fields", True, f"{first} {last}", "labeled_fields", first, last, conf, 1)


def test_suggestion_is_corrected_and_reading_kept_as_alternative():
    rec = rb.build_record(_form(), _hand(first=None, last=None, conf=0.3),
                          ee.NameCheck(needed=True, readings=[_crop("מאור", "אצולאי")]), today=TODAY)
    assert rec.values["first_name"] == rec.values["last_name"] == ""          # still never a value
    assert rec.suggestions == {"first_name": "מאור", "last_name": "אזולאי"}
    assert rec.alternatives["last_name"] == ["אצולאי"] and "first_name" not in rec.alternatives


def test_dictionary_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(config, "NAME_DICTIONARY_ENABLED", False)
    rec = rb.build_record(_form(), _hand(first=None, last=None, conf=0.3),
                          ee.NameCheck(needed=True, readings=[_crop("מאור", "אצולאי")]), today=TODAY)
    assert rec.suggestions["last_name"] == "אצולאי"


# ---- batch name sharing ---------------------------------------------------------------------

def _record(index, first="", last="", ident="000000018", flags=None, suggestions=None):
    values = {k: "" for k in rb.FIELD_KEYS} | {"first_name": first, "last_name": last, "id_number": ident}
    return rb.Record(index, values, dict(flags or {}), {}, {}, suggestions=dict(suggestions or {}))


def test_batch_sharing_is_off_by_default():
    a, b = _record(1, "טובה", "גרין"), _record(2)
    assert rb.apply_batch_name_sharing([a, b]) == 0 and b.suggestions == {}


def test_batch_sharing_gives_a_suggestion_never_a_value(monkeypatch):
    monkeypatch.setattr(config, "BATCH_NAME_SHARING", True)
    a, b = _record(1, "טובה", "גרין"), _record(2, flags={"first_name": "name_unverified"})
    other = _record(3, ident="000000026")
    assert rb.apply_batch_name_sharing([a, b, other]) == 1
    assert b.suggestions == {"first_name": "טובה", "last_name": "גרין"} and b.values["first_name"] == ""
    assert other.suggestions == {}


def test_batch_sharing_needs_a_valid_unflagged_id(monkeypatch):
    monkeypatch.setattr(config, "BATCH_NAME_SHARING", True)
    a, b = _record(1, "טובה", "גרין", ident="000000019"), _record(2, ident="000000019")   # bad check digit
    c, d = _record(3, "טובה", "גרין"), _record(4, flags={"id_number": "low_confidence"})
    assert rb.apply_batch_name_sharing([a, b, c, d]) == 0


def test_batch_sharing_uses_another_forms_suggestion(monkeypatch):
    monkeypatch.setattr(config, "BATCH_NAME_SHARING", True)
    a = _record(1, suggestions={"first_name": "מאור", "last_name": "אזולאי"})
    b = _record(2)
    assert rb.apply_batch_name_sharing([a, b]) == 1 and b.suggestions["last_name"] == "אזולאי"


def test_swap_names():
    entered = {"first_name": "גרין", "last_name": "טובה", "id_number": "1"}
    assert rs.swap_names(entered) == {"first_name": "טובה", "last_name": "גרין", "id_number": "1"}


def test_low_confidence_crop_reading_is_not_suggested():
    weak = ee.NameReading("patient_sticker", True, "לשסט שוצה", "convention", "שוצה", "לשסט", 0.15, 3)
    rec = rb.build_record(_form(), _hand(first=None, last=None, conf=0.3),
                          ee.NameCheck(needed=True, readings=[weak]), today=TODAY)
    assert rec.suggestions == {} and rec.flags["first_name"] == "name_unverified"


# ---- name crop views: enlargement and ink-only (image processing only) ------------------------

def _name_crop(width=600, height=120):
    """White crop, yellow highlighter band, blue 'handwriting' strokes on it."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, height // 3, width, 2 * height // 3], fill=(250, 240, 90))
    for x in range(40, width - 40, 60):
        draw.line([(x, height // 4), (x + 30, 3 * height // 4)], fill=(20, 40, 160), width=6)
    return img


def test_small_crop_is_enlarged_and_large_one_kept():
    small = ee.enlarge_name_crop(_name_crop(400, 120))
    assert small.size == (1200, 360)                           # capped by NAME_CROP_MAX_UPSCALE (3x)
    tall = ee.enlarge_name_crop(_name_crop(400, 300))
    assert tall.height == ee.NAME_CROP_TARGET_HEIGHT           # enlarged to the target height
    wide = ee.enlarge_name_crop(_name_crop(1500, 150))
    assert max(wide.size) <= ee.NAME_CROP_MAX_SIDE             # never above the max side
    big = _name_crop(1200, 500)
    assert ee.enlarge_name_crop(big) is big                    # never shrinks here


def test_ink_only_view_removes_highlighter():
    views = ee.name_crop_views(_name_crop())
    assert [label for label, _ in views] == [" - colour", " - ink only"]
    ink = views[1][1].convert("L")
    w, h = ink.size
    band = [ink.getpixel((x, h // 2)) for x in range(0, 30)]   # highlighter only, left of the first stroke
    assert min(band) > 200                                     # the yellow band became white


def test_plain_crop_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "NAME_CROP_ENHANCE", False)
    crop = _name_crop()
    assert ee.name_crop_views(crop) == [("", crop)]


def test_verify_names_sends_both_views_and_note(monkeypatch):
    from PIL import Image

    sent = {}
    monkeypatch.setattr(ee, "generate_json", lambda contents, *a, **k: sent.setdefault("c", contents) and {"crops": []})
    page = Image.new("RGB", (2000, 3000), "white")
    page.paste(_name_crop(1600, 120), (200, 900))
    ee.verify_names([page], _hand())
    texts = [c for c in sent["c"] if isinstance(c, str)]
    assert "Crop 1 - colour:" in texts and "Crop 1 - ink only:" in texts
    assert ee.NAME_VIEWS_NOTE.strip() in texts[0]


def test_disabled_views_keep_the_earlier_request(monkeypatch):
    from PIL import Image

    monkeypatch.setattr(config, "NAME_CROP_ENHANCE", False)
    sent = {}
    monkeypatch.setattr(ee, "generate_json", lambda contents, *a, **k: sent.setdefault("c", contents) and {"crops": []})
    ee.verify_names([Image.new("RGB", (2000, 3000), "white")], _hand())
    texts = [c for c in sent["c"] if isinstance(c, str)]
    assert texts[1:] == ["Crop 1:"]                            # same labels as before: dev-cache keys unchanged
    assert "Ignore yellow highlighter. Do not output anything except the JSON." in texts[0]


def test_ink_view_keeps_green_and_blue_pen():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (300, 100), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 30, 300, 70], fill=(250, 240, 90))                  # yellow highlighter
    draw.line([(50, 20), (50, 80)], fill=(40, 140, 60), width=6)          # green pen (erased by ink_only)
    draw.line([(150, 20), (150, 80)], fill=(20, 40, 160), width=6)        # blue pen
    view = ee.ink_view(img)
    assert view.getpixel((50, 50)) < 80 and view.getpixel((150, 50)) < 80  # both pens dark
    assert view.getpixel((250, 50)) > 200                                  # highlighter gone
