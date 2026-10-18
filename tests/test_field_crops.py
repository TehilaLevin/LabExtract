"""Source crops: every located field keeps the image area its value was read from (in memory only).

All Gemini calls are faked - these tests never touch the API.
"""
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from src import extraction_engine as ee
from src import field_crops as fc
from src import record_builder as rb
from src.constants import FIELD_KEYS, LOCATED_FIELDS

BOXES = {  # (ymin, xmin, ymax, xmax), 0-1000, one distinct colour per field
    "id_number": ((100, 100, 150, 400), "red"),
    "central_numbers": ((200, 600, 260, 900), "blue"),
    "received_date": ((300, 600, 360, 900), "green"),
    "commitment_numbers": ((500, 100, 550, 400), "purple"),
    "sending_entity": ((20, 700, 80, 980), "orange"),
}
NAME_BOX = (700, 100, 760, 900)


def _page(size=(1000, 1400), marks=()):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    w, h = size
    for (y0, x0, y1, x1), colour in marks:
        draw.rectangle([x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000], fill=colour)
    return img


def _model_json(field_locations=None, name_locations=None):
    return {
        "form_type": {"value": "maccabi_lab_request", "confidence": 0.97},
        "first_name": {"value": "ישראל", "confidence": 0.9, "script": "printed", "status": "read"},
        "last_name": {"value": "ישראלי", "confidence": 0.9, "script": "printed", "status": "read"},
        "name_order_basis": "separate_fields",
        "id_number": {"value": "000000018", "confidence": 0.95, "status": "read"},
        "central_numbers": {"value": ["937309"], "confidence": 0.95, "status": "read"},
        "received_date": {"value": "01-06-2026", "confidence": 0.9, "received_date_source": "received_stamp",
                          "status": "read"},
        "sending_entity": {"value": "מכבי", "confidence": 0.95, "status": "read"},
        "commitment_numbers": {"value": ["50364186"], "confidence": 0.95, "status": "read"},
        "field_locations": field_locations if field_locations is not None else [
            {"field": k, "page": 1, "box_2d": list(box)} for k, (box, _) in BOXES.items()],
        "name_locations": name_locations if name_locations is not None else [
            {"page": 1, "box_2d": list(NAME_BOX), "kind": "labeled_fields"}],
    }


def _ext(**kwargs):
    return ee.parse_extraction(_model_json(**kwargs))


def _decode(crop: fc.SourceCrop) -> Image.Image:
    assert crop.jpeg[:2] == b"\xff\xd8"  # JPEG
    return Image.open(BytesIO(crop.jpeg)).convert("RGB")


def _centre(img: Image.Image) -> tuple:
    return img.getpixel((img.width // 2, img.height // 2))


COLOURS = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 128, 0), "purple": (128, 0, 128),
           "orange": (255, 165, 0)}


def _close(rgb, name, tol=40):
    return all(abs(a - b) <= tol for a, b in zip(rgb, COLOURS[name]))


# ---- schema / parsing -------------------------------------------------------------------

def test_every_field_can_be_located():
    assert set(LOCATED_FIELDS) | {"first_name", "last_name"} == set(FIELD_KEYS)


def test_sender_location_is_parsed_and_capped_at_one():
    locs = [{"field": "sending_entity", "page": 1, "box_2d": [i, 0, i + 10, 10]} for i in range(3)]
    ext = _ext(field_locations=locs)
    assert ext.field_locations == [{"field": "sending_entity", "page": 1, "box": (0, 0, 10, 10)}]


def test_prompt_asks_for_the_sender_location():
    prompt = ee.load_prompt(["מכבי"])
    assert "sending_entity: ONE entry for the place you identified the sender from" in prompt


def test_sender_location_never_triggers_a_reread():
    from src.field_reread import select_reread_fields

    assert select_reread_fields(_ext(), {"sending_entity": "missing"}) == []


# ---- source_crops ----------------------------------------------------------------------

def test_every_located_field_gets_a_crop_of_its_own_area():
    page = _page(marks=[*BOXES.values(), (NAME_BOX, "red")])
    crops = fc.source_crops([page], _ext())
    assert list(crops) == list(FIELD_KEYS)  # all seven fields, in field order
    for key, (box, colour) in BOXES.items():
        [crop] = crops[key]
        assert crop.page == 1 and crop.box == box
        assert _close(_centre(_decode(crop)), colour), key


def test_both_names_share_the_name_location_crops():
    crops = fc.source_crops([_page(marks=[(NAME_BOX, "red")])], _ext())
    assert crops["first_name"] == crops["last_name"]
    assert len(crops["first_name"]) == 1 and crops["first_name"][0].box == NAME_BOX
    assert _close(_centre(_decode(crops["first_name"][0])), "red")


def test_several_locations_of_one_field_are_all_kept_with_their_page():
    page1 = _page(marks=[((200, 600, 260, 900), "blue")])
    page2 = _page(marks=[((400, 100, 460, 400), "green")])
    ext = _ext(field_locations=[
        {"field": "central_numbers", "page": 1, "box_2d": [200, 600, 260, 900]},
        {"field": "central_numbers", "page": 2, "box_2d": [400, 100, 460, 400]},
    ])
    crops = fc.source_crops([page1, page2], ext)["central_numbers"]
    assert [c.page for c in crops] == [1, 2]
    assert _close(_centre(_decode(crops[0])), "blue") and _close(_centre(_decode(crops[1])), "green")


def test_padding_keeps_the_label_beside_the_value():
    page = _page(marks=[((100, 100, 150, 400), "red")])
    img = _decode(fc.source_crops([page], _ext())["id_number"][0])
    w, h = page.size
    assert img.width > 300 * w / 1000 and img.height > 50 * h / 1000  # larger than the bare box


def test_unusable_boxes_are_skipped_and_the_field_is_absent():
    ext = _ext(field_locations=[
        {"field": "id_number", "page": 3, "box_2d": [100, 100, 150, 400]},  # page out of range
        {"field": "received_date", "page": 1, "box_2d": [300, 600, 360, 900]},
    ], name_locations=[{"page": 5, "box_2d": list(NAME_BOX), "kind": "other"}])
    crops = fc.source_crops([_page()], ext)
    assert list(crops) == ["received_date"]


def test_degenerate_box_on_a_tiny_page_is_skipped():
    ext = _ext(field_locations=[{"field": "id_number", "page": 1, "box_2d": [0, 0, 1, 1]}], name_locations=[])
    assert fc.source_crops([Image.new("RGB", (8, 8), "white")], ext) == {}


def test_fields_without_a_location_have_no_crop():
    crops = fc.source_crops([_page()], _ext(field_locations=[], name_locations=[]))
    assert crops == {}


def test_failed_extraction_has_no_crops():
    assert fc.source_crops([_page()], ee.empty_result("boom")) == {}


def test_crops_are_bounded_for_display():
    big = Image.new("RGB", (6000, 8000), "white")
    ext = _ext(field_locations=[{"field": "sending_entity", "page": 1, "box_2d": [0, 0, 600, 1000]}],
               name_locations=[])
    img = _decode(fc.source_crops([big], ext)["sending_entity"][0])
    assert max(img.size) <= fc.MAX_SIDE


# ---- record / pipeline ------------------------------------------------------------------

def _form():
    from pathlib import Path

    from src.file_converter import Page
    from src.form_splitter import Form

    return Form(index=1, source_file="batch.pdf", pages=[Page("batch.pdf", 3, Path("x.jpg"))], rotations=[0])


def test_record_keeps_crops_but_never_exports_them():
    ext = _ext()
    crops = fc.source_crops([_page()], ext)
    rec = rb.build_record(_form(), ext, field_crops=crops)
    assert rec.field_crops == crops
    row = rec.to_row()
    assert not any(isinstance(v, (bytes, list, dict)) for v in row.values())
    assert "field_crops" not in rb.records_to_dataframe([rec]).columns


def test_record_without_crops_defaults_to_empty():
    assert rb.build_record(_form(), _ext()).field_crops == {}


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "TEMP_DIR", tmp_path)
    return tmp_path


def _png_bytes(marks=()):
    buf = BytesIO()
    _page(marks=marks).save(buf, "PNG")
    return buf.getvalue()


def _fake_generate(contents, schema, **kwargs):
    if "upright" in schema["properties"]:  # orientation request
        return {"upright": "A", "confidence": 0.99}
    if "pages" in schema["properties"]:  # split request
        n = sum(1 for c in contents if isinstance(c, str) and c.startswith("PAGE "))
        return {"pages": [{"page_index": i, "is_blank": False, "starts_new_form": True, "confidence": 0.9}
                          for i in range(1, n + 1)]}
    return _model_json()


@pytest.mark.parametrize("reread_enabled", [True, False])
def test_process_files_returns_a_crop_for_every_field(monkeypatch, temp_dir, reread_enabled):
    monkeypatch.setattr(ee, "generate_json", _fake_generate)
    monkeypatch.setattr("src.config.FIELD_REREAD_ENABLED", reread_enabled)
    [rec] = rb.process_files([("a.png", _png_bytes(marks=[*BOXES.values(), (NAME_BOX, "red")]))])
    assert list(rec.field_crops) == list(FIELD_KEYS)
    for key, (_, colour) in BOXES.items():
        assert _close(_centre(_decode(rec.field_crops[key][0])), colour), key
    assert not list(temp_dir.iterdir())  # crops live in memory; the run folder is deleted


@pytest.mark.parametrize("crop_source, expected", [("original", "red"), ("processed", "blue")])
def test_process_files_crops_follow_crop_source(monkeypatch, temp_dir, crop_source, expected):
    """By default the review crop shows the page as scanned, not the adjusted copy the model read."""
    from src import image_preprocessor as ip

    box = BOXES["id_number"][0]

    def fake_preprocess(image):  # original marked red, processed copy marked blue: same geometry
        return ip.PreprocessResult(original=_page(marks=[(box, "red")]), processed=_page(marks=[(box, "blue")]),
                                   steps=["fake"])

    monkeypatch.setattr(ee, "generate_json", _fake_generate)
    monkeypatch.setattr(ip, "preprocess", fake_preprocess)
    monkeypatch.setattr("src.config.CROP_SOURCE", crop_source)
    [rec] = rb.process_files([("a.png", _png_bytes())])
    assert _close(_centre(_decode(rec.field_crops["id_number"][0])), expected)


def test_quota_stop_gives_records_without_crops(monkeypatch, temp_dir):
    def fake(contents, schema, **kwargs):
        if "upright" in schema["properties"] or "pages" in schema["properties"]:
            return _fake_generate(contents, schema)
        raise ee.GeminiQuotaError("Gemini API quota exhausted")

    monkeypatch.setattr(ee, "generate_json", fake)
    monkeypatch.setattr("src.config.MAX_PARALLEL_REQUESTS", 1)
    records = rb.process_files([(f"{i}.png", _png_bytes()) for i in range(2)])
    assert all(r.error and r.field_crops == {} for r in records)
