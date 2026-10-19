"""Targeted re-read of flagged number fields: boxes, crops, combining, decisions, diagnostics.

All model calls are faked - these tests never touch the API.
"""
from datetime import date
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from src import extraction_engine as ee
from src import field_reread as fr
from src import image_preprocessor as ip
from src import record_builder as rb
from src.file_converter import Page
from src.form_splitter import Form

TODAY = date(2026, 10, 13)
VALID_ID = "000000018"
OTHER_VALID_ID = "000000026"
INVALID_ID = "000000019"


def _form():
    return Form(index=1, source_file="batch.pdf", pages=[Page("batch.pdf", 3, Path("x.jpg"))], rotations=[0])


def _model_json(locations=None, **overrides):
    data = {
        "form_type": {"value": "maccabi_lab_request", "confidence": 0.97},
        "first_name": {"value": "ישראל", "confidence": 0.9, "script": "printed", "status": "read"},
        "last_name": {"value": "ישראלי", "confidence": 0.9, "script": "printed", "status": "read"},
        "name_order_basis": "separate_fields",
        "id_number": {"value": VALID_ID, "confidence": 0.95, "status": "read"},
        "central_numbers": {"value": ["937309", "937310"], "confidence": 0.95, "status": "read"},
        "received_date": {"value": "01-06-2026", "confidence": 0.9, "received_date_source": "received_stamp",
                          "status": "read"},
        "sending_entity": {"value": "מכבי", "confidence": 0.95, "status": "read"},
        "commitment_numbers": {"value": ["50364186"], "confidence": 0.95, "status": "read"},
        "field_locations": locations if locations is not None else [
            {"field": k, "page": 1, "box_2d": [100, 100, 200, 400]}
            for k in ("id_number", "central_numbers", "received_date", "commitment_numbers")],
    }
    data.update(overrides)
    return data


def _ext(**overrides):
    return ee.parse_extraction(_model_json(**overrides))


def _reading(key, values, status="read", conf=0.95, source=None):
    if key == "received_date" and source is None:
        source = "received_stamp"
    return fr.CropReading(key, status, list(values), conf, source)


def _recheck(*readings, error=None):
    return fr.FieldRecheck(attempted=sorted({r.field for r in readings}, key=fr.REREAD_FIELDS.index),
                           readings=list(readings), error=error)


def _rec(ext, recheck):
    return rb.build_record(_form(), ext, recheck=recheck, today=TODAY)


# ---- schema / parsing ----------------------------------------------------------------

def test_schema_has_status_and_field_locations():
    schema = ee.build_schema(["כללית"])
    assert schema["properties"]["id_number"]["properties"]["status"]["enum"] == ["read", "unreadable", "not_present"]
    assert "field_locations" in schema["required"]
    item = schema["properties"]["field_locations"]["items"]["properties"]
    assert item["field"]["enum"] == [*fr.REREAD_FIELDS, "sending_entity"]


def test_prompt_explains_status_and_field_locations():
    prompt = ee.load_prompt(["כללית"])
    assert '"unreadable": the field IS on the form' in prompt
    assert "field_locations" in prompt and "ALSO when the value is unreadable" in prompt


def test_invalid_field_location_boxes_are_dropped():
    ext = _ext(locations=[
        {"field": "id_number", "page": 1, "box_2d": [100, 100, 200, 300]},         # ok
        {"field": "id_number", "page": 1, "box_2d": [100, 100, 200, 300]},         # duplicate
        {"field": "id_number", "page": 0, "box_2d": [1, 2, 3, 4]},                 # bad page
        {"field": "id_number", "page": True, "box_2d": [1, 2, 3, 4]},              # bool is not a page
        {"field": "central_numbers", "page": 1, "box_2d": [300, 100, 200, 300]},   # ymin > ymax
        {"field": "central_numbers", "page": 1, "box_2d": [0, 0, 1001, 10]},       # out of range
        {"field": "central_numbers", "page": 1, "box_2d": [1, 2, 3]},              # 3 numbers
        {"field": "central_numbers", "page": 1, "box_2d": ["a", 2, 3, 4]},         # not numbers
        {"field": "first_name", "page": 1, "box_2d": [1, 2, 3, 4]},                # names: name_locations
        "garbage",
    ])
    assert ext.field_locations == [{"field": "id_number", "page": 1, "box": (100, 100, 200, 300)}]


def test_field_locations_capped_per_field():
    locs = [{"field": "id_number", "page": 1, "box_2d": [i, 0, i + 10, 10]} for i in range(5)]
    assert len(_ext(locations=locs).field_locations) == 2


def test_missing_locations_and_status_are_tolerated():
    data = _model_json()
    del data["field_locations"]
    del data["id_number"]["status"]
    ext = ee.parse_extraction(data)
    assert ext.field_locations == [] and ext.fields["id_number"].status is None


def test_unreadable_is_distinguished_from_missing():
    rec = _rec(_ext(id_number={"value": None, "confidence": 0.2, "status": "unreadable"},
                    commitment_numbers={"value": [], "confidence": 0.1, "status": "not_present"}), None)
    assert rec.flags["id_number"] == "unreadable"
    assert rec.flags["commitment_numbers"] == "missing"


# ---- selection and crops ----------------------------------------------------------------

def test_select_only_flagged_and_located_fields():
    ext = _ext(id_number={"value": INVALID_ID, "confidence": 0.95},
               central_numbers={"value": ["937309"], "confidence": 0.5},
               received_date={"value": "01-06-2026", "confidence": 0.9, "received_date_source": "none"},
               locations=[{"field": "id_number", "page": 1, "box_2d": [1, 1, 50, 50]},
                          {"field": "received_date", "page": 1, "box_2d": [1, 1, 50, 50]}])
    flags = rb.field_flags(ext, today=TODAY)[1]
    assert flags["received_date"] == "date_source_rejected"
    # central is flagged but has no box; the date's source rule is never re-litigated
    assert fr.select_reread_fields(ext, flags) == ["id_number"]


def test_no_reread_after_extraction_error():
    assert fr.select_reread_fields(ee.empty_result("boom"), {"id_number": "extraction_failed"}) == []


def test_crops_near_image_boundaries_are_clamped():
    page = Image.new("RGB", (1000, 1400), "white")
    locs = [{"field": "id_number", "page": 1, "box": (0, 0, 20, 30)},                  # top-left corner
            {"field": "central_numbers", "page": 1, "box": (980, 970, 1000, 1000)}]    # bottom-right corner
    crops, skipped = fr.crop_field_regions([page], locs, ["id_number", "central_numbers"])
    assert skipped == {} and len(crops) == 2
    for _, crop in crops:
        assert 10 <= crop.size[0] <= 1000 and 10 <= crop.size[1] <= 1400
    assert crops[1][1].size == (30 + 60, round(0.02 * 1400) + 60)  # padded only towards the inside


def test_unusable_boxes_are_skipped_with_a_reason():
    page = Image.new("RGB", (1000, 1400), "white")
    locs = [{"field": "id_number", "page": 2, "box": (0, 0, 100, 100)},          # page does not exist
            {"field": "central_numbers", "page": 1, "box": (0, 0, 900, 900)},    # most of the page
            {"field": "commitment_numbers", "page": 1, "box": (10, 10, 20, 20)}]
    crops, skipped = fr.crop_field_regions([page], locs, ["id_number", "central_numbers", "commitment_numbers",
                                                           "received_date"])
    assert [k for k, _ in crops] == ["commitment_numbers"]
    assert skipped == {"id_number": "page_out_of_range", "central_numbers": "box_too_large",
                       "received_date": "no_location"}


def test_crops_never_modify_the_page_images():
    page = Image.new("RGB", (800, 1000), "white")
    ImageDraw.Draw(page).rectangle([100, 100, 300, 140], fill="black")
    before = page.tobytes()
    fr.crop_field_regions([page], [{"field": "id_number", "page": 1, "box": (90, 100, 150, 400)}], ["id_number"])
    assert page.tobytes() == before


# ---- combining crop readings ----------------------------------------------------------------

def test_list_field_union_keeps_every_number():
    t = fr.combine_readings("central_numbers", [_reading("central_numbers", ["937309", "937310"]),
                                                _reading("central_numbers", ["937310", "937311"], conf=0.85)])
    assert t.status == "read" and t.value == ["937309", "937310", "937311"] and t.confidence == 0.85


def test_list_field_with_one_unreadable_crop_is_unusable():
    t = fr.combine_readings("central_numbers", [_reading("central_numbers", ["937309"]),
                                                _reading("central_numbers", [], status="unreadable")])
    assert t.status == "unreadable" and t.value is None


def test_scalar_crops_that_disagree_are_inconsistent():
    t = fr.combine_readings("id_number", [_reading("id_number", [VALID_ID]), _reading("id_number", [OTHER_VALID_ID])])
    assert t.status == "inconsistent"
    assert fr.combine_readings("id_number", [_reading("id_number", [VALID_ID, OTHER_VALID_ID])]).status == "inconsistent"


@pytest.mark.parametrize("statuses, expected", [
    (["not_in_crop"], "box_miss"),
    (["no_answer"], "failed"),
    (["unreadable", "not_in_crop"], "unreadable"),
])
def test_scalar_not_read(statuses, expected):
    assert fr.combine_readings("id_number", [_reading("id_number", [], status=s) for s in statuses]).status == expected


# ---- reread_fields with a fake model ----------------------------------------------------------

def test_reread_is_one_request_and_never_sees_the_page_value(monkeypatch):
    calls = []

    def fake_generate(contents, schema, **kwargs):
        calls.append((contents, kwargs))
        return {"crops": [{"crop_index": 1, "status": "read", "values": [VALID_ID], "confidence": 0.9,
                           "received_date_source": "none"},
                          {"crop_index": 2, "status": "read", "values": [], "confidence": 0.9,   # read but empty
                           "received_date_source": "none"}]}                                  # crop 3: no answer

    monkeypatch.setattr(ee, "generate_json", fake_generate)
    ext = _ext(id_number={"value": "123456789", "confidence": 0.5},
               locations=[{"field": "id_number", "page": 1, "box_2d": [100, 100, 150, 400]},
                          {"field": "central_numbers", "page": 1, "box_2d": [500, 500, 600, 700]},
                          {"field": "central_numbers", "page": 1, "box_2d": [700, 500, 800, 700]}])
    recheck = fr.reread_fields([Image.new("RGB", (1000, 1400), "white")], ext, ["id_number", "central_numbers"])
    assert len(calls) == 1
    contents, kwargs = calls[0]
    assert kwargs["purpose"] == "reread_fields" and "media_resolution" not in kwargs
    texts = " ".join(c for c in contents if isinstance(c, str))
    assert "123456789" not in texts and "937309" not in texts   # independent reading
    assert sum(1 for c in contents if not isinstance(c, str)) == 3
    assert [r.status for r in recheck.readings] == ["read", "unreadable", "no_answer"]
    assert recheck.result("id_number").value == VALID_ID
    assert recheck.result("central_numbers").status == "unreadable"


def test_failed_reread_request_is_recorded(monkeypatch):
    def boom(*a, **k):
        raise ee.LLMError("Claude API error 500")

    monkeypatch.setattr(ee, "generate_json", boom)
    recheck = fr.reread_fields([Image.new("RGB", (1000, 1400))], _ext(), ["id_number"])
    assert recheck.error and recheck.result("id_number").status == "failed"


def test_quota_error_is_raised(monkeypatch):
    def quota(*a, **k):
        raise ee.LLMQuotaError("quota")

    monkeypatch.setattr(ee, "generate_json", quota)
    with pytest.raises(ee.LLMQuotaError):
        fr.reread_fields([Image.new("RGB", (1000, 1400))], _ext(), ["id_number"])


def test_nothing_to_reread_makes_no_request(monkeypatch):
    monkeypatch.setattr(ee, "generate_json", lambda *a, **k: pytest.fail("no call expected"))
    assert fr.reread_fields([Image.new("RGB", (100, 100))], _ext(), []).attempted == []
    recheck = fr.reread_fields([Image.new("RGB", (100, 100))], _ext(locations=[]), ["id_number"])
    assert recheck.attempted == [] and recheck.skipped == {"id_number": "no_location"}


# ---- decisions ----------------------------------------------------------------------------

def test_failed_targeted_extraction_changes_nothing():
    ext = _ext(id_number={"value": INVALID_ID, "confidence": 0.95})
    rec = _rec(ext, fr.FieldRecheck(attempted=["id_number"], error="Claude API error 500"))
    assert rec.values["id_number"] == INVALID_ID and rec.flags["id_number"] == "invalid_id"
    assert rec.diagnostics["fields"]["id_number"]["reread"] == "failed"


def test_low_confidence_number_that_agrees_stays_flagged():
    ext = _ext(central_numbers={"value": ["937309"], "confidence": 0.6})
    rec = _rec(ext, _recheck(_reading("central_numbers", ["937309"])))
    assert rec.values["central_numbers"] == "937309"
    assert rec.flags["central_numbers"] == "low_confidence"     # no strong check for central numbers
    assert rec.diagnostics["fields"]["central_numbers"]["reread"] == "agreed"


def test_low_confidence_id_confirmed_by_checksum_and_agreement():
    ext = _ext(id_number={"value": VALID_ID, "confidence": 0.6})
    rec = _rec(ext, _recheck(_reading("id_number", [VALID_ID[1:]])))   # same number without the leading zero
    assert rec.values["id_number"] == VALID_ID and "id_number" not in rec.flags


def test_low_confidence_reread_does_not_confirm_id():
    rec = _rec(_ext(id_number={"value": VALID_ID, "confidence": 0.6}),
               _recheck(_reading("id_number", [VALID_ID], conf=0.6)))
    assert rec.flags["id_number"] == "low_confidence"


@pytest.mark.parametrize("orig", [INVALID_ID, "0011223344"])   # bad check digit / 10 digits
def test_id_failing_check_digit_replaced_by_valid_crop_reading(orig):
    rec = _rec(_ext(id_number={"value": orig, "confidence": 0.9}), _recheck(_reading("id_number", [VALID_ID])))
    assert rec.values["id_number"] == VALID_ID
    assert rec.flags["id_number"] == "reread_replaced"           # still flagged for review
    assert rec.alternatives["id_number"] == [orig]               # the full-page reading stays visible
    assert rec.diagnostics["fields"]["id_number"]["reread"] == "replaced"


@pytest.mark.parametrize("crop, expected_value", [
    (OTHER_VALID_ID, ""),         # two valid, different readings: emptied (conflict), never swapped
    (INVALID_ID, VALID_ID),       # crop fails the check digit: the valid ID stays
])
def test_id_passing_check_digit_is_never_replaced(crop, expected_value):
    rec = _rec(_ext(id_number={"value": VALID_ID, "confidence": 0.6}), _recheck(_reading("id_number", [crop])))
    assert rec.values["id_number"] == expected_value and rec.values["id_number"] != crop
    assert rec.flags["id_number"] == "reread_conflict"
    assert crop in rec.alternatives["id_number"]


def test_confident_valid_id_is_not_selected_for_reread():
    ext = _ext(id_number={"value": VALID_ID, "confidence": 0.95})
    assert "id_number" not in fr.select_reread_fields(ext, rb.field_flags(ext, today=TODAY)[1])


@pytest.mark.parametrize("reading", [
    _reading("id_number", ["000000027"]),                # crop also fails the check digit
    _reading("id_number", [VALID_ID], conf=0.6),          # crop passes, but not confidently
])
def test_id_not_replaced_without_a_confident_valid_crop_reading(reading):
    rec = _rec(_ext(id_number={"value": INVALID_ID, "confidence": 0.9}), _recheck(reading))
    assert rec.values["id_number"] == INVALID_ID and rec.flags["id_number"] in ("invalid_id", "reread_conflict")
    assert rec.alternatives["id_number"] == reading.values


def test_id_never_replaced_by_a_central_or_commitment_number():
    ext = _ext(id_number={"value": INVALID_ID, "confidence": 0.9},
               commitment_numbers={"value": [VALID_ID], "confidence": 0.95})
    rec = _rec(ext, _recheck(_reading("id_number", [VALID_ID])))
    assert rec.values["id_number"] == INVALID_ID and "id_number" in rec.flags


def test_invalid_date_replaced_by_valid_crop_reading_but_still_flagged():
    ext = _ext(received_date={"value": "31-02-2026", "confidence": 0.9, "received_date_source": "received_stamp"})
    rec = _rec(ext, _recheck(_reading("received_date", ["21-02-2026"])))
    assert rec.values["received_date"] == "21/02/2026" and rec.flags["received_date"] == "reread_replaced"
    assert rec.alternatives["received_date"] == ["31-02-2026"]   # the original reading stays visible


def test_disagreement_between_two_valid_readings_empties_the_field():
    rec = _rec(_ext(id_number={"value": VALID_ID, "confidence": 0.6}),
               _recheck(_reading("id_number", [OTHER_VALID_ID])))
    assert rec.values["id_number"] == "" and rec.flags["id_number"] == "reread_conflict"
    assert rec.alternatives["id_number"] == [VALID_ID, OTHER_VALID_ID]


def test_validated_value_contradicted_by_weak_reading_is_kept_but_flagged():
    rec = _rec(_ext(central_numbers={"value": ["937309"], "confidence": 0.6}),
               _recheck(_reading("central_numbers", ["93730"])))           # 5 digits: fails the format
    assert rec.values["central_numbers"] == "937309"
    assert rec.flags["central_numbers"] == "reread_conflict"


def test_missing_field_filled_only_with_valid_confident_reading():
    ext = _ext(commitment_numbers={"value": [], "confidence": 0.1, "status": "unreadable"})
    rec = _rec(ext, _recheck(_reading("commitment_numbers", ["50364186"])))
    assert rec.values["commitment_numbers"] == "50364186"
    assert rec.flags["commitment_numbers"] == "reread_single_reading"

    weak = _rec(ext, _recheck(_reading("commitment_numbers", ["50364186"], conf=0.5)))
    assert weak.values["commitment_numbers"] == "" and weak.flags["commitment_numbers"] == "unreadable"
    assert weak.alternatives["commitment_numbers"] == ["50364186"]


def test_missing_field_with_unreadable_crop_is_flagged_unreadable():
    ext = _ext(central_numbers={"value": [], "confidence": 0.1, "status": "not_present"})
    rec = _rec(ext, _recheck(_reading("central_numbers", [], status="unreadable")))
    assert rec.values["central_numbers"] == "" and rec.flags["central_numbers"] == "unreadable"


def test_reread_that_drops_a_number_never_replaces_the_list():
    ext = _ext(central_numbers={"value": ["937309", "93731"], "confidence": 0.9})   # second one invalid
    rec = _rec(ext, _recheck(_reading("central_numbers", ["937309"])))
    assert rec.values["central_numbers"] == "937309, 93731"          # both original numbers kept
    assert rec.flags["central_numbers"] == "reread_conflict"
    assert rec.alternatives["central_numbers"] == ["937309"]


def test_reread_that_finds_more_numbers_keeps_all_of_them():
    ext = _ext(central_numbers={"value": ["937309"], "confidence": 0.6})
    rec = _rec(ext, _recheck(_reading("central_numbers", ["937309", "937310", "937311"])))
    assert rec.values["central_numbers"] == "937309, 937310, 937311"
    assert rec.flags["central_numbers"] == "reread_replaced"


def test_invalid_list_replaced_by_complete_valid_list():
    ext = _ext(central_numbers={"value": ["937309", "93731"], "confidence": 0.9})
    rec = _rec(ext, _recheck(_reading("central_numbers", ["937309", "937311"])))
    assert rec.values["central_numbers"] == "937309, 937311" and rec.flags["central_numbers"] == "reread_replaced"


def test_crop_reading_of_another_number_is_never_accepted_as_the_id():
    ext = _ext(id_number={"value": None, "confidence": 0.1, "status": "unreadable"},
               commitment_numbers={"value": [VALID_ID], "confidence": 0.95})
    rec = _rec(ext, _recheck(_reading("id_number", [VALID_ID])))
    assert rec.values["id_number"] == "" and rec.flags["id_number"] == "unreadable"


def test_received_date_from_wrong_source_is_never_accepted():
    ext = _ext(received_date={"value": None, "confidence": 0.1, "received_date_source": "none",
                              "status": "unreadable"})
    rec = _rec(ext, _recheck(_reading("received_date", ["02-06-2026"], source="none")))
    assert rec.values["received_date"] == "" and rec.flags["received_date"] == "unreadable"
    ok = _rec(ext, _recheck(_reading("received_date", ["02-06-2026"], source="handwritten_near_central")))
    assert ok.values["received_date"] == "02/06/2026" and ok.flags["received_date"] == "reread_single_reading"


def test_reread_commitment_equal_to_id_is_still_removed():
    ext = _ext(commitment_numbers={"value": [], "confidence": 0.1, "status": "unreadable"})
    rec = _rec(ext, _recheck(_reading("commitment_numbers", [VALID_ID[2:]])))
    assert rec.values["commitment_numbers"] == "" and rec.flags["commitment_numbers"] == "commitment_is_id"


@pytest.mark.parametrize("orig, reading", [
    ({"value": INVALID_ID, "confidence": 0.9}, _reading("id_number", [], status="unreadable")),
    ({"value": INVALID_ID, "confidence": 0.9}, _reading("id_number", [], status="not_in_crop")),
    ({"value": VALID_ID, "confidence": 0.5}, _reading("id_number", [OTHER_VALID_ID])),
    ({"value": None, "confidence": 0.1}, _reading("id_number", [INVALID_ID])),
    ({"value": INVALID_ID, "confidence": 0.9}, _reading("id_number", [VALID_ID])),
])
def test_unresolved_or_changed_values_always_remain_flagged(orig, reading):
    rec = _rec(_ext(id_number=orig), _recheck(reading))
    assert "id_number" in rec.flags and rec.needs_review


def test_apply_never_mutates_the_extraction():
    ext = _ext(id_number={"value": INVALID_ID, "confidence": 0.9})
    rb.apply_field_recheck(ext, _recheck(_reading("id_number", [VALID_ID])), today=TODAY)
    assert ext.fields["id_number"].value == INVALID_ID


# ---- diagnostics ----------------------------------------------------------------------------

def test_diagnostics_explain_flags_without_values():
    ext = _ext(id_number={"value": INVALID_ID, "confidence": 0.9}, central_numbers={"value": ["937309"],
                                                                                     "confidence": 0.6})
    recheck = _recheck(_reading("id_number", [VALID_ID]), _reading("central_numbers", ["937309"]))
    rec = rb.build_record(_form(), ext, recheck=recheck, today=TODAY,
                          page_info=[{"page": 3, "size": [2480, 3508], "steps": ["autocontrast"]}])
    d = rec.diagnostics["fields"]
    assert d["id_number"]["reread"] == "replaced" and d["id_number"]["flag"] == "reread_replaced"
    assert d["central_numbers"]["reread"] == "agreed" and d["central_numbers"]["crops"] == 1
    text = "\n".join(rb.diagnostic_lines(rec))
    assert "size=2480x3508 preprocessing=autocontrast" in text and "id_number: flag=reread_replaced" in text
    for secret in (VALID_ID, INVALID_ID, "937309", "ישראל", "batch.pdf"):
        assert secret not in text
    assert "batch.pdf p3" in "\n".join(rb.diagnostic_lines(rec, include_source=True))


def test_diagnostics_without_reread():
    rec = _rec(_ext(id_number={"value": INVALID_ID, "confidence": 0.9}), None)
    assert rec.diagnostics["fields"]["id_number"]["reread"] == "not_run"


# ---- preprocessing: originals kept, faint strokes survive ------------------------------------------

def _dark_page_with_sticker():
    img = Image.new("RGB", (1000, 1400), (120, 120, 120))
    draw = ImageDraw.Draw(img)
    for y in range(100, 1300, 40):
        draw.rectangle([100, y, 900, y + 10], fill=(20, 20, 20))
    draw.rectangle([600, 600, 900, 800], fill=(230, 230, 230))    # white sticker
    draw.rectangle([650, 680, 850, 700], fill=(175, 175, 175))    # faint pencil digits on it
    return img


def test_brightening_keeps_faint_strokes_on_white_stickers():
    result = ip.preprocess(_dark_page_with_sticker())
    gray = result.processed.convert("L")
    assert any(s.startswith("brighten") for s in result.steps)
    assert gray.getpixel((620, 620)) - gray.getpixel((750, 690)) >= 20   # linear x1.6 left 0


def test_aligned_original_has_processed_geometry_and_original_tones():
    page = _dark_page_with_sticker().rotate(3, expand=True, fillcolor=(120, 120, 120))
    before = page.tobytes()
    result = ip.preprocess(page)
    assert any(s.startswith("deskew") for s in result.steps)
    assert result.aligned.size == result.processed.size            # boxes map 1:1
    assert page.tobytes() == before and result.original is page     # original untouched
    # Original tones: still the dark scan, not the brightened one.
    assert ip.mean_luminance(result.aligned) < ip.mean_luminance(result.processed) - 20


def test_crop_source_is_original_when_not_deskewed():
    page = Image.new("RGB", (400, 500), "white")
    assert ip.preprocess(page).crop_source is page


# ---- pipeline wiring ------------------------------------------------------------------------

def _png_bytes():
    from io import BytesIO

    buf = BytesIO()
    Image.new("RGB", (800, 1000), "white").save(buf, "PNG")
    return buf.getvalue()


@pytest.mark.parametrize("enabled, expected_calls", [(True, 1), (False, 0)])
def test_process_files_rereads_once_per_flagged_form(monkeypatch, tmp_path, enabled, expected_calls):
    from src import config

    monkeypatch.setattr(config, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(config, "FIELD_REREAD_ENABLED", enabled)
    calls = {"reread": 0}

    def fake_generate(contents, schema, **kwargs):
        if "upright" in schema["properties"]:
            return {"upright": "A", "confidence": 0.99}
        if "pages" in schema["properties"]:
            return {"pages": [{"page_index": 1, "is_blank": False, "starts_new_form": True, "confidence": 0.9}]}
        if kwargs.get("purpose") == "reread_fields":
            calls["reread"] += 1
            return {"crops": [{"crop_index": 1, "status": "read", "values": [VALID_ID], "confidence": 0.95,
                               "received_date_source": "none"}]}
        return _model_json(id_number={"value": INVALID_ID, "confidence": 0.9})

    monkeypatch.setattr(ee, "generate_json", fake_generate)
    records = rb.process_files([("a.png", _png_bytes())])
    assert calls["reread"] == expected_calls
    assert records[0].values["id_number"] == (VALID_ID if enabled else INVALID_ID)
    assert records[0].alternatives == ({"id_number": [INVALID_ID]} if enabled else {})
    assert "id_number" in records[0].flags
    assert not list(tmp_path.iterdir())


def test_quota_during_reread_keeps_the_extraction(monkeypatch, tmp_path):
    from src import config

    monkeypatch.setattr(config, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(config, "FIELD_REREAD_ENABLED", True)

    def fake_generate(contents, schema, **kwargs):
        if "upright" in schema["properties"]:
            return {"upright": "A", "confidence": 0.99}
        if "pages" in schema["properties"]:
            return {"pages": [{"page_index": 1, "is_blank": False, "starts_new_form": True, "confidence": 0.9}]}
        if kwargs.get("purpose") == "reread_fields":
            raise ee.LLMQuotaError("quota")
        return _model_json(id_number={"value": INVALID_ID, "confidence": 0.9})

    monkeypatch.setattr(ee, "generate_json", fake_generate)
    rec = rb.process_files([("a.png", _png_bytes())])[0]
    assert rec.error is None and rec.values["sending_entity"] == "מכבי"
    assert rec.values["id_number"] == INVALID_ID and rec.flags["id_number"] == "invalid_id"
    assert rec.diagnostics["reread_error"] is True
