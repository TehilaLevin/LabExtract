from datetime import date
from pathlib import Path

import pytest

from src import extraction_engine as ee
from src import record_builder as rb
from src.constants import FIELD_KEYS, NEEDS_REVIEW_KEY
from src.file_converter import Page
from src.form_splitter import Form

TODAY = date(2026, 10, 13)
VALID_ID = "000000018"


def _form(split_flags=None):
    return Form(index=1, source_file="batch.pdf", pages=[Page("batch.pdf", 3, Path("x.jpg"))],
                rotations=[0], split_flags=split_flags or [])


def _model_json(**overrides):
    data = {
        "form_type": {"value": "maccabi_lab_request", "confidence": 0.97},
        "first_name": {"value": "ישראל", "confidence": 0.9},
        "last_name": {"value": "ישראלי", "confidence": 0.9},
        "name_order_basis": "separate_fields",
        "id_number": {"value": VALID_ID, "confidence": 0.95},
        "central_numbers": {"value": ["937309", "937310", "937311"], "confidence": 0.95},
        "received_date": {"value": "01-06-2026", "confidence": 0.9, "received_date_source": "received_stamp"},
        "sending_entity": {"value": "מכבי", "confidence": 0.95},
        "commitment_numbers": {"value": ["50364186"], "confidence": 0.95},
    }
    data.update(overrides)
    return data


def _record(**overrides):
    return rb.build_record(_form(), ee.parse_extraction(_model_json(**overrides)), today=TODAY)


# ---- parse_extraction ----------------------------------------------------------

def test_parse_extraction_tolerates_garbage():
    res = ee.parse_extraction({"form_type": "x", "id_number": {"value": "  ", "confidence": "sure"},
                               "central_numbers": {"value": "939376", "confidence": 2}})
    assert res.form_type == "unknown"
    assert res.fields["id_number"].value is None
    assert res.fields["id_number"].confidence == 0.0
    assert res.fields["central_numbers"].value == ["939376"]
    assert res.fields["central_numbers"].confidence == 1.0
    assert res.fields["first_name"].value is None
    assert res.fields["received_date"].source == "none"


def test_parse_extraction_list_cleanup():
    res = ee.parse_extraction({"commitment_numbers": {"value": ["2648 00879", "", None, "2648 00879"], "confidence": 0.7}})
    assert res.fields["commitment_numbers"].value == ["264800879"]


def test_schema_restricts_sender_to_closed_list():
    schema = ee.build_schema(["כללית", "מכבי"])
    sender = schema["properties"]["sending_entity"]["properties"]["value"]
    assert sender["enum"] == ["כללית", "מכבי"] and sender["nullable"] is True
    assert schema["properties"]["central_numbers"]["properties"]["value"]["type"] == "ARRAY"
    assert set(FIELD_KEYS) <= set(schema["required"])


def test_prompt_contains_rules_and_senders():
    prompt = ee.load_prompt(["כללית", "רבין"])
    assert "- רבין" in prompt and "{{" not in prompt
    for must in ("NEVER GUESS", "GLC", "התקבל", "דרישה", "דגימה", "ADMISSION"):
        assert must in prompt


# ---- build_record --------------------------------------------------------------

def test_clean_record_has_no_flags_and_lists_are_comma_separated():
    rec = _record()
    assert rec.flags == {}
    assert not rec.needs_review
    assert rec.values["central_numbers"] == "937309, 937310, 937311"
    assert rec.values["received_date"] == "01/06/2026"
    assert rec.metadata == {"source_file": "batch.pdf", "page_numbers": "3", "form_type": "מכבי – מעבדה שולחת"}


def test_missing_and_low_confidence_are_flagged():
    rec = _record(commitment_numbers={"value": [], "confidence": 0.9},
                  first_name={"value": "ישראל", "confidence": 0.55})
    assert rec.flags == {"commitment_numbers": "missing", "first_name": "low_confidence"}
    assert rec.values["first_name"] == "ישראל"  # low confidence: kept, but orange
    assert rec.needs_review


def test_invalid_id_is_kept_and_flagged():
    rec = _record(id_number={"value": "123456789", "confidence": 0.99})
    assert rec.flags == {"id_number": "invalid_id"}
    assert rec.values["id_number"] == "123456789"


def test_received_date_from_unapproved_source_is_emptied():
    rec = _record(received_date={"value": "21/08/2026", "confidence": 0.99, "received_date_source": "none"})
    assert rec.values["received_date"] == ""
    assert rec.flags == {"received_date": "date_source_rejected"}


def test_handwritten_near_central_date_is_accepted():
    rec = _record(received_date={"value": "16/6/26", "confidence": 0.85,
                                 "received_date_source": "handwritten_near_central"})
    assert rec.values["received_date"] == "16/06/2026" and "received_date" not in rec.flags


@pytest.mark.parametrize("raw, reason, shown", [
    ("31/02/2026", "invalid_date", ""),
    ("18-06-20", "date_out_of_range", "18/06/2020"),
    ("01/12/2026", "date_out_of_range", "01/12/2026"),
    ("1?/08/2026", "invalid_date", ""),
])
def test_bad_received_dates(raw, reason, shown):
    rec = _record(received_date={"value": raw, "confidence": 0.95, "received_date_source": "received_stamp"})
    assert rec.flags == {"received_date": reason}
    assert rec.values["received_date"] == shown


def test_sender_outside_closed_list_is_emptied():
    rec = _record(sending_entity={"value": "קליניקה פרטית", "confidence": 0.99})
    assert rec.values["sending_entity"] == ""
    assert rec.flags == {"sending_entity": "sender_not_in_list"}


def test_hospital_form_label_includes_sender():
    rec = _record(form_type={"value": "hospital_form", "confidence": 0.9},
                  sending_entity={"value": "רמב\"ם", "confidence": 0.95})
    assert rec.metadata["form_type"] == 'טופס בית חולים – רמב"ם'


def test_number_format_checks():
    rec = _record(central_numbers={"value": ["93730", "937310"], "confidence": 0.95},
                  commitment_numbers={"value": ["X132767", "12-34"], "confidence": 0.95})
    assert rec.flags == {"central_numbers": "invalid_format", "commitment_numbers": "invalid_format"}
    assert rec.values["commitment_numbers"] == "X132767, 12-34"


@pytest.mark.parametrize("basis, flagged", [
    ("separate_fields", False),
    ("form_convention", False),  # e.g. Sheba printout: last name first, known
    ("unclear", True),
    (None, True),                # missing / unknown basis is treated as unclear
    ("guess", True),
])
def test_names_flagged_only_when_order_unclear(basis, flagged):
    rec = _record(name_order_basis=basis)
    expected = {"first_name": "name_order_unclear", "last_name": "name_order_unclear"} if flagged else {}
    assert rec.flags == expected
    assert rec.values["first_name"] == "ישראל"  # the split is still shown, only flagged


def test_low_reading_confidence_wins_over_known_order():
    rec = _record(name_order_basis="form_convention", last_name={"value": "ישראלי", "confidence": 0.5})
    assert rec.flags == {"last_name": "low_confidence"}


def test_unclear_order_with_missing_name_reports_missing():
    rec = _record(name_order_basis="unclear", first_name={"value": None, "confidence": 0.2})
    assert rec.flags == {"first_name": "missing", "last_name": "name_order_unclear"}


def test_prompt_lists_known_name_orders():
    from src.constants import NAME_ORDER_CONVENTIONS

    prompt = ee.load_prompt(["כללית"])
    assert "name_order_basis" in prompt and "<= 0.6 for both" not in prompt
    for where, order in NAME_ORDER_CONVENTIONS:
        assert f"- {where} => {order}" in prompt
    assert any(w.startswith("sheba_system_printout") and o.startswith("LAST") for w, o in NAME_ORDER_CONVENTIONS)


def test_schema_has_name_order_basis():
    from src.constants import NAME_ORDER_BASES

    schema = ee.build_schema(["כללית"])
    assert schema["properties"]["name_order_basis"]["enum"] == list(NAME_ORDER_BASES)
    assert "name_order_basis" in schema["required"]


def test_name_with_digits_is_flagged():
    assert _record(last_name={"value": "ישראלי2", "confidence": 0.99}).flags == {"last_name": "invalid_format"}


def test_extraction_error_flags_every_field():
    rec = rb.build_record(_form(), ee.empty_result("Claude API error 500"), today=TODAY)
    assert set(rec.flags) == set(FIELD_KEYS)
    assert set(rec.flags.values()) == {"extraction_failed"}
    assert rec.error and rec.needs_review


def test_split_flag_marks_record_for_review():
    rec = rb.build_record(_form(["split_uncertain"]), ee.parse_extraction(_model_json()), today=TODAY)
    assert rec.flags == {} and rec.needs_review


def test_dataframe_one_row_per_form():
    df = rb.records_to_dataframe([_record(), _record(id_number={"value": None, "confidence": 0})])
    assert len(df) == 2
    assert list(df[NEEDS_REVIEW_KEY]) == [False, True]
    assert rb.count_flagged_fields([_record(), _record(id_number={"value": None, "confidence": 0})]) == 1


# ---- process_files (fake model, no API calls) ------------------------------------

@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "TEMP_DIR", tmp_path)
    return tmp_path


def _png_bytes():
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (80, 100), "white").save(buf, "PNG")
    return buf.getvalue()


def test_process_files_end_to_end_and_cleans_temp(monkeypatch, temp_dir):
    def fake_generate(contents, schema, **kwargs):
        if "upright" in schema["properties"]:  # orientation request
            return {"upright": "A", "confidence": 0.99}
        if "pages" in schema["properties"]:  # split request
            n = sum(1 for c in contents if isinstance(c, str) and c.startswith("PAGE "))
            return {"pages": [{"page_index": i, "is_blank": False, "starts_new_form": True, "confidence": 0.9}
                              for i in range(1, n + 1)]}
        return _model_json()

    monkeypatch.setattr(ee, "generate_json", fake_generate)
    stages = []
    records = rb.process_files([("a.png", _png_bytes()), ("b.png", _png_bytes())],
                               progress=lambda s, d, t: stages.append((s, d, t)))
    assert len(records) == 2
    assert records[0].values["sending_entity"] == "מכבי"
    assert ("extract", 2, 2) in stages
    assert not [p for p in temp_dir.iterdir()]  # run folder deleted


def test_process_files_cleans_temp_on_failure(monkeypatch, temp_dir):
    def boom(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr("src.form_splitter.split_forms", boom)
    with pytest.raises(RuntimeError):
        rb.process_files([("a.png", _png_bytes())])
    assert not [p for p in temp_dir.iterdir()]


def test_quota_error_stops_further_requests(monkeypatch, temp_dir):
    calls = {"extract": 0}

    def fake_generate(contents, schema, **kwargs):
        if "upright" in schema["properties"]:
            return {"upright": "A", "confidence": 0.99}
        if "pages" in schema["properties"]:
            return {"pages": [{"page_index": 1, "is_blank": False, "starts_new_form": True, "confidence": 0.9}]}
        calls["extract"] += 1
        raise ee.LLMQuotaError("API quota exhausted")

    monkeypatch.setattr(ee, "generate_json", fake_generate)
    monkeypatch.setattr("src.config.MAX_PARALLEL_REQUESTS", 1)
    records = rb.process_files([(f"{i}.png", _png_bytes()) for i in range(4)])
    assert calls["extract"] == 1
    assert all(r.error for r in records)


# ---- dev-only response cache --------------------------------------------------------

@pytest.fixture
def fake_client(fake_claude):
    """Claude with a fake SDK client (tests/conftest.py); .calls counts the requests that left the app."""
    return fake_claude


def _contents(color="white"):
    from PIL import Image

    return ["prompt", ee.image_part(Image.new("RGB", (20, 20), color), 100)]


def test_dev_cache_off_by_default(fake_client, monkeypatch):
    from src import config

    monkeypatch.delenv("DEV_CACHE", raising=False)
    assert config.DEV_CACHE in (False, True)  # value comes from .env; force off below
    monkeypatch.setattr(config, "DEV_CACHE", False)
    ee.generate_json(_contents(), {"type": "OBJECT"})
    ee.generate_json(_contents(), {"type": "OBJECT"})
    assert fake_client.calls == 2
    assert not config.DEV_CACHE_DIR.exists()


def test_dev_cache_hits_same_images_and_misses_changed_ones(fake_client, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "DEV_CACHE", True)
    assert ee.generate_json(_contents(), {"type": "OBJECT"}) == {"ok": True}
    assert ee.generate_json(_contents(), {"type": "OBJECT"}) == {"ok": True}
    assert fake_client.calls == 1                       # second call served from disk
    ee.generate_json(_contents("black"), {"type": "OBJECT"})
    assert fake_client.calls == 2                       # different image -> new request
    ee.generate_json(_contents(), {"type": "OBJECT"}, system_instruction="changed prompt")
    assert fake_client.calls == 3                       # prompt change invalidates
    monkeypatch.setattr(config, "CLAUDE_MODEL", "claude-sonnet-5-5")
    ee.generate_json(_contents(), {"type": "OBJECT"})
    assert fake_client.calls == 4                       # another model -> new request
    assert ee.clear_dev_cache() == 4


def test_token_usage_is_counted_for_live_calls_only(fake_client, monkeypatch):
    from src import config
    from tests.conftest import fake_reply

    fake_client.reply = fake_reply(input_tokens=200, output_tokens=25, cache_read=100)
    monkeypatch.setattr(config, "DEV_CACHE", True)
    ee.reset_usage()
    ee.generate_json(_contents(), {"type": "OBJECT"})
    ee.generate_json(_contents(), {"type": "OBJECT"})  # cache hit: not counted
    assert ee.get_usage() == {"requests": 1, "prompt_tokens": 300, "output_tokens": 25}


def test_invalid_json_is_retried_once_then_fails(fake_client):
    from tests.conftest import fake_reply

    fake_client.reply = fake_reply(text='{"ok": tr')
    with pytest.raises(ee.LLMError, match="invalid JSON"):
        ee.generate_json(_contents(), {"type": "OBJECT"})
    assert fake_client.calls == 2


def test_sdk_client_retries_with_backoff_and_times_out(monkeypatch):
    """Retries (429/529/5xx/connection, exponential backoff) and timeouts are the SDK's: check the settings."""
    import anthropic

    from src import claude_client, config

    seen = {}
    monkeypatch.setattr(claude_client, "_client", None)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: seen.update(kw) or object())
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-test")
    claude_client.anthropic_client()
    assert seen["max_retries"] == config.CLAUDE_MAX_RETRIES >= 2 and seen["timeout"] == config.CLAUDE_TIMEOUT_S
    monkeypatch.setattr(claude_client, "_client", None)


# ---- handwritten-name safeguards -------------------------------------------------

def _hand(first="ישראל", last="ישראלי", conf=0.9, basis="separate_fields", script="handwritten", locations=None):
    data = _model_json(first_name={"value": first, "confidence": conf, "script": script},
                       last_name={"value": last, "confidence": conf, "script": script},
                       name_order_basis=basis)
    data["name_locations"] = locations if locations is not None else [
        {"page": 1, "box_2d": [300, 100, 340, 900], "kind": "labeled_fields"}]
    return ee.parse_extraction(data)


def _reading(full="ישראל ישראלי", first=None, last=None, assignment="single_field", legible=True, conf=0.95,
             kind="labeled_fields"):
    return ee.NameReading(kind, legible, full, assignment, first, last, conf)


def _check(*readings, error=None):
    return ee.NameCheck(needed=True, readings=list(readings), error=error)


def _rec(ext, check):
    return rb.build_record(_form(), ext, check, today=TODAY)


def test_parse_name_script_and_locations():
    ext = ee.parse_extraction({
        "first_name": {"value": "ישראל", "confidence": 0.9, "script": "handwritten"},
        "last_name": {"value": "ישראלי", "confidence": 0.9},
        "name_locations": [
            {"page": 1, "box_2d": [100, 200, 150, 800], "kind": "patient_sticker"},
            {"page": 1, "box_2d": [150, 200, 100, 800], "kind": "other"},   # ymin > ymax
            {"page": 0, "box_2d": [1, 2, 3, 4], "kind": "other"},           # bad page
            {"page": 1, "box_2d": [1, 2, 3], "kind": "other"},              # 3 numbers
            {"page": 2, "box_2d": [0, 0, 1000, 1000], "kind": "weird"},
            {"page": 1, "box_2d": [5, 5, 50, 50], "kind": "single_field"},
            {"page": 1, "box_2d": [6, 6, 60, 60], "kind": "single_field"},
        ],
    })
    assert ext.fields["first_name"].script == "handwritten"
    assert ext.fields["last_name"].script == "unknown"
    assert [(loc["page"], loc["kind"]) for loc in ext.name_locations] == [
        (1, "patient_sticker"), (2, "other"), (1, "single_field")]  # capped at MAX_NAME_LOCATIONS


def test_schema_has_script_and_name_locations():
    schema = ee.build_schema(["כללית"])
    assert schema["properties"]["first_name"]["properties"]["script"]["enum"] == ["printed", "handwritten", "unknown"]
    assert "script" in schema["properties"]["last_name"]["required"]
    assert "name_locations" in schema["required"]


def test_prompt_has_rtl_label_rule_and_handwriting_rule():
    prompt = ee.load_prompt(["כללית"])
    assert "LABEL / VALUE DIRECTION" in prompt and "immediately to its RIGHT" in prompt
    assert "HANDWRITTEN NAMES" in prompt and "plausible-looking name" in prompt


@pytest.mark.parametrize("first_script, last_script, values, expected", [
    ("printed", "printed", ("א", "ב"), False),
    ("handwritten", "printed", ("א", "ב"), True),
    ("unknown", "unknown", ("א", "ב"), True),
    ("handwritten", "handwritten", (None, None), False),  # nothing read: already "missing"
])
def test_needs_name_check(first_script, last_script, values, expected):
    ext = ee.parse_extraction({"first_name": {"value": values[0], "confidence": 0.9, "script": first_script},
                               "last_name": {"value": values[1], "confidence": 0.9, "script": last_script}})
    assert ee.needs_name_check(ext) is expected


def test_crop_name_regions_maps_box_with_padding():
    from PIL import Image

    page = Image.new("RGB", (2000, 3000), "white")
    crops = ee.crop_name_regions([page], [{"page": 1, "box": (300, 100, 340, 900), "kind": "labeled_fields"},
                                          {"page": 2, "box": (0, 0, 10, 10), "kind": "other"}])
    assert len(crops) == 1  # page 2 does not exist
    kind, crop = crops[0]
    # box: x 200-1800, y 900-1020 (120 px high) -> padded by 240 px (x) and 72 px (y)
    assert kind == "labeled_fields" and crop.size[0] == 2000 and abs(crop.size[1] - 264) <= 1


def test_printed_names_need_no_verification():
    rec = _rec(_hand(script="printed"), ee.NameCheck(needed=False))
    assert rec.flags == {}


def test_crop_agrees_with_page_reading_no_flag():
    rec = _rec(_hand(), _check(_reading(first="ישראל", last="ישראלי", assignment="labeled_fields")))
    assert rec.flags == {} and rec.values["first_name"] == "ישראל"


def test_invented_name_is_emptied_and_flagged():
    # Page reading invented a plausible name (pages 9 / 8 / 3); the high-res crop reads differently.
    rec = _rec(_hand(first="יסמין", last="יואל"), _check(_reading("ישראל ישראלי")))
    assert rec.values["first_name"] == rec.values["last_name"] == ""
    assert rec.flags["first_name"] == rec.flags["last_name"] == "name_mismatch"


def test_illegible_crop_empties_names():
    rec = _rec(_hand(first="חסן", last="חסן", basis="form_convention"),
               _check(_reading(None, legible=False, conf=0.2)))
    assert rec.values["first_name"] == "" and rec.flags["first_name"] == "name_unverified"


def test_verification_error_or_no_location_empties_names():
    rec = _rec(_hand(), _check(error="no name location to verify"))
    assert rec.flags == {"first_name": "name_unverified", "last_name": "name_unverified"}


def test_two_locations_that_disagree_are_flagged():
    rec = _rec(_hand(), _check(_reading("ישראל ישראלי"), _reading("ישראל כהן", kind="patient_sticker")))
    assert rec.flags["first_name"] == "name_conflict" and rec.values["last_name"] == ""


def test_known_convention_still_flagged_when_crop_is_uncertain():
    rec = _rec(_hand(basis="form_convention", conf=0.95), _check(_reading("ישראל ישראלי", conf=0.6)))
    assert rec.flags == {"first_name": "low_confidence", "last_name": "low_confidence"}


def test_name_tokens_ignore_order_and_punctuation():
    assert rb._name_tokens("ישראל", "ישראלי") == rb._name_tokens("ישראלי.", "ישראל")


def test_verify_names_with_fake_model(monkeypatch):
    from PIL import Image

    seen = {}

    def fake_generate(contents, schema, **kwargs):
        seen.update(kwargs)
        seen["crops"] = sum(1 for c in contents if not isinstance(c, str))
        return {"crops": [{"crop_index": 1, "legible": True, "full_name": "ישראל ישראלי", "assignment": "single_field",
                           "first_name": None, "last_name": None, "confidence": 0.9}]}  # crop 2 missing

    monkeypatch.setattr(ee, "generate_json", fake_generate)
    ext = _hand(locations=[{"page": 1, "box_2d": [100, 100, 150, 900], "kind": "single_field"},
                           {"page": 1, "box_2d": [500, 100, 550, 900], "kind": "patient_sticker"}])
    check = ee.verify_names([Image.new("RGB", (1000, 1400), "white")], ext)
    assert seen["crops"] == 4 and seen["purpose"] == "verify_names"  # 2 crops, each in colour + ink view
    assert [r.legible for r in check.readings] == [True, False]  # a missing answer counts as illegible
    assert check.readings[0].full_name == "ישראל ישראלי"


def test_verify_names_skips_printed(monkeypatch):
    monkeypatch.setattr(ee, "generate_json", lambda *a, **k: pytest.fail("no call expected"))
    assert ee.verify_names([], _hand(script="printed")).needed is False


def test_usage_by_purpose(fake_client, monkeypatch):
    from src import config
    from tests.conftest import fake_reply

    fake_client.reply = fake_reply(input_tokens=100, output_tokens=10, cache_read=0)
    monkeypatch.setattr(config, "DEV_CACHE", False)
    ee.reset_usage()
    ee.generate_json(_contents(), {"type": "OBJECT"}, purpose="extract")
    ee.generate_json(_contents(), {"type": "OBJECT"}, purpose="verify_names")
    ee.generate_json(_contents(), {"type": "OBJECT"}, purpose="verify_names")
    assert ee.get_usage("verify_names") == {"requests": 2, "prompt_tokens": 200, "output_tokens": 20}
    assert ee.get_usage()["requests"] == 3
    assert set(ee.get_usage_by_purpose()) == {"extract", "verify_names"}


# ---- unverified-name suggestions, kept crops, commitment == ID ---------------------

def test_mismatch_suggests_crop_reading_never_as_value():
    rec = _rec(_hand(first="יסמין", last="יואל"),
               _check(_reading("ישראל ישראלי", first="ישראל", last="ישראלי", assignment="convention")))
    assert rec.values["first_name"] == "" and rec.values["last_name"] == ""   # never approved
    assert rec.flags["first_name"] == "name_mismatch"
    assert rec.suggestions == {"first_name": "ישראל", "last_name": "ישראלי"}
    assert rec.display_value("first_name") == "ישראל (הצעה – לא מאומת)"
    assert rec.to_row()["first_name"] == ""  # export rows carry approved values only


def test_illegible_crop_falls_back_to_page_reading_as_suggestion():
    rec = _rec(_hand(first="חסן", last="חסן"), _check(_reading(None, legible=False, conf=0.2)))
    assert rec.flags["first_name"] == "name_unverified"
    assert rec.suggestions == {"first_name": "חסן", "last_name": "חסן"}


def test_conflict_suggests_most_confident_crop():
    rec = _rec(_hand(), _check(_reading("ישראל כהן", first="ישראל", last="כהן", conf=0.7),
                               _reading("ישראל ישראלי", first="ישראל", last="ישראלי", conf=0.9)))
    assert rec.flags["last_name"] == "name_conflict"
    assert rec.suggestions["last_name"] == "ישראלי"


def test_reversed_labeled_fields_are_flagged():
    # Page 13: labeled fields, page reading swapped first/last with confidence 0.9.
    rec = _rec(_hand(first="ישראלי", last="ישראל"),
               _check(_reading(first="ישראל", last="ישראלי", assignment="labeled_fields")))
    assert rec.flags == {"first_name": "name_order_unclear", "last_name": "name_order_unclear"}
    assert rec.values["first_name"] == ""
    assert rec.suggestions == {"first_name": "ישראל", "last_name": "ישראלי"}  # order from the labels


def test_no_page_reading_but_crop_gives_suggestion():
    ext = _hand(first=None, last=None, conf=0.3)
    assert ee.needs_name_check(ext) is True  # a name location exists
    rec = _rec(ext, _check(_reading("ישראל ישראלי", first="ישראל", last="ישראלי", assignment="convention",
                                    conf=0.85)))
    assert rec.flags["first_name"] == "name_unverified" and rec.values["first_name"] == ""
    assert rec.suggestions["first_name"] == "ישראל"


def _crop(first="ישראל", last="ישראלי", conf=0.95, uncertain=0, assignment="labeled_fields", kind="labeled_fields"):
    return ee.NameReading(kind, True, f"{first} {last}", assignment, first, last, conf, uncertain)


def test_no_page_reading_confident_crops_are_approved():
    rec = _rec(_hand(first=None, last=None, conf=0.3), _check(_crop(), _crop(kind="other", assignment="convention")))
    assert rec.values["first_name"] == "ישראל" and rec.values["last_name"] == "ישראלי"
    assert "first_name" not in rec.flags and "last_name" not in rec.flags and rec.suggestions == {}


@pytest.mark.parametrize("crops", [
    [_crop(conf=0.89)],                                     # below NAME_AUTO_APPROVE_CONFIDENCE
    [_crop(uncertain=1)],                                   # a letter in doubt
    [_crop(assignment="unclear")],                          # first / last split not known
    [_crop(), _crop(first="ישראלי", last="ישראל")],         # crops disagree on the order
    [_crop(), _crop(last="ישראלית")],                       # crops disagree on a letter
    [_crop(), ee.NameReading("other", False, None, "unclear", None, None, 0.2)],  # one crop illegible
])
def test_no_page_reading_weak_crops_stay_suggestions(crops):
    rec = _rec(_hand(first=None, last=None, conf=0.3), _check(*crops))
    assert rec.values["first_name"] == rec.values["last_name"] == ""
    assert rec.flags["first_name"] == rec.flags["last_name"] == "name_unverified"
    assert rec.suggestions["first_name"]


def test_uncertain_letter_flags_an_agreed_name():
    rec = _rec(_hand(conf=0.95), _check(_crop(uncertain=1)))
    assert rec.values["first_name"] == "ישראל"
    assert rec.flags == {"first_name": "low_confidence", "last_name": "low_confidence"}


def test_name_check_parses_uncertain_letters(monkeypatch):
    from PIL import Image

    answer = {"crops": [{"crop_index": 1, "legible": True, "full_name": "ישראל ישראלי", "assignment": "labeled_fields",
                         "first_name": "ישראל", "last_name": "ישראלי", "uncertain_letters": 2, "confidence": 0.7},
                        {"crop_index": 2, "legible": True, "full_name": "ישראל ישראלי", "assignment": "labeled_fields",
                         "first_name": "ישראל", "last_name": "ישראלי", "confidence": 0.95}]}
    monkeypatch.setattr(ee, "generate_json", lambda *a, **k: answer)
    ext = _hand(locations=[{"page": 1, "box_2d": [300, 100, 340, 900], "kind": "labeled_fields"},
                           {"page": 1, "box_2d": [500, 100, 540, 900], "kind": "other"}])
    check = ee.verify_names([Image.new("RGB", (1000, 1400), "white")], ext)
    assert [r.uncertain_letters for r in check.readings] == [2, 1]  # missing count => uncertain
    assert "uncertain_letters" in ee.NAME_CHECK_SCHEMA["properties"]["crops"]["items"]["required"]
    assert "ALWAYS give your best reading" in ee.NAME_CHECK_PROMPT


def test_unsplit_crop_reading_is_shown_whole():
    rec = _rec(_hand(first="א", last="ב"), _check(_reading("ישראל ישראלי")))
    assert rec.suggestions == {"first_name": "ישראל ישראלי", "last_name": "ישראל ישראלי"}


def test_verified_names_have_no_suggestions():
    rec = _rec(_hand(), _check(_reading(first="ישראל", last="ישראלי", assignment="labeled_fields")))
    assert rec.suggestions == {} and rec.flags == {}


def test_verify_names_writes_nothing_and_sends_form_type(monkeypatch, tmp_path):
    from PIL import Image

    from src import config

    monkeypatch.setattr(config, "TEMP_DIR", tmp_path)
    sent = {}

    def fake_generate(contents, schema, **kwargs):
        sent["prompt"] = contents[0]
        return {"crops": [{"crop_index": 1, "legible": True, "full_name": "ישראל ישראלי", "assignment": "convention",
                           "first_name": "ישראל", "last_name": "ישראלי", "confidence": 0.9}]}

    monkeypatch.setattr(ee, "generate_json", fake_generate)
    check = ee.verify_names([Image.new("RGB", (1000, 1400), "white")], _hand())
    assert "Form type reported by the full-page pass: maccabi_lab_request" in sent["prompt"]
    assert "sheba_system_printout" in sent["prompt"]  # known name orders are included
    assert len(check.readings) == 1
    assert not list(tmp_path.iterdir())  # nothing written to disk


@pytest.mark.parametrize("commitments, expected_value", [
    (["9806981", "11223344"], "9806981"),
    (["0011223344"], ""),                 # leading zeros: still the ID
])
def test_commitment_equal_to_id_is_removed_and_flagged(commitments, expected_value):
    rec = _record(id_number={"value": "11223344", "confidence": 0.95},
                  commitment_numbers={"value": commitments, "confidence": 0.95})
    assert rec.values["commitment_numbers"] == expected_value
    assert rec.flags["commitment_numbers"] == "commitment_is_id"


def test_commitment_kept_when_no_id_or_different():
    assert "commitment_numbers" not in _record(id_number={"value": None, "confidence": 0}).flags
    assert _record().values["commitment_numbers"] == "50364186"


# ---- thinking level ---------------------------------------------------------------------

def test_thinking_level_sets_the_effort_and_keys_the_cache(fake_client):
    ee.generate_json(["x"], {"type": "OBJECT"}, thinking_level="medium")
    assert fake_client.kwargs["output_config"]["effort"] == "medium"
    with pytest.raises(ValueError):
        ee.generate_json(["x"], {"type": "OBJECT"}, thinking_level="max")
    plain = ee._cache_key(["x"], {}, None, "high", None, "m")
    assert ee._cache_key(["x"], {}, None, "high", "high", "m") != plain
