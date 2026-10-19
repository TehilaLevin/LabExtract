"""Validation added after extraction: central prefix, received-date window / collection date, batch ID
consistency, raw_text and collection_date parsing. No API calls."""
from datetime import date

import pytest

from src import config
from src import extraction_engine as ee
from src import record_builder as rb
from src.constants import FIELD_KEYS
from src.validators import check_received_date, date_before

TODAY = date(2026, 10, 13)


def _extraction(collection_date=None, **overrides):
    fields = {k: ee.FieldResult(value=None, confidence=0.95, status="read") for k in FIELD_KEYS}
    fields["central_numbers"] = ee.FieldResult(["939376"], 0.95, status="read")
    fields["commitment_numbers"] = ee.FieldResult(["730578626"], 0.95, status="read")
    fields["received_date"] = ee.FieldResult("01/10/2026", 0.95, source="received_stamp", status="read")
    fields["id_number"] = ee.FieldResult("000000018", 0.95, status="read")
    fields["first_name"] = ee.FieldResult("ישראל", 0.95, status="read")
    fields["last_name"] = ee.FieldResult("ישראלי", 0.95, status="read")
    fields["sending_entity"] = ee.FieldResult("מאוחדת", 0.95, status="read")
    for key, value in overrides.items():
        fields[key] = value
    return ee.ExtractionResult("meuhedet_commitment", 0.9, fields, collection_date=collection_date,
                               name_order_basis="separate_fields")


def _flags(extraction):
    return rb.field_flags(extraction, today=TODAY)[1]


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setattr(config, "STRICT_VALIDATION", True)
    monkeypatch.setattr(config, "CENTRAL_NUMBER_PREFIXES", ("93", "94"))
    monkeypatch.setattr(config, "RECEIVED_DATE_MAX_AGE_DAYS", 90)


def test_clean_form_has_no_flags(strict):
    assert _flags(_extraction()) == {}


def test_central_prefix_is_a_warning_not_a_rejection(strict):
    ex = _extraction(central_numbers=ee.FieldResult(["939376", "123456"], 0.95, status="read"))
    values, flags = rb.field_flags(ex, today=TODAY)
    assert flags["central_numbers"] == "central_prefix"
    assert values["central_numbers"] == "939376, 123456"             # value kept for the reviewer
    ex = _extraction(central_numbers=ee.FieldResult(["93937"], 0.95, status="read"))
    assert _flags(ex)["central_numbers"] == "invalid_format"          # 6 digits still required


def test_received_date_window(strict):
    old = _extraction(received_date=ee.FieldResult("01/06/2026", 0.95, source="received_stamp", status="read"))
    assert _flags(old)["received_date"] == "date_out_of_range"       # 134 days > 90
    future = _extraction(received_date=ee.FieldResult("01/11/2026", 0.95, source="received_stamp", status="read"))
    assert _flags(future)["received_date"] == "date_out_of_range"
    assert check_received_date("01/06/2026", TODAY, max_age_days=200).reason is None


def test_received_date_before_collection_date(strict):
    assert _flags(_extraction(collection_date="05/10/2026"))["received_date"] == "date_before_collection"
    assert "received_date" not in _flags(_extraction(collection_date="30/09/2026"))
    assert "received_date" not in _flags(_extraction(collection_date="unreadable"))
    assert date_before("05/10/2026", "01/10/2026") and not date_before("01/10/2026", "01/10/2026")


def test_received_date_at_most_14_days_after_collection(strict):
    assert "received_date" not in _flags(_extraction(collection_date="17/09/2026"))        # 14 days: ok
    assert _flags(_extraction(collection_date="16/09/2026"))["received_date"] == "date_long_after_collection"
    assert "received_date" not in _flags(_extraction(collection_date="01/10/2026"))        # same day: ok


def test_collection_date_with_time_or_without_year():
    from src.validators import days_between

    assert days_between("17/06/2026 10:18", "18/06/2026") == 1      # time after the date is ignored
    assert days_between("25.08", "25/08/2026") is None               # no year: no check, never a guess


def test_eval_mode_skips_only_the_age_from_today_check(strict, monkeypatch):
    old = ee.FieldResult("10/06/2026", 0.95, source="received_stamp", status="read")
    assert _flags(_extraction(received_date=old))["received_date"] == "date_out_of_range"
    monkeypatch.setattr(config, "EVAL_MODE", True)
    assert "received_date" not in _flags(_extraction(received_date=old))
    future = ee.FieldResult("01/11/2026", 0.95, source="received_stamp", status="read")
    assert _flags(_extraction(received_date=future))["received_date"] == "date_out_of_range"   # still checked
    assert _flags(_extraction(received_date=old, collection_date="01/05/2026"))["received_date"] == \
        "date_long_after_collection"                                                         # still checked


def test_strict_validation_off_restores_earlier_checks(strict, monkeypatch):
    monkeypatch.setattr(config, "STRICT_VALIDATION", False)
    ex = _extraction(central_numbers=ee.FieldResult(["123456"], 0.95, status="read"), collection_date="05/10/2026",
                     received_date=ee.FieldResult("01/06/2026", 0.95, source="received_stamp", status="read"))
    assert _flags(ex) == {}                                           # one-year window, no prefix / collection


def test_new_flags_trigger_one_targeted_reread(strict):
    from src.field_reread import REREAD_REASONS

    assert {"central_prefix", "date_before_collection", "date_out_of_range"} <= REREAD_REASONS


def _record(first, last, ident):
    values = {k: "" for k in FIELD_KEYS}
    values.update(first_name=first, last_name=last, id_number=ident)
    return rb.Record(1, values, {}, {}, {})


def test_batch_id_consistency(strict):
    a, b, c = _record("ישראל", "ישראלי", "000000018"), _record("ישראל", "ישראלי", "000000026"), _record("דנה", "כהן", "1")
    assert rb.apply_batch_id_consistency([a, b, c]) == 2
    assert a.flags["id_number"] == b.flags["id_number"] == "id_batch_mismatch" and not c.flags
    same = [_record("ישראל", "ישראלי", "18"), _record("ישראל", "ישראלי", "000000018")]  # leading zeros
    assert rb.apply_batch_id_consistency(same) == 0
    nameless = [_record("", "", "18"), _record("", "", "26")]
    assert rb.apply_batch_id_consistency(nameless) == 0


def test_raw_text_and_collection_date_are_parsed():
    data = {k: {"value": None, "confidence": "low", "raw_text": None, "source_note": ""} for k in FIELD_KEYS}
    data["id_number"] = {"value": None, "confidence": "low", "raw_text": "11?2334?", "source_note": "next to ת.ז."}
    data["collection_date"] = {"value": "28/09/26", "confidence": "high"}
    res = ee.parse_extraction(data)
    assert res.fields["id_number"].value is None and res.fields["id_number"].raw_text == "11?2334?"
    assert res.collection_date == "28/09/26"
    rec = rb.build_record(type("F", (), {"index": 1, "source_file": "a.pdf", "page_numbers": [1], "split_flags": []})(),
                          res, today=TODAY)
    assert rec.raw_texts == {"id_number": "11?2334?"} and "id_number" not in rec.to_row() or rec.to_row()["id_number"] == ""


def test_claude_schema_has_raw_text_and_collection_date():
    from src import llm_provider as lp
    import json

    out = lp.to_json_schema(ee.build_schema(["כללית"], labelled_confidence=True))
    assert out["properties"]["id_number"]["properties"]["raw_text"] == {"type": "string"}
    assert "collection_date" in out["required"]
    assert json.dumps(out).count('"null"') <= 16                     # API limit on union-typed parameters
    prompt = ee.load_prompt(["כללית"], labelled_confidence=True)
    assert "If it is not legible, return null. Do not guess. Do not complete names to common names." in prompt
    assert "collection_date" not in json.dumps(ee.build_schema(["כללית"]))  # Gemini schema unchanged
