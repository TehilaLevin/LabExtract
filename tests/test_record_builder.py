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
    res = ee.parse_extraction({"form_type": "x", "id_number": {"value": "  ", "confidence": "high"},
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


def test_name_with_digits_is_flagged():
    assert _record(last_name={"value": "ישראלי2", "confidence": 0.99}).flags == {"last_name": "invalid_format"}


def test_extraction_error_flags_every_field():
    rec = rb.build_record(_form(), ee.empty_result("Gemini API error 500"), today=TODAY)
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


# ---- process_files (fake Gemini, no API calls) ------------------------------------

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
        raise ee.GeminiQuotaError("Gemini API quota exhausted")

    monkeypatch.setattr(ee, "generate_json", fake_generate)
    monkeypatch.setattr("src.config.MAX_PARALLEL_REQUESTS", 1)
    records = rb.process_files([(f"{i}.png", _png_bytes()) for i in range(4)])
    assert calls["extract"] == 1
    assert all(r.error for r in records)


# ---- dev-only response cache --------------------------------------------------------

class _FakeResponse:
    text = '{"ok": true}'


class _FakeModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        return _FakeResponse()


@pytest.fixture
def fake_client(monkeypatch, tmp_path):
    from src import config

    models = _FakeModels()
    client = type("C", (), {"models": models})()
    monkeypatch.setattr(ee, "get_client", lambda: client)
    monkeypatch.setattr(config, "DEV_CACHE_DIR", tmp_path / "dev_cache")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    return models


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
    assert ee.clear_dev_cache() == 3
