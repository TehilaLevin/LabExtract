"""Streamlit app flows with streamlit.testing (no browser). Results are injected into session_state;
any model call fails the test - reruns must never call the API."""
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src import extraction_engine as ee
from src import record_builder as rb
from src import review_session as rs
from src.constants import FIELD_KEYS, FLAG_REASONS, SUGGESTION_LABEL
from src.field_crops import PagePreview, SourceCrop

VALID_ID = "000000018"
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _jpeg(colour="white"):
    buf = BytesIO()
    Image.new("RGB", (60, 40), colour).save(buf, "JPEG")
    return buf.getvalue()


def _record(index, flags=None, suggestions=None, **values):
    base = {"first_name": "ישראל", "last_name": "ישראלי", "id_number": VALID_ID, "central_numbers": "937309",
            "received_date": "01/10/2026", "sending_entity": "מכבי", "commitment_numbers": "50364186"}
    base.update(values)
    crops = {k: [SourceCrop(1, (0, 0, 10, 10), _jpeg("red"))] for k in FIELD_KEYS}
    return rb.Record(index, base, dict(flags or {}), {k: 0.9 for k in FIELD_KEYS},
                     {"source_file": "batch.pdf", "page_numbers": str(index), "form_type": "מכבי – מעבדה שולחת"},
                     suggestions=dict(suggestions or {}), field_crops=crops,
                     page_images=[PagePreview(index, _jpeg())])


@pytest.fixture(autouse=True)
def no_model_calls(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("the model must not be called")

    monkeypatch.setattr(ee, "generate_json", fail)


def _app(records, screen="summary", pos=0):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["reviews"] = rs.new_session(records)
    at.session_state["screen"] = screen
    at.session_state["pos"] = pos
    return at.run()


def _input(at, key):
    return next(w for w in at.text_input if w.key.startswith(f"in_") and f"_{key}_" in w.key)


def test_upload_screen_without_results():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.button(key="extract").disabled  # nothing uploaded yet


def test_summary_counts():
    at = _app([_record(1), _record(2, flags={"id_number": "invalid_id"}, id_number="000000019")])
    assert not at.exception
    assert any("2 טפסים עובדו, 1 שדות סומנו לבדיקה, 0 טפסים אושרו" in m.value for m in at.markdown)


def test_review_shows_values_and_flag_reason():
    at = _app([_record(1, flags={"id_number": "invalid_id"}, id_number="000000019")], screen="review")
    assert not at.exception
    assert _input(at, "id_number").value == "000000019"
    assert any(FLAG_REASONS["invalid_id"] in m.value for m in at.markdown)
    assert at.selectbox[0].value == "מכבי"  # sender: closed list


def test_approve_moves_to_next_form():
    at = _app([_record(1, flags={"id_number": "low_confidence"}), _record(2)], screen="review")
    at.button(key="approve").click().run()
    assert not at.exception
    reviews = at.session_state["reviews"]
    assert reviews[0].approved and not reviews[1].approved
    assert at.session_state["pos"] == 1


def test_correction_is_saved_and_normalized():
    at = _app([_record(1, flags={"received_date": "missing"}, received_date="")], screen="review")
    _input(at, "received_date").set_value("1/10/26")
    at.button(key="approve").click().run()
    review = at.session_state["reviews"][0]
    assert review.approved and review.value("received_date") == "01/10/2026"
    assert at.session_state["screen"] == "export"  # last form approved


def test_invalid_value_needs_approve_anyway():
    at = _app([_record(1, flags={"id_number": "invalid_id"}, id_number="000000019")], screen="review")
    at.button(key="approve").click().run()
    assert not at.session_state["reviews"][0].approved
    assert any("בדיקת התקינות נכשלה" in w.value for w in at.warning)
    at.button(key="force_approve").click().run()
    assert at.session_state["reviews"][0].approved


def test_save_without_approving_keeps_the_edit():
    at = _app([_record(1)], screen="review")
    _input(at, "first_name").set_value("יעל")
    at.button(key="save").click().run()
    review = at.session_state["reviews"][0]
    assert not review.approved and review.value("first_name") == "יעל"
    assert _input(at, "first_name").value == "יעל"  # shown again after the rerun


def test_unverified_name_is_a_suggestion_not_a_value():
    rec = _record(1, flags={"first_name": "name_unverified", "last_name": "name_unverified"},
                  suggestions={"first_name": "ישראל"}, first_name="", last_name="")
    at = _app([rec], screen="review")
    assert _input(at, "first_name").value == ""
    assert any(SUGGESTION_LABEL in m.value for m in at.markdown)
    assert any(c.value == "ישראל" for c in at.code)


def test_export_needs_confirmation_while_forms_are_unapproved():
    at = _app([_record(1), _record(2)], screen="export")
    assert not at.exception
    assert any("2 טפסים עדיין לא אושרו" in w.value for w in at.warning)
    assert at.checkbox(key="export_anyway").value is False


def test_export_without_warning_when_all_approved():
    records = [_record(1), _record(2)]
    at = AppTest.from_file(APP, default_timeout=30)
    reviews = rs.new_session(records)
    for r in reviews:
        rs.approve(r, {})
    at.session_state["reviews"] = reviews
    at.session_state["screen"] = "export"
    at.run()
    assert not at.exception
    assert not at.warning and not at.checkbox


@pytest.mark.parametrize("enabled", [True, False])
def test_dev_cache_banner_only_while_the_cache_is_on(monkeypatch, enabled):
    monkeypatch.setattr("src.config.DEV_CACHE", enabled)
    at = AppTest.from_file(APP, default_timeout=30).run()
    shown = any("DEV_CACHE פעיל" in e.value for e in at.error)
    assert shown is enabled


def test_clear_all_returns_to_upload():
    at = _app([_record(1)])
    at.button(key="clear_all").click().run()
    assert at.session_state["reviews"] is None and at.session_state["screen"] == "upload"
