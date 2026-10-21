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
