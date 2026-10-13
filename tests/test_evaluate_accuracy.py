import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "evaluate_accuracy", Path(__file__).resolve().parent.parent / "evaluation" / "evaluate_accuracy.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

TRUTH = {"source_file": "f.pdf", "page_numbers": "13", "first_name": "ישראל", "last_name": "ישראלי",
         "id_number": "11223344", "central_numbers": "937309;937310", "received_date": "",
         "sending_entity": "שערי צדק", "commitment_numbers": "251493"}
GOOD = {"first_name": "ישראל", "last_name": "ישראלי", "id_number": "11223344",
        "central_numbers": "937310, 937309", "received_date": "", "sending_entity": "שערי צדק",
        "commitment_numbers": "251493"}


def test_all_fields_match_after_normalization():
    assert all(ev.compare_row(TRUTH, GOOD).values())


def test_names_count_and_swapped_names_fail():
    swapped = dict(GOOD, first_name="ישראלי", last_name="ישראל")
    out = ev.compare_row(TRUTH, swapped)
    assert out["first_name"] is False and out["last_name"] is False


def test_invented_name_fails_and_empty_name_fails():
    assert ev.compare_row(TRUTH, dict(GOOD, first_name="יסמין"))["first_name"] is False
    assert ev.compare_row(TRUTH, dict(GOOD, first_name=""))["first_name"] is False


def test_expected_empty_matches_only_empty():
    assert ev.compare_row(TRUTH, GOOD)["received_date"] is True
    assert ev.compare_row(TRUTH, dict(GOOD, received_date="21/08/2026"))["received_date"] is False


@pytest.mark.parametrize("key, a, b", [
    ("received_date", "18-06-2026", "18/06/2026"),
    ("id_number", "1122 3344", "11223344"),
    ("sending_entity", "בית חולים שערי צדק", "שערי צדק"),
    ("commitment_numbers", "1;2", "2, 1"),
])
def test_normalization(key, a, b):
    assert ev.normalize_value(key, a) == ev.normalize_value(key, b)


def test_summary_counts_missing_record_as_all_failed():
    summary = ev.summarize([(TRUTH, GOOD), (TRUTH, None)])
    assert summary["per_field"]["first_name"] == (1, 2)
    assert len(summary["failures"]) == len(ev.FIELD_KEYS)


def test_report_never_prints_personal_values(capsys):
    bad = dict(GOOD, first_name="יסמין", id_number="999", central_numbers="111111")
    ev.print_report(ev.summarize([(TRUTH, bad)]))
    out = capsys.readouterr().out
    assert "יסמין" not in out and "999" not in out and "11223344" not in out
    assert "first_name: mismatch" in out
    assert "expected=937309;937310 got=111111" in out  # non-personal fields show values


def test_ground_truth_file_has_names_for_the_five_forms():
    rows = ev.load_ground_truth(ev.DEFAULT_GROUND_TRUTH)
    assert len(rows) >= 5
    assert all(r["first_name"] and r["last_name"] for r in rows)


def test_suggestions_counted_separately_and_never_as_correct():
    empty_names = dict(GOOD, first_name="", last_name="")
    summary = ev.summarize([
        (TRUTH, empty_names, {"first_name": "ישראל", "last_name": "ישראלי"}),   # correct suggestions
        (TRUTH, empty_names, {"first_name": "יסמין", "last_name": "ישראלי"}),   # one wrong
    ])
    assert summary["per_field"]["first_name"] == (0, 2)          # suggestions are not values
    assert summary["suggestions"]["first_name"] == (1, 1)
    assert summary["suggestions"]["last_name"] == (2, 0)


def test_suggestion_ignored_when_value_present():
    summary = ev.summarize([(TRUTH, GOOD, {"first_name": "יסמין"})])
    assert summary["suggestions"]["first_name"] == (0, 0)


def test_report_shows_suggestion_counts_without_values(capsys):
    ev.print_report(ev.summarize([(TRUTH, dict(GOOD, first_name=""), {"first_name": "יסמין"})]))
    out = capsys.readouterr().out
    assert "correct suggestions: 0   wrong suggestions: 1" in out
    assert "first_name: mismatch (suggestion wrong)" in out and "יסמין" not in out
