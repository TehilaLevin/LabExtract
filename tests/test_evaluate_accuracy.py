import importlib.util
import json
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
    ev.print_report(ev.summarize([(TRUTH, bad)]), show_values=True)
    out = capsys.readouterr().out
    assert "יסמין" not in out and "999" not in out and "11223344" not in out
    assert "first_name: mismatch" in out
    assert "expected=937309;937310 got=111111" in out  # non-personal fields show values on request


def test_report_prints_no_values_by_default(capsys):
    bad = dict(GOOD, central_numbers="111111")
    ev.print_report(ev.summarize([(TRUTH, bad)]))
    out = capsys.readouterr().out
    assert "111111" not in out and "937309" not in out
    assert "central_numbers: mismatch" in out


def test_saved_summary_has_counts_only_and_comparison(capsys):
    bad = dict(GOOD, central_numbers="111111", first_name="יסמין")
    base = ev.counts_only(ev.summarize([(TRUTH, bad)]), truth_rows=[TRUTH])
    now = ev.counts_only(ev.summarize([(TRUTH, GOOD)]), {"central_numbers": {"agreed": 1}}, truth_rows=[TRUTH])
    text = json.dumps(base, ensure_ascii=False) + json.dumps(now, ensure_ascii=False)
    assert "111111" not in text and "יסמין" not in text and "11223344" not in text and "ישראל" not in text
    assert ev.print_comparison(base, now) is True
    out = capsys.readouterr().out
    assert "central_numbers      0/1 -> 1/1 (+1)" in out and "wrong_unflagged 1->0" in out
    assert "Fixed: f.pdf p13 first_name" in out


@pytest.mark.parametrize("change, reason", [
    (lambda s: s.update(labels_sha256="other"), "different ground-truth rows or labels"),
    (lambda s: s.update(rows=[["g.pdf", "1"]]), "different ground-truth rows or labels"),
    (lambda s: s.update(run={"model_sha256": "x", "complete": True}), "different model"),
    (lambda s: s.update(run={"complete": False}), "baseline run is incomplete"),
])
def test_comparison_refuses_different_runs(capsys, change, reason):
    now = ev.counts_only(ev.summarize([(TRUTH, GOOD)]), truth_rows=[TRUTH],
                         info={"model_sha256": "m", "complete": True})
    base = json.loads(json.dumps(now))
    change(base)
    assert ev.print_comparison(base, now) is False
    assert reason in capsys.readouterr().out


def test_breakdown_separates_flagged_missing_and_silent_errors():
    values = dict(GOOD, first_name="", id_number="99999999", central_numbers="111111", sending_entity="מכבי")
    flags = {"id_number": "invalid_id", "commitment_numbers": "low_confidence"}
    b = ev.summarize([(TRUTH, values, None, flags), (TRUTH, None)])["breakdown"]
    assert b["first_name"] == dict.fromkeys(ev.OUTCOMES, 0) | {"missing": 1, "no_record": 1}
    assert b["id_number"]["wrong_flagged"] == 1
    assert b["central_numbers"]["wrong_unflagged"] == 1             # a silent error
    assert b["commitment_numbers"]["correct_flagged"] == 1           # right, but still shown for review
    assert b["received_date"]["correct"] == 1                        # expected empty, got empty


def test_validation_never_makes_a_value_correct():
    # A checksum-valid ID that differs from the label is wrong, flagged or not.
    out = ev.summarize([(TRUTH, dict(GOOD, id_number="000000018"), None, {})])
    assert out["per_field"]["id_number"] == (0, 1) and out["breakdown"]["id_number"]["wrong_unflagged"] == 1


def test_id_leading_zero_is_the_same_id():
    assert ev.normalize_value("id_number", "011223344") == ev.normalize_value("id_number", "11223344")
    assert ev.normalize_value("id_number", "11223344") != ev.normalize_value("id_number", "11223345")


def test_labels_fingerprint_changes_with_any_label():
    assert ev.labels_fingerprint([TRUTH]) == ev.labels_fingerprint([dict(TRUTH)])
    assert ev.labels_fingerprint([TRUTH]) != ev.labels_fingerprint([dict(TRUTH, commitment_numbers="251494")])


def test_multi_page_rows_send_every_page_and_map_by_first_page(monkeypatch):
    from types import SimpleNamespace

    sent = {}

    def fake_subset(source, pages):
        sent["pages"] = pages
        return b"%PDF"

    def fake_process(files, progress=None):
        # subset pages: 1 -> p4, 2 -> p5 (one 2-page form), 3 -> p9
        return [SimpleNamespace(metadata={"page_numbers": "1, 2"}, tag="A"),
                SimpleNamespace(metadata={"page_numbers": "3"}, tag="B")]

    monkeypatch.setattr(ev, "_subset_pdf", fake_subset)
    monkeypatch.setattr("src.record_builder.process_files", fake_process)
    rows = [dict(TRUTH, page_numbers="4,5"), dict(TRUTH, page_numbers="9")]
    results = ev.run_engine(rows)
    assert sent["pages"] == [4, 5, 9]
    assert [rec.tag for _, rec in results] == ["A", "B"]


@pytest.fixture
def guarded(monkeypatch):
    from src import config
    from src import extraction_engine as ee

    monkeypatch.setattr(config, "DEV_CACHE", False)
    monkeypatch.setattr(config, "DEV_CACHE_DIR", Path(__file__).parent / "_no_such_cache_dir")
    monkeypatch.setattr(ee, "_generate_with_transport_retry", ee._generate_with_transport_retry)  # restored after
    monkeypatch.setattr(ee, "generate_json", ee.generate_json)                                    # restored after
    monkeypatch.setattr(ee, "get_client", lambda: pytest.fail("live API call attempted"))
    monkeypatch.setattr(config, "GEMINI_MODEL", "test-model", raising=False)
    return ee


def test_cache_only_never_calls_the_api(guarded):
    from src import config

    ee = guarded
    stats = {"cache_misses": 0}
    ev._block_live_calls(stats)
    with pytest.raises(ee.GeminiError, match="cache-only"):
        ee.generate_json(["hello"], {"type": "OBJECT"}, purpose="extract")
    assert stats["cache_misses"] == 1 and stats["misses_by_stage"] == {"extract": 1} and config.DEV_CACHE is True


def test_request_budget_stops_live_calls(guarded, monkeypatch):
    ee = guarded

    class _Response:
        text = '{"ok": true}'

    calls = []
    monkeypatch.setattr(ee, "_generate_with_transport_retry", lambda contents, cfg: calls.append(1) or _Response())
    stats = ev.new_stats()
    ev.install_request_guard(stats, max_requests=2)
    ee.generate_json(["a"], {"type": "OBJECT"}, purpose="extract")
    ee.generate_json(["b"], {"type": "OBJECT"}, purpose="reread_fields")
    with pytest.raises(ee.GeminiQuotaError, match="max-requests"):
        ee.generate_json(["c"], {"type": "OBJECT"}, purpose="extract")
    assert len(calls) == 2 and stats["budget_exhausted"] is True
    assert stats["live_by_stage"] == {"extract": 1, "reread_fields": 1}


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


# ---- UNREADABLE labels ------------------------------------------------------------------

UNREADABLE_TRUTH = dict(TRUTH, id_number="UNREADABLE", central_numbers="unreadable ")


def test_unreadable_label_empty_and_flagged_is_correct():
    values = dict(GOOD, id_number="", central_numbers="")
    out = ev.summarize([(UNREADABLE_TRUTH, values, None, {"id_number": "unreadable", "central_numbers": "missing"})])
    assert out["per_field"]["id_number"] == (1, 1) and out["per_field"]["central_numbers"] == (1, 1)
    assert out["breakdown"]["id_number"]["correct_flagged"] == 1
    assert not [f for f in out["failures"] if f["field"] in ("id_number", "central_numbers")]


@pytest.mark.parametrize("flags", [{}, {"id_number": "low_confidence"}])
def test_unreadable_label_any_value_is_a_silent_error(flags):
    out = ev.summarize([(UNREADABLE_TRUTH, dict(GOOD, id_number="11223344"), None, flags)])
    assert out["per_field"]["id_number"] == (0, 1)
    assert out["breakdown"]["id_number"]["wrong_unflagged"] == 1      # a guess, even when flagged


def test_unreadable_label_empty_but_unflagged_is_missing():
    out = ev.summarize([(UNREADABLE_TRUTH, dict(GOOD, id_number=""), None, {})])
    assert out["breakdown"]["id_number"]["missing"] == 1 and out["per_field"]["id_number"] == (0, 1)


def test_unreadable_label_never_matches_the_literal_text():
    assert ev.compare_row(UNREADABLE_TRUTH, dict(GOOD, id_number="UNREADABLE"))["id_number"] is False


# ---- tags / split -------------------------------------------------------------------------

def test_old_ground_truth_without_tags_or_split_still_loads(tmp_path):
    path = tmp_path / "gt.csv"
    path.write_text("source_file,page_numbers,id_number\nf.pdf,3,1\n", encoding="utf-8")
    rows = ev.load_ground_truth(path)
    assert rows[0]["tags"] == "" and ev.row_split(rows[0]) == "unassigned" and ev.row_tags(rows[0]) == []


def test_select_rows_by_split_and_tag():
    rows = [dict(TRUTH, split="test", tags="handwritten; multi_central"), dict(TRUTH, split="Tune", tags="printed"),
            dict(TRUTH, split="", tags="")]
    assert len(ev.select_rows(rows, split="test")) == 1
    assert ev.select_rows(rows, split="tune")[0]["tags"] == "printed"
    assert len(ev.select_rows(rows, tag="MULTI_CENTRAL")) == 1
    assert len(ev.select_rows(rows, split="unassigned")) == 1


def test_groups_report_per_split_and_tag(capsys):
    a = dict(TRUTH, split="test", tags="handwritten;multi_central")
    b = dict(TRUTH, split="tune", tags="handwritten")
    groups = ev.summarize_groups([(a, GOOD, None, {}), (b, dict(GOOD, central_numbers="111111"), None, {})])
    assert set(groups["split"]) == {"test", "tune"} and set(groups["tag"]) == {"handwritten", "multi_central"}
    assert groups["tag"]["handwritten"]["per_field"]["central_numbers"] == (1, 2)
    saved = ev.counts_only(ev.summarize([(a, GOOD)]), groups=groups)
    assert saved["groups"]["split"]["tune"]["silent_errors"] == 1 and "111111" not in json.dumps(saved)
    ev.print_groups(groups)
    out = capsys.readouterr().out
    assert "Per split:" in out and "silent_errors=1" in out and "111111" not in out
