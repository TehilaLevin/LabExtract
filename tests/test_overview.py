"""Overview aggregation (sender kinds and counts) and the database screen (temp DB from conftest, no API calls)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import overview
from src import record_store as store
from tests.test_record_store import _review

APP = str(Path(__file__).resolve().parent.parent / "app.py")


# ------------------------------------------------------------------ overview

def test_breakdown_counts_percentages_and_kinds():
    rows = overview.breakdown({"מכבי": 3, "איכילוב": 1, "": 2, "גורם מוזר": 2, "כללית": 0})
    by_name = {r["name"]: r for r in rows}
    assert by_name["מכבי"]["kind"] == overview.KIND_HMO and by_name["מכבי"]["percent"] == 37.5
    assert by_name["איכילוב"]["kind"] == overview.KIND_HOSPITAL
    assert by_name["גורם מוזר"]["kind"] == overview.KIND_OTHER
    assert by_name[overview.UNKNOWN_NAME]["kind"] == overview.KIND_UNKNOWN and by_name[overview.UNKNOWN_NAME]["count"] == 2
    assert "כללית" not in by_name                       # zero counts are dropped
    assert rows[0]["name"] == "מכבי"                    # sorted by count
    assert sum(r["percent"] for r in rows) == 100.0


def test_breakdown_sorting_ties_missing_names_and_denominator():
    rows = overview.breakdown({"מכבי": 2, "כללית": 2, "איכילוב": 5, None: 1, "  ": 1})
    assert [r["count"] for r in rows] == sorted((r["count"] for r in rows), reverse=True)
    assert [r["name"] for r in rows[:4]] == ["איכילוב", "כללית", overview.UNKNOWN_NAME, "מכבי"]  # ties broken by name
    unknown = next(r for r in rows if r["kind"] == overview.KIND_UNKNOWN)
    assert unknown["name"] == overview.UNKNOWN_NAME and unknown["count"] == 2    # None and blank merge
    assert all(r["percent"] == round(100 * r["count"] / 11, 1) for r in rows)    # denominator = all forms


def test_breakdown_empty_and_totals_by_kind():
    assert overview.breakdown({}) == []
    totals = overview.totals_by_kind(overview.breakdown({"מכבי": 1, "כללית": 1, "איכילוב": 2}))
    assert totals[overview.KIND_HMO] == {"count": 2, "percent": 50.0}
    assert totals[overview.KIND_HOSPITAL]["count"] == 2 and totals[overview.KIND_OTHER]["count"] == 0
    assert overview.totals_by_kind([])[overview.KIND_HMO] == {"count": 0, "percent": 0.0}


def test_sender_counts_from_the_database():
    store.save_all([_review(1, sender="מכבי"), _review(2, sender="מכבי"), _review(3, sender="כללית")], "b1")
    assert store.sender_counts() == {"מכבי": 2, "כללית": 1}


def test_sender_counts_approved_only():
    store.save_all([_review(1, sender="מכבי"), _review(2, sender="כללית")], "b1")
    assert store.sender_counts(approved_only=True) == {}


def test_database_screen_has_no_activity_overview():
    """The per-sender overview chart was removed from the database screen (the module still serves the archive assistant)."""
    store.save_all([_review(1, sender="מכבי")], "b1")
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["screen"] = "database"
    at.run()
    assert not at.exception and not any("סקירת פעילות" in m.value for m in at.markdown)
