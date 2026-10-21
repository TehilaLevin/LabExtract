"""Overview aggregation and achievement conditions/persistence (temp DB from conftest, no API calls)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import achievements as achv
from src import overview
from src import record_store as store
from src import review_session as rs
from tests.test_record_store import _review
from tests.test_app import _record

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
    assert overview.known_sender_count(overview.breakdown(store.sender_counts())) == 2


def test_sender_counts_approved_only():
    store.save_all([_review(1, sender="מכבי"), _review(2, sender="כללית")], "b1")
    assert store.sender_counts(approved_only=True) == {}


# ------------------------------------------------------------------ achievements

def test_conditions_are_only_met_when_true():
    assert achv.evaluate(achv.Stats()) == set()
    assert achv.evaluate(achv.Stats(processed_total=1)) == {"first_form"}
    assert "batch_complete" not in achv.evaluate(achv.Stats(batch_size=1, batch_finished=True))
    assert "batch_complete" in achv.evaluate(achv.Stats(batch_size=3, batch_finished=True))
    assert "batch_complete" not in achv.evaluate(achv.Stats(batch_size=3, batch_finished=False))
    assert "review_complete" not in achv.evaluate(achv.Stats(batch_size=3, batch_approved=2))
    assert "review_complete" in achv.evaluate(achv.Stats(batch_size=3, batch_approved=3))
    assert "review_complete" in achv.evaluate(achv.Stats(batch_size=1, batch_approved=1))
    assert "review_complete" not in achv.evaluate(achv.Stats(batch_size=0, batch_approved=0))
    assert "century_club" not in achv.evaluate(achv.Stats(processed_total=99))
    assert {"century_club"} <= achv.evaluate(achv.Stats(processed_total=100))
    assert {"five_hundred", "thousand"} <= achv.evaluate(achv.Stats(processed_total=1000))
    assert "wide_reach" not in achv.evaluate(achv.Stats(known_senders=4))
    assert "wide_reach" in achv.evaluate(achv.Stats(known_senders=5))


def test_night_shift_only_for_an_approval_between_midnight_and_five():
    assert "night_shift" in achv.evaluate(achv.Stats(hour=2))
    assert "night_shift" not in achv.evaluate(achv.Stats(hour=5))
    assert "night_shift" not in achv.evaluate(achv.Stats(hour=None))


def test_unlock_is_idempotent_and_counter_accumulates():
    assert achv.unlock({"first_form", "century_club"}) == ["first_form", "century_club"]
    assert achv.unlock({"first_form", "thousand"}) == ["thousand"]   # no duplicates
    assert set(achv.unlocked()) == {"first_form", "century_club", "thousand"}
    assert achv.add_processed(40) == 40 and achv.add_processed(70) == 110 and achv.processed_total() == 110


def test_achievements_survive_deleting_the_forms():
    store.save_all([_review(1)], "b1")
    achv.unlock({"first_form"})
    store.delete_all()
    assert set(achv.unlocked()) == {"first_form"}


# ------------------------------------------------------------------ app

def _approved_session():
    at = AppTest.from_file(APP, default_timeout=30)
    items = rs.new_session([_record(1), _record(2)])
    at.session_state["reviews"] = items
    at.session_state["batch_id"] = "b1"
    at.session_state["screen"] = "review"
    at.session_state["pos"] = 0
    return at


def test_review_complete_toast_once_and_not_repeated_on_rerun():
    at = _approved_session().run()
    for pos in (0, 1):
        at.session_state["pos"] = pos
        at.run()
        next(b for b in at.button if b.key == "approve").click()
        at.run()
    assert not at.exception
    assert "review_complete" in achv.unlocked()
    toasts = [t.value for t in at.toast]
    assert sum("סקירה הושלמה" in t for t in toasts) <= 1
    at.run()                                           # a plain rerun announces nothing new
    assert not at.session_state["toasts"]


def test_database_screen_shows_overview_with_stored_data_only():
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["screen"] = "database"
    at.run()
    assert not at.exception and any("עדיין אין נתונים" in m.value for m in at.markdown)   # empty database
    store.save_all([_review(1, sender="מכבי")], "b1")
    at.run()
    assert not at.exception
