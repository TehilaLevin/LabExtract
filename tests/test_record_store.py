"""SQLite record store: upsert, search, filters, export frame, delete. Uses the temp DB from conftest."""
import pytest

from src import record_builder as rb
from src import record_store as store
from src import review_session as rs
from src.constants import FIELD_KEYS, REVIEW_STATUS_APPROVED, REVIEW_STATUS_KEY

VALID_ID = "000000018"


def _review(index, sender="מכבי", flags=None, **values):
    base = {"first_name": "ישראל", "last_name": "ישראלי", "id_number": VALID_ID, "central_numbers": "937309, 937310",
            "received_date": "01/10/2026", "sending_entity": sender, "commitment_numbers": "50364186"}
    base.update(values)
    record = rb.Record(index, base, dict(flags or {}), {k: 0.9 for k in FIELD_KEYS},
                       {"source_file": "a.pdf", "page_numbers": str(index), "form_type": "טופס"})
    return rs.FormReview(record)


def test_save_is_an_upsert_and_tracks_status():
    review = _review(1, flags={"first_name": "missing"})
    store.save_review(review, "b1")
    store.save_review(review, "b1")
    assert store.counts() == {"total": 1, "approved": 0, "pending": 1}
    review.approved = True
    store.save_review(review, "b1")
    assert store.counts() == {"total": 1, "approved": 1, "pending": 0}
    row = store.search().iloc[0]
    assert row["status"] == store.STATUS_APPROVED and row["open_flags"] == {}


def test_same_form_index_in_another_batch_is_a_new_row():
    store.save_all([_review(1), _review(2)], "b1")
    store.save_all([_review(1)], "b2")
    assert store.counts()["total"] == 3


def test_edits_are_stored_not_the_engine_value():
    review = _review(1)
    review.edits["first_name"] = "דנה"
    store.save_review(review, "b1")
    assert store.search().iloc[0]["first_name"] == "דנה"


def test_search_by_text_status_and_sender():
    a, b = _review(1, sender="מכבי"), _review(2, sender="כללית", first_name="דנה", central_numbers="947000")
    b.approved = True
    store.save_all([a, b], "b1")
    assert len(store.search("937310")) == 1
    assert store.search("דנה").iloc[0]["sending_entity"] == "כללית"
    assert len(store.search(status=store.STATUS_APPROVED)) == 1
    assert store.search(sender="מכבי").iloc[0]["central_numbers"] == "937309, 937310"
    assert store.search("%").empty and store.search("no such value").empty
    assert store.senders() == ["מכבי", "כללית"] or set(store.senders()) == {"מכבי", "כללית"}


def test_export_frame_highlights_open_flags_of_pending_forms_only():
    pending = _review(1, flags={"id_number": "invalid_id"})
    done = _review(2, flags={"id_number": "invalid_id"})
    done.approved = True
    store.save_all([pending, done], "b1")
    stored = store.search().sort_values("id").reset_index(drop=True)
    frame, highlight = store.export_frame(stored)
    assert (0, "id_number") in highlight and (0, REVIEW_STATUS_KEY) in highlight
    assert not any(row == 1 for row, _ in highlight)
    assert frame.loc[1, REVIEW_STATUS_KEY] == REVIEW_STATUS_APPROVED


def test_delete_and_delete_all():
    store.save_all([_review(1), _review(2), _review(3)], "b1")
    ids = list(store.search()["id"])
    assert store.delete(ids[:1]) == 1 and store.delete([]) == 0
    assert store.delete_all() == 2 and store.counts()["total"] == 0


def test_empty_database_search_has_columns():
    frame = store.search()
    assert frame.empty and "first_name" in frame.columns


def test_no_images_or_suggestions_are_stored():
    import sqlite3
    store.save_review(_review(1), "b1")
    conn = sqlite3.connect(store.config.DB_PATH)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(forms)")}
    conn.close()
    assert not any(w in c for c in cols for w in ("image", "crop", "suggest", "note"))
