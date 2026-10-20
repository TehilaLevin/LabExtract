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


def _jpeg(colour="white"):
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (30, 20), colour).save(buf, "JPEG")
    return buf.getvalue()


def test_saved_batch_reopens_with_crops_page_previews_and_edits():
    from src.field_crops import PagePreview, SourceCrop

    review = _review(1, flags={"last_name": "name_unverified"}, last_name="")
    review.record.suggestions["last_name"] = "ישראלי"
    review.record.field_crops["central_numbers"] = [SourceCrop(1, (1, 2, 3, 4), _jpeg("red")),
                                                    SourceCrop(2, (5, 6, 7, 8), _jpeg("blue"))]
    review.record.page_images.extend([PagePreview(7, _jpeg()), PagePreview(8, _jpeg("gray"))])
    store.save_review(review, "b1")
    review.edits["last_name"] = "ישראלי"
    store.save_review(review, "b1")  # an update keeps the images written on insert

    loaded = store.load_batch("b1")
    assert len(loaded) == 1
    again = loaded[0]
    assert again.value("last_name") == "ישראלי" and again.record.flags == {"last_name": "name_unverified"}
    assert again.record.suggestions == {"last_name": "ישראלי"}
    crops = again.record.field_crops["central_numbers"]
    assert [c.page for c in crops] == [1, 2] and crops[1].box == (5, 6, 7, 8)
    assert crops[0].jpeg == review.record.field_crops["central_numbers"][0].jpeg
    assert [p.page_number for p in again.record.page_images] == [7, 8]


def test_delete_removes_the_images_too():
    from src import config
    from src.field_crops import SourceCrop

    review = _review(1)
    review.record.field_crops["id_number"] = [SourceCrop(1, (1, 2, 3, 4), _jpeg())]
    store.save_review(review, "b1")
    store.delete(list(store.search()["id"]))
    with store._connect(config.DB_PATH) as conn:
        assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0


def test_database_from_before_the_images_is_upgraded_in_place():
    import sqlite3

    from src import config

    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    old = sqlite3.connect(config.DB_PATH)
    old.execute("CREATE TABLE forms (id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT NOT NULL, "
                "form_index INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending', open_flags TEXT NOT NULL "
                "DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL, approved_at TEXT, "
                + ", ".join(f"{c} TEXT NOT NULL DEFAULT ''" for c in store._COLUMNS)
                + ", UNIQUE (batch_id, form_index))")
    old.commit()
    old.close()
    store.save_review(_review(1), "b1")
    assert store.counts()["total"] == 1 and len(store.load_batch("b1")) == 1


# ---- what is stored, and when it is removed ---------------------------------------------------

def _full_review(index=1):
    """A form with a crop, two page previews and every reading aid, as the engine produces it."""
    from src.field_crops import PagePreview, SourceCrop

    review = _review(index, flags={"last_name": "name_unverified"}, last_name="")
    rec = review.record
    rec.suggestions["last_name"] = "ישראלי"
    rec.alternatives["id_number"] = ["000000026"]
    rec.notes["last_name"] = "כתב יד בראש הטופס"
    rec.raw_texts["last_name"] = "ישר?לי"
    rec.field_crops["last_name"] = [SourceCrop(1, (1, 2, 3, 4), _jpeg("red"))]
    rec.page_images.extend([PagePreview(1, _jpeg()), PagePreview(2, _jpeg("gray"))])
    return review


def _image_kinds():
    from src import config

    with store._connect(config.DB_PATH) as conn:
        return sorted(r[0] for r in conn.execute("SELECT kind FROM images"))


def _record_json(form_id=1):
    import json

    from src import config

    with store._connect(config.DB_PATH) as conn:
        return json.loads(conn.execute("SELECT record_json FROM forms WHERE id = ?", (form_id,)).fetchone()[0])


def test_pending_form_keeps_crops_page_previews_and_suggestions():
    store.save_review(_full_review(), "b1")
    assert _image_kinds() == ["crop", "page", "page"]
    stored = _record_json()
    assert stored["suggestions"] == {"last_name": "ישראלי"} and stored["alternatives"] and stored["notes"]


def test_raw_texts_are_never_stored():
    review = _full_review()
    store.save_review(review, "b1")
    review.approved = True
    store.save_review(review, "b1")
    assert "raw_texts" not in _record_json()
    assert store.load_batch("b1")[0].record.raw_texts == {}


def test_approval_deletes_page_previews_and_keeps_crops():
    review = _full_review()
    store.save_review(review, "b1")
    review.approved = True
    store.save_review(review, "b1")
    assert _image_kinds() == ["crop"]
    again = store.load_batch("b1")[0]
    assert again.record.page_images == [] and len(again.record.field_crops["last_name"]) == 1


def test_approval_strips_suggestions_alternatives_and_notes_but_keeps_values_flags_and_edits():
    review = _full_review()
    store.save_review(review, "b1")
    review.edits["last_name"] = "ישראלי"
    review.approved = True
    store.save_review(review, "b1")
    stored = _record_json()
    assert stored["suggestions"] == {} and stored["alternatives"] == {} and stored["notes"] == {}
    assert stored["flags"] == {"last_name": "name_unverified"} and stored["values"]["first_name"] == "ישראל"
    again = store.load_batch("b1")[0]
    assert again.value("last_name") == "ישראלי" and again.approved
    assert store.search().iloc[0]["last_name"] == "ישראלי"


def test_db_store_crops_false_stores_no_images(monkeypatch):
    monkeypatch.setattr(store.config, "DB_STORE_CROPS", False)
    store.save_review(_full_review(), "b1")
    assert _image_kinds() == []
    again = store.load_batch("b1")[0]
    assert again.record.field_crops == {} and again.record.page_images == []
    assert again.value("first_name") == "ישראל"          # the values are stored regardless


def test_delete_images_only_keeps_the_values():
    store.save_review(_full_review(1), "b1")
    store.save_review(_full_review(2), "b1")
    assert store.delete_images() == 6
    assert _image_kinds() == []
    assert store.counts()["total"] == 2
    assert set(store.search()["first_name"]) == {"ישראל"}
    assert len(store.load_batch("b1")) == 2


def test_delete_one_record_removes_its_images_and_keeps_the_others():
    store.save_review(_full_review(1), "b1")
    store.save_review(_full_review(2), "b1")
    first = int(store.search().sort_values("id").iloc[0]["id"])
    assert store.delete([first]) == 1
    assert store.counts()["total"] == 1
    assert _image_kinds() == ["crop", "page", "page"]       # only the second form's images remain
