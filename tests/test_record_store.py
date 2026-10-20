"""Saved records: a run is stored with its images, edits and approval, and comes back unchanged."""
from io import BytesIO

from PIL import Image

from src import record_builder as rb
from src import record_store as store
from src import review_session as rs
from src.constants import FIELD_KEYS
from src.field_crops import PagePreview, SourceCrop


def _jpeg(colour="white"):
    buf = BytesIO()
    Image.new("RGB", (30, 20), colour).save(buf, "JPEG")
    return buf.getvalue()


def _record(index, **flags):
    values = {k: "" for k in FIELD_KEYS}
    values.update(first_name="ישראל", id_number="000000018", central_numbers="482913, 482914")
    return rb.Record(index, values, dict(flags), {k: 0.95 for k in FIELD_KEYS},
                     {"source_file": "batch.pdf", "page_numbers": f"{index}, {index + 1}", "form_type": "מכבי – התחייבות"},
                     suggestions={"last_name": "ישראלי"},
                     field_crops={"id_number": [SourceCrop(1, (100, 200, 150, 400), _jpeg("red"))],
                                  "central_numbers": [SourceCrop(1, (1, 2, 3, 4), _jpeg("blue")),
                                                      SourceCrop(2, (5, 6, 7, 8), _jpeg("green"))]},
                     page_images=[PagePreview(index, _jpeg()), PagePreview(index + 1, _jpeg("gray"))])


def test_run_round_trip_keeps_values_flags_and_images(tmp_path):
    db = tmp_path / "records.db"
    reviews = rs.new_session([_record(1, last_name="name_unverified"), _record(3)])
    run_id = store.save_run(reviews, db)
    assert [r.store_id for r in reviews] == [1, 2]

    loaded = store.load_run(run_id, db)
    assert [r.form_index for r in loaded] == [1, 3]
    first = loaded[0].record
    assert first.values == reviews[0].record.values and first.flags == {"last_name": "name_unverified"}
    assert first.suggestions == {"last_name": "ישראלי"}
    assert [c.page for c in first.field_crops["central_numbers"]] == [1, 2]
    assert first.field_crops["id_number"][0].box == (100, 200, 150, 400)
    assert first.field_crops["id_number"][0].jpeg == reviews[0].record.field_crops["id_number"][0].jpeg
    assert [p.page_number for p in first.page_images] == [1, 2]


def test_edits_and_approval_are_saved(tmp_path):
    db = tmp_path / "records.db"
    reviews = rs.new_session([_record(1)])
    run_id = store.save_run(reviews, db)
    rs.approve(reviews[0], {"last_name": "ישראלי"})
    store.update_review(reviews[0], db)

    loaded = store.load_run(run_id, db)[0]
    assert loaded.approved and loaded.value("last_name") == "ישראלי"
    listed = store.list_records(db)
    assert listed[0]["approved"] and listed[0]["review"].value("last_name") == "ישראלי"


def test_unsaved_review_is_not_written(tmp_path):
    db = tmp_path / "records.db"
    store.update_review(rs.new_session([_record(1)])[0], db)
    assert store.list_records(db) == []


def test_delete_removes_records_and_images(tmp_path):
    db = tmp_path / "records.db"
    reviews = rs.new_session([_record(1), _record(2)])
    store.save_run(reviews, db)
    assert store.delete_records([reviews[0].store_id], db) == 1
    assert [r["id"] for r in store.list_records(db)] == [reviews[1].store_id]
    with store.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM images WHERE record_id = ?", (reviews[0].store_id,)).fetchone()[0] == 0
