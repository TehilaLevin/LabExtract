"""Saved records: every extracted form is kept in a local SQLite database, one row per commitment.

A row holds the engine's record (values, flags, confidences, metadata and the reading aids), the reviewer's
corrections and the approval status. The images shown on the review screen - the crop of each field and
the page previews - are kept with it, so a saved record can be opened later with a crop next to every field.

Privacy: the database holds personal data. It lives in config.RECORDS_DB (data/records/, git-ignored) on
the lab computer only, is never logged or sent anywhere, and rows are removed with delete_records.
"""
import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Optional

from src import config
from src.constants import FIELD_KEYS
from src.field_crops import PagePreview, SourceCrop
from src.record_builder import Record
from src.review_session import FormReview

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,            -- one extraction run (the files uploaded together)
    position INTEGER NOT NULL,       -- order of the form within its run
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    approved INTEGER NOT NULL DEFAULT 0,
    record_json TEXT NOT NULL,       -- record_builder.Record without its images
    edits_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS images (
    record_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,              -- 'crop' (one field) or 'page' (page preview)
    field TEXT,
    page INTEGER NOT NULL,           -- crop: page within the form; page: page number in the source file
    box TEXT,                        -- crop: (ymin, xmin, ymax, xmax) 0-1000, JSON
    seq INTEGER NOT NULL,
    jpeg BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS images_record ON images(record_id);
"""
_IMAGE_FIELDS = ("field_crops", "page_images")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(path or config.RECORDS_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def _record_json(record: Record) -> str:
    data = {f.name: getattr(record, f.name) for f in fields(record) if f.name not in _IMAGE_FIELDS}
    return json.dumps(data, ensure_ascii=False, default=str)


def _insert(conn: sqlite3.Connection, review: FormReview, run_id: str, position: int) -> int:
    now = _now()
    cur = conn.execute(
        "INSERT INTO records (run_id, position, created_at, updated_at, approved, record_json, edits_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, position, now, now, int(review.approved), _record_json(review.record),
         json.dumps(review.edits, ensure_ascii=False)))
    record_id = cur.lastrowid
    rows = []
    for key, crops in review.record.field_crops.items():
        rows += [(record_id, "crop", key, c.page, json.dumps(list(c.box)), i, c.jpeg) for i, c in enumerate(crops)]
    rows += [(record_id, "page", None, p.page_number, None, i, p.jpeg)
             for i, p in enumerate(review.record.page_images)]
    conn.executemany("INSERT INTO images (record_id, kind, field, page, box, seq, jpeg) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     rows)
    return record_id


def save_run(reviews: list, path: Optional[Path] = None) -> str:
    """Save a new extraction run; sets each review's store_id. Returns the run id."""
    run_id = uuid.uuid4().hex
    with closing(connect(path)) as conn, conn:
        for position, review in enumerate(reviews):
            review.store_id = _insert(conn, review, run_id, position)
    return run_id


def update_review(review: FormReview, path: Optional[Path] = None) -> None:
    """Store the reviewer's corrections and approval status of a saved review (no-op when it is not saved)."""
    if review.store_id is None:
        return
    with closing(connect(path)) as conn, conn:
        conn.execute("UPDATE records SET edits_json = ?, approved = ?, updated_at = ? WHERE id = ?",
                     (json.dumps(review.edits, ensure_ascii=False), int(review.approved), _now(), review.store_id))


def _review(conn: sqlite3.Connection, row: sqlite3.Row) -> FormReview:
    data = json.loads(row["record_json"])
    crops: dict = {}
    pages = []
    for img in conn.execute("SELECT kind, field, page, box, jpeg FROM images WHERE record_id = ? ORDER BY kind, seq",
                            (row["id"],)):
        if img["kind"] == "crop":
            crops.setdefault(img["field"], []).append(SourceCrop(img["page"], tuple(json.loads(img["box"])), img["jpeg"]))
        else:
            pages.append(PagePreview(img["page"], img["jpeg"]))
    record = Record(**data, field_crops={k: crops[k] for k in FIELD_KEYS if k in crops}, page_images=pages)
    return FormReview(record, edits=json.loads(row["edits_json"]), approved=bool(row["approved"]), store_id=row["id"])


def load_run(run_id: str, path: Optional[Path] = None) -> list:
    """[FormReview] of one run, in their original order."""
    with closing(connect(path)) as conn:
        rows = conn.execute("SELECT * FROM records WHERE run_id = ? ORDER BY position", (run_id,)).fetchall()
        return [_review(conn, row) for row in rows]


def list_records(path: Optional[Path] = None) -> list:
    """Every saved record, newest first: [{id, run_id, created_at, updated_at, approved, review}].

    `review` is a FormReview without images (enough for the Records table); open a run with load_run.
    """
    with closing(connect(path)) as conn:
        rows = conn.execute("SELECT * FROM records ORDER BY created_at DESC, position").fetchall()
    out = []
    for row in rows:
        record = Record(**json.loads(row["record_json"]))
        review = FormReview(record, edits=json.loads(row["edits_json"]), approved=bool(row["approved"]),
                            store_id=row["id"])
        out.append({"id": row["id"], "run_id": row["run_id"], "created_at": row["created_at"],
                    "updated_at": row["updated_at"], "approved": bool(row["approved"]), "review": review})
    return out


def delete_records(ids: list, path: Optional[Path] = None) -> int:
    """Delete saved records and their images. Returns how many were deleted."""
    if not ids:
        return 0
    with closing(connect(path)) as conn, conn:
        cur = conn.execute(f"DELETE FROM records WHERE id IN ({','.join('?' * len(ids))})", [int(i) for i in ids])
        return cur.rowcount
