"""Local SQLite store for reviewed records (pure Python, no Streamlit).

One row per form, keyed by (batch_id, form_index), so saving the same form again updates it. Every form is
saved as pending when extracted and updated when the reviewer saves or approves it, so the lab can search
and filter what was approved and what still waits, across sessions.

Stored (the file holds personal data: data/db/, git-ignored, opened by this process only, NOT encrypted):
- the 7 field values as the exporter would write them, the 3 metadata columns, the review status, the reason
  codes of the flags still open, and the reviewer's edits;
- record_json: the engine's record without images and WITHOUT raw_texts (never stored). While a form is not
  approved it also holds suggestions (plain-text readings of handwritten names, unverified), alternatives and
  notes; approving the form strips all three, leaving the final values, flag codes and edits;
- images (JPEG, quality 85): the field crops (up to 1200 px on the long side) of every form, so a crop sits
  next to each field also after approval and when a saved form is reopened; and the page previews (up to
  1600 px) of forms NOT approved yet - approving deletes them.
DB_STORE_CROPS=false stores no image at all. DB_ENABLED=false turns the store off. Nothing is ever deleted
automatically: `delete`, `delete_images` and `delete_all` are manual.
"""
import json
import sqlite3
from contextlib import closing
from dataclasses import fields
from datetime import datetime
from typing import Optional

import pandas as pd

from src import config
from src.excel_exporter import expand_per_commitment
from src.constants import (
    FIELD_KEYS,
    METADATA_KEYS,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_KEY,
    REVIEW_STATUS_PENDING,
)

STATUS_APPROVED = "approved"
STATUS_PENDING = "pending"
_COLUMNS = (*FIELD_KEYS, *METADATA_KEYS)
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS forms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    form_index INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT '{STATUS_PENDING}',
    open_flags TEXT NOT NULL DEFAULT '{{}}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    approved_at TEXT,
    {", ".join(f"{c} TEXT NOT NULL DEFAULT ''" for c in _COLUMNS)},
    record_json TEXT NOT NULL DEFAULT '',
    edits_json TEXT NOT NULL DEFAULT '{{}}',
    UNIQUE (batch_id, form_index)
);
CREATE INDEX IF NOT EXISTS idx_forms_status ON forms (status);
CREATE TABLE IF NOT EXISTS images (
    form_id INTEGER NOT NULL REFERENCES forms(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,              -- 'crop' (one field) or 'page' (page preview)
    field TEXT,
    page INTEGER NOT NULL,           -- crop: page within the form; page: page number in the source file
    box TEXT,                        -- crop: (ymin, xmin, ymax, xmax) 0-1000, JSON
    seq INTEGER NOT NULL,
    jpeg BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_form ON images (form_id);
"""
# Columns added after the first release: added to an existing database file when it is opened.
_ADDED_COLUMNS = {"record_json": "TEXT NOT NULL DEFAULT ''", "edits_json": "TEXT NOT NULL DEFAULT '{}'"}
_IMAGE_FIELDS = ("field_crops", "page_images")
_NEVER_STORED = ("raw_texts",)                           # model readings of the text as written
_PENDING_ONLY = ("suggestions", "alternatives", "notes")  # reading aids: removed on approval


def _connect(path=None) -> sqlite3.Connection:
    path = path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    existing = {row[1] for row in conn.execute("PRAGMA table_info(forms)")}
    if existing:
        for column, kind in _ADDED_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE forms ADD COLUMN {column} {kind}")
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def save_review(review, batch_id: str, path=None) -> None:
    """Insert or update one reviewed form (a review_session.FormReview) as pending or approved."""
    values = {key: review.value(key) for key in FIELD_KEYS}
    values.update({key: str(review.record.metadata.get(key, "")) for key in METADATA_KEYS})
    now = _now()
    status = STATUS_APPROVED if review.approved else STATUS_PENDING
    flags = json.dumps(review.open_flags(), sort_keys=True)
    record = {f.name: getattr(review.record, f.name) for f in fields(review.record)
              if f.name not in _IMAGE_FIELDS and f.name not in _NEVER_STORED}
    if review.approved:
        record.update({name: {} for name in _PENDING_ONLY})
    record_json = json.dumps(record, ensure_ascii=False, default=str)
    edits_json = json.dumps(review.edits, ensure_ascii=False)
    cols = ", ".join(_COLUMNS)
    marks = ", ".join("?" for _ in _COLUMNS)
    updates = ", ".join(f"{c}=excluded.{c}" for c in _COLUMNS)
    with closing(_connect(path)) as conn, conn:
        conn.execute(
            f"INSERT INTO forms (batch_id, form_index, status, open_flags, created_at, updated_at, approved_at, "
            f"record_json, edits_json, {cols}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, {marks}) "
            f"ON CONFLICT (batch_id, form_index) DO UPDATE SET status=excluded.status, "
            f"open_flags=excluded.open_flags, updated_at=excluded.updated_at, "
            f"approved_at=excluded.approved_at, record_json=excluded.record_json, "
            f"edits_json=excluded.edits_json, {updates}",
            (batch_id, review.form_index, status, flags, now, now, now if review.approved else None,
             record_json, edits_json, *(values[c] for c in _COLUMNS)))
        form_id = conn.execute("SELECT id FROM forms WHERE batch_id = ? AND form_index = ?",
                               (batch_id, review.form_index)).fetchone()[0]
        if review.approved:  # approved forms keep only their field crops
            conn.execute("DELETE FROM images WHERE form_id = ? AND kind = 'page'", (form_id,))
        if config.DB_STORE_CROPS:
            _insert_images(conn, form_id, review.record, with_pages=not review.approved)


def _insert_images(conn: sqlite3.Connection, form_id: int, record, with_pages: bool) -> None:
    """Crops (and, for forms not approved, page previews): each kind is written once, the record never changes."""
    have = {row[0] for row in conn.execute("SELECT DISTINCT kind FROM images WHERE form_id = ?", (form_id,))}
    rows = []
    if "crop" not in have:
        for key, crops in record.field_crops.items():
            rows += [(form_id, "crop", key, c.page, json.dumps(list(c.box)), i, c.jpeg) for i, c in enumerate(crops)]
    if with_pages and "page" not in have:
        rows += [(form_id, "page", None, p.page_number, None, i, p.jpeg) for i, p in enumerate(record.page_images)]
    conn.executemany("INSERT INTO images (form_id, kind, field, page, box, seq, jpeg) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     rows)


def load_batch(batch_id: str, path=None) -> list:
    """[FormReview] of one extraction run, in form order, with their crops and page previews, ready for the
    review screen. Rows saved before record_json existed cannot be reopened and are skipped."""
    from src.field_crops import PagePreview, SourceCrop
    from src.record_builder import Record
    from src.review_session import FormReview

    out = []
    with closing(_connect(path)) as conn:
        rows = conn.execute("SELECT id, status, record_json, edits_json FROM forms WHERE batch_id = ? "
                            "AND record_json != '' ORDER BY form_index", (batch_id,)).fetchall()
        for row in rows:
            crops: dict = {}
            pages = []
            for img in conn.execute("SELECT kind, field, page, box, jpeg FROM images WHERE form_id = ? "
                                    "ORDER BY kind, seq", (row["id"],)):
                if img["kind"] == "crop":
                    crops.setdefault(img["field"], []).append(
                        SourceCrop(img["page"], tuple(json.loads(img["box"])), img["jpeg"]))
                else:
                    pages.append(PagePreview(img["page"], img["jpeg"]))
            record = Record(**json.loads(row["record_json"]),
                            field_crops={k: crops[k] for k in FIELD_KEYS if k in crops}, page_images=pages)
            out.append(FormReview(record, edits=json.loads(row["edits_json"]),
                                  approved=row["status"] == STATUS_APPROVED))
    return out


def save_all(reviews: list, batch_id: str, path=None) -> None:
    for review in reviews:
        save_review(review, batch_id, path)


def search(text: str = "", status: Optional[str] = None, sender: Optional[str] = None,
           path=None) -> pd.DataFrame:
    """Stored forms, newest first. `text` matches any field or metadata column (substring); `status` is
    STATUS_APPROVED / STATUS_PENDING; `sender` is an exact sending_entity. Columns: id, batch_id, form_index,
    status, open_flags (a dict), updated_at, then the field and metadata keys."""
    where, args = [], []
    text = " ".join((text or "").split())
    if text:
        escaped = text.replace("!", "!!").replace("%", "!%").replace("_", "!_")  # "!" is the LIKE escape char
        like = f"%{escaped}%"
        where.append("(" + " OR ".join(f"{c} LIKE ? ESCAPE '!'" for c in _COLUMNS) + ")")
        args += [like] * len(_COLUMNS)
    if status:
        where.append("status = ?")
        args.append(status)
    if sender:
        where.append("sending_entity = ?")
        args.append(sender)
    sql = (f"SELECT id, batch_id, form_index, status, open_flags, updated_at, {', '.join(_COLUMNS)} FROM forms"
           + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY updated_at DESC, id DESC")
    with closing(_connect(path)) as conn:
        rows = [dict(r) for r in conn.execute(sql, args)]
    frame = pd.DataFrame(rows, columns=["id", "batch_id", "form_index", "status", "open_flags", "updated_at", *_COLUMNS])
    frame["open_flags"] = frame["open_flags"].map(json.loads)
    return frame


def counts(path=None) -> dict:
    with closing(_connect(path)) as conn:
        rows = conn.execute("SELECT status, COUNT(*) FROM forms GROUP BY status").fetchall()
    found = {row[0]: row[1] for row in rows}
    approved, pending = found.get(STATUS_APPROVED, 0), found.get(STATUS_PENDING, 0)
    return {"total": approved + pending, "approved": approved, "pending": pending}


def senders(path=None) -> list:
    with closing(_connect(path)) as conn:
        return [r[0] for r in conn.execute(
            "SELECT DISTINCT sending_entity FROM forms WHERE sending_entity != '' ORDER BY sending_entity")]


def sender_counts(path=None, approved_only: bool = False) -> dict:
    """{sending_entity: number of stored forms}; forms without a sender are counted under "".

    approved_only=True counts only forms the reviewer approved (used by the activity overview).
    """
    where = "WHERE status = ? " if approved_only else ""
    args = (STATUS_APPROVED,) if approved_only else ()
    with closing(_connect(path)) as conn:
        return {row[0]: row[1] for row in conn.execute(
            f"SELECT sending_entity, COUNT(*) FROM forms {where}GROUP BY sending_entity", args)}


def archive_rows(path=None) -> list:
    """Every stored form as a plain dict, newest first, for the archive assistant (src/archive_assistant.py).

    Keys: id, batch_id, form_index, status, open_flags (a dict), created_at, approved_at, then the field and
    metadata columns. Read only; never record_json, edits or images.
    """
    sql = (f"SELECT id, batch_id, form_index, status, open_flags, created_at, approved_at, {', '.join(_COLUMNS)} "
           "FROM forms ORDER BY created_at DESC, id DESC")
    with closing(_connect(path)) as conn:
        rows = [dict(r) for r in conn.execute(sql)]
    for row in rows:
        row["open_flags"] = json.loads(row["open_flags"] or "{}")
    return rows


def export_frame(stored: pd.DataFrame) -> tuple:
    """(DataFrame, highlight) for excel_exporter from a `search` result: same as the review export (one row per
    commitment; the database itself stays one row per form);
    flagged fields of forms not yet approved are highlighted."""
    frame = stored[[*FIELD_KEYS, *METADATA_KEYS]].copy()
    frame[REVIEW_STATUS_KEY] = stored["status"].map(
        {STATUS_APPROVED: REVIEW_STATUS_APPROVED, STATUS_PENDING: REVIEW_STATUS_PENDING})
    frame = frame.reset_index(drop=True)
    highlight = set()
    for pos, (status, flags) in enumerate(zip(stored["status"], stored["open_flags"])):
        if status != STATUS_APPROVED:
            highlight |= {(pos, key) for key in flags}
            highlight.add((pos, REVIEW_STATUS_KEY))
    return expand_per_commitment(frame, highlight)


def delete(ids: list, path=None) -> int:
    if not ids:
        return 0
    with closing(_connect(path)) as conn, conn:
        cur = conn.execute(f"DELETE FROM forms WHERE id IN ({', '.join('?' for _ in ids)})", list(ids))
        return cur.rowcount


def delete_images(path=None) -> int:
    """Delete every stored image (crops and page previews); values, edits and status stay. Returns the count."""
    with closing(_connect(path)) as conn, conn:
        return conn.execute("DELETE FROM images").rowcount


def delete_all(path=None) -> int:
    with closing(_connect(path)) as conn, conn:
        deleted = conn.execute("DELETE FROM forms").rowcount
        conn.execute("DELETE FROM images")
        return deleted
