"""Local SQLite store for reviewed records (pure Python, no Streamlit).

One row per form, keyed by (batch_id, form_index), so saving the same form again updates it. Every form is
saved as pending when extracted and updated when the reviewer saves or approves it, so the lab can search
and filter what was approved and what still waits, across sessions.

Stored: the 7 field values as the exporter would write them, the 3 metadata columns, the review status and
the reason codes of the flags still open (codes only, never values). Never stored: images, crops, model
suggestions, notes or readings. The file holds personal data: it lives in data/db/ (git-ignored), is
opened by this process only, and `delete_all` empties it. DB_ENABLED=false turns the store off.
"""
import json
import sqlite3
from contextlib import closing
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
    UNIQUE (batch_id, form_index)
);
CREATE INDEX IF NOT EXISTS idx_forms_status ON forms (status);
"""


def _connect(path=None) -> sqlite3.Connection:
    path = path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
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
    cols = ", ".join(_COLUMNS)
    marks = ", ".join("?" for _ in _COLUMNS)
    updates = ", ".join(f"{c}=excluded.{c}" for c in _COLUMNS)
    with closing(_connect(path)) as conn, conn:
        conn.execute(
            f"INSERT INTO forms (batch_id, form_index, status, open_flags, created_at, updated_at, approved_at, "
            f"{cols}) VALUES (?, ?, ?, ?, ?, ?, ?, {marks}) "
            f"ON CONFLICT (batch_id, form_index) DO UPDATE SET status=excluded.status, "
            f"open_flags=excluded.open_flags, updated_at=excluded.updated_at, "
            f"approved_at=excluded.approved_at, {updates}",
            (batch_id, review.form_index, status, flags, now, now, now if review.approved else None,
             *(values[c] for c in _COLUMNS)))


def save_all(reviews: list, batch_id: str, path=None) -> None:
    for review in reviews:
        save_review(review, batch_id, path)


def search(text: str = "", status: Optional[str] = None, sender: Optional[str] = None,
           path=None) -> pd.DataFrame:
    """Stored forms, newest first. `text` matches any field or metadata column (substring); `status` is
    STATUS_APPROVED / STATUS_PENDING; `sender` is an exact sending_entity. Columns: id, status, open_flags
    (a dict), updated_at, then the field and metadata keys."""
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
    sql = (f"SELECT id, status, open_flags, updated_at, {', '.join(_COLUMNS)} FROM forms"
           + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY updated_at DESC, id DESC")
    with closing(_connect(path)) as conn:
        rows = [dict(r) for r in conn.execute(sql, args)]
    frame = pd.DataFrame(rows, columns=["id", "status", "open_flags", "updated_at", *_COLUMNS])
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


def sender_counts(path=None) -> dict:
    """{sending_entity: number of stored forms}; forms without a sender are counted under ""."""
    with closing(_connect(path)) as conn:
        return {row[0]: row[1] for row in conn.execute(
            "SELECT sending_entity, COUNT(*) FROM forms GROUP BY sending_entity")}


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


def delete_all(path=None) -> int:
    with closing(_connect(path)) as conn, conn:
        return conn.execute("DELETE FROM forms").rowcount
