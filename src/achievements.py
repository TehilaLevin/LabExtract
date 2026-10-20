"""Achievements: a few quiet milestones (pure Python, no Streamlit).

`evaluate` decides which achievements are earned from plain numbers; `unlock` records them once in the
existing SQLite file (tables `achievements` and `usage_counters`), so a rerun or a page refresh never
announces the same one twice. Nothing personal is stored: only achievement ids, their unlock time and the
number of forms processed. The tables are separate from `forms`, so "delete the whole database" (which
removes patient records) keeps them. Hidden achievements show no description until unlocked.
"""
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src import config

PROCESSED_COUNTER = "forms_processed"
BATCH_MIN_FORMS = 2          # a "batch" is at least this many forms
SENDERS_GOAL = 5             # distinct approved senders for "wide reach"
NIGHT_HOURS = range(0, 5)    # local hours 00:00-04:59

_SCHEMA = """
CREATE TABLE IF NOT EXISTS achievements (id TEXT PRIMARY KEY, unlocked_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS usage_counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL DEFAULT 0);
"""


@dataclass(frozen=True)
class Achievement:
    id: str
    title: str
    description: str
    hidden: bool = False


ACHIEVEMENTS = (
    Achievement("first_form", "הטופס הראשון", "טופס ראשון עבר חילוץ בהצלחה"),
    Achievement("batch_complete", "אצווה הושלמה", f"אצווה של {BATCH_MIN_FORMS} טפסים ויותר סיימה עיבוד"),
    Achievement("review_complete", "סקירה הושלמה", "כל הטפסים באצווה נבדקו ואושרו"),
    Achievement("wide_reach", "טווח רחב", f"טפסים מ-{SENDERS_GOAL} גורמים שולחים מהרשימה המאושרת"),
    Achievement("century_club", "מועדון המאה", "100 טפסים עובדו"),
    Achievement("five_hundred", "חמש מאות", "500 טפסים עובדו"),
    Achievement("thousand", "אלף טפסים", "1,000 טפסים עובדו"),
    Achievement("night_shift", "משמרת לילה", "אישור טופס בין חצות לחמש בבוקר", hidden=True),
)
BY_ID = {a.id: a for a in ACHIEVEMENTS}


@dataclass(frozen=True)
class Stats:
    """The facts the conditions look at."""
    processed_total: int = 0      # forms processed successfully, all time
    batch_size: int = 0           # forms in the current batch
    batch_approved: int = 0       # of them, approved
    batch_finished: bool = False  # the current batch finished extraction
    known_senders: int = 0        # distinct approved senders among stored forms
    hour: Optional[int] = None    # local hour of an approval that just happened, else None


def evaluate(stats: Stats) -> set:
    """Ids of every achievement whose condition holds right now."""
    earned = set()
    if stats.processed_total >= 1:
        earned.add("first_form")
    if stats.batch_finished and stats.batch_size >= BATCH_MIN_FORMS:
        earned.add("batch_complete")
    if stats.batch_size >= BATCH_MIN_FORMS and stats.batch_approved == stats.batch_size:
        earned.add("review_complete")
    if stats.known_senders >= SENDERS_GOAL:
        earned.add("wide_reach")
    for id_, goal in (("century_club", 100), ("five_hundred", 500), ("thousand", 1000)):
        if stats.processed_total >= goal:
            earned.add(id_)
    if stats.hour in NIGHT_HOURS:
        earned.add("night_shift")
    return earned


# ------------------------------------------------------------------ persistence

def _connect(path=None) -> sqlite3.Connection:
    path = path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    return conn


def unlocked(path=None) -> dict:
    """{achievement id: unlock time} of everything recorded so far."""
    with closing(_connect(path)) as conn:
        return dict(conn.execute("SELECT id, unlocked_at FROM achievements"))


def unlock(ids, path=None) -> list:
    """Record ids once; return only the ones that were new (in ACHIEVEMENTS order). Idempotent."""
    new = []
    now = datetime.now().isoformat(timespec="seconds")
    with closing(_connect(path)) as conn, conn:
        for a in ACHIEVEMENTS:
            if a.id in ids and conn.execute("INSERT OR IGNORE INTO achievements (id, unlocked_at) VALUES (?, ?)",
                                            (a.id, now)).rowcount:
                new.append(a.id)
    return new


def processed_total(path=None) -> int:
    with closing(_connect(path)) as conn:
        row = conn.execute("SELECT value FROM usage_counters WHERE name = ?", (PROCESSED_COUNTER,)).fetchone()
    return row[0] if row else 0


def add_processed(n: int, path=None) -> int:
    """Add n successfully processed forms to the all-time counter; return the new total."""
    with closing(_connect(path)) as conn, conn:
        conn.execute("INSERT INTO usage_counters (name, value) VALUES (?, ?) "
                     "ON CONFLICT (name) DO UPDATE SET value = value + excluded.value",
                     (PROCESSED_COUNTER, max(n, 0)))
    return processed_total(path)
