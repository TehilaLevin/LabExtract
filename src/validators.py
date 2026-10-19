"""Pure-Python validators and normalizers. Never guess: return None / False when unsure."""
import json
import re
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional, Union

DEFAULT_SENDERS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "reference" / "approved_senders.json"
)


# --------------------------------------------------------------------------- ID

def validate_israeli_id(id_str: str) -> bool:
    """Return True if id_str is a valid Israeli ID (pad to 9 digits, weighted checksum)."""
    if not isinstance(id_str, str):
        return False
    s = id_str.strip()
    if not s.isascii() or not s.isdigit() or len(s) > 9:
        return False
    s = s.zfill(9)
    if int(s) == 0:
        return False
    total = 0
    for i, ch in enumerate(s):
        n = int(ch) * (1 if i % 2 == 0 else 2)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


# ------------------------------------------------------------------ sending entity

# Quotes, geresh, gershayim and dashes are removed; matching is whole-token based.
_PUNCT_RE = re.compile("[\"'`׳״“”‘’]")
_DASH_RE = re.compile("[‐-―-]")
_SEPARATORS_RE = re.compile(r"[()\[\],:;.]")
_SPACES_RE = re.compile(r"\s+")

# Common prefixes (already punctuation-free), longest first.
_PREFIXES = tuple(
    sorted(
        ("בית חולים", "ביח", "ביהח", "קופת חולים", "קופח", "מרכז רפואי", "המרכז הרפואי"),
        key=len,
        reverse=True,
    )
)


def _normalize_entity(text: str) -> str:
    s = _PUNCT_RE.sub("", _SEPARATORS_RE.sub(" ", _DASH_RE.sub(" ", text or ""))).lower()
    s = _SPACES_RE.sub(" ", s).strip()
    stripped = True
    while stripped and s:
        stripped = False
        for prefix in _PREFIXES:
            if s == prefix:
                break  # nothing left to match on; keep as is
            if s.startswith(prefix + " "):
                s = s[len(prefix):].strip()
                stripped = True
                break
    return s


@lru_cache(maxsize=4)
def _load_senders(path: str) -> tuple:
    """Return (alias_map, canonical_names, parent_of).

    alias_map: normalized alias -> set of canonical names.
    parent_of: campus canonical name -> parent canonical name (from "campus_of").
    Every list-valued top-level key (hmos, hospitals, other_entities, ...) is a group.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    alias_map: dict = {}
    names = []
    parent_of = {}
    for group, entries in data.items():
        if group.startswith("_") or not isinstance(entries, list):
            continue
        for entry in entries:
            names.append(entry["name"])
            if entry.get("campus_of"):
                parent_of[entry["name"]] = entry["campus_of"]
            for alias in [entry["name"], *entry.get("aliases", [])]:
                key = _normalize_entity(alias)
                if key:
                    alias_map.setdefault(key, set()).add(entry["name"])
    return alias_map, tuple(names), parent_of


def list_approved_senders(senders_path: Union[str, Path, None] = None) -> list:
    """Return the canonical names of all approved senders."""
    return list(_load_senders(str(senders_path or DEFAULT_SENDERS_PATH))[1])


def _resolve_campuses(found: set, parent_of: dict) -> set:
    """A campus beats its parent; several campuses of one parent resolve to the parent."""
    campuses = {n for n in found if n in parent_of}
    found = found - {parent_of[c] for c in campuses}
    parents = {parent_of.get(n) for n in found}
    if len(found) > 1 and len(parents) == 1 and None not in parents:
        return {parents.pop()}
    return found


def match_sending_entity(raw_text: str, senders_path: Union[str, Path, None] = None) -> Optional[str]:
    """Map raw text to a canonical approved sender, or None.

    Exact (normalized) match first; otherwise aliases that appear as whole-word
    sequences inside the text. An alias nested inside a longer matched alias is
    ignored (e.g. "השרון" inside "לב השרון"). If the text still matches more than
    one sender, return None - never guess.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None
    alias_map, _, parent_of = _load_senders(str(senders_path or DEFAULT_SENDERS_PATH))
    norm = _normalize_entity(raw_text)
    if not norm:
        return None
    if norm in alias_map:
        hits = alias_map[norm]
        return next(iter(hits)) if len(hits) == 1 else None

    padded = f" {norm} "
    spans = []  # (start, end, names)
    for alias, names in alias_map.items():
        needle = f" {alias} "
        pos = padded.find(needle)
        while pos != -1:
            spans.append((pos, pos + len(needle), names))
            pos = padded.find(needle, pos + 1)
    found = set()
    for start, end, names in spans:
        nested = any(
            s <= start and end <= e and (e - s) > (end - start) for s, e, _ in spans
        )
        if not nested:
            found |= names
    found = _resolve_campuses(found, parent_of)
    return next(iter(found)) if len(found) == 1 else None


# -------------------------------------------------------------------------- dates

class DateCheck(NamedTuple):
    value: Optional[str]   # "DD/MM/YYYY", or None when unreadable / not a valid date
    needs_review: bool
    reason: Optional[str]  # "unreadable" | "invalid" | "future" | "too_old" | None


_DATE_RE = re.compile(r"^(\d{1,4})[./\-\s](\d{1,2})[./\-\s](\d{1,4})$")


def _parse_parts(date_str) -> Optional[tuple]:
    """Return (y, m, d) for a syntactically complete date, else None (unreadable)."""
    if not date_str or not isinstance(date_str, str) or not date_str.isascii():
        return None
    m = _DATE_RE.match(date_str.strip())
    if not m:
        return None
    a, b, c = m.group(1), m.group(2), m.group(3)
    if len(a) == 4:  # ISO: YYYY-MM-DD
        if len(c) > 2:
            return None
        return int(a), int(b), int(c)
    if len(a) > 2 or len(c) not in (2, 4):
        return None
    year = int(c) + 2000 if len(c) == 2 else int(c)  # 2-digit year means 20YY
    return year, int(b), int(a)  # all forms are Israeli: DD/MM/YYYY, day first


def normalize_date(date_str: str) -> Optional[str]:
    """Return "DD/MM/YYYY", or None if unparseable or not a real calendar date.

    Always day-first (Israeli forms); 2-digit years mean 20YY. Also accepts YYYY-MM-DD.
    """
    parts = _parse_parts(date_str)
    if parts is None:
        return None
    y, mo, d = parts
    try:
        date(y, mo, d)
    except ValueError:
        return None
    return f"{d:02d}/{mo:02d}/{y:04d}"


def _one_year_before(today: date) -> date:
    try:
        return today.replace(year=today.year - 1)
    except ValueError:  # 29 Feb
        return today.replace(year=today.year - 1, day=28)


def check_received_date(date_str: str, today: Optional[date] = None,
                        max_age_days: Optional[int] = None, check_age: bool = True) -> DateCheck:
    """Normalize a received date and decide whether it needs review.

    Flagged when: unreadable / partial text, not a valid date, in the future, or older than
    max_age_days (default: one year; check_age=False skips the age check). A flagged-but-valid date keeps
    its value for display.
    """
    today = today or date.today()
    if _parse_parts(date_str) is None:
        return DateCheck(None, True, "unreadable")
    value = normalize_date(date_str)
    if value is None:
        return DateCheck(None, True, "invalid")
    d, mo, y = (int(x) for x in value.split("/"))
    parsed = date(y, mo, d)
    if parsed > today:
        return DateCheck(value, True, "future")
    oldest = _one_year_before(today) if max_age_days is None else today - timedelta(days=max_age_days)
    if check_age and parsed < oldest:
        return DateCheck(value, True, "too_old")
    return DateCheck(value, False, None)


# ------------------------------------------------------------------ number lists

_SPLIT_RE = re.compile(r"[,;\n\r]+")


def normalize_number_list(value: Union[str, list, tuple, None]) -> list:
    """Return a de-duplicated, order-preserving list of non-empty number strings.

    Accepts None, a single string (separated by comma/semicolon/newline), or a list.
    """
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value)
    out: list = []
    for item in items:
        if item is None:
            continue
        for part in _SPLIT_RE.split(str(item)):
            part = part.strip()
            if part and part not in out:
                out.append(part)
    return out


def format_number_list(value: Union[str, list, tuple, None], separator: str = ", ") -> str:
    """Display a list field as comma-separated text (empty string when no values)."""
    return separator.join(normalize_number_list(value))


def date_before(earlier_str: str, later_str: str) -> bool:
    """True when both are valid dates and later_str is strictly before earlier_str (e.g. a sample received
    before it was collected). Unreadable dates never trigger it."""
    a, b = normalize_date(earlier_str), normalize_date(later_str)
    if a is None or b is None:
        return False
    to_date = lambda v: date(int(v[6:]), int(v[3:5]), int(v[:2]))  # noqa: E731
    return to_date(b) < to_date(a)


_DATE_TOKEN_RE = re.compile(r"(?<!\d)\d{1,4}[./\-]\d{1,2}[./\-]\d{2,4}(?!\d)")


def _date_part(text) -> Optional[str]:
    """The first complete date inside text ("17/06/2026 10:18" -> "17/06/2026"), else None."""
    if not isinstance(text, str):
        return None
    m = _DATE_TOKEN_RE.search(text)
    return m.group(0) if m else None


def days_between(earlier_str: str, later_str: str) -> Optional[int]:
    """(later - earlier) in days for two valid dates (a time after the date is ignored), else None."""
    a, b = normalize_date(_date_part(earlier_str)), normalize_date(_date_part(later_str))
    if a is None or b is None:
        return None
    to_date = lambda v: date(int(v[6:]), int(v[3:5]), int(v[:2]))  # noqa: E731
    return (to_date(b) - to_date(a)).days
