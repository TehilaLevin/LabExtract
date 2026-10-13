"""Pure-Python validators and normalizers. Never guess: return None / False when unsure."""
import json
import re
from datetime import date
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
    s = _PUNCT_RE.sub("", _DASH_RE.sub(" ", text or "")).lower()
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
    """Return (alias_map, canonical_names): normalized alias -> set of canonical names."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    alias_map: dict = {}
    names = []
    for group in ("hmos", "hospitals"):
        for entry in data.get(group, []):
            names.append(entry["name"])
            for alias in [entry["name"], *entry.get("aliases", [])]:
                key = _normalize_entity(alias)
                if key:
                    alias_map.setdefault(key, set()).add(entry["name"])
    return alias_map, tuple(names)


def list_approved_senders(senders_path: Union[str, Path, None] = None) -> list:
    """Return the canonical names of all approved senders."""
    return list(_load_senders(str(senders_path or DEFAULT_SENDERS_PATH))[1])


def match_sending_entity(raw_text: str, senders_path: Union[str, Path, None] = None) -> Optional[str]:
    """Map raw text to a canonical approved sender, or None.

    Exact (normalized) match first; otherwise an alias that appears as a whole-word
    sequence inside the text. If the text matches more than one sender, return None.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None
    alias_map, _ = _load_senders(str(senders_path or DEFAULT_SENDERS_PATH))
    norm = _normalize_entity(raw_text)
    if not norm:
        return None
    if norm in alias_map:
        hits = alias_map[norm]
        return next(iter(hits)) if len(hits) == 1 else None

    padded = f" {norm} "
    found = set()
    for alias, names in alias_map.items():
        if f" {alias} " in padded:
            found |= names
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


def check_received_date(date_str: str, today: Optional[date] = None) -> DateCheck:
    """Normalize a received date and decide whether it needs review.

    Flagged when: unreadable / partial text, not a valid date, in the future, or more
    than one year in the past. A flagged-but-valid date keeps its value for display.
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
    if parsed < _one_year_before(today):
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
