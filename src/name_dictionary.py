"""Dictionary check of handwritten-name readings, like the check digit for an ID number.

The model often confuses Hebrew cursive letters that look alike (seen: ז read as צ, "אזולאי" -> "אצולאי") and
drops or adds a ו / י. A reading word that is not a known name is compared with the dictionary
(data/reference/israeli_names.json): the variants within MAX_EDITS look-alike edits are generated, and when
exactly one variant at the smallest distance is a known name, that name is the correction.

A correction is NEVER an approved value: record_builder uses it only for the orange "suggestion - unverified",
and keeps the model's own reading as an alternative for the reviewer. A word that is already a known name,
or with no / several dictionary matches, is left as read. The dictionary is public data (no patient data).
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

MAX_EDITS = 2

# Letter pairs that look alike in Hebrew handwriting (cursive and print-like), in both directions.
CONFUSABLE_PAIRS = (
    ("ז", "צ"), ("ז", "ו"), ("ו", "ן"), ("ו", "י"), ("ר", "ד"), ("ר", "ך"), ("ד", "ך"), ("ה", "ח"),
    ("ח", "ת"), ("ה", "ת"), ("ב", "כ"), ("ג", "נ"), ("ס", "ם"), ("ע", "צ"), ("ט", "מ"),
)
# Vowel letters (אמות קריאה) that handwriting adds or drops.
OPTIONAL_LETTERS = ("ו", "י")
_FINALS = str.maketrans("ךםןףץ", "כמנפצ")

_SUBSTITUTES: dict = {}
for _a, _b in CONFUSABLE_PAIRS:
    _SUBSTITUTES.setdefault(_a, set()).add(_b)
    _SUBSTITUTES.setdefault(_b, set()).add(_a)


def lookup_key(word: str) -> str:
    """Comparison form: final letters folded to their regular form (a final-form slip is no difference)."""
    return word.translate(_FINALS)


@lru_cache(maxsize=4)
def load_names(path: str) -> frozenset:
    """All first names and surnames as lookup keys. The forms do not write names in a fixed order, so a
    word is checked against both lists."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return frozenset(lookup_key(w) for key in ("first_names", "last_names") for w in data.get(key, []))


def _one_edit(word: str) -> set:
    out = set()
    for i, ch in enumerate(word):
        for sub in _SUBSTITUTES.get(ch, ()):
            out.add(word[:i] + sub + word[i + 1:])
        if ch in OPTIONAL_LETTERS:
            out.add(word[:i] + word[i + 1:])
    for i in range(1, len(word) + 1):  # not before the first letter
        for letter in OPTIONAL_LETTERS:
            out.add(word[:i] + letter + word[i:])
    return {lookup_key(w) for w in out if w}


def correct_word(word: str, names: frozenset) -> Optional[str]:
    """The single dictionary name closest to `word` by look-alike edits, or None (already a name, no match,
    or several equally close matches)."""
    key = lookup_key(word)
    if len(key) < 2 or key in names:
        return None
    seen, frontier = {key}, {key}
    for _ in range(MAX_EDITS):
        frontier = {v for w in frontier for v in _one_edit(w)} - seen
        seen |= frontier
        hits = frontier & names
        if len(hits) == 1:
            return _with_final(next(iter(hits)))
        if hits:
            return None
    return None


def _with_final(key: str) -> str:
    finals = {"כ": "ך", "מ": "ם", "נ": "ן", "פ": "ף", "צ": "ץ"}
    return key[:-1] + finals.get(key[-1], key[-1]) if key else key


def correct_name(text: str, names: frozenset) -> Optional[str]:
    """Correct each word of a name; None when no word changed."""
    words = text.split()
    corrected = [correct_word(w, names) or w for w in words]
    return " ".join(corrected) if corrected != words else None
