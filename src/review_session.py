"""Review state for the Streamlit app: reviewer edits, approvals, export rows (pure Python, no Streamlit).

One FormReview per extracted record. The record is never changed: the reviewer's saved values live in
`edits`, so the screen can always show what the engine read. Rules:
- A field's value is the reviewer's saved value, else the engine's approved value. An unverified
  suggestion is NEVER a value: the reviewer has to copy it into the field themselves.
- Approving a form re-validates every saved value. Failed validation (bad ID check digit, invalid date,
  sender not in the closed list...) needs an explicit "approve anyway"; empty fields are only reported.
- Export: one row per form. A flagged cell is highlighted while its form is not approved; after approval
  the reviewer has seen and accepted every value, including the ones left empty.
"""
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from src import config
from src.constants import (
    FIELD_KEYS,
    LIST_FIELDS,
    LIST_SEPARATOR,
    METADATA_KEYS,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_KEY,
    REVIEW_STATUS_PENDING,
)
from src.validators import (
    check_received_date,
    format_number_list,
    match_sending_entity,
    normalize_number_list,
    validate_israeli_id,
)

CENTRAL_NUMBER_RE = re.compile(r"^\d{6}$")          # same rules as record_builder
COMMITMENT_NUMBER_RE = re.compile(r"^[A-Za-z0-9]+$")
_DIGIT_RE = re.compile(r"\d")
_DATE_WARNINGS = {"unreadable": "invalid_date", "invalid": "invalid_date",
                  "future": "date_out_of_range", "too_old": "date_out_of_range"}


@dataclass
class FormReview:
    record: object                                # record_builder.Record (never mutated)
    edits: dict = field(default_factory=dict)     # field key -> value saved by the reviewer
    approved: bool = False
    store_id: Optional[int] = None                # row id in record_store (None: not saved)

    @property
    def form_index(self) -> int:
        return self.record.form_index

    def value(self, key: str) -> str:
        """The value that will be exported: the reviewer's saved value, else the engine's approved one."""
        if key in self.edits:
            return self.edits[key]
        return self.record.values.get(key, "")

    def open_flags(self) -> dict:
        """Flags still waiting for the reviewer: all of them until the form is approved, none after."""
        return {} if self.approved else dict(self.record.flags)

    def label(self) -> str:
        meta = self.record.metadata
        return f"{meta.get('source_file', '')} – עמ' {meta.get('page_numbers', '')}"


# ------------------------------------------------------------------ validation of reviewer input

def normalize_input(key: str, text: Optional[str], senders_path=None, today: Optional[date] = None) -> tuple:
    """(normalized value, warning code or None) for one reviewer-entered value.

    Warning codes are constants.FLAG_REASONS keys; "missing" means the field was left empty.
    Values that fail validation are kept as typed (cleaned), so nothing the reviewer wrote is lost.
    """
    text = " ".join(str(text or "").split())
    if not text:
        return "", "missing"
    if key in ("first_name", "last_name"):
        return text, ("invalid_format" if _DIGIT_RE.search(text) else None)
    if key == "id_number":
        digits = "".join(text.split())
        return digits, (None if validate_israeli_id(digits) else "invalid_id")
    if key in LIST_FIELDS:
        numbers = ["".join(n.split()) for n in normalize_number_list(text)]
        pattern = CENTRAL_NUMBER_RE if key == "central_numbers" else COMMITMENT_NUMBER_RE
        ok = bool(numbers) and all(pattern.match(n) for n in numbers)
        return format_number_list(numbers, LIST_SEPARATOR), (None if ok else "invalid_format")
    if key == "received_date":
        check = check_received_date(text, today=today or date.today())
        return (check.value or text), _DATE_WARNINGS.get(check.reason)
    if key == "sending_entity":
        canonical = match_sending_entity(text, senders_path or config.SENDERS_PATH)
        return (canonical, None) if canonical else (text, "sender_not_in_list")
    return text, None


@dataclass
class ApprovalResult:
    approved: bool
    warnings: dict      # field -> warning code: failed validation, needs "approve anyway"
    empty: list         # fields left empty (reported, never blocking)


def save_edits(review: FormReview, entered: dict, senders_path=None, today: Optional[date] = None) -> tuple:
    """Store the reviewer's values (normalized). Returns (warnings, empty fields) for those values."""
    warnings, empty = {}, []
    for key in FIELD_KEYS:
        if key not in entered:
            continue
        value, warning = normalize_input(key, entered[key], senders_path, today)
        review.edits[key] = value
        if warning == "missing":
            empty.append(key)
        elif warning:
            warnings[key] = warning
    return warnings, empty


def approve(review: FormReview, entered: dict, force: bool = False, senders_path=None,
            today: Optional[date] = None) -> ApprovalResult:
    """Save the entered values and approve the form, unless a value fails validation and not `force`.

    Fields not in `entered` are validated with their current value, so approval always checks the whole form.
    """
    values = {key: entered.get(key, review.value(key)) for key in FIELD_KEYS}
    warnings, empty = save_edits(review, values, senders_path, today)
    review.approved = force or not warnings
    return ApprovalResult(review.approved, warnings, empty)


def reopen(review: FormReview) -> None:
    review.approved = False


# ------------------------------------------------------------------ session

def new_session(records: list) -> list:
    return [FormReview(r) for r in records]


def unapproved(reviews: list) -> list:
    return [r for r in reviews if not r.approved]


def next_unapproved(reviews: list, after: int) -> Optional[int]:
    """Position of the next unapproved form after position `after` (wrapping around), or None."""
    n = len(reviews)
    for step in range(1, n + 1):
        pos = (after + step) % n
        if not reviews[pos].approved:
            return pos
    return None


def summary_frame(reviews: list) -> pd.DataFrame:
    """One row per form for the summary screen (English keys; the app sets Hebrew headers)."""
    rows = []
    for r in reviews:
        meta = r.record.metadata
        rows.append({
            "form_index": r.form_index,
            "source_file": meta.get("source_file", ""),
            "page_numbers": meta.get("page_numbers", ""),
            "form_type": meta.get("form_type", ""),
            "flagged_fields": len(r.record.flags),
            "form_warnings": len(r.record.split_flags) + bool(r.record.error),
            REVIEW_STATUS_KEY: REVIEW_STATUS_APPROVED if r.approved else REVIEW_STATUS_PENDING,
        })
    return pd.DataFrame(rows, columns=["form_index", "source_file", "page_numbers", "form_type", "flagged_fields",
                                       "form_warnings", REVIEW_STATUS_KEY])


def export_frame(reviews: list) -> tuple:
    """(DataFrame, highlight) for the export: one row per form, fields + metadata + review status.

    highlight is a set of (row position, column key) cells to fill orange: flagged fields of forms that
    are not approved. Suggestions, crops and images are never part of the export.
    """
    columns = [*FIELD_KEYS, *METADATA_KEYS, REVIEW_STATUS_KEY]
    rows, highlight = [], set()
    for pos, r in enumerate(reviews):
        row = {key: r.value(key) for key in FIELD_KEYS}
        row.update({key: r.record.metadata.get(key, "") for key in METADATA_KEYS})
        row[REVIEW_STATUS_KEY] = REVIEW_STATUS_APPROVED if r.approved else REVIEW_STATUS_PENDING
        rows.append(row)
        highlight |= {(pos, key) for key in r.open_flags()}
        if not r.approved:
            highlight.add((pos, REVIEW_STATUS_KEY))
    return pd.DataFrame(rows, columns=columns), highlight
