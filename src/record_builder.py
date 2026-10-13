"""Merge extraction + validation into review-ready records (one record per form).

A field is flagged for review when it is missing, fails validation, or its confidence
is below CONFIDENCE_THRESHOLD. Values that fail validation are kept for display (the
reviewer sees what was read), except where showing them would be a guess: a received
date from a non-approved source and a sender outside the closed list are emptied.
"""
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

import pandas as pd

from src import config
from src.constants import (
    ACCEPTED_RECEIVED_DATE_SOURCES,
    ENTITY_LABELLED_FORM_TYPES,
    FIELD_KEYS,
    FORM_TYPES,
    LIST_FIELDS,
    LIST_SEPARATOR,
    METADATA_KEYS,
    NEEDS_REVIEW_KEY,
    SUGGESTION_LABEL,
)
from src.validators import (
    check_received_date,
    format_number_list,
    match_sending_entity,
    normalize_number_list,
    validate_israeli_id,
)

logger = logging.getLogger(__name__)

CENTRAL_NUMBER_RE = re.compile(r"^\d{6}$")          # GLC stickers: 6 digits
COMMITMENT_NUMBER_RE = re.compile(r"^[A-Za-z0-9]+$")
_DIGIT_RE = re.compile(r"\d")

_DATE_REASONS = {"unreadable": "invalid_date", "invalid": "invalid_date",
                 "future": "date_out_of_range", "too_old": "date_out_of_range"}


@dataclass
class Record:
    form_index: int
    values: dict                 # field key -> display text ("" when empty)
    flags: dict                  # field key -> reason code (see constants.FLAG_REASONS)
    confidences: dict            # field key -> model confidence
    metadata: dict               # source_file, page_numbers, form_type (display text)
    split_flags: list = field(default_factory=list)
    error: Optional[str] = None
    # Unverified readings of handwritten names (field -> text). NEVER approved values:
    # values[field] stays "" and the field stays flagged; shown only as an orange suggestion.
    suggestions: dict = field(default_factory=dict)
    # [(kind, jpeg_bytes)] crops of the name area, in memory only, for the review UI.
    name_crops: list = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.flags or self.split_flags or self.error)

    def display_value(self, key: str) -> str:
        """Approved value, or the unverified suggestion clearly labelled as such."""
        if self.values.get(key):
            return self.values[key]
        if self.suggestions.get(key):
            return f"{self.suggestions[key]} ({SUGGESTION_LABEL})"
        return ""

    def to_row(self) -> dict:
        row = {k: self.values.get(k, "") for k in FIELD_KEYS}
        row.update({k: self.metadata.get(k, "") for k in METADATA_KEYS})
        row[NEEDS_REVIEW_KEY] = self.needs_review
        return row


def _form_type_label(form_type: str, sender: Optional[str]) -> str:
    label = FORM_TYPES.get(form_type, FORM_TYPES["unknown"])
    if form_type in ENTITY_LABELLED_FORM_TYPES and sender:
        return f"{label} – {sender}"
    return label


def _validate(key: str, value, source: Optional[str], today: date) -> tuple:
    """Return (display_value, reason_or_None) for one non-empty field value."""
    if key in ("first_name", "last_name"):
        return value, ("invalid_format" if _DIGIT_RE.search(value) else None)

    if key == "id_number":
        return value, (None if validate_israeli_id(value) else "invalid_id")

    if key == "central_numbers":
        ok = all(CENTRAL_NUMBER_RE.match(n) for n in value)
        return format_number_list(value, LIST_SEPARATOR), (None if ok else "invalid_format")

    if key == "commitment_numbers":
        ok = all(COMMITMENT_NUMBER_RE.match(n) for n in value)
        return format_number_list(value, LIST_SEPARATOR), (None if ok else "invalid_format")

    if key == "received_date":
        if source not in ACCEPTED_RECEIVED_DATE_SOURCES:
            return "", "date_source_rejected"  # never use admission/print dates
        check = check_received_date(value, today=today)
        return (check.value or ""), _DATE_REASONS.get(check.reason)

    if key == "sending_entity":
        canonical = match_sending_entity(value, config.SENDERS_PATH)
        return (canonical, None) if canonical else ("", "sender_not_in_list")

    return value, None


def build_record(form, extraction, name_check=None, threshold: float = config.CONFIDENCE_THRESHOLD,
                 today: Optional[date] = None) -> Record:
    """Merge one form's extraction with validation. `form` is a form_splitter.Form.

    name_check is the extraction_engine.NameCheck for this form (None = not run).
    """
    today = today or date.today()
    values, flags, confidences = {}, {}, {}
    for key in FIELD_KEYS:
        res = extraction.fields[key]
        confidences[key] = res.confidence
        raw = normalize_number_list(res.value) if key in LIST_FIELDS else res.value
        if extraction.error:
            values[key], flags[key] = "", "extraction_failed"
            continue
        if not raw:
            values[key], flags[key] = "", "missing"
            continue
        display, reason = _validate(key, raw, res.source, today)
        values[key] = display
        if reason:
            flags[key] = reason
        elif res.confidence < threshold:
            flags[key] = "low_confidence"
        elif key in ("first_name", "last_name") and extraction.name_order_basis == "unclear":
            flags[key] = "name_order_unclear"  # read fine, but first/last may be swapped

    suggestions = {}
    if not extraction.error:
        _drop_commitments_equal_to_id(values, flags, extraction)
        suggestions = _apply_name_check(values, flags, extraction, name_check, threshold)

    sender = values.get("sending_entity") or None
    metadata = {
        "source_file": form.source_file,
        "page_numbers": LIST_SEPARATOR.join(str(n) for n in form.page_numbers),
        "form_type": _form_type_label(extraction.form_type, sender),
    }
    return Record(form.index, values, flags, confidences, metadata,
                  split_flags=list(form.split_flags), error=extraction.error,
                  suggestions=suggestions, name_crops=list(name_check.crops) if name_check else [])


def _id_key(text) -> str:
    return re.sub(r"\D", "", text or "").lstrip("0")


def _drop_commitments_equal_to_id(values: dict, flags: dict, extraction) -> None:
    """A commitment number identical to the patient's ID was misread from the ID field
    (seen on Sheba printouts): remove it and flag the field."""
    id_key = _id_key(extraction.fields["id_number"].value)
    numbers = normalize_number_list(extraction.fields["commitment_numbers"].value)
    if not id_key or not numbers:
        return
    kept = [n for n in numbers if _id_key(n) != id_key]
    if len(kept) == len(numbers):
        return
    values["commitment_numbers"] = format_number_list(kept, LIST_SEPARATOR)
    flags["commitment_numbers"] = "commitment_is_id"


# ------------------------------------------------------------------ name safeguards

NAME_KEYS = ("first_name", "last_name")
_NAME_CLEAN_RE = re.compile(r"[\"'`׳״.,:;()\-]")


def _name_tokens(*parts) -> tuple:
    """Order-insensitive comparison key: the words of the name, punctuation removed."""
    words = []
    for part in parts:
        if part:
            words += _NAME_CLEAN_RE.sub(" ", part).split()
    return tuple(sorted(words))


def _reading_tokens(reading) -> tuple:
    return _name_tokens(reading.first_name, reading.last_name) if (
        reading.first_name or reading.last_name) else _name_tokens(reading.full_name)


def _best_reading(readings: list):
    """Most confident legible crop reading, or None."""
    legible = [r for r in readings if r.legible and _reading_tokens(r)]
    return max(legible, key=lambda r: r.confidence) if legible else None


def _suggestion(reading, extraction) -> dict:
    """Best available reading as a suggestion: the crop reading first, else the page reading."""
    if reading is not None:
        if reading.first_name or reading.last_name:
            return {k: v for k, v in (("first_name", reading.first_name), ("last_name", reading.last_name)) if v}
        return {"first_name": reading.full_name, "last_name": reading.full_name}  # unsplit: show whole name
    return {k: extraction.fields[k].value for k in NAME_KEYS if extraction.fields[k].value}


def _apply_name_check(values: dict, flags: dict, extraction, name_check, threshold: float) -> dict:
    """Safeguards for handwritten (or unknown-script) names. Mutates values / flags and
    returns suggestions {field: text} for names that could not be verified.

    A name is approved only when the full-page reading and EVERY crop reading agree and are
    confident. Otherwise both names are emptied, flagged, and the best available reading
    (crop first) is returned as an unverified suggestion:
    - "name_unverified": verification failed, or a crop is illegible, or no page reading.
    - "name_conflict":   crops of different name locations disagree.
    - "name_mismatch":   crops agree with each other but not with the full-page reading.
    - "name_order_unclear": same words, but a crop with labeled fields assigns first/last
      the other way round.
    A known name-order convention never overrides an uncertain reading (low_confidence).
    """
    if name_check is None or not name_check.needed:
        return {}
    readings = name_check.readings
    best = _best_reading(readings)

    def unverified(reason: str) -> dict:
        for k in NAME_KEYS:
            values[k], flags[k] = "", reason
        return _suggestion(best, extraction)

    page_key = _name_tokens(extraction.fields["first_name"].value, extraction.fields["last_name"].value)
    if name_check.error or not readings or not page_key:
        return unverified("name_unverified")
    if not all(r.legible and _reading_tokens(r) for r in readings):
        return unverified("name_unverified")
    if len({_reading_tokens(r) for r in readings}) > 1:
        return unverified("name_conflict")
    if _reading_tokens(readings[0]) != page_key:
        return unverified("name_mismatch")
    first, last = extraction.fields["first_name"].value, extraction.fields["last_name"].value
    for r in readings:
        if r.assignment == "labeled_fields" and r.first_name and r.last_name and (
                _name_tokens(r.first_name), _name_tokens(r.last_name)) != (_name_tokens(first), _name_tokens(last)):
            best = r  # the labeled crop is the better source for the order
            return unverified("name_order_unclear")
    present = [k for k in NAME_KEYS if values.get(k)]
    lowest = min([extraction.fields[k].confidence for k in present] + [r.confidence for r in readings])
    if lowest < threshold:
        for k in present:
            if flags.get(k) in (None, "name_order_unclear"):
                flags[k] = "low_confidence"
    return {}


def records_to_dataframe(records: list) -> pd.DataFrame:
    """One row per form, English column keys (Hebrew headers are applied on display/export)."""
    columns = [*FIELD_KEYS, *METADATA_KEYS, NEEDS_REVIEW_KEY]
    return pd.DataFrame([r.to_row() for r in records], columns=columns)


def count_flagged_fields(records: list) -> int:
    return sum(len(r.flags) for r in records)


# ------------------------------------------------------------------ orchestration

ProgressCallback = Callable[[str, int, int], None]  # (stage, done, total)


def process_files(files, progress: Optional[ProgressCallback] = None) -> list:
    """Full pipeline: convert -> split -> preprocess -> extract -> validate.

    Page images live in a per-run folder under data/temp/ that is always deleted,
    even when the run fails.
    """
    # Imported here so the pure record-building code above has no API dependency.
    from src.extraction_engine import GeminiQuotaError, NameCheck, empty_result, extract_form, verify_names
    from src.file_converter import cleanup_temp, convert_files, new_run_dir
    from src.form_splitter import split_forms
    from src.image_preprocessor import preprocess
    from src.validators import list_approved_senders

    run_dir = new_run_dir()
    try:
        pages = convert_files(files, run_dir)
        forms = split_forms(pages, (lambda d, t: progress("split", d, t)) if progress else None)
        senders = list_approved_senders(config.SENDERS_PATH)
        quota_hit = threading.Event()

        def work(form):
            if quota_hit.is_set():
                return empty_result("Gemini API quota exhausted"), None
            images = [preprocess(img).processed for img in form.images()]
            try:
                extraction = extract_form(images, senders)
                return extraction, verify_names(images, extraction)
            except GeminiQuotaError as exc:
                quota_hit.set()
                return empty_result(str(exc)), NameCheck(needed=False)

        results = [None] * len(forms)
        with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_REQUESTS) as pool:
            futures = {pool.submit(work, f): i for i, f in enumerate(forms)}
            for done, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if progress:
                    progress("extract", done, len(forms))

        records = [build_record(f, e, c) for f, (e, c) in zip(forms, results)]
        logger.info("Processed %d form(s); %d field(s) flagged",
                    len(records), count_flagged_fields(records))
        return records
    finally:
        cleanup_temp(run_dir)
