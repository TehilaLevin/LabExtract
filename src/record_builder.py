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

    @property
    def needs_review(self) -> bool:
        return bool(self.flags or self.split_flags or self.error)

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


def build_record(form, extraction, threshold: float = config.CONFIDENCE_THRESHOLD,
                 today: Optional[date] = None) -> Record:
    """Merge one form's extraction with validation. `form` is a form_splitter.Form."""
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

    sender = values.get("sending_entity") or None
    metadata = {
        "source_file": form.source_file,
        "page_numbers": LIST_SEPARATOR.join(str(n) for n in form.page_numbers),
        "form_type": _form_type_label(extraction.form_type, sender),
    }
    return Record(form.index, values, flags, confidences, metadata,
                  split_flags=list(form.split_flags), error=extraction.error)


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
    from src.extraction_engine import GeminiQuotaError, empty_result, extract_form
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
                return empty_result("Gemini API quota exhausted")
            images = [preprocess(img).processed for img in form.images()]
            try:
                return extract_form(images, senders)
            except GeminiQuotaError as exc:
                quota_hit.set()
                return empty_result(str(exc))

        extractions = [None] * len(forms)
        with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_REQUESTS) as pool:
            futures = {pool.submit(work, f): i for i, f in enumerate(forms)}
            for done, future in enumerate(as_completed(futures), start=1):
                extractions[futures[future]] = future.result()
                if progress:
                    progress("extract", done, len(forms))

        records = [build_record(f, e) for f, e in zip(forms, extractions)]
        logger.info("Processed %d form(s); %d field(s) flagged",
                    len(records), count_flagged_fields(records))
        return records
    finally:
        cleanup_temp(run_dir)
