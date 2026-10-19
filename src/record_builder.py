"""Merge extraction + validation into review-ready records (one record per form).

A field is flagged for review when it is missing, fails validation, or its confidence
is below CONFIDENCE_THRESHOLD. Values that fail validation are kept for display (the
reviewer sees what was read), except where showing them would be a guess: a received
date from a non-approved source and a sender outside the closed list are emptied.
"""
import dataclasses
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
    days_between,
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
    # field key -> [field_crops.SourceCrop]: where each value was read from, in memory only (never exported
    # or logged), for the review UI. Fields the model did not locate are absent.
    field_crops: dict = field(default_factory=dict)
    # [field_crops.PagePreview] of the form's pages, in memory only, for the review UI.
    page_images: list = field(default_factory=list)
    # Other readings from the targeted re-read (field -> [text]), in memory only, never exported or
    # logged: shown to the reviewer next to a flagged field. Never an approved value.
    alternatives: dict = field(default_factory=dict)
    # Why fields were flagged and what was tried (codes / counts only, no values): see diagnostic_lines.
    diagnostics: dict = field(default_factory=dict)
    # The model's note on where each value was read (field -> text; Claude only), in memory only,
    # never exported or logged: shown to the reviewer next to the field.
    notes: dict = field(default_factory=dict)
    # The text exactly as the model saw it written, even partial (field -> text; Claude only), in memory only,
    # never exported or logged. NEVER an approved value: shown to the reviewer as a reading aid.
    raw_texts: dict = field(default_factory=dict)

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


def _validate(key: str, value, source: Optional[str], today: date, collection_date: Optional[str] = None) -> tuple:
    """Return (display_value, reason_or_None) for one non-empty field value.

    With STRICT_VALIDATION: a central number outside CENTRAL_NUMBER_PREFIXES is a warning flag (the value is
    kept); a received date is flagged when older than RECEIVED_DATE_MAX_AGE_DAYS (not in EVAL_MODE), or not
    0..RECEIVED_AFTER_COLLECTION_MAX_DAYS days after the collection date.
    """
    if key in ("first_name", "last_name"):
        return value, ("invalid_format" if _DIGIT_RE.search(value) else None)

    if key == "id_number":
        return value, (None if validate_israeli_id(value) else "invalid_id")

    if key == "central_numbers":
        ok = all(CENTRAL_NUMBER_RE.match(n) for n in value)
        if ok and config.STRICT_VALIDATION and config.CENTRAL_NUMBER_PREFIXES and not all(
                n.startswith(config.CENTRAL_NUMBER_PREFIXES) for n in value):
            return format_number_list(value, LIST_SEPARATOR), "central_prefix"
        return format_number_list(value, LIST_SEPARATOR), (None if ok else "invalid_format")

    if key == "commitment_numbers":
        ok = all(COMMITMENT_NUMBER_RE.match(n) for n in value)
        return format_number_list(value, LIST_SEPARATOR), (None if ok else "invalid_format")

    if key == "received_date":
        if source not in ACCEPTED_RECEIVED_DATE_SOURCES:
            return "", "date_source_rejected"  # never use admission/print dates
        check = check_received_date(value, today=today, check_age=not config.EVAL_MODE, max_age_days=(
            config.RECEIVED_DATE_MAX_AGE_DAYS if config.STRICT_VALIDATION else None))
        if check.reason is None and config.STRICT_VALIDATION and collection_date:
            gap = days_between(collection_date, value)
            if gap is not None and gap < 0:
                return check.value, "date_before_collection"
            if gap is not None and gap > config.RECEIVED_AFTER_COLLECTION_MAX_DAYS:
                return check.value, "date_long_after_collection"
        return (check.value or ""), _DATE_REASONS.get(check.reason)

    if key == "sending_entity":
        canonical = match_sending_entity(value, config.SENDERS_PATH)
        return (canonical, None) if canonical else ("", "sender_not_in_list")

    return value, None


def _check_field(key: str, res, extraction, threshold: float, today: date) -> tuple:
    """(display_value, reason_or_None) for one extracted field: validation, then confidence."""
    if extraction.error:
        return "", "extraction_failed"
    raw = normalize_number_list(res.value) if key in LIST_FIELDS else res.value
    if not raw:
        # The extractor tells "on the form but illegible" apart from "not on the form".
        return "", ("unreadable" if res.status == "unreadable" else "missing")
    display, reason = _validate(key, raw, res.source, today, getattr(extraction, "collection_date", None))
    if reason:
        return display, reason
    if res.confidence < threshold:
        return display, "low_confidence"
    if key in NAME_KEYS and extraction.name_order_basis == "unclear":
        return display, "name_order_unclear"  # read fine, but first/last may be swapped
    return display, None


def field_flags(extraction, threshold: float = config.CONFIDENCE_THRESHOLD, today: Optional[date] = None) -> tuple:
    """(values, flags) from validation and confidence alone: no re-read, no name check.
    Used to decide which fields get a targeted re-read."""
    today = today or date.today()
    values, flags = {}, {}
    for key in FIELD_KEYS:
        values[key], reason = _check_field(key, extraction.fields[key], extraction, threshold, today)
        if reason:
            flags[key] = reason
    return values, flags


def build_record(form, extraction, name_check=None, threshold: float = config.CONFIDENCE_THRESHOLD,
                 today: Optional[date] = None, recheck=None, page_info: Optional[list] = None,
                 field_crops: Optional[dict] = None, page_images: Optional[list] = None) -> Record:
    """Merge one form's extraction with validation. `form` is a form_splitter.Form.

    name_check is the extraction_engine.NameCheck for this form (None = not run); recheck is the
    field_reread.FieldRecheck (None = not run); page_info is [{"page", "size", "steps"}] for diagnostics;
    field_crops / page_images are field_crops.source_crops() / page_previews() for this form.
    """
    today = today or date.today()
    overrides, alternatives, outcomes = {}, {}, {}
    effective = extraction
    if recheck is not None and not extraction.error:
        effective, overrides, alternatives, outcomes = apply_field_recheck(extraction, recheck, threshold, today)

    values, flags = field_flags(effective, threshold, today)
    for key, reason in overrides.items():
        if reason is None:
            flags.pop(key, None)
        else:
            flags[key] = reason
    confidences = {key: extraction.fields[key].confidence for key in FIELD_KEYS}

    suggestions = {}
    if not extraction.error:
        _drop_commitments_equal_to_id(values, flags, effective)
        suggestions = _apply_name_check(values, flags, extraction, name_check, threshold)

    sender = values.get("sending_entity") or None
    metadata = {
        "source_file": form.source_file,
        "page_numbers": LIST_SEPARATOR.join(str(n) for n in form.page_numbers),
        "form_type": _form_type_label(extraction.form_type, sender),
    }
    return Record(form.index, values, flags, confidences, metadata,
                  split_flags=list(form.split_flags), error=extraction.error,
                  suggestions=suggestions, field_crops=dict(field_crops or {}),
                  page_images=list(page_images or []),
                  alternatives=alternatives,
                  diagnostics=_diagnostics(extraction, flags, recheck, outcomes, page_info),
                  notes={k: f.note for k, f in extraction.fields.items() if getattr(f, "note", None)},
                  raw_texts={k: f.raw_text for k, f in extraction.fields.items() if getattr(f, "raw_text", None)})


def apply_batch_id_consistency(records: list) -> int:
    """The same patient (identical first + last name) must carry the same ID on every form of the batch.
    Records whose ID differs from another record of that patient get the id_number flag id_batch_mismatch
    (the value is kept for the reviewer). Returns the number of records flagged. No-op without
    STRICT_VALIDATION or when names are missing (handwritten names are often empty)."""
    if not config.STRICT_VALIDATION:
        return 0
    by_patient: dict = {}
    for rec in records:
        first, last = rec.values.get("first_name", "").strip(), rec.values.get("last_name", "").strip()
        ident = _id_key(rec.values.get("id_number", ""))
        if first and last and ident:
            by_patient.setdefault((first, last), []).append((rec, ident))
    flagged = 0
    for entries in by_patient.values():
        if len({ident for _, ident in entries}) > 1:
            for rec, _ in entries:
                rec.flags["id_number"] = "id_batch_mismatch"
                flagged += 1
    return flagged


# ------------------------------------------------------------------ targeted re-read decisions
# A crop reading (field_reread) is one more reading, not the truth. Rules, per re-read field:
# - Re-read failed / unreadable / box missed / crops inconsistent: nothing changes (an empty
#   field whose crop was unreadable is flagged "unreadable").
# - Same value as the full page: kept as it was. Only for the ID (checksum) does agreement of two
#   confident readings clear a "low_confidence" flag; other fields have no check strong enough.
# - The full page had nothing: the crop value is shown only if it passes validation with
#   confidence >= threshold, and stays flagged "reread_single_reading".
# - The full page failed validation, the crop passes: the crop value replaces it, flagged
#   "reread_replaced", the full-page reading kept as an alternative (lists only when no number
#   would be lost). For the ID this means: the full-page ID fails the check digit AND the crop
#   reading passes it (with confidence >= threshold, and not equal to a central / commitment
#   number). An ID that passes the check digit is never replaced. (Seen on sample page 13: the
#   full page misread the ID, the crop read it correctly.)
# - Both pass validation but differ: the field is emptied and flagged "reread_conflict".
# - Otherwise the full-page value stays, flagged "reread_conflict" when either reading passes
#   validation. A failed second pass never overwrites a better-supported value.
# Every other reading is kept in Record.alternatives (memory only) for the reviewer. Every changed
# field stays flagged: model confidence is a heuristic, not a probability of being right.

REREAD_CONFIRM_FIELDS = frozenset({"id_number"})
REREAD_REPLACE_FIELDS = frozenset({"id_number", "central_numbers", "commitment_numbers", "received_date"})
_KEEP = object()  # flag override sentinel: leave the flag computed from the effective value


def _as_text(key: str, value) -> str:
    return format_number_list(value, LIST_SEPARATOR) if key in LIST_FIELDS else str(value or "")


def _other_numbers(extraction) -> set:
    nums = normalize_number_list(extraction.fields["central_numbers"].value) + normalize_number_list(
        extraction.fields["commitment_numbers"].value)
    return {_id_key(n) for n in nums} - {""}


def _decide(key: str, orig, target, extraction, threshold: float, today: date) -> tuple:
    """-> (outcome, replacement FieldResult or None, flag override (_KEEP / None / reason), alternatives)."""
    from src.extraction_engine import FieldResult
    from src.field_reread import comparable

    o_raw = normalize_number_list(orig.value) if key in LIST_FIELDS else orig.value
    if target.status != "read":
        if target.status == "unreadable" and not o_raw:
            return "unreadable", None, "unreadable", []
        return target.status, None, _KEEP, []

    collected = getattr(extraction, "collection_date", None)
    t_reason = _validate(key, target.value, target.source, today, collected)[1]
    if t_reason is None and target.confidence < threshold:
        t_reason = "low_confidence"
    if t_reason is None and key == "id_number" and _id_key(target.value) in _other_numbers(extraction):
        t_reason = "matches_other_field"  # the crop read a central / commitment number as the ID
    t_ok = t_reason is None
    t_field = FieldResult(value=target.value, confidence=target.confidence, source=target.source, status="read")
    t_text = _as_text(key, target.value)

    if not o_raw:
        if t_ok:
            return "filled", t_field, "reread_single_reading", []
        return "unverified", None, _KEEP, [t_text]

    o_reason, o_text = _validate(key, o_raw, orig.source, today, collected)[1], _as_text(key, o_raw)
    o_ok = o_reason is None
    if comparable(key, o_raw) == comparable(key, target.value):
        if key in REREAD_CONFIRM_FIELDS and o_ok and t_ok and orig.confidence < threshold:
            return "confirmed", None, None, []
        return "agreed", None, _KEEP, []
    if key in LIST_FIELDS and t_ok and comparable(key, o_raw) < comparable(key, target.value):
        return "extended", t_field, "reread_replaced", [o_text]   # every original number re-read, plus more
    if t_ok and not o_ok and key in REREAD_REPLACE_FIELDS and (
            key not in LIST_FIELDS or len(target.value) >= len(o_raw)):
        return "replaced", t_field, "reread_replaced", [o_text]
    if o_ok and t_ok:
        empty = FieldResult(value=[] if key in LIST_FIELDS else None, confidence=0.0,
                            source=orig.source, status=orig.status)
        return "conflict", empty, "reread_conflict", [o_text, t_text]
    return "mismatch", None, ("reread_conflict" if o_ok or t_ok else _KEEP), [t_text]


def apply_field_recheck(extraction, recheck, threshold: float = config.CONFIDENCE_THRESHOLD,
                        today: Optional[date] = None) -> tuple:
    """Apply a targeted re-read. Returns (effective extraction, flag overrides {field: reason or None},
    alternatives {field: [other readings]}, outcomes {field: code}). Never mutates `extraction`."""
    today = today or date.today()
    fields = dict(extraction.fields)
    overrides, alternatives, outcomes = {}, {}, {}
    for key in recheck.attempted:
        outcome, replacement, override, alts = _decide(
            key, extraction.fields[key], recheck.result(key), extraction, threshold, today)
        outcomes[key] = outcome
        if replacement is not None:
            fields[key] = replacement
        if override is not _KEEP:
            overrides[key] = override
        if alts:
            alternatives[key] = alts
    return dataclasses.replace(extraction, fields=fields), overrides, alternatives, outcomes


# ------------------------------------------------------------------ safe diagnostics

def _diagnostics(extraction, flags: dict, recheck, outcomes: dict, page_info: Optional[list]) -> dict:
    """Why each field was flagged and what the pipeline tried. Codes and counts only - never values."""
    from src.constants import REREAD_FIELDS

    fields = {}
    for key in FIELD_KEYS:
        if key not in flags and key not in outcomes:
            continue
        res = extraction.fields[key]
        if key not in REREAD_FIELDS:
            reread = "n/a"
        elif recheck is None:
            reread = "not_run"
        elif key in outcomes:
            reread = outcomes[key]
        elif key in recheck.skipped:
            reread = f"skipped:{recheck.skipped[key]}"
        else:
            reread = "not_needed"
        fields[key] = {
            "flag": flags.get(key),
            "status": res.status or "unknown",
            "confidence": round(res.confidence, 2),
            "locations": sum(1 for loc in extraction.field_locations if loc["field"] == key),
            "reread": reread,
            "crops": sum(1 for r in recheck.readings if r.field == key) if recheck else 0,
        }
    return {"pages": list(page_info or []), "fields": fields,
            "reread_error": bool(recheck and recheck.error),
            "highlight_crops": getattr(extraction, "highlight_crops", 0)}


def diagnostic_lines(record, include_source: bool = False) -> list:
    """Human-readable diagnostics for one record: no field values. Logs use include_source=False,
    because an uploaded file's name may itself contain a patient's name."""
    pages = record.metadata.get("page_numbers", "")
    where = (f"{record.metadata.get('source_file', '')} p{pages}" if include_source
             else f"form {record.form_index} (p{pages})")
    lines = [f"{where} page {p['page']}: size={p['size'][0]}x{p['size'][1]} preprocessing="
             f"{','.join(p['steps']) or 'none'}" for p in record.diagnostics.get("pages", [])]
    for key, d in record.diagnostics.get("fields", {}).items():
        lines.append(f"{where} {key}: flag={d['flag'] or 'none'} status={d['status']} conf={d['confidence']} "
                     f"locations={d['locations']} reread={d['reread']} crops={d['crops']}")
    if record.diagnostics.get("reread_error"):
        lines.append(f"{where}: targeted re-read request failed")
    if record.diagnostics.get("highlight_crops"):
        lines.append(f"{where}: {record.diagnostics['highlight_crops']} highlight crop(s) sent with the page")
    return lines


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
    """Full pipeline: convert -> split -> preprocess -> extract -> verify names -> re-read
    flagged number fields -> validate. Each record keeps page previews and the source crop of every
    located field, in memory only.

    The full-page pass reads the processed pages; name, field and source crops are cut from
    config.CROP_SOURCE ("original" pixels by default, same geometry). Page images live in a
    per-run folder under data/temp/ that is always deleted, even when the run fails.
    """
    # Imported here so the pure record-building code above has no API dependency.
    from src.extraction_engine import LLMQuotaError, NameCheck, empty_result, extract_form, verify_names
    from src.field_crops import page_previews, source_crops
    from src.field_reread import FieldRecheck, reread_fields, select_reread_fields
    from src.file_converter import cleanup_temp, convert_files, new_run_dir
    from src.form_splitter import split_forms
    from src.highlight_crops import clear_debug_crops
    from src.image_preprocessor import preprocess
    from src.validators import list_approved_senders

    clear_debug_crops()  # debug crops (HIGHLIGHT_DEBUG) of the previous run are never kept longer
    run_dir = new_run_dir()
    try:
        pages = convert_files(files, run_dir)
        forms = split_forms(pages, (lambda d, t: progress("split", d, t)) if progress else None)
        senders = list_approved_senders(config.SENDERS_PATH)
        quota_hit = threading.Event()

        def work(form):
            prepared = [preprocess(img) for img in form.images()]
            images = [p.processed for p in prepared]
            crop_images = images if config.CROP_SOURCE == "processed" else [p.crop_source for p in prepared]
            page_info = [{"page": n, "size": list(p.processed.size), "steps": list(p.steps)}
                         for n, p in zip(form.page_numbers, prepared)]
            # Previews even when extraction fails: the reviewer then types the form in by hand.
            previews = page_previews(crop_images, form.page_numbers)
            if quota_hit.is_set():
                return empty_result("API quota exhausted"), None, None, page_info, {}, previews
            try:
                extraction = extract_form(images, senders, crop_images,
                                          form_label=f"form{form.index}_p{'-'.join(map(str, form.page_numbers))}")
                name_check = verify_names(crop_images, extraction)
            except LLMQuotaError as exc:
                quota_hit.set()
                return empty_result(str(exc)), NameCheck(needed=False), None, page_info, {}, previews
            crops = source_crops(crop_images, extraction)
            if not config.FIELD_REREAD_ENABLED:
                return extraction, name_check, None, page_info, crops, previews
            keys = select_reread_fields(extraction, field_flags(extraction)[1])
            try:
                recheck = reread_fields(crop_images, extraction, keys)
            except LLMQuotaError as exc:
                quota_hit.set()  # keep the full-page extraction; the re-read simply did not happen
                recheck = FieldRecheck(attempted=keys, error=str(exc))
            return extraction, name_check, recheck, page_info, crops, previews

        results = [None] * len(forms)
        with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_REQUESTS) as pool:
            futures = {pool.submit(work, f): i for i, f in enumerate(forms)}
            for done, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if progress:
                    progress("extract", done, len(forms))

        records = [build_record(f, e, c, recheck=r, page_info=info, field_crops=crops, page_images=previews)
                   for f, (e, c, r, info, crops, previews) in zip(forms, results)]
        mismatched = apply_batch_id_consistency(records)
        if mismatched:
            logger.info("Batch ID check: %d record(s) flagged id_batch_mismatch", mismatched)
        for record in records:
            for line in diagnostic_lines(record):
                logger.info(line)
        logger.info("Processed %d form(s); %d field(s) flagged",
                    len(records), count_flagged_fields(records))
        return records
    finally:
        cleanup_temp(run_dir)
