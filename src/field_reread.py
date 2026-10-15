"""Targeted second reading of flagged number fields: ID, central numbers, received date, commitment numbers.

The full-page pass sees a 300-dpi page downscaled to EXTRACTION_IMAGE_MAX_SIDE, so the small digits on
stickers, stamps and handwriting get few pixels. When one of these fields is flagged (missing, unreadable,
low confidence, failed validation) and the full-page pass reported where it is (field_locations), each
location is cropped from the full-resolution page and re-read - all fields of one form in ONE request, and
never more than one re-read request per form.

The crop reading is independent: the prompt never contains the full-page value. Whether it may change
anything is decided by deterministic rules in record_builder.apply_field_recheck - this module only reads.
Model boxes are not trusted blindly: out-of-range pages, degenerate boxes and boxes too large to gain any
resolution are skipped, and the model must say when a crop does not show the requested field.

Privacy: only the crops and a fixed prompt are sent; nothing from the answers is logged.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.constants import LIST_FIELDS, RECEIVED_DATE_SOURCES, REREAD_FIELDS
from src.validators import normalize_date, normalize_number_list

logger = logging.getLogger(__name__)

# Flag reasons that trigger a re-read. Not "date_source_rejected" (the source rule is not re-litigated),
# not "commitment_is_id" or "extraction_failed".
REREAD_REASONS = frozenset({"missing", "unreadable", "low_confidence", "invalid_id", "invalid_format",
                            "invalid_date", "date_out_of_range"})

CROP_PAD_X = 0.25           # horizontal padding, fraction of the box width (labels sit beside values)
CROP_PAD_Y = 0.5            # vertical padding, fraction of the box height
CROP_MIN_PAD_PX = 60        # ~5 mm at 300 dpi: enough to show the label next to a tight box
CROP_MAX_SIDE = 1600
MAX_BOX_AREA = 0.25         # a box over 25% of the page gains no resolution: not a targeted crop
MAX_CROPS_PER_FORM = 8

CROP_STATUSES = ("read", "unreadable", "not_in_crop")

FIELD_DESCRIPTIONS = {
    "id_number": 'the PATIENT\'s Israeli ID number (labels "ת.ז.", "ת\\"ז", "מספר זהות", "מ.ז.", "ID / PASS"). '
                 "Never a phone, case, visit, file, commitment or doctor-license number.",
    "central_numbers": 'the lab\'s central number: a 6-digit number on a small barcode sticker marked "GLC" '
                       '(or handwritten next to "מספר מרכזי" / "מרכזי"). Several stickers => all numbers.',
    "received_date": 'the date the lab RECEIVED the sample: either inside the rectangular "התקבל" stamp of '
                     '"המעבדה המרכזית לנגיפים" (received_date_source = "received_stamp"), or handwritten right '
                     'below / next to the central number (received_date_source = "handwritten_near_central"). '
                     "Any other date (printed admission, print, sampling, birth date) does NOT count: "
                     'received_date_source = "none" and status = "not_in_crop".',
    "commitment_numbers": 'the payment commitment number (labels "מספר התחייבות", "התחייבות", "דרישה", "דגימה", '
                          '"מספר דגימה", "אסמכתא"). Never an ID, a GLC central number, a case/visit number, a '
                          "service code or a phone number.",
}

REREAD_PROMPT = """You re-read ONE field per crop from scanned Israeli medical lab forms. Each crop is a close-up of
the area where the full page said the field is. Read each crop independently.

For each crop:
- status = "read": the requested field is visible and EVERY character is legible. values = the value(s) exactly as
  written, digits (and letters, if printed so) without spaces. Dates: copy as written, e.g. "18-06-2026", "16/6/26".
- status = "unreadable": the field is in the crop but at least one character is unclear, covered or cut off.
  values = []. Never complete or guess a digit - an unreadable field is reviewed by a person, a guessed one is not.
- status = "not_in_crop": the crop does not show the requested field (another number, an empty area). values = [].
- confidence: 0-1, how sure you are of every character.
- received_date_source: only for received_date (see below); "none" for every other field.
Do not output anything except the JSON.

Fields:
{descriptions}"""

REREAD_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "crops": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "crop_index": {"type": "INTEGER"},
                    "status": {"type": "STRING", "enum": list(CROP_STATUSES)},
                    "values": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "confidence": {"type": "NUMBER"},
                    "received_date_source": {"type": "STRING", "enum": list(RECEIVED_DATE_SOURCES)},
                },
                "required": ["crop_index", "status", "values", "confidence", "received_date_source"],
                "propertyOrdering": ["crop_index", "status", "values", "confidence", "received_date_source"],
            },
        }
    },
    "required": ["crops"],
}


@dataclass
class CropReading:
    field: str
    status: str                     # one of CROP_STATUSES, or "no_answer"
    values: list
    confidence: float
    source: Optional[str] = None    # received_date only


@dataclass
class TargetedValue:
    """The combined crop reading of one field."""
    status: str                     # "read" | "unreadable" | "box_miss" | "inconsistent" | "failed"
    value: object = None            # str (scalar fields) or list[str] (LIST_FIELDS) when status == "read"
    confidence: float = 0.0
    source: Optional[str] = None
    crops: int = 0


@dataclass
class FieldRecheck:
    attempted: list = field(default_factory=list)   # field keys sent for re-reading
    readings: list = field(default_factory=list)    # list[CropReading]
    skipped: dict = field(default_factory=dict)     # field -> why it was not re-read (no values)
    error: Optional[str] = None

    def result(self, key: str) -> TargetedValue:
        if key not in self.attempted:
            return TargetedValue("failed")
        if self.error:
            return TargetedValue("failed", crops=sum(r.field == key for r in self.readings))
        return combine_readings(key, [r for r in self.readings if r.field == key])


# ------------------------------------------------------------------ pure helpers

def comparable(key: str, value) -> object:
    """Comparison key for agreement checks (never written back as a value)."""
    if key in LIST_FIELDS:
        return frozenset(n.replace(" ", "") for n in normalize_number_list(value))
    text = "".join(str(value or "").split())
    if key == "id_number" and text.isdigit() and len(text) <= 9:
        return text.zfill(9)
    if key == "received_date":
        return normalize_date(text) or text
    return text


def combine_readings(key: str, readings: list) -> TargetedValue:
    """Combine the crop readings of one field. Conservative:
    - list fields: every crop must be "read" (an unreadable or misplaced crop may hide a number);
      the value is the de-duplicated union in crop order.
    - scalar fields: crops that were read must agree and hold exactly one value each.
    """
    n = len(readings)
    if not readings:
        return TargetedValue("failed")
    read = [r for r in readings if r.status == "read" and r.values]
    unread = [r for r in readings if r not in read]

    def not_read() -> TargetedValue:
        statuses = {r.status for r in unread}
        status = ("unreadable" if "unreadable" in statuses else
                  "failed" if statuses == {"no_answer"} else "box_miss")
        return TargetedValue(status, crops=n)

    if key in LIST_FIELDS:
        if unread:
            return not_read()
        values = list(dict.fromkeys("".join(v.split()) for v in normalize_number_list(
            [v for r in read for v in r.values])))
        return TargetedValue("read", values, min(r.confidence for r in read), crops=n)
    if not read:
        return not_read()
    if any(len(r.values) != 1 for r in read) or len({comparable(key, r.values[0]) for r in read}) > 1:
        return TargetedValue("inconsistent", crops=n)
    sources = [r.source for r in read]
    source = sources[0] if len(set(sources)) == 1 else "none"
    return TargetedValue("read", "".join(read[0].values[0].split()), min(r.confidence for r in read),
                         source=source if key == "received_date" else None, crops=n)


def select_reread_fields(extraction, flags: dict) -> list:
    """Fields worth one targeted re-read: flagged for a re-readable reason AND located on the page."""
    if extraction.error:
        return []
    located = {loc["field"] for loc in extraction.field_locations}
    return [k for k in REREAD_FIELDS if flags.get(k) in REREAD_REASONS and k in located]


def crop_field_regions(images: list, locations: list, keys: list) -> tuple:
    """Crop each location of the requested fields. Returns ([(field, crop)], {field: skip_reason}).

    A field whose every box was unusable gets a skip reason; a field with at least one crop does not.
    """
    from src.extraction_engine import padded_pixel_box

    crops, problems = [], {}
    for loc in locations:
        key = loc["field"]
        if key not in keys:
            continue
        if loc["page"] > len(images):
            problems.setdefault(key, "page_out_of_range")
            continue
        y0, x0, y1, x1 = loc["box"]
        if (y1 - y0) * (x1 - x0) > MAX_BOX_AREA * 1_000_000:
            problems.setdefault(key, "box_too_large")
            continue
        img = images[loc["page"] - 1]
        box = padded_pixel_box(img.size, loc["box"], CROP_PAD_X, CROP_PAD_Y, CROP_MIN_PAD_PX)
        if box is None:
            problems.setdefault(key, "box_degenerate")
            continue
        if len(crops) >= MAX_CROPS_PER_FORM:
            problems.setdefault(key, "too_many_crops")
            continue
        crops.append((key, img.crop(box)))
    cropped = {k for k, _ in crops}
    skipped = {k: problems.get(k, "no_location") for k in keys if k not in cropped}
    return crops, skipped


# ------------------------------------------------------------------ API call

def _unit(value) -> float:
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, c)) if c == c else 0.0


def reread_fields(images: list, extraction, keys: list) -> FieldRecheck:
    """Re-read the given fields from high-resolution crops in one request. Never raises on API
    problems except GeminiQuotaError (the caller stops the run)."""
    from src.extraction_engine import GeminiError, GeminiQuotaError, generate_json, image_part

    if not keys:
        return FieldRecheck()
    crops, skipped = crop_field_regions(images, extraction.field_locations, keys)
    recheck = FieldRecheck(attempted=sorted({k for k, _ in crops}, key=REREAD_FIELDS.index), skipped=skipped)
    if not crops:
        return recheck
    descriptions = "\n".join(f"- {k}: {FIELD_DESCRIPTIONS[k]}" for k in recheck.attempted)
    contents: list = [REREAD_PROMPT.format(descriptions=descriptions)]
    for i, (key, crop) in enumerate(crops, start=1):
        contents += [f"Crop {i} - field: {key}", image_part(crop, CROP_MAX_SIDE)]
    try:
        data = generate_json(contents, REREAD_SCHEMA, media_resolution="high", purpose="reread_fields")
    except GeminiQuotaError:
        raise
    except GeminiError as exc:
        logger.warning("Targeted re-read failed: %s", exc)
        recheck.error = str(exc)
        return recheck
    by_index = {}
    for c in data.get("crops") or []:
        if isinstance(c, dict) and isinstance(c.get("crop_index"), int):
            by_index.setdefault(c["crop_index"], c)
    for i, (key, _) in enumerate(crops, start=1):
        c = by_index.get(i)
        if c is None or c.get("status") not in CROP_STATUSES:
            recheck.readings.append(CropReading(key, "no_answer", [], 0.0))
            continue
        values = [v for v in (c.get("values") if isinstance(c.get("values"), list) else [])
                  if isinstance(v, str) and v.strip()]
        source = c.get("received_date_source")
        recheck.readings.append(CropReading(
            field=key,
            status=c["status"] if c["status"] != "read" or values else "unreadable",
            values=values if c["status"] == "read" else [],
            confidence=_unit(c.get("confidence")),
            source=(source if source in RECEIVED_DATE_SOURCES else "none") if key == "received_date" else None,
        ))
    logger.info("Re-read %d field(s) from %d crop(s); statuses=%s", len(recheck.attempted), len(crops),
                [r.status for r in recheck.readings])
    return recheck
