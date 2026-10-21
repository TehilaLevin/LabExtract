"""Model calls (via src.claude_client), prompt, JSON schema, per-field confidence.

Privacy: only form images (and the fixed prompt) are sent to the API. Nothing from
the model's answer is logged - only field names, confidences and error types.
"""
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional, Union

from PIL import Image, ImageOps

from src import config
from src.constants import (
    CONFIDENCE_LABELS,
    FIELD_KEYS,
    FIELD_STATUSES,
    FORM_TYPES,
    LIST_FIELDS,
    LOCATED_FIELDS,
    MAX_FIELD_LOCATIONS,
    MAX_NAME_LOCATIONS,
    NAME_LOCATION_KINDS,
    NAME_ORDER_BASES,
    NAME_ORDER_CONVENTIONS,
    NAME_SCRIPTS,
    RECEIVED_DATE_SOURCES,
)
from src.image_preprocessor import resize_for_model
from src import claude_client
from src.claude_client import THINKING_LEVELS, ImagePart, LLMError, LLMQuotaError, ModelReply, ModelRequest
from src.validators import list_approved_senders

logger = logging.getLogger(__name__)

NAME_KEYS = ("first_name", "last_name")

def image_part(image: Image.Image, max_side: int) -> ImagePart:
    """Encode an image as JPEG, downscaled to max_side."""
    img = resize_for_model(image.convert("RGB"), max_side)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=90)
    return ImagePart(buf.getvalue(), "image/jpeg")


def _parse_json(text: Optional[str]) -> dict:
    if not text:
        raise ValueError("empty response")
    s = text.strip()
    if s.startswith("```"):  # tolerate fenced output despite structured output
        s = s.strip("`")
        s = s[s.find("{"):]
    data = json.loads(s)
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


# ------------------------------------------------------------------ token usage

# Image-detail tag per request purpose, hashed into the dev-cache key. It no longer changes the request; it
# stays in the key so the cached answers of earlier runs keep their keys (they save money).
_CACHE_DETAIL = {"orientation": "low", "split": "medium"}

_usage_lock = threading.Lock()
_usage: dict = {}  # purpose -> {"requests", "prompt_tokens", "output_tokens", "cache_*_tokens", "models"}


def _record_usage(reply, purpose: str) -> None:
    """prompt_tokens counts all input tokens, cached ones included (see llm_provider.cost_usd)."""
    cache_read = getattr(reply, "cache_read_tokens", 0) or 0
    cache_write = getattr(reply, "cache_write_tokens", 0) or 0
    with _usage_lock:
        u = _usage.setdefault(purpose, {"requests": 0, "prompt_tokens": 0, "output_tokens": 0,
                                        "cache_read_tokens": 0, "cache_write_tokens": 0, "thinking_tokens": 0,
                                        "seconds": 0.0,
                                        "models": {}})
        u["requests"] += 1
        u["prompt_tokens"] += (getattr(reply, "input_tokens", 0) or 0) + cache_read + cache_write
        u["output_tokens"] += getattr(reply, "output_tokens", 0) or 0
        u["cache_read_tokens"] += cache_read
        u["cache_write_tokens"] += cache_write
        u["thinking_tokens"] += getattr(reply, "thinking_tokens", 0) or 0
        u["seconds"] += getattr(reply, "seconds", 0.0) or 0.0
        model = getattr(reply, "model", "") or "unknown"
        u["models"][model] = u["models"].get(model, 0) + 1


def get_usage(purpose: Optional[str] = None) -> dict:
    """Token totals for live (non-cached) calls since the last reset - all purposes, or one.

    Counts only, never content. Purposes: "extract", "verify_names", "reread_fields", "orientation",
    "split".
    """
    with _usage_lock:
        rows = [_usage.get(purpose, {})] if purpose else list(_usage.values())
        return {k: sum(r.get(k, 0) for r in rows) for k in ("requests", "prompt_tokens", "output_tokens")}


def get_usage_by_purpose() -> dict:
    with _usage_lock:
        return {k: {**v, "models": dict(v["models"])} for k, v in _usage.items()}


def reset_usage() -> None:
    with _usage_lock:
        _usage.clear()


# ------------------------------------------------------- dev-only response cache

_cache_warned = threading.Event()


def _cache_key(contents: list, schema: dict, system_instruction: Optional[str], detail: str,
               thinking_level: Optional[str], model_id: str) -> str:
    """SHA-256 of everything that determines the answer: model and settings (claude_client.cache_id), schema,
    prompt, images, thinking level. The thinking level is hashed only when set (stable keys)."""
    h = hashlib.sha256()
    pieces = [model_id, json.dumps(schema, sort_keys=True, ensure_ascii=False), system_instruction or "", detail]
    if thinking_level:
        pieces.append(f"thinking_level={thinking_level}")
    for piece in pieces:
        h.update(piece.encode("utf-8"))
        h.update(b"\x00")
    for item in contents:
        if isinstance(item, str):
            h.update(b"T" + item.encode("utf-8"))
        elif isinstance(item, ImagePart):
            h.update(b"I" + hashlib.sha256(item.data).digest())
        else:
            h.update(b"?" + repr(item).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    if not _cache_warned.is_set():
        _cache_warned.set()
        logger.warning("DEV_CACHE is enabled: model responses (may contain personal data) are "
                       "stored in %s. Development only - never enable in production.",
                       config.DEV_CACHE_DIR.name)
    path = config.DEV_CACHE_DIR / f"{key}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cache_put(key: str, data: dict) -> None:
    config.DEV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DEV_CACHE_DIR / f"{key}.json"
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def clear_dev_cache() -> int:
    """Delete all cached responses; returns the number of files removed."""
    if not config.DEV_CACHE_DIR.exists():
        return 0
    files = list(config.DEV_CACHE_DIR.glob("*.json"))
    for f in files:
        f.unlink(missing_ok=True)
    return len(files)


def _call_model(contents: list, request: ModelRequest) -> ModelReply:
    """The single place where a request leaves the app (the evaluation scripts guard this call)."""
    return claude_client.call(contents, request)


def generate_json(contents: list, schema: dict, *, system_instruction: Optional[str] = None,
                  purpose: str = "other", thinking_level: Optional[str] = None) -> dict:
    """Call Claude with structured JSON output and return the parsed object.

    contents: prompt strings and ImageParts (see image_part). schema: compact notation, converted to JSON
    Schema by claude_client. Transient HTTP errors are retried by the SDK; unparsable JSON is retried
    INVALID_JSON_RETRIES more times. Raises LLMError (LLMQuotaError: stop the run) on failure.
    purpose ("extract", "split", ...) sets effort and thinking; thinking_level (one of THINKING_LEVELS)
    overrides the effort. With DEV_CACHE on (development only), identical requests are answered from disk.
    """
    if thinking_level is not None and thinking_level not in THINKING_LEVELS:
        raise ValueError(f"thinking_level must be one of {THINKING_LEVELS}")
    request = ModelRequest(schema=schema, system_instruction=system_instruction, thinking_level=thinking_level,
                           purpose=purpose)
    cache_key = None
    if config.DEV_CACHE:
        cache_key = _cache_key(contents, schema, system_instruction, _CACHE_DETAIL.get(purpose, "high"),
                               thinking_level, model_id=claude_client.cache_id(request))
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
    last_error = "unknown"
    for attempt in range(1 + config.INVALID_JSON_RETRIES):
        reply = _call_model(contents, request)
        _record_usage(reply, purpose)
        try:
            data = _parse_json(reply.text)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = type(exc).__name__
            logger.warning("Invalid JSON from the model (attempt %d): %s", attempt + 1, last_error)
            continue
        if cache_key:
            _cache_put(cache_key, data)
        return data
    raise LLMError(f"Model returned invalid JSON ({last_error})")


# ------------------------------------------------------------------ extraction

@dataclass
class FieldResult:
    value: Union[str, list, None]  # list[str] for LIST_FIELDS
    confidence: float
    source: Optional[str] = None    # received_date only: one of RECEIVED_DATE_SOURCES
    script: Optional[str] = None    # first_name / last_name only: one of NAME_SCRIPTS
    status: Optional[str] = None    # one of FIELD_STATUSES; None = not reported
    note: Optional[str] = None      # Claude only: where on the form the value was read (never the value)
    raw_text: Optional[str] = None  # Claude only: the text exactly as written, even partial ('?' = illegible
                                    # character). Shown to the reviewer only - never a value, never logged.


@dataclass
class ExtractionResult:
    form_type: str
    form_type_confidence: float
    fields: dict                    # field key -> FieldResult
    error: Optional[str] = None
    name_order_basis: str = "unclear"  # one of NAME_ORDER_BASES
    name_locations: list = field(default_factory=list)  # [{"page", "box", "kind"}], box = 0-1000
    field_locations: list = field(default_factory=list)  # [{"field", "page", "box"}], LOCATED_FIELDS only
    # Claude only: the sample collection date as written (validation of received_date only, never exported).
    collection_date: Optional[str] = None
    highlight_crops: int = 0        # number of yellow-highlight crops sent with the page(s)


def _value_schema(key: str, sender_names: list) -> dict:
    if key in LIST_FIELDS:
        return {"type": "ARRAY", "items": {"type": "STRING"}}
    if key == "sending_entity":
        return {"type": "STRING", "enum": list(sender_names), "nullable": True}
    return {"type": "STRING", "nullable": True}


def build_schema(sender_names: list) -> dict:
    """Response schema: form_type plus, per field, value, confidence ("high" | "medium" | "low"), status,
    source_note and (single-value fields) raw_text; collection_date for the received-date check."""
    confidence_schema = {"type": "STRING", "enum": list(CONFIDENCE_LABELS)}
    props = {
        "form_type": {
            "type": "OBJECT",
            "properties": {
                "value": {"type": "STRING", "enum": list(FORM_TYPES)},
                "confidence": {"type": "NUMBER"},
            },
            "required": ["value", "confidence"],
        }
    }
    for key in FIELD_KEYS:
        field_props = {"value": _value_schema(key, sender_names), "confidence": dict(confidence_schema),
                       "status": {"type": "STRING", "enum": list(FIELD_STATUSES)}}
        required = ["value", "confidence", "status"]
        # raw_text: plain string ("" = none) and single-value fields only - with nullable unions or on the
        # list fields too, the compiled grammar was too large for the API (400).
        if key not in LIST_FIELDS:
            field_props["raw_text"] = {"type": "STRING"}
            required.append("raw_text")
        field_props["source_note"] = {"type": "STRING"}
        required.append("source_note")
        if key == "received_date":
            field_props["received_date_source"] = {"type": "STRING", "enum": list(RECEIVED_DATE_SOURCES)}
            required.append("received_date_source")
        if key in NAME_KEYS:
            field_props["script"] = {"type": "STRING", "enum": list(NAME_SCRIPTS)}
            required.append("script")
        props[key] = {
            "type": "OBJECT",
            "properties": field_props,
            "required": required,
            "propertyOrdering": list(field_props),
        }
        if key == "last_name":
            props["name_order_basis"] = {"type": "STRING", "enum": list(NAME_ORDER_BASES)}
            props["name_locations"] = {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "page": {"type": "INTEGER"},
                        "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                        "kind": {"type": "STRING", "enum": list(NAME_LOCATION_KINDS)},
                    },
                    "required": ["page", "box_2d", "kind"],
                    "propertyOrdering": ["page", "box_2d", "kind"],
                },
            }
    props["field_locations"] = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "field": {"type": "STRING", "enum": list(LOCATED_FIELDS)},
                "page": {"type": "INTEGER"},
                "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
            },
            "required": ["field", "page", "box_2d"],
            "propertyOrdering": ["field", "page", "box_2d"],
        },
    }
    props["collection_date"] = {
        "type": "OBJECT",
        "properties": {"value": {"type": "STRING"}, "confidence": dict(confidence_schema)},
        "required": ["value", "confidence"],
    }
    return {
        "type": "OBJECT",
        "properties": props,
        "required": list(props),
        "propertyOrdering": list(props),
    }


# Appended to prompts/extraction_prompt.txt. The rules in that file stay word for word; this maps their
# numeric confidence bands onto the labels the schema asks for, and adds the output-only fields.
EXTRACTION_RULES = """
==========================================================================
CONFIDENCE LABELS AND SOURCE NOTES (for this response format)
==========================================================================
For the seven fields (first_name ... commitment_numbers), confidence is a label instead of a number:
- "high":   the 0.80-1.0 band above (clearly printed or clear handwriting, and sure it is the right field).
- "medium": the 0.50-0.79 band above (readable but some doubt).
- "low":    below 0.50 (could not read it or could not find it) -> value must be null / [].
Wherever a rule above caps the confidence (e.g. "confidence <= 0.7" or "<= 0.6"), answer "medium" or lower;
a cap of 0.4 or below means "low". form_type.confidence stays a number 0-1.
If it is not legible, return null. Do not guess. Do not complete names to common names.
(If any letter or digit of a value would be a guess, the value is null with "low".)
raw_text (names, id_number, received_date, sending_entity): the text EXACTLY as written in that field, even if partial or unclear - write "?" for each
character you cannot read (e.g. "11?233?4"). It is shown to a human reviewer only and is never used as the value,
so it may be incomplete; value stays null when the text is not fully legible. raw_text is an empty string when
nothing is written there at all.
source_note (per field): a few words on WHERE you read the value (label, sticker, stamp, handwriting, page) or why
it is empty, e.g. "GLC sticker, page 1 top left" or "stamp present, date smudged". Never write the value, a name or
a number in source_note. Use an empty string when there is nothing to say.
collection_date: the date the SAMPLE WAS TAKEN, only when the form states it ("תאריך לקיחה", "תאריך לקיחת הדגימה",
"נלקח בתאריך"), copied as written; empty string otherwise. It is NOT the received date and is only used to check it.

ZOOMED CROPS (when the request includes them, after the full page):
The zoomed crops are only magnified views to help legibility. Highlighting does NOT indicate which field a value belongs to. Some highlighted values are decoys (visit date, intake date, form date). Identify every field only from its printed label and position on the full page.
"""


def load_prompt(sender_names: list) -> str:
    template = config.PROMPT_PATH.read_text(encoding="utf-8")
    name_rules = "\n".join(f"- {where} => {order}" for where, order in NAME_ORDER_CONVENTIONS)
    prompt = (template
              .replace("{{NAME_ORDER_RULES}}", name_rules)
              .replace("{{APPROVED_SENDERS}}", "\n".join(f"- {n}" for n in sender_names)))
    return prompt + EXTRACTION_RULES


def _confidence(value) -> float:
    if isinstance(value, str) and value.strip().lower() in CONFIDENCE_LABELS:
        return CONFIDENCE_LABELS[value.strip().lower()]
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, c)) if c == c else 0.0  # NaN -> 0


def _count(value) -> int:
    """A non-negative count from the model; missing or malformed => 1 (unknown counts as uncertain)."""
    if isinstance(value, bool):
        return 1
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 1
    return max(0, n)


def _clean_scalar(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return None if s == "" or s.lower() in ("null", "none", "n/a") else s


def _clean_list(value) -> list:
    items = value if isinstance(value, list) else ([] if value is None else [value])
    out = []
    for item in items:
        s = _clean_scalar(item)
        if s is not None:
            s = "".join(s.split())  # numbers: drop inner spaces ("2648 00879" -> "264800879")
            if s not in out:
                out.append(s)
    return out


def parse_extraction(data: dict) -> ExtractionResult:
    """Normalize the model's JSON into an ExtractionResult, tolerating missing/odd values."""
    data = data if isinstance(data, dict) else {}
    ft = data.get("form_type") if isinstance(data.get("form_type"), dict) else {}
    form_type = ft.get("value") if ft.get("value") in FORM_TYPES else "unknown"
    fields = {}
    for key in FIELD_KEYS:
        raw = data.get(key) if isinstance(data.get(key), dict) else {}
        value = _clean_list(raw.get("value")) if key in LIST_FIELDS else _clean_scalar(raw.get("value"))
        status = raw.get("status")
        note = _clean_scalar(raw.get("source_note"))
        raw_text = _clean_scalar(raw.get("raw_text"))
        result = FieldResult(value=value, confidence=_confidence(raw.get("confidence")),
                             status=status if status in FIELD_STATUSES else None,
                             note=note[:160] if note else None,
                             raw_text=raw_text[:120] if raw_text else None)
        if key == "received_date":
            source = raw.get("received_date_source")
            result.source = source if source in RECEIVED_DATE_SOURCES else "none"
        if key in NAME_KEYS:
            script = raw.get("script")
            result.script = script if script in NAME_SCRIPTS else "unknown"
        fields[key] = result
    basis = data.get("name_order_basis")
    collection = data.get("collection_date") if isinstance(data.get("collection_date"), dict) else {}
    return ExtractionResult(form_type, _confidence(ft.get("confidence")), fields,
                            name_order_basis=basis if basis in NAME_ORDER_BASES else "unclear",
                            name_locations=_parse_locations(data.get("name_locations")),
                            field_locations=_parse_field_locations(data.get("field_locations")),
                            collection_date=_clean_scalar(collection.get("value")))


def _parse_box(item) -> Optional[tuple]:
    """(page, (ymin, xmin, ymax, xmax)) for a well-formed item: page >= 1, box within 0-1000."""
    if not isinstance(item, dict):
        return None
    box, page = item.get("box_2d"), item.get("page")
    if not (isinstance(page, int) and not isinstance(page, bool) and page >= 1
            and isinstance(box, list) and len(box) == 4):
        return None
    try:
        y0, x0, y1, x1 = (int(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not (0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000):
        return None
    return page, (y0, x0, y1, x1)


def _parse_locations(raw) -> list:
    """Keep only well-formed name boxes (see _parse_box), at most MAX_NAME_LOCATIONS."""
    out = []
    for item in raw if isinstance(raw, list) else []:
        parsed = _parse_box(item)
        if parsed is None:
            continue
        kind = item.get("kind") if item.get("kind") in NAME_LOCATION_KINDS else "other"
        out.append({"page": parsed[0], "box": parsed[1], "kind": kind})
    return out[:MAX_NAME_LOCATIONS]


def _parse_field_locations(raw) -> list:
    """Well-formed boxes of LOCATED_FIELDS, at most MAX_FIELD_LOCATIONS per field, duplicates dropped."""
    out, per_field = [], {}
    for item in raw if isinstance(raw, list) else []:
        parsed = _parse_box(item)
        key = item.get("field") if isinstance(item, dict) else None
        if parsed is None or key not in LOCATED_FIELDS:
            continue
        loc = {"field": key, "page": parsed[0], "box": parsed[1]}
        if loc in out or per_field.get(key, 0) >= MAX_FIELD_LOCATIONS[key]:
            continue
        per_field[key] = per_field.get(key, 0) + 1
        out.append(loc)
    return out


def empty_result(error: str) -> ExtractionResult:
    """All fields empty with zero confidence: used when extraction failed."""
    fields = {k: FieldResult([] if k in LIST_FIELDS else None, 0.0) for k in FIELD_KEYS}
    fields["received_date"].source = "none"
    return ExtractionResult("unknown", 0.0, fields, error=error)


def extract_form(images: list, sender_names: Optional[list] = None, crop_images: Optional[list] = None,
                 form_label: str = "form") -> ExtractionResult:
    """Extract all fields from one form's page images. Never raises on API problems.

    crop_images: the same pages at full 300 DPI resolution (original pixels). With HIGHLIGHT_CROPS on, zoomed
    crops of their highlighted regions are added to the request; a page without highlighting adds nothing.
    """
    from src.highlight_crops import CROP_IMAGE_MAX_SIDE, crop_contents, highlight_crops, save_debug_crops

    sender_names = sender_names or list_approved_senders(config.SENDERS_PATH)
    contents: list = []
    for i, img in enumerate(images, start=1):
        if len(images) > 1:
            contents.append(f"Page {i} of {len(images)}:")
        contents.append(image_part(img, config.EXTRACTION_IMAGE_MAX_SIDE))
    crops = []
    if crop_images and config.HIGHLIGHT_CROPS_ENABLED:
        crops = highlight_crops(crop_images)
        save_debug_crops(crops, form_label)
        contents += crop_contents(crops, len(images), image_part, CROP_IMAGE_MAX_SIDE)
    contents.append("Extract the fields from this form now.")
    try:
        data = generate_json(contents, build_schema(sender_names),
                             system_instruction=load_prompt(sender_names), purpose="extract")
    except LLMQuotaError:
        raise  # the caller stops the run; no point in sending more requests
    except LLMError as exc:
        logger.warning("Extraction failed: %s", exc)
        return empty_result(str(exc))
    result = parse_extraction(data)
    result.highlight_crops = len(crops)
    logger.info(
        "Extracted form_type=%s confidences=%s highlight_crops=%d",
        result.form_type,
        {k: round(f.confidence, 2) for k, f in result.fields.items()},
        len(crops),
    )
    return result


# ------------------------------------------------------------ name verification
# Handwritten names are the riskiest field: the model can turn unclear cursive into a
# plausible, confident, wrong name. Each name location is cropped from the full-resolution
# page and re-read at high resolution in a separate request. The crop pass always returns its best
# reading (with a count of uncertain letters), so the reviewer gets a suggestion instead of an empty field;
# record_builder decides what is approved and flags (or empties) the names when the readings differ.

# How a crop reading assigned first / last name.
NAME_ASSIGNMENTS = ("labeled_fields", "convention", "unclear")

NAME_CHECK_PROMPT = """You verify PATIENT NAMES on scanned Israeli medical forms. Each crop below comes from one form
and should contain the patient's name, often handwritten in Hebrew cursive, possibly with its labels.

For EACH crop, independently:
- full_name: the name exactly as written, words in reading order (right-to-left), or null.
- first_name / last_name: always fill them when legible.
  If the crop shows separate labeled fields ("שם פרטי", "שם משפחה"), use the labels: assignment = "labeled_fields".
  In Hebrew a value is written to the LEFT of its label: each value belongs to the label immediately to its RIGHT.
  If the name is in one field / sticker line, split it by the KNOWN NAME ORDERS below: assignment = "convention";
  if that layout is not listed, split with the family name first and set assignment = "unclear".
- Read letter by letter. Hebrew cursive letter shapes differ from print. ALWAYS give your best reading of the letters
  you see, even when some letters are unclear: a person reviews every uncertain reading, so a best reading helps and
  an empty answer does not. Never replace what is written with a common or plausible name - transcribe the strokes.
- legible = false only when the crop shows no name at all (empty, cut off entirely, covered); then the names are null.
- uncertain_letters: how many letters of the name you are NOT sure about (0 = every letter is clear).
- confidence: 0-1, how sure you are of every letter (any uncertain letter => below 0.8).
Ignore yellow highlighter.{views_note} Do not output anything except the JSON.

Form type reported by the full-page pass: {form_type}
KNOWN NAME ORDERS (single-field names):
{name_rules}"""

NAME_CHECK_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "crops": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "crop_index": {"type": "INTEGER"},
                    "legible": {"type": "BOOLEAN"},
                    "full_name": {"type": "STRING", "nullable": True},
                    "assignment": {"type": "STRING", "enum": list(NAME_ASSIGNMENTS)},
                    "first_name": {"type": "STRING", "nullable": True},
                    "last_name": {"type": "STRING", "nullable": True},
                    "uncertain_letters": {"type": "INTEGER"},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["crop_index", "legible", "full_name", "assignment", "first_name", "last_name",
                             "uncertain_letters", "confidence"],
                "propertyOrdering": ["crop_index", "legible", "full_name", "assignment", "first_name",
                                     "last_name", "uncertain_letters", "confidence"],
            },
        }
    },
    "required": ["crops"],
}

NAME_CROP_PAD_Y = 0.6     # vertical padding, as a fraction of the box height
NAME_CROP_PAD_X = 0.35    # horizontal padding (fraction of box width): labels sit beside values and
                          # model boxes can be too narrow (seen on sample page 9: name cut at the left)
NAME_CROP_MIN_PAD_PX = 40
NAME_CROP_MAX_SIDE = 1600
# NAME_CROP_ENHANCE: small name crops are enlarged so the crop is at least NAME_CROP_TARGET_HEIGHT px high
# (at most NAME_CROP_MAX_UPSCALE x, never above NAME_CROP_MAX_SIDE), and an "ink only" grey view (yellow
# highlighter removed, pen ink dark - ink_view) is sent next to the colour crop. Image processing only:
# the model's reading is not changed afterwards.
NAME_CROP_TARGET_HEIGHT = 450
NAME_CROP_MAX_UPSCALE = 3.0
NAME_VIEWS_NOTE = (" Each crop comes twice: in colour, and as an 'ink only' grey view of the SAME crop (yellow "
                   "highlighter removed, pen ink dark). Read both and give ONE answer per crop.")


@dataclass
class NameReading:
    kind: str
    legible: bool
    full_name: Optional[str]
    assignment: str
    first_name: Optional[str]
    last_name: Optional[str]
    confidence: float
    uncertain_letters: int = 0      # letters the model was not sure about; unknown counts as uncertain


@dataclass
class NameCheck:
    needed: bool
    readings: list = field(default_factory=list)  # list[NameReading], one per crop
    error: Optional[str] = None


def needs_name_check(extraction: ExtractionResult) -> bool:
    """Verify unless both names are machine-printed. With no page reading, verify only when
    a name location was reported (the crop reading can then become a suggestion)."""
    if extraction.error:
        return False
    names = [extraction.fields[k] for k in NAME_KEYS]
    if all(f.script == "printed" for f in names):
        return False
    return any(f.value for f in names) or bool(extraction.name_locations)


def enlarge_name_crop(crop: Image.Image) -> Image.Image:
    """Upscale a small name crop towards NAME_CROP_TARGET_HEIGHT (never shrinks; capped by max upscale and side)."""
    factor = min(NAME_CROP_MAX_UPSCALE, NAME_CROP_TARGET_HEIGHT / max(1, crop.height),
                 NAME_CROP_MAX_SIDE / max(crop.size))
    if factor <= 1.0:
        return crop
    return crop.resize((round(crop.width * factor), round(crop.height * factor)), Image.LANCZOS)


def ink_view(crop: Image.Image) -> Image.Image:
    """Grey view of a name crop without yellow highlighter: the red channel, contrast-stretched. Yellow
    highlighter and paper have high red (white); blue, green and black pen have low red (dark). Not
    highlight_crops.ink_only: that mask is tuned for blue / black ink and erased green pen (seen on a form)."""
    return ImageOps.autocontrast(crop.convert("RGB").getchannel("R"), cutoff=1)


def name_crop_views(crop: Image.Image) -> list:
    """[(label suffix, image)] sent for one name crop: the crop as is, or - with NAME_CROP_ENHANCE - the
    enlarged colour crop plus its enlarged ink view (highlighter removed)."""
    if not config.NAME_CROP_ENHANCE:
        return [("", crop)]
    big = enlarge_name_crop(crop)
    return [(" - colour", big), (" - ink only", ink_view(big))]


def crop_name_regions(images: list, locations: list) -> list:
    """Crop each reported name location (0-1000 box) from the full-resolution page, with padding."""
    crops = []
    for loc in locations:
        if loc["page"] > len(images):
            continue
        img = images[loc["page"] - 1]
        box = padded_pixel_box(img.size, loc["box"], NAME_CROP_PAD_X, NAME_CROP_PAD_Y, NAME_CROP_MIN_PAD_PX)
        if box is None:
            continue
        crops.append((loc["kind"], img.crop(box)))
    return crops


def padded_pixel_box(size: tuple, box: tuple, pad_x: float, pad_y: float, min_pad: int) -> Optional[tuple]:
    """Map a 0-1000 [ymin, xmin, ymax, xmax] box to a padded (left, top, right, bottom) pixel box,
    clamped to the image. None when the clamped box is degenerate (under 10 px on a side)."""
    w, h = size
    y0, x0, y1, x1 = (v / 1000 for v in box)
    bw, bh = (x1 - x0) * w, (y1 - y0) * h
    px, py = max(min_pad, bw * pad_x), max(min_pad, bh * pad_y)
    out = (max(0, int(x0 * w - px)), max(0, int(y0 * h - py)),
           min(w, int(x1 * w + px)), min(h, int(y1 * h + py)))
    if out[2] - out[0] < 10 or out[3] - out[1] < 10:
        return None
    return out


def verify_names(images: list, extraction: ExtractionResult) -> NameCheck:
    """Re-read every name location from high-resolution crops. Never raises on API problems."""
    if not needs_name_check(extraction):
        return NameCheck(needed=False)
    crops = crop_name_regions(images, extraction.name_locations)
    if not crops:
        return NameCheck(needed=True, error="no name location to verify")
    name_rules = "\n".join(f"- {where} => {order}" for where, order in NAME_ORDER_CONVENTIONS)
    views_note = NAME_VIEWS_NOTE if config.NAME_CROP_ENHANCE else ""
    contents: list = [NAME_CHECK_PROMPT.format(form_type=extraction.form_type, name_rules=name_rules,
                                               views_note=views_note)]
    for i, (_, crop) in enumerate(crops, start=1):
        for label, view in name_crop_views(crop):
            contents += [f"Crop {i}{label}:", image_part(view, NAME_CROP_MAX_SIDE)]
    try:
        data = generate_json(contents, NAME_CHECK_SCHEMA, purpose="verify_names")
    except LLMQuotaError:
        raise
    except LLMError as exc:
        logger.warning("Name verification failed: %s", exc)
        return NameCheck(needed=True, error=str(exc))
    by_index = {c.get("crop_index"): c for c in data.get("crops") or [] if isinstance(c, dict)}
    readings = []
    for i, (kind, _) in enumerate(crops, start=1):
        c = by_index.get(i)
        if c is None:
            readings.append(NameReading(kind, False, None, "unclear", None, None, 0.0))
            continue
        legible = bool(c.get("legible"))
        readings.append(NameReading(
            kind=kind,
            legible=legible,
            full_name=_clean_scalar(c.get("full_name")) if legible else None,
            assignment=c.get("assignment") if c.get("assignment") in NAME_ASSIGNMENTS else "unclear",
            first_name=_clean_scalar(c.get("first_name")) if legible else None,
            last_name=_clean_scalar(c.get("last_name")) if legible else None,
            confidence=_confidence(c.get("confidence")),
            uncertain_letters=_count(c.get("uncertain_letters")),
        ))
    logger.info("Verified names from %d crop(s); legible=%s", len(readings), [r.legible for r in readings])
    return NameCheck(needed=True, readings=readings)
