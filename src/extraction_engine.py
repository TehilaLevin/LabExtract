"""Gemini client, prompt, JSON schema, per-field confidence.

Privacy: only form images (and the fixed prompt) are sent to the API. Nothing from
the model's answer is logged - only field names, confidences and error types.
"""
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional, Union

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

from src import config
from src.constants import (
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
from src.validators import list_approved_senders

logger = logging.getLogger(__name__)

NAME_KEYS = ("first_name", "last_name")


class GeminiError(RuntimeError):
    """A Gemini call failed. The message never contains form content."""


class GeminiQuotaError(GeminiError):
    """429 RESOURCE_EXHAUSTED after retries: stop sending requests for this run."""


# ------------------------------------------------------------------ Gemini client

_client: Optional[genai.Client] = None
_client_lock = threading.Lock()


def get_client() -> genai.Client:
    # One shared client, created under a lock: concurrent first calls must not create
    # several clients (a garbage-collected client closes its HTTP pool mid-request).
    global _client
    with _client_lock:
        if _client is None:
            _client = _new_client()
        return _client


def _new_client() -> genai.Client:
    return genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(
            timeout=config.REQUEST_TIMEOUT_S * 1000,  # milliseconds
            retry_options=types.HttpRetryOptions(
                attempts=config.MAX_ATTEMPTS,
                initial_delay=2.0,
                max_delay=30.0,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        ),
    )


def image_part(image: Image.Image, max_side: int) -> types.Part:
    """Encode an image as an inline JPEG part, downscaled to max_side."""
    img = resize_for_model(image.convert("RGB"), max_side)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=90)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def _parse_json(text: Optional[str]) -> dict:
    if not text:
        raise ValueError("empty response")
    s = text.strip()
    if s.startswith("```"):  # tolerate fenced output despite response_mime_type
        s = s.strip("`")
        s = s[s.find("{"):]
    data = json.loads(s)
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


# ------------------------------------------------------------------ token usage

_MEDIA_RESOLUTIONS = {
    "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
    "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
}

TRANSPORT_RETRY_DELAY_S = 2.0

_usage_lock = threading.Lock()
_usage: dict = {}  # purpose -> {"requests", "prompt_tokens", "output_tokens"}


def _record_usage(response, purpose: str) -> None:
    meta = getattr(response, "usage_metadata", None)
    with _usage_lock:
        u = _usage.setdefault(purpose, {"requests": 0, "prompt_tokens": 0, "output_tokens": 0})
        u["requests"] += 1
        if meta is not None:
            u["prompt_tokens"] += meta.prompt_token_count or 0
            u["output_tokens"] += (meta.candidates_token_count or 0) + (
                getattr(meta, "thoughts_token_count", 0) or 0)


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
        return {k: dict(v) for k, v in _usage.items()}


def reset_usage() -> None:
    with _usage_lock:
        _usage.clear()


# ------------------------------------------------------- dev-only response cache

_cache_warned = threading.Event()


def _cache_key(contents: list, schema: dict, system_instruction: Optional[str],
               media_resolution: str) -> str:
    """SHA-256 of everything that determines the answer: model, schema, prompt, images."""
    h = hashlib.sha256()
    for piece in (config.GEMINI_MODEL, json.dumps(schema, sort_keys=True, ensure_ascii=False),
                  system_instruction or "", media_resolution):
        h.update(piece.encode("utf-8"))
        h.update(b"\x00")
    for item in contents:
        if isinstance(item, str):
            h.update(b"T" + item.encode("utf-8"))
        elif getattr(item, "inline_data", None) is not None:
            h.update(b"I" + hashlib.sha256(item.inline_data.data).digest())
        else:
            h.update(b"?" + repr(item).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    if not _cache_warned.is_set():
        _cache_warned.set()
        logger.warning("DEV_CACHE is enabled: Gemini responses (may contain personal data) are "
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


def _generate_with_transport_retry(contents: list, cfg: types.GenerateContentConfig):
    """One generate_content call. HttpRetryOptions only retries HTTP status codes, so
    dropped connections / transport timeouts (e.g. RemoteProtocolError) are retried here."""
    for attempt in range(1, config.MAX_ATTEMPTS + 1):
        try:
            return get_client().models.generate_content(
                model=config.GEMINI_MODEL, contents=contents, config=cfg
            )
        except genai_errors.APIError as exc:
            if exc.code == 429:
                raise GeminiQuotaError(f"Gemini API quota exhausted (429 {exc.status})") from None
            raise GeminiError(f"Gemini API error {exc.code}: {exc.status}") from None
        except httpx.TransportError as exc:
            if attempt == config.MAX_ATTEMPTS:
                raise GeminiError(f"Gemini request failed: {type(exc).__name__}") from None
            logger.warning("Transport error %s (attempt %d), retrying", type(exc).__name__, attempt)
            time.sleep(TRANSPORT_RETRY_DELAY_S * attempt)
        except Exception as exc:  # anything else: do not retry blindly
            raise GeminiError(f"Gemini request failed: {type(exc).__name__}") from None


def generate_json(contents: list, schema: dict, *, system_instruction: Optional[str] = None,
                  media_resolution: str = "high", purpose: str = "other") -> dict:
    """Call Gemini with structured JSON output and return the parsed object.

    Transient HTTP errors are retried by the client (HttpRetryOptions); unparsable
    JSON is retried INVALID_JSON_RETRIES more times. Raises GeminiError on failure.
    media_resolution ("low" | "medium" | "high") sets how many tokens each image costs.
    With DEV_CACHE on (development only), identical requests are answered from disk.
    """
    if media_resolution not in _MEDIA_RESOLUTIONS:
        raise ValueError(f"media_resolution must be one of {sorted(_MEDIA_RESOLUTIONS)}")
    cache_key = None
    if config.DEV_CACHE:
        cache_key = _cache_key(contents, schema, system_instruction, media_resolution)
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        system_instruction=system_instruction,
        media_resolution=_MEDIA_RESOLUTIONS[media_resolution],
    )
    last_error = "unknown"
    for attempt in range(1 + config.INVALID_JSON_RETRIES):
        response = _generate_with_transport_retry(contents, cfg)
        _record_usage(response, purpose)
        try:
            data = _parse_json(response.text)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = type(exc).__name__
            logger.warning("Invalid JSON from Gemini (attempt %d): %s", attempt + 1, last_error)
            continue
        if cache_key:
            _cache_put(cache_key, data)
        return data
    raise GeminiError(f"Gemini returned invalid JSON ({last_error})")


# ------------------------------------------------------------------ extraction

@dataclass
class FieldResult:
    value: Union[str, list, None]  # list[str] for LIST_FIELDS
    confidence: float
    source: Optional[str] = None    # received_date only: one of RECEIVED_DATE_SOURCES
    script: Optional[str] = None    # first_name / last_name only: one of NAME_SCRIPTS
    status: Optional[str] = None    # one of FIELD_STATUSES; None = not reported


@dataclass
class ExtractionResult:
    form_type: str
    form_type_confidence: float
    fields: dict                    # field key -> FieldResult
    error: Optional[str] = None
    name_order_basis: str = "unclear"  # one of NAME_ORDER_BASES
    name_locations: list = field(default_factory=list)  # [{"page", "box", "kind"}], box = 0-1000
    field_locations: list = field(default_factory=list)  # [{"field", "page", "box"}], LOCATED_FIELDS only


def _value_schema(key: str, sender_names: list) -> dict:
    if key in LIST_FIELDS:
        return {"type": "ARRAY", "items": {"type": "STRING"}}
    if key == "sending_entity":
        return {"type": "STRING", "enum": list(sender_names), "nullable": True}
    return {"type": "STRING", "nullable": True}


def build_schema(sender_names: list) -> dict:
    """Response schema: form_type plus {value, confidence} per field."""
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
        field_props = {"value": _value_schema(key, sender_names), "confidence": {"type": "NUMBER"},
                       "status": {"type": "STRING", "enum": list(FIELD_STATUSES)}}
        required = ["value", "confidence", "status"]
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
    return {
        "type": "OBJECT",
        "properties": props,
        "required": list(props),
        "propertyOrdering": list(props),
    }


def load_prompt(sender_names: list) -> str:
    template = config.PROMPT_PATH.read_text(encoding="utf-8")
    name_rules = "\n".join(f"- {where} => {order}" for where, order in NAME_ORDER_CONVENTIONS)
    return (template
            .replace("{{NAME_ORDER_RULES}}", name_rules)
            .replace("{{APPROVED_SENDERS}}", "\n".join(f"- {n}" for n in sender_names)))


def _confidence(value) -> float:
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, c)) if c == c else 0.0  # NaN -> 0


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
        result = FieldResult(value=value, confidence=_confidence(raw.get("confidence")),
                             status=status if status in FIELD_STATUSES else None)
        if key == "received_date":
            source = raw.get("received_date_source")
            result.source = source if source in RECEIVED_DATE_SOURCES else "none"
        if key in NAME_KEYS:
            script = raw.get("script")
            result.script = script if script in NAME_SCRIPTS else "unknown"
        fields[key] = result
    basis = data.get("name_order_basis")
    return ExtractionResult(form_type, _confidence(ft.get("confidence")), fields,
                            name_order_basis=basis if basis in NAME_ORDER_BASES else "unclear",
                            name_locations=_parse_locations(data.get("name_locations")),
                            field_locations=_parse_field_locations(data.get("field_locations")))


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


def extract_form(images: list, sender_names: Optional[list] = None) -> ExtractionResult:
    """Extract all fields from one form's page images. Never raises on API problems."""
    sender_names = sender_names or list_approved_senders(config.SENDERS_PATH)
    contents: list = []
    for i, img in enumerate(images, start=1):
        if len(images) > 1:
            contents.append(f"Page {i} of {len(images)}:")
        contents.append(image_part(img, config.EXTRACTION_IMAGE_MAX_SIDE))
    contents.append("Extract the fields from this form now.")
    try:
        data = generate_json(contents, build_schema(sender_names),
                             system_instruction=load_prompt(sender_names), purpose="extract")
    except GeminiQuotaError:
        raise  # the caller stops the run; no point in sending more requests
    except GeminiError as exc:
        logger.warning("Extraction failed: %s", exc)
        return empty_result(str(exc))
    result = parse_extraction(data)
    logger.info(
        "Extracted form_type=%s confidences=%s",
        result.form_type,
        {k: round(f.confidence, 2) for k, f in result.fields.items()},
    )
    return result


# ------------------------------------------------------------ name verification
# Handwritten names are the riskiest field: the model can turn unclear cursive into a
# plausible, confident, wrong name. Each name location is cropped from the full-resolution
# page and re-read at high resolution in a separate request; record_builder compares the
# readings with the full-page reading and flags (or empties) the names when they differ.

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
- Read letter by letter. Hebrew cursive letter shapes differ from print. If you would have to guess ANY letter, set
  legible = false and full_name / first_name / last_name = null. Never output a plausible-looking name you cannot see.
- confidence: 0-1, how sure you are of every letter.
Ignore yellow highlighter. Do not output anything except the JSON.

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
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["crop_index", "legible", "full_name", "assignment", "first_name", "last_name",
                             "confidence"],
                "propertyOrdering": ["crop_index", "legible", "full_name", "assignment", "first_name",
                                     "last_name", "confidence"],
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


@dataclass
class NameReading:
    kind: str
    legible: bool
    full_name: Optional[str]
    assignment: str
    first_name: Optional[str]
    last_name: Optional[str]
    confidence: float


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
    contents: list = [NAME_CHECK_PROMPT.format(form_type=extraction.form_type, name_rules=name_rules)]
    for i, (_, crop) in enumerate(crops, start=1):
        contents += [f"Crop {i}:", image_part(crop, NAME_CROP_MAX_SIDE)]
    try:
        data = generate_json(contents, NAME_CHECK_SCHEMA, media_resolution="high", purpose="verify_names")
    except GeminiQuotaError:
        raise
    except GeminiError as exc:
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
        ))
    logger.info("Verified names from %d crop(s); legible=%s", len(readings), [r.legible for r in readings])
    return NameCheck(needed=True, readings=readings)
