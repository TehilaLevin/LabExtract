"""Gemini client, prompt, JSON schema, per-field confidence.

Privacy: only form images (and the fixed prompt) are sent to the API. Nothing from
the model's answer is logged - only field names, confidences and error types.
"""
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Union

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

from src import config
from src.constants import FIELD_KEYS, FORM_TYPES, LIST_FIELDS, RECEIVED_DATE_SOURCES
from src.image_preprocessor import resize_for_model
from src.validators import list_approved_senders

logger = logging.getLogger(__name__)


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


# ------------------------------------------------------- dev-only response cache

_cache_warned = threading.Event()


def _cache_key(contents: list, schema: dict, system_instruction: Optional[str],
               high_resolution: bool) -> str:
    """SHA-256 of everything that determines the answer: model, schema, prompt, images."""
    h = hashlib.sha256()
    for piece in (config.GEMINI_MODEL, json.dumps(schema, sort_keys=True, ensure_ascii=False),
                  system_instruction or "", str(high_resolution)):
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


def generate_json(contents: list, schema: dict, *, system_instruction: Optional[str] = None,
                  high_resolution: bool = True) -> dict:
    """Call Gemini with structured JSON output and return the parsed object.

    Transient HTTP errors are retried by the client (HttpRetryOptions); unparsable
    JSON is retried INVALID_JSON_RETRIES more times. Raises GeminiError on failure.
    With DEV_CACHE on (development only), identical requests are answered from disk.
    """
    cache_key = None
    if config.DEV_CACHE:
        cache_key = _cache_key(contents, schema, system_instruction, high_resolution)
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        system_instruction=system_instruction,
        media_resolution=(
            types.MediaResolution.MEDIA_RESOLUTION_HIGH if high_resolution
            else types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
        ),
    )
    last_error = "unknown"
    for attempt in range(1 + config.INVALID_JSON_RETRIES):
        try:
            response = get_client().models.generate_content(
                model=config.GEMINI_MODEL, contents=contents, config=cfg
            )
        except genai_errors.APIError as exc:
            if exc.code == 429:
                raise GeminiQuotaError(f"Gemini API quota exhausted (429 {exc.status})") from None
            raise GeminiError(f"Gemini API error {exc.code}: {exc.status}") from None
        except Exception as exc:  # network errors, timeouts after retries
            raise GeminiError(f"Gemini request failed: {type(exc).__name__}") from None
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


@dataclass
class ExtractionResult:
    form_type: str
    form_type_confidence: float
    fields: dict                    # field key -> FieldResult
    error: Optional[str] = None


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
        field_props = {"value": _value_schema(key, sender_names), "confidence": {"type": "NUMBER"}}
        required = ["value", "confidence"]
        if key == "received_date":
            field_props["received_date_source"] = {"type": "STRING", "enum": list(RECEIVED_DATE_SOURCES)}
            required.append("received_date_source")
        props[key] = {
            "type": "OBJECT",
            "properties": field_props,
            "required": required,
            "propertyOrdering": list(field_props),
        }
    return {
        "type": "OBJECT",
        "properties": props,
        "required": list(props),
        "propertyOrdering": list(props),
    }


def load_prompt(sender_names: list) -> str:
    template = config.PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{{APPROVED_SENDERS}}", "\n".join(f"- {n}" for n in sender_names))


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
        result = FieldResult(value=value, confidence=_confidence(raw.get("confidence")))
        if key == "received_date":
            source = raw.get("received_date_source")
            result.source = source if source in RECEIVED_DATE_SOURCES else "none"
        fields[key] = result
    return ExtractionResult(form_type, _confidence(ft.get("confidence")), fields)


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
                             system_instruction=load_prompt(sender_names))
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
