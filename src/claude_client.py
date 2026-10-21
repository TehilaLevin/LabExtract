"""Claude client: the only model provider (Anthropic SDK).

Every model call in the app (orientation, splitting, extraction, name check, field re-read) goes through
extraction_engine.generate_json, which builds a ModelRequest and calls call(contents, request). contents is
a list of str (prompt text) and ImagePart (an encoded image). Response schemas are written in a compact
notation (uppercase types, "nullable") and converted to JSON Schema for structured outputs by to_json_schema.

Privacy: only the form images and the fixed prompts are sent. Only token counts and timing are logged -
never images, prompts, answers or API error bodies that could echo content.
"""
import base64
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from PIL import Image

from src import config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """A model call failed. The message never contains form content."""


class LLMQuotaError(LLMError):
    """The run must stop: rate limit still exceeded after retries, no credit, or the API key or model is
    rejected. Every further request would fail the same way."""


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    media_type: str = "image/jpeg"


@dataclass(frozen=True)
class ModelRequest:
    schema: dict                          # compact schema notation (see module doc)
    system_instruction: Optional[str] = None
    thinking_level: Optional[str] = None  # one of THINKING_LEVELS (mapped to effort); None = default
    purpose: str = "other"


@dataclass
class ModelReply:
    text: Optional[str]
    input_tokens: int = 0           # uncached input
    output_tokens: int = 0          # incl. thinking
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    thinking_tokens: int = 0        # part of output_tokens spent on thinking, when reported
    model: str = ""                 # the model that actually answered
    seconds: float = 0.0


THINKING_LEVELS = ("minimal", "low", "medium", "high")

# USD per 1M tokens: input, output, cache read, cache write (5-minute TTL = 1.25 x input).
# platform.claude.com pricing (Oct 2026).
PRICES_PER_MTOK = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-fable-5-1": (10.00, 50.00, 0.25, 12.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}


def cost_usd(model: str, usage: dict) -> Optional[float]:
    """Cost of a usage dict (see extraction_engine.get_usage_by_purpose); None for an unknown model."""
    price = next((p for name, p in PRICES_PER_MTOK.items() if model == name or model.startswith(name + "-")), None)
    if price is None:
        return None
    p_in, p_out, p_read, p_write = price
    uncached = usage.get("prompt_tokens", 0) - usage.get("cache_read_tokens", 0) - usage.get("cache_write_tokens", 0)
    return (uncached * p_in + usage.get("output_tokens", 0) * p_out
            + usage.get("cache_read_tokens", 0) * p_read + usage.get("cache_write_tokens", 0) * p_write) / 1e6


# ------------------------------------------------------------------ request settings

# Image limits: platform.claude.com/docs/en/build-with-claude/vision (high-resolution tier, Claude 4.7 and
# later). Larger images would be downscaled by the API anyway; doing it here keeps the payload small and
# the result deterministic.
CLAUDE_MAX_LONG_EDGE = 2576
CLAUDE_MAX_VISUAL_TOKENS = 4784          # ceil(w/28) * ceil(h/28)
CLAUDE_MANY_IMAGES = 20                  # above this many images per request ...
CLAUDE_MANY_IMAGES_MAX_SIDE = 2000       # ... every image must be at most this on both sides
CLAUDE_MAX_IMAGE_B64_BYTES = 10 * 1024 * 1024
CLAUDE_MEDIA_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
# Server-side refusal fallback ("default" routes by refusal category). Data entry from forms should never be
# declined; if it is, the request is re-run on another Claude model instead of failing.
CLAUDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Cheap layout questions: thinking adds cost there, not accuracy.
_LOW_EFFORT_PURPOSES = ("orientation", "split")
# Reading fields: thinking is requested explicitly (adaptive - the documented mode on current models;
# budget_tokens and temperature are rejected). Effort stays CLAUDE_EFFORT (high by default).
THINKING_PURPOSES = ("extract", "reread_fields")
THINKING_CONFIG = {"type": "adaptive"}
NAME_CHECK_PURPOSE = "verify_names"
_THINKING_TO_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}


def check_configured() -> None:
    """Raise RuntimeError with a clear message when the API key or effort setting is wrong."""
    config.CLAUDE_API_KEY  # noqa: B018 - raises when missing
    for name in ("CLAUDE_EFFORT", "NAME_CHECK_EFFORT"):
        if getattr(config, name) not in CLAUDE_EFFORTS:
            raise RuntimeError(f"{name} must be one of {CLAUDE_EFFORTS}, got {getattr(config, name)!r}")


def model_id() -> str:
    return config.CLAUDE_MODEL


def model_for(request: ModelRequest) -> str:
    """The name check may run on its own model (config.NAME_CHECK_MODEL); every other call uses CLAUDE_MODEL."""
    return config.NAME_CHECK_MODEL if request.purpose == NAME_CHECK_PURPOSE else config.CLAUDE_MODEL


def effort_for(request: ModelRequest) -> str:
    if request.thinking_level:
        return _THINKING_TO_EFFORT[request.thinking_level]
    if request.purpose in _LOW_EFFORT_PURPOSES:
        return "low"
    if request.purpose == NAME_CHECK_PURPOSE:
        return config.NAME_CHECK_EFFORT
    return config.CLAUDE_EFFORT


def cache_id(request: ModelRequest) -> str:
    """Model and request settings that change the answer (part of the dev-cache key)."""
    thinking = "|thinking=adaptive" if request.purpose in THINKING_PURPOSES else ""
    return f"claude|{model_for(request)}|effort={effort_for(request)}{thinking}"


# ------------------------------------------------------------------ images and schemas

def visual_tokens(width: int, height: int) -> int:
    return math.ceil(width / 28) * math.ceil(height / 28)


def fit_claude_image(part: ImagePart, max_side: int = CLAUDE_MAX_LONG_EDGE) -> ImagePart:
    """Downscale (never upscale) so the image is within Claude's per-image limits."""
    img = Image.open(BytesIO(part.data))
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if visual_tokens(round(w * scale), round(h * scale)) > CLAUDE_MAX_VISUAL_TOKENS:
        scale = min(scale, math.sqrt(CLAUDE_MAX_VISUAL_TOKENS * 28 * 28 / (w * h)))
        while scale > 0.05 and visual_tokens(round(w * scale), round(h * scale)) > CLAUDE_MAX_VISUAL_TOKENS:
            scale *= 0.99  # ceil() rounding can leave it a few tokens over
    media_type = part.media_type if part.media_type in CLAUDE_MEDIA_TYPES else "image/jpeg"
    data = part.data
    if scale < 1.0 or media_type != part.media_type:
        img = img.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        data, media_type = _jpeg(img, 90), "image/jpeg"
    quality = 85
    while len(base64.b64encode(data)) > CLAUDE_MAX_IMAGE_B64_BYTES and quality >= 40:
        data, media_type = _jpeg(Image.open(BytesIO(data)).convert("RGB"), quality), "image/jpeg"
        quality -= 15
    return ImagePart(data, media_type)


def _jpeg(img: Image.Image, quality: int) -> bytes:
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


_SCHEMA_TYPES = {"OBJECT": "object", "ARRAY": "array", "STRING": "string", "NUMBER": "number",
                 "INTEGER": "integer", "BOOLEAN": "boolean"}


def to_json_schema(schema: dict) -> dict:
    """Convert a compact response schema to JSON Schema for Claude structured outputs.

    Every object gets additionalProperties: false and lists ALL its properties as required (optional
    properties count toward a small per-request limit and would reorder the output). "nullable" becomes a
    ["type", "null"] union; a nullable enum becomes anyOf [enum, null] (the API rejects an enum next to a
    type array). "propertyOrdering" is dropped (properties keep their order).
    """
    out: dict = {}
    t = schema.get("type")
    json_type = _SCHEMA_TYPES.get(str(t).upper(), str(t).lower()) if t else None
    if json_type == "object":
        props = schema.get("properties", {})
        out = {"type": "object", "properties": {k: to_json_schema(v) for k, v in props.items()},
               "required": list(props), "additionalProperties": False}
    elif json_type == "array":
        out = {"type": "array", "items": to_json_schema(schema.get("items", {"type": "STRING"}))}
    elif json_type:
        out = {"type": json_type}
    if "enum" in schema:
        out["enum"] = list(schema["enum"])
    if "description" in schema:
        out["description"] = schema["description"]
    if schema.get("nullable") and "type" in out:
        if "enum" in out:
            description = out.pop("description", None)
            out = {"anyOf": [out, {"type": "null"}]}
            if description:
                out["description"] = description
        else:
            out["type"] = [out["type"], "null"]
    return out


# An enum value with an ASCII double quote (Hebrew acronyms: רמב"ם, צה"ל) comes back unescaped under
# structured outputs, which breaks the JSON (seen on every Rambam form). Such values are sent as an alias
# with the Hebrew gershayim "״" instead, and the alias is mapped back in the answer text.
_QUOTE_ALIAS = "״"


def alias_quoted_enums(schema: dict) -> tuple:
    """(schema copy with quote-free enum values, {alias: original}). Works on converted JSON Schema."""
    aliases: dict = {}

    def walk(node):
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        out = {k: walk(v) for k, v in node.items()}
        if isinstance(out.get("enum"), list):
            taken = set(out["enum"])
            values = []
            for v in out["enum"]:
                if isinstance(v, str) and ('"' in v or "\\" in v):
                    alias = v.replace('"', _QUOTE_ALIAS).replace("\\", "")
                    while alias in taken:
                        alias += _QUOTE_ALIAS
                    taken.add(alias)
                    aliases[alias] = v
                    v = alias
                values.append(v)
            out["enum"] = values
        return out

    return walk(schema), aliases


def restore_quoted_enums(text: Optional[str], aliases: dict) -> Optional[str]:
    """Put the original enum values (JSON-escaped) back in place of their aliases."""
    if not text or not aliases:
        return text
    for alias, original in aliases.items():
        escaped = json.dumps(original, ensure_ascii=False)
        text = text.replace(json.dumps(alias, ensure_ascii=False), escaped)
        text = text.replace(json.dumps(alias), escaped)  # the model may \\u-escape non-ASCII
    return text


def content_blocks(contents: list) -> list:
    n_images = sum(isinstance(c, ImagePart) for c in contents)
    max_side = CLAUDE_MANY_IMAGES_MAX_SIDE if n_images > CLAUDE_MANY_IMAGES else CLAUDE_MAX_LONG_EDGE
    blocks = []
    for item in contents:
        if isinstance(item, ImagePart):
            part = fit_claude_image(item, max_side)
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": part.media_type,
                "data": base64.standard_b64encode(part.data).decode("ascii")}})
        elif isinstance(item, str):
            if item.strip():  # the API rejects empty text blocks
                blocks.append({"type": "text", "text": item})
        else:
            raise TypeError(f"Unsupported content item: {type(item).__name__}")
    return blocks


# ------------------------------------------------------------------ the call

_client = None
_client_lock = threading.Lock()


def anthropic_client():
    """One shared SDK client, created under a lock (concurrent first calls must not create several)."""
    import anthropic

    global _client
    with _client_lock:
        if _client is None:
            _client = anthropic.Anthropic(
                api_key=config.CLAUDE_API_KEY,
                timeout=config.CLAUDE_TIMEOUT_S,
                # The SDK retries 408/409/429/5xx (incl. 529 overloaded) and connection errors with
                # exponential backoff and honours retry-after.
                max_retries=config.CLAUDE_MAX_RETRIES,
            )
        return _client


def call(contents: list, request: ModelRequest) -> ModelReply:
    """One Messages API request with structured JSON output. Raises LLMError / LLMQuotaError."""
    import anthropic

    schema, aliases = alias_quoted_enums(to_json_schema(request.schema))
    kwargs = {
        "model": model_for(request),
        "max_tokens": config.CLAUDE_MAX_TOKENS,
        "messages": [{"role": "user", "content": content_blocks(contents)}],
        # Sampling parameters (temperature) are rejected by current models; the JSON schema and the effort
        # level are what keep answers deterministic in shape.
        "output_config": {"effort": effort_for(request),
                          "format": {"type": "json_schema", "schema": schema}},
        "betas": [CLAUDE_FALLBACK_BETA],
        "fallbacks": "default",
    }
    if request.purpose in THINKING_PURPOSES:
        kwargs["thinking"] = dict(THINKING_CONFIG)
    if request.system_instruction:
        # The extraction prompt is identical for every form: cache it (reads cost 10% of input).
        kwargs["system"] = [{"type": "text", "text": request.system_instruction,
                             "cache_control": {"type": "ephemeral"}}]
    start = time.monotonic()
    try:
        response = anthropic_client().beta.messages.create(**kwargs)
    except anthropic.RateLimitError:
        raise LLMQuotaError("Claude API rate limit (429) still exceeded after retries") from None
    except anthropic.AuthenticationError:
        raise LLMQuotaError("Claude API key rejected (401): check CLAUDE_API_KEY (.env, or Secrets on Streamlit Cloud)") from None
    except anthropic.PermissionDeniedError:
        raise LLMQuotaError("Claude API key has no access (403)") from None
    except anthropic.NotFoundError:
        raise LLMQuotaError(f"Claude model not found (404): check model {model_for(request)}") from None
    except anthropic.BadRequestError as exc:
        # 400 messages describe the request shape (schema, limits); they never echo the images.
        if "credit balance" in str(exc.message).lower():  # every further request would fail: stop the run
            raise LLMQuotaError("Claude API credit balance too low: add credit in the Anthropic Console "
                                "(Plans & Billing)") from None
        raise LLMError(f"Claude API error 400: {str(exc.message)[:200]}") from None
    except anthropic.APITimeoutError:
        raise LLMError(f"Claude request timed out after {config.CLAUDE_TIMEOUT_S}s (with retries)") from None
    except anthropic.APIConnectionError:
        raise LLMError("Claude API connection failed (with retries)") from None
    except anthropic.APIStatusError as exc:
        if exc.status_code == 529:
            raise LLMError("Claude API overloaded (529) after retries - try again later") from None
        raise LLMError(f"Claude API error {exc.status_code}") from None
    usage = response.usage
    reply = ModelReply(
        text=restore_quoted_enums("".join(b.text for b in response.content if b.type == "text") or None, aliases),
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        thinking_tokens=getattr(getattr(usage, "output_tokens_details", None), "thinking_tokens", 0) or 0,
        model=response.model,
        seconds=time.monotonic() - start,
    )
    _log_call(request.purpose, reply)
    if response.stop_reason == "refusal":
        raise LLMError("Claude declined the request (refusal)")
    if response.stop_reason == "max_tokens":
        raise LLMError("Claude output was cut off (max_tokens)")
    return reply


def _log_call(purpose: str, reply: ModelReply) -> None:
    """Token usage and timing only - nothing from the request or the answer."""
    logger.info("model call purpose=%s model=%s input=%d cache_read=%d cache_write=%d output=%d (thinking=%d) "
                "time=%.1fs", purpose, reply.model, reply.input_tokens, reply.cache_read_tokens,
                reply.cache_write_tokens, reply.output_tokens, reply.thinking_tokens, reply.seconds)
