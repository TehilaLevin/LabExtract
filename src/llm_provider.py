"""Model providers behind one interface: Claude (default) and Gemini.

LLM_PROVIDER=claude|gemini selects the provider; rolling back to Gemini is that one variable.
Every model call in the app goes through extraction_engine.generate_json, which builds a
ModelRequest and calls get_provider().call(contents, request). contents is a list of str
(prompt text) and ImagePart (an encoded image); request.schema is written in the Gemini schema
dialect ("OBJECT", "nullable", ...) and converted for Claude by to_json_schema.

Privacy: providers send only the form images and the fixed prompts. Only token counts and timing
are logged - never images, prompts, answers or API error bodies that could echo content.
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
    """The run must stop: rate limit / quota still exhausted after retries, or the API key or
    model is rejected. Every further request would fail the same way."""


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    media_type: str = "image/jpeg"


@dataclass(frozen=True)
class ModelRequest:
    schema: dict                         # Gemini schema dialect (see module doc)
    system_instruction: Optional[str] = None
    media_resolution: str = "high"       # Gemini only: "low" | "medium" | "high"
    thinking_level: Optional[str] = None  # one of THINKING_LEVELS; None = provider default
    purpose: str = "other"


@dataclass
class ModelReply:
    text: Optional[str]
    input_tokens: int = 0           # uncached input
    output_tokens: int = 0          # incl. thinking
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    thinking_tokens: int = 0        # part of output_tokens spent on thinking (Claude, when reported)
    model: str = ""                 # the model that actually answered
    seconds: float = 0.0


THINKING_LEVELS = ("minimal", "low", "medium", "high")

# USD per 1M tokens: input, output, cache read, cache write (5-minute TTL = 1.25 x input).
# Claude: platform.claude.com pricing (Oct 2026). Gemini: ai.google.dev/gemini-api/docs/pricing.
PRICES_PER_MTOK = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "gemini-3.6-flash": (0.75, 3.75, 0.0, 0.0),
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


def get_provider():
    """The configured provider (read on every call, so tests and scripts can switch it)."""
    return get_provider_named(config.LLM_PROVIDER)


def get_provider_named(name: str):
    if name not in _PROVIDERS:
        raise RuntimeError(f"LLM_PROVIDER must be one of {sorted(_PROVIDERS)}, got {name!r}")
    return _PROVIDERS[name]


def check_configured() -> None:
    """Raise RuntimeError with a clear message when the selected provider's key or model is missing."""
    get_provider().check_configured()


# ------------------------------------------------------------------ Claude

# Image limits: platform.claude.com/docs/en/build-with-claude/vision (high-resolution tier,
# Claude 4.7 and later). Larger images would be downscaled by the API anyway; doing it here keeps
# the payload small and the result deterministic.
CLAUDE_MAX_LONG_EDGE = 2576
CLAUDE_MAX_VISUAL_TOKENS = 4784          # ceil(w/28) * ceil(h/28)
CLAUDE_MANY_IMAGES = 20                  # above this many images per request ...
CLAUDE_MANY_IMAGES_MAX_SIDE = 2000       # ... every image must be at most this on both sides
CLAUDE_MAX_IMAGE_B64_BYTES = 10 * 1024 * 1024
CLAUDE_MEDIA_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
# Server-side refusal fallback ("default" routes by refusal category). Data entry from forms should
# never be declined; if it is, the request is re-run on another Claude model instead of failing.
CLAUDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Cheap layout questions: thinking adds cost there, not accuracy.
_LOW_EFFORT_PURPOSES = ("orientation", "split")
# Reading fields: thinking is requested explicitly (adaptive - the documented mode on current models;
# budget_tokens and temperature are rejected). Effort stays CLAUDE_EFFORT (high by default).
THINKING_PURPOSES = ("extract", "reread_fields")
THINKING_CONFIG = {"type": "adaptive"}
_THINKING_TO_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}


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


_GEMINI_TYPES = {"OBJECT": "object", "ARRAY": "array", "STRING": "string", "NUMBER": "number",
                 "INTEGER": "integer", "BOOLEAN": "boolean"}


def to_json_schema(schema: dict) -> dict:
    """Convert a Gemini-dialect response schema to JSON Schema for Claude structured outputs.

    Every object gets additionalProperties: false and lists ALL its properties as required
    (optional properties count toward a small per-request limit and would reorder the output).
    "nullable" becomes a ["type", "null"] union; a nullable enum becomes anyOf [enum, null] (the API
    rejects an enum next to a type array).
    """
    out: dict = {}
    t = schema.get("type")
    json_type = _GEMINI_TYPES.get(str(t).upper(), str(t).lower()) if t else None
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


class ClaudeProvider:
    name = "claude"
    # The extraction schema asks Claude for high / medium / low confidence plus a source note.
    labelled_confidence = True

    def __init__(self):
        self._client = None
        self._lock = threading.Lock()

    def check_configured(self) -> None:
        config.CLAUDE_API_KEY  # noqa: B018 - raises when missing
        if config.CLAUDE_EFFORT not in CLAUDE_EFFORTS:
            raise RuntimeError(f"CLAUDE_EFFORT must be one of {CLAUDE_EFFORTS}, got {config.CLAUDE_EFFORT!r}")

    def model_id(self) -> str:
        return config.CLAUDE_MODEL

    def effort(self, request: ModelRequest) -> str:
        if request.thinking_level:
            return _THINKING_TO_EFFORT[request.thinking_level]
        if request.purpose in _LOW_EFFORT_PURPOSES:
            return "low"
        return config.CLAUDE_EFFORT

    def cache_id(self, request: ModelRequest) -> str:
        """Identifies everything provider-specific that changes the answer (dev cache key)."""
        thinking = "|thinking=adaptive" if request.purpose in THINKING_PURPOSES else ""
        return f"claude|{config.CLAUDE_MODEL}|effort={self.effort(request)}{thinking}"

    def client(self):
        # One shared client, created under a lock (concurrent first calls must not create several).
        import anthropic

        with self._lock:
            if self._client is None:
                self._client = anthropic.Anthropic(
                    api_key=config.CLAUDE_API_KEY,
                    timeout=config.CLAUDE_TIMEOUT_S,
                    # The SDK retries 408/409/429/5xx (incl. 529 overloaded) and connection errors with
                    # exponential backoff and honours retry-after.
                    max_retries=config.CLAUDE_MAX_RETRIES,
                )
            return self._client

    def content_blocks(self, contents: list) -> list:
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

    def call(self, contents: list, request: ModelRequest) -> ModelReply:
        import anthropic

        schema, aliases = alias_quoted_enums(to_json_schema(request.schema))
        kwargs = {
            "model": config.CLAUDE_MODEL,
            "max_tokens": config.CLAUDE_MAX_TOKENS,
            "messages": [{"role": "user", "content": self.content_blocks(contents)}],
            # Sampling parameters (temperature) are rejected by current models; the JSON schema and
            # the effort level are what keep answers deterministic in shape.
            "output_config": {"effort": self.effort(request),
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
            response = self.client().beta.messages.create(**kwargs)
        except anthropic.RateLimitError:
            raise LLMQuotaError("Claude API rate limit (429) still exceeded after retries") from None
        except anthropic.AuthenticationError:
            raise LLMQuotaError("Claude API key rejected (401): check CLAUDE_API_KEY in .env") from None
        except anthropic.PermissionDeniedError:
            raise LLMQuotaError("Claude API key has no access (403)") from None
        except anthropic.NotFoundError:
            raise LLMQuotaError(f"Claude model not found (404): check CLAUDE_MODEL={config.CLAUDE_MODEL}") from None
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
        seconds = time.monotonic() - start
        usage = response.usage
        reply = ModelReply(
            text=restore_quoted_enums("".join(b.text for b in response.content if b.type == "text") or None, aliases),
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            thinking_tokens=getattr(getattr(usage, "output_tokens_details", None), "thinking_tokens", 0) or 0,
            model=response.model,
            seconds=seconds,
        )
        _log_call(request.purpose, reply)
        if response.stop_reason == "refusal":
            raise LLMError("Claude declined the request (refusal)")
        if response.stop_reason == "max_tokens":
            raise LLMError("Claude output was cut off (max_tokens)")
        return reply


# ------------------------------------------------------------------ Gemini

_MEDIA_RESOLUTIONS = ("low", "medium", "high")
TRANSPORT_RETRY_DELAY_S = 2.0


class GeminiProvider:
    name = "gemini"
    labelled_confidence = False

    def __init__(self):
        self._client = None
        self._lock = threading.Lock()

    def check_configured(self) -> None:
        config.GEMINI_API_KEY, config.GEMINI_MODEL  # noqa: B018 - raises when missing

    def model_id(self) -> str:
        return config.GEMINI_MODEL

    def cache_id(self, request: ModelRequest) -> str:
        return config.GEMINI_MODEL  # unchanged, so the existing dev cache still answers

    def client(self):
        # One shared client, created under a lock: concurrent first calls must not create
        # several clients (a garbage-collected client closes its HTTP pool mid-request).
        from google import genai
        from google.genai import types

        with self._lock:
            if self._client is None:
                self._client = genai.Client(
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
            return self._client

    def config_for(self, request: ModelRequest):
        from google.genai import types

        resolutions = {
            "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
            "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
            "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        }
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=request.schema,
            system_instruction=request.system_instruction,
            media_resolution=resolutions[request.media_resolution],
            thinking_config=(types.ThinkingConfig(thinking_level=types.ThinkingLevel(request.thinking_level.upper()))
                             if request.thinking_level else None),
        )

    def call(self, contents: list, request: ModelRequest) -> ModelReply:
        from google.genai import types

        parts = [types.Part.from_bytes(data=c.data, mime_type=c.media_type) if isinstance(c, ImagePart) else c
                 for c in contents]
        start = time.monotonic()
        response = self._generate_with_transport_retry(parts, self.config_for(request))
        meta = getattr(response, "usage_metadata", None)
        reply = ModelReply(
            text=response.text,
            input_tokens=(meta.prompt_token_count or 0) if meta is not None else 0,
            output_tokens=((meta.candidates_token_count or 0) + (getattr(meta, "thoughts_token_count", 0) or 0)
                           if meta is not None else 0),
            model=config.GEMINI_MODEL,
            seconds=time.monotonic() - start,
        )
        _log_call(request.purpose, reply)
        return reply

    def _generate_with_transport_retry(self, contents: list, cfg):
        """One generate_content call. HttpRetryOptions only retries HTTP status codes, so
        dropped connections / transport timeouts (e.g. RemoteProtocolError) are retried here."""
        import httpx
        from google.genai import errors as genai_errors

        for attempt in range(1, config.MAX_ATTEMPTS + 1):
            try:
                return self.client().models.generate_content(
                    model=config.GEMINI_MODEL, contents=contents, config=cfg
                )
            except genai_errors.APIError as exc:
                if exc.code == 429:
                    raise LLMQuotaError(f"Gemini API quota exhausted (429 {exc.status})") from None
                raise LLMError(f"Gemini API error {exc.code}: {exc.status}") from None
            except httpx.TransportError as exc:
                if attempt == config.MAX_ATTEMPTS:
                    raise LLMError(f"Gemini request failed: {type(exc).__name__}") from None
                logger.warning("Transport error %s (attempt %d), retrying", type(exc).__name__, attempt)
                time.sleep(TRANSPORT_RETRY_DELAY_S * attempt)
            except Exception as exc:  # anything else: do not retry blindly
                raise LLMError(f"Gemini request failed: {type(exc).__name__}") from None


def _log_call(purpose: str, reply: ModelReply) -> None:
    """Token usage and timing only - nothing from the request or the answer."""
    logger.info("model call purpose=%s model=%s input=%d cache_read=%d cache_write=%d output=%d (thinking=%d) "
                "time=%.1fs", purpose, reply.model, reply.input_tokens, reply.cache_read_tokens,
                reply.cache_write_tokens, reply.output_tokens, reply.thinking_tokens, reply.seconds)


_PROVIDERS = {"claude": ClaudeProvider(), "gemini": GeminiProvider()}
