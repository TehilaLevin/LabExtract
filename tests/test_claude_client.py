"""Claude client: request shape, schema conversion, image limits, error mapping, cache keys.

The Anthropic client is faked - these tests never touch the API.
"""
import base64
import json
from io import BytesIO
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from PIL import Image

from src import config
from src import extraction_engine as ee
from src import claude_client as lp
from src.constants import CONFIDENCE_LABELS, FIELD_KEYS
from tests.conftest import fake_reply as _reply


def _jpeg(w: int, h: int) -> lp.ImagePart:
    buf = BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "JPEG")
    return lp.ImagePart(buf.getvalue())


# ---- configuration -------------------------------------------------------------------------

def test_claude_key_falls_back_to_anthropic_api_key(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert config.CLAUDE_API_KEY == "sk-ant-test"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(RuntimeError, match="CLAUDE_API_KEY"):
        config.CLAUDE_API_KEY  # noqa: B018


# ---- schema conversion ---------------------------------------------------------------------

def test_compact_schema_converts_to_strict_json_schema():
    schema = ee.build_schema(["כללית", "מכבי"])
    out = lp.to_json_schema(schema)
    assert out["type"] == "object" and out["additionalProperties"] is False
    assert out["required"] == list(out["properties"])          # every property required
    sender = out["properties"]["sending_entity"]["properties"]["value"]
    assert sender == {"anyOf": [{"type": "string", "enum": ["כללית", "מכבי"]}, {"type": "null"}]}
    assert out["properties"]["id_number"]["properties"]["value"]["type"] == ["string", "null"]
    central = out["properties"]["central_numbers"]["properties"]["value"]
    assert central == {"type": "array", "items": {"type": "string"}}
    conf = out["properties"]["id_number"]["properties"]["confidence"]
    assert conf == {"type": "string", "enum": list(CONFIDENCE_LABELS)}
    assert "source_note" in out["properties"]["id_number"]["required"]
    assert "propertyOrdering" not in json.dumps(out) and "nullable" not in json.dumps(out)
    # Structured outputs allow at most 16 union-typed parameters per request.
    assert json.dumps(out).count('"null"') <= 16


def test_extraction_prompt_has_the_confidence_labels_and_the_decoy_warning():
    prompt = ee.load_prompt(["כללית"])
    assert "CONFIDENCE LABELS" in prompt
    assert ("Highlighting does NOT indicate which field a value belongs to. Some highlighted values are decoys "
            "(visit date, intake date, form date).") in prompt


def test_confidence_labels_map_onto_the_threshold():
    data = {k: {"value": None, "confidence": "low", "source_note": ""} for k in FIELD_KEYS}
    data["id_number"] = {"value": "123", "confidence": "high", "source_note": "label ת.ז., page 1"}
    data["last_name"] = {"value": "x", "confidence": "Medium"}
    res = ee.parse_extraction(data)
    assert res.fields["id_number"].confidence >= config.CONFIDENCE_THRESHOLD
    assert res.fields["last_name"].confidence < config.CONFIDENCE_THRESHOLD
    assert res.fields["first_name"].confidence < config.CONFIDENCE_THRESHOLD
    assert res.fields["id_number"].note == "label ת.ז., page 1"
    assert res.fields["first_name"].note is None


# ---- image limits ----------------------------------------------------------------------------

def test_small_images_are_sent_unchanged():
    part = _jpeg(1000, 800)
    assert lp.fit_claude_image(part) == part


def test_large_images_are_downscaled_within_limits():
    out = lp.fit_claude_image(_jpeg(3500, 2480))
    w, h = Image.open(BytesIO(out.data)).size
    assert max(w, h) <= lp.CLAUDE_MAX_LONG_EDGE
    assert lp.visual_tokens(w, h) <= lp.CLAUDE_MAX_VISUAL_TOKENS
    assert abs(w / h - 3500 / 2480) < 0.01                       # aspect ratio kept
    assert out.media_type == "image/jpeg"


def test_many_images_use_the_stricter_side_limit():
    blocks = lp.content_blocks(["intro", ""] + [_jpeg(2400, 1000)] * 21)
    assert blocks[0] == {"type": "text", "text": "intro"}            # empty text is dropped
    img = Image.open(BytesIO(base64.b64decode(blocks[1]["source"]["data"])))
    assert max(img.size) <= lp.CLAUDE_MANY_IMAGES_MAX_SIDE


# ---- Claude call with a fake client -------------------------------------------------------------

@pytest.fixture
def claude(fake_claude):
    return fake_claude


def test_claude_request_shape(claude):
    ee.reset_usage()
    data = ee.generate_json(["read this", _jpeg(50, 50)], {"type": "OBJECT", "properties": {"ok": {"type": "BOOLEAN"}}},
                            system_instruction="rules", purpose="extract")
    assert data == {"ok": True}
    kw = claude.kwargs
    assert kw["model"] == config.CLAUDE_MODEL and "temperature" not in kw
    assert kw["output_config"]["effort"] == config.CLAUDE_EFFORT
    assert kw["output_config"]["format"]["schema"]["properties"]["ok"] == {"type": "boolean"}
    assert kw["system"][0]["text"] == "rules" and kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    content = kw["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "read this"}
    assert content[1]["type"] == "image" and content[1]["source"]["media_type"] == "image/jpeg"
    assert ee.get_usage() == {"requests": 1, "prompt_tokens": 1000, "output_tokens": 40}
    usage = ee.get_usage_by_purpose()["extract"]
    assert usage["cache_read_tokens"] == 900 and usage["models"] == {"claude-sonnet-5-5": 1}
    assert lp.cost_usd("claude-sonnet-5-5", usage) == pytest.approx((100 * 2 + 40 * 10 + 900 * 0.2) / 1e6)


def test_thinking_is_requested_for_extraction_and_reread_only(claude):
    for purpose in ("extract", "reread_fields"):
        ee.generate_json(["x", purpose], {"type": "OBJECT"}, purpose=purpose)
        assert claude.kwargs["thinking"] == {"type": "adaptive"}
        assert claude.kwargs["output_config"]["effort"] == config.CLAUDE_EFFORT
        assert "temperature" not in claude.kwargs
    for purpose in ("orientation", "split", "verify_names"):
        ee.generate_json(["x", purpose], {"type": "OBJECT"}, purpose=purpose)
        assert "thinking" not in claude.kwargs


def test_default_claude_model_is_opus(monkeypatch):
    import importlib

    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    fresh = importlib.reload(config)
    try:
        assert fresh.CLAUDE_MODEL == "claude-opus-5-5"
    finally:
        importlib.reload(config)


def test_layout_questions_run_at_low_effort(claude):
    ee.generate_json(["x"], {"type": "OBJECT"}, purpose="orientation")
    assert claude.kwargs["output_config"]["effort"] == "low"
    ee.generate_json(["x"], {"type": "OBJECT"}, purpose="extract", thinking_level="medium")
    assert claude.kwargs["output_config"]["effort"] == "medium"


def _status_error(cls, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return cls("error", response=response, body=None)


@pytest.mark.parametrize("error, expected", [
    (lambda: _status_error(anthropic.RateLimitError, 429), lp.LLMQuotaError),
    (lambda: _status_error(anthropic.AuthenticationError, 401), lp.LLMQuotaError),
    (lambda: _status_error(anthropic.NotFoundError, 404), lp.LLMQuotaError),
    (lambda: _status_error(anthropic.InternalServerError, 529), lp.LLMError),
    (lambda: anthropic.APITimeoutError(request=httpx2.Request("POST", "https://api.anthropic.com")), lp.LLMError),
])
def test_api_errors_are_mapped(claude, error, expected):
    claude.error = error()
    with pytest.raises(expected) as info:
        ee.generate_json(["x"], {"type": "OBJECT"})
    assert "Claude" in str(info.value)
    if expected is lp.LLMError:
        assert not isinstance(info.value, lp.LLMQuotaError)


def test_low_credit_stops_the_run(claude):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    claude.error = anthropic.BadRequestError(
        "Your credit balance is too low to access the Anthropic API.",
        response=httpx2.Response(400, request=request), body=None)
    with pytest.raises(lp.LLMQuotaError, match="credit"):
        ee.generate_json(["x"], {"type": "OBJECT"})


def test_refusal_and_truncation_are_errors(claude):
    claude.reply = _reply(stop_reason="refusal")
    with pytest.raises(lp.LLMError, match="declined"):
        ee.generate_json(["x"], {"type": "OBJECT"})
    claude.reply = _reply(text='{"ok": tr', stop_reason="max_tokens")
    with pytest.raises(lp.LLMError, match="max_tokens"):
        ee.generate_json(["x"], {"type": "OBJECT"})


def test_enum_values_with_quotes_are_aliased_and_restored(claude):
    schema = {"type": "OBJECT", "properties": {"s": {"type": "STRING", "enum": ['רמב"ם', "כללית"], "nullable": True}}}
    sent, aliases = lp.alias_quoted_enums(lp.to_json_schema(schema))
    assert '"' not in "".join(sent["properties"]["s"]["anyOf"][0]["enum"])
    alias = next(iter(aliases))
    # The model answers with the alias (plain or \u-escaped); the caller gets the original name back.
    claude.reply = _reply(text='{"s": "' + alias + '"}')
    assert ee.generate_json(["x"], schema) == {"s": 'רמב"ם'}
    claude.reply = _reply(text='{"s": ' + json.dumps(alias) + "}")
    assert ee.generate_json(["y"], schema) == {"s": 'רמב"ם'}
    assert claude.kwargs["output_config"]["format"]["schema"] == sent


def test_cache_key_depends_on_model_effort_and_purpose(monkeypatch):
    request = lp.ModelRequest(schema={}, purpose="extract")
    key = ee._cache_key(["x"], {}, None, "high", None, lp.cache_id(request))
    assert lp.cache_id(request) == f"claude|{config.CLAUDE_MODEL}|effort={config.CLAUDE_EFFORT}|thinking=adaptive"
    assert lp.cache_id(lp.ModelRequest(schema={}, purpose="orientation")).endswith("effort=low")
    monkeypatch.setattr(config, "CLAUDE_MODEL", "claude-sonnet-5-5")
    assert ee._cache_key(["x"], {}, None, "high", None, lp.cache_id(request)) != key
    assert ee._cache_key(["x"], {}, None, "high", "high", lp.cache_id(request)) != \
        ee._cache_key(["x"], {}, None, "high", None, lp.cache_id(request))


def test_cache_detail_tag_keeps_earlier_keys():
    # Orientation / split answers cached by earlier runs were keyed with these tags: they must not change.
    assert ee._CACHE_DETAIL == {"orientation": "low", "split": "medium"}


def test_name_check_can_use_its_own_model_and_effort(claude, monkeypatch):
    monkeypatch.setattr(config, "NAME_CHECK_MODEL", "claude-fable-5-1")
    monkeypatch.setattr(config, "NAME_CHECK_EFFORT", "max")
    ee.generate_json(["x"], {"type": "OBJECT"}, purpose="verify_names")
    assert claude.kwargs["model"] == "claude-fable-5-1" and claude.kwargs["output_config"]["effort"] == "max"
    ee.generate_json(["x"], {"type": "OBJECT"}, purpose="extract")
    assert claude.kwargs["model"] == config.CLAUDE_MODEL
    assert claude.kwargs["output_config"]["effort"] == config.CLAUDE_EFFORT
    assert lp.cache_id(lp.ModelRequest(schema={}, purpose="verify_names")) == "claude|claude-fable-5-1|effort=max"


def test_default_name_check_keeps_earlier_cache_keys():
    # Same model and effort as before the setting existed: dev-cache answers stay valid.
    assert config.NAME_CHECK_MODEL == config.CLAUDE_MODEL and config.NAME_CHECK_EFFORT == config.CLAUDE_EFFORT
    assert lp.cache_id(lp.ModelRequest(schema={}, purpose="verify_names")) == \
        f"claude|{config.CLAUDE_MODEL}|effort={config.CLAUDE_EFFORT}"


def test_fable_price_is_known():
    usage = {"prompt_tokens": 1000, "output_tokens": 100, "cache_read_tokens": 0, "cache_write_tokens": 0}
    assert lp.cost_usd("claude-fable-5-1", usage) == pytest.approx((1000 * 10 + 100 * 50) / 1e6)
