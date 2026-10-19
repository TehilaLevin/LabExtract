"""Shared test setup. No test ever calls the API: model calls go to fake_claude or a faked generate_json."""
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _wide_received_date_window(monkeypatch):
    """Fixtures use June 2026 sample dates with "today" in October; the 90-day production window would flag
    them all. Tests of the window itself (test_strict_validation.py) set it explicitly."""
    from src import config

    monkeypatch.setattr(config, "RECEIVED_DATE_MAX_AGE_DAYS", 365)


class FakeMessages:
    """Stands in for anthropic_client().beta.messages: records the request, returns `reply` or raises `error`."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.kwargs, self.calls = reply, error, None, 0

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.reply


def fake_reply(text='{"ok": true}', stop_reason="end_turn", input_tokens=100, output_tokens=40,
               cache_read=900, cache_write=0, model="claude-sonnet-5-5"):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens, cache_read_input_tokens=cache_read,
                            cache_creation_input_tokens=cache_write)
    return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
                           usage=usage, model=model, stop_reason=stop_reason)


@pytest.fixture
def fake_claude(monkeypatch, tmp_path):
    """Claude with a fake SDK client (no network). DEV_CACHE off, cache folder in tmp_path."""
    from src import claude_client, config

    messages = FakeMessages(fake_reply())
    monkeypatch.setattr(config, "DEV_CACHE", False)
    monkeypatch.setattr(config, "DEV_CACHE_DIR", tmp_path / "dev_cache")
    monkeypatch.setattr(claude_client, "anthropic_client",
                        lambda: SimpleNamespace(beta=SimpleNamespace(messages=messages)))
    return messages
