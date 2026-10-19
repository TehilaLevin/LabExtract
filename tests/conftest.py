"""Shared test setup."""
import pytest


@pytest.fixture(autouse=True)
def _wide_received_date_window(monkeypatch):
    """Fixtures use June 2026 sample dates with "today" in October; the 90-day production window would flag
    them all. Tests of the window itself (test_strict_validation.py) set it explicitly."""
    from src import config

    monkeypatch.setattr(config, "RECEIVED_DATE_MAX_AGE_DAYS", 365)
