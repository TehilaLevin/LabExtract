"""Handwriting experiment helpers - no API calls."""
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

_spec = importlib.util.spec_from_file_location(
    "handwriting_experiment", Path(__file__).resolve().parent.parent / "evaluation" / "handwriting_experiment.py")
hx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hx)

TRUTH = ("אבג", "דהו")


def _reading(name, legible=True, conf=0.9):
    return {"legible": legible, "name": name, "confidence": conf}


# ---- prompts never carry the answer -------------------------------------------------

def test_prompts_contain_no_ground_truth_and_no_hebrew():
    for prompt in (hx.READ_PROMPT, hx.LOCATE_PROMPT):
        for word in hx.SAMPLE_TRUTH:
            assert word not in prompt
        assert not re.search(r"[֐-׿]", prompt)  # no Hebrew at all: nothing to copy from


# ---- scoring ----------------------------------------------------------------------------

@pytest.mark.parametrize("reading, expected", [
    (_reading("אבג דהו"), "correct"),
    (_reading("דהו, אבג"), "correct"),               # order and punctuation do not matter
    (_reading("אבג"), "partial"),
    (_reading("אבג דהז"), "invented"),
    (_reading("אבג", legible=False), "empty"),
    (_reading(None), "empty"),
    ({"error": "GeminiError"}, "error"),
    (None, "error"),
])
def test_categorize(reading, expected):
    assert hx.categorize(reading, TRUTH) == expected


def test_accepted_needs_legible_name_and_confidence():
    assert hx.accepted(_reading("אבג", conf=0.8))
    assert not hx.accepted(_reading("אבג", conf=0.79))
    assert not hx.accepted(_reading("אבג", legible=False))
    assert not hx.accepted(_reading(None))
    assert not hx.accepted({"error": "x"})


def test_consensus_accepts_only_agreement():
    agreed = hx.consensus(_reading("אבג דהו"), _reading("דהו אבג", conf=0.85))
    assert hx.accepted(agreed) and agreed["confidence"] == 0.85
    assert not hx.accepted(hx.consensus(_reading("אבג דהו"), _reading("אבג דהז")))
    assert not hx.accepted(hx.consensus(_reading("אבג דהו"), _reading("אבג דהו", conf=0.5)))


def test_summary_counts_silent_errors_separately():
    rows = [{"arm": "a", "set": "sample", "category": "invented", "accepted": True},
            {"arm": "a", "set": "sample", "category": "invented", "accepted": False},
            {"arm": "a", "set": "sample", "category": "correct", "accepted": True},
            {"arm": "a", "set": "sample", "category": "correct", "accepted": False},
            {"arm": "a", "set": "sample", "category": "empty", "accepted": False}]
    c = hx.summarize_round(rows)[("a", "sample")]
    assert (c["n"], c["silent_errors"], c["wrong_flagged"], c["correct_accepted"], c["empty"]) == (5, 1, 1, 1, 1)


# ---- preprocessing variants ------------------------------------------------------------------

def _crop_with_table_line():
    img = Image.new("RGB", (600, 120), "white")
    d = ImageDraw.Draw(img)
    d.line([(0, 90), (599, 90)], fill="black", width=2)          # table line across the whole crop
    d.line([(300, 30), (300, 70)], fill=(20, 40, 160), width=3)  # a short handwritten stroke
    return img


@pytest.mark.parametrize("variant", list(hx.VARIANTS))
def test_variants_never_modify_the_input(variant):
    img = _crop_with_table_line()
    before = img.tobytes()
    out = hx.apply_variant(img, variant)
    assert img.tobytes() == before and out is not img
    scale = 2 if variant in ("upscale", "combo") else 1
    assert out.size == (img.width * scale, img.height * scale)


def test_line_removal_keeps_the_stroke_and_removes_the_line():
    out = np.asarray(hx.apply_variant(_crop_with_table_line(), "lines").convert("L"))
    assert out[90, 100] > 200                                    # the long line is gone
    assert out[50, 300] < 120                                    # the short stroke is kept


# ---- cost ----------------------------------------------------------------------------------

def test_cost_and_estimate():
    assert hx.cost_usd(1_000_000, 1_000_000, (0.75, 3.75)) == pytest.approx(4.5)
    rows = hx.estimate(10, ["flash:none"], {"flash": "gemini-3.6-flash"}, {})
    assert rows[0][1] == 10 and rows[0][2] == pytest.approx(10 * hx.cost_usd(1500, 2000, (0.75, 3.75)))
    with pytest.raises(SystemExit):
        hx.price_for("unknown-model", {})
    assert hx.price_for("unknown-model", {"unknown-model": (1.0, 2.0)}) == (1.0, 2.0)


# ---- test sets -------------------------------------------------------------------------------

def test_empty_template_has_no_rows(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("file,first_name,last_name,notes\n", encoding="utf-8")
    assert hx.load_labels(path) == [] and hx.load_labels(tmp_path / "missing.csv") == []


def test_photo_crops_read_labels_and_skip_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(hx, "HW_DIR", tmp_path)
    Image.new("RGB", (4000, 1000), "white").save(tmp_path / "a.jpg")
    crops = hx.photo_crops([{"file": "a.jpg", "first_name": "אבג", "last_name": "דהו"},
                            {"file": "missing.jpg", "first_name": "x", "last_name": "y"}])
    assert len(crops) == 1
    crop_id, set_name, truth, img = crops[0]
    assert set_name == "photo" and truth == TRUTH and max(img.size) == hx.PHOTO_MAX_SIDE


# ---- request guard ------------------------------------------------------------------------------

@pytest.fixture
def restore_engine(monkeypatch):
    from src import config
    from src import extraction_engine as ee

    for name in ("generate_json", "_generate_with_transport_retry", "_cache_get", "_cache_put"):
        monkeypatch.setattr(ee, name, getattr(ee, name))
    monkeypatch.setattr(config, "DEV_CACHE", False)
    monkeypatch.setattr(config, "DEV_CACHE_DIR", Path(__file__).parent / "_no_such_cache_dir")
    monkeypatch.setattr(config, "GEMINI_MODEL", "test-model", raising=False)
    return ee


def test_guard_blocks_other_stages_and_caps_requests(restore_engine, monkeypatch):
    ee = restore_engine

    class _Response:
        text = '{"legible": false, "name": null, "confidence": 0}'

    calls = []
    monkeypatch.setattr(ee, "_generate_with_transport_retry", lambda c, cfg: calls.append(1) or _Response())
    stats = hx.install_guard(max_requests=1, allowed_stages=("hw_read",), fresh=True)
    with pytest.raises(ee.GeminiError, match="blocked"):
        ee.generate_json(["x"], {"type": "OBJECT"}, purpose="extract")
    ee.generate_json(["a"], {"type": "OBJECT"}, purpose="hw_read:flash:none")
    with pytest.raises(ee.GeminiQuotaError):
        ee.generate_json(["b"], {"type": "OBJECT"}, purpose="hw_read:flash:none")
    assert len(calls) == 1 and stats == {"live": 1, "blocked": 1, "cap_reached": True}


def test_use_model_restores_the_configured_model():
    from src import config

    had = "GEMINI_MODEL" in vars(config)
    with hx.use_model("other-model"):
        assert config.GEMINI_MODEL == "other-model"
    assert ("GEMINI_MODEL" in vars(config)) == had
