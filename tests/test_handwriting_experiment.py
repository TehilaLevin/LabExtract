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

    for name in ("generate_json", "_call_model", "_cache_get", "_cache_put"):
        monkeypatch.setattr(ee, name, getattr(ee, name))
    monkeypatch.setattr(config, "DEV_CACHE", False)
    monkeypatch.setattr(config, "DEV_CACHE_DIR", Path(__file__).parent / "_no_such_cache_dir")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")  # setattr would read the missing value first
    return ee


def test_guard_blocks_other_stages_and_caps_requests(restore_engine, monkeypatch):
    ee = restore_engine

    class _Response:
        text = '{"legible": false, "name": null, "confidence": 0}'

    calls = []
    monkeypatch.setattr(ee, "_call_model", lambda c, cfg: calls.append(1) or _Response())
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


# ---- labelled forms: DPI x preprocessing x thinking ------------------------------------

def test_control_prompt_contains_no_ground_truth_and_no_hebrew():
    for word in hx.SAMPLE_TRUTH:
        assert word not in hx.CONTROL_PROMPT
    assert not re.search(r"[֐-׿]", hx.CONTROL_PROMPT)


def test_parse_arm():
    assert hx.parse_arm("pro:600:clahe:high") == ("pro", 600, "clahe", "high")
    for bad in ("flash:450:none:default", "flash:300:sharpen:default", "flash:300:none:max",
                "ultra:300:none:high", "300:none:high", "flash:x:none:high"):
        with pytest.raises(SystemExit):
            hx.parse_arm(bad)


def test_deskew_angle():
    assert hx.deskew_angle(["autocontrast", "deskew -1.25"]) == -1.25
    assert hx.deskew_angle(["autocontrast"]) == 0.0


def _page_with_ink(boxes):
    img = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(img)
    for y0, x0, y1, x1 in boxes:
        draw.rectangle([x0, y0 * 1.4, x1, y1 * 1.4], fill="black")
    return img


def test_empty_box_crop_area_is_blank_and_avoids_names():
    ink = [(150, 150, 850, 480)]                       # everything left of x=480 is ink
    img = _page_with_ink(ink)
    name = (600, 600, 640, 800)
    box = hx.find_empty_box(img, (40, 150), [hx._padded(name, img.size)])
    assert box is not None
    shown = hx._padded(box, img.size)
    assert not hx._overlaps(shown, ink[0]) and not hx._overlaps(shown, hx._padded(name, img.size))
    assert min(shown[:2]) >= hx.EMPTY_MARGIN and max(shown[2:]) <= 1000 - hx.EMPTY_MARGIN


def test_empty_box_none_when_nothing_is_blank():
    img = _page_with_ink([(0, 0, 1000, 1000)])
    assert hx.find_empty_box(img, (40, 150), []) is None


def test_crop_target_covers_the_same_area_at_both_dpis():
    box = (200, 300, 260, 700)
    small, big = Image.new("RGB", (1000, 1400)), Image.new("RGB", (2000, 2800))
    a, b = hx.crop_target(small, box, 300), hx.crop_target(big, box, 600)
    assert abs(b.size[0] - 2 * a.size[0]) <= 1 and abs(b.size[1] - 2 * a.size[1]) <= 1  # int rounding


@pytest.mark.parametrize("kind, reading, expected", [
    ("name", _reading("ישראל ישראלי"), "correct"),
    ("name", _reading("ישראל"), "partial"),
    ("name", _reading("יסמין יוסף"), "invented"),
    ("control_empty", _reading(None, legible=False), "abstain"),
    ("control_empty", _reading("יוסי"), "guess"),
    ("control_text", _reading("939273"), "guess"),       # any text returned as a name is a guess
    ("control_text", _reading("", legible=True), "abstain"),
    ("control_text", {"error": "GeminiError"}, "error"),
])
def test_score_reading(kind, reading, expected):
    assert hx.score_reading(kind, reading) == expected


def _row(arm, kind, name, legible=True, conf=0.9):
    reading = _reading(name, legible, conf)
    return {"arm": arm, "kind": kind, "category": hx.score_reading(kind, reading),
            "accepted": hx.accepted(reading)}


def test_summarize_arms_counts_silent_errors_and_guesses():
    rows = [_row("a", "name", "ישראל ישראלי"), _row("a", "name", "יסמין יוסף"),
            _row("a", "name", "יסמין יוסף", conf=0.5), _row("a", "control_empty", "יוסי"),
            _row("a", "control_text", None, legible=False)]
    c = hx.summarize_arms(rows)["a"]
    assert (c["name_n"], c["name_correct"], c["name_invented"], c["name_silent"]) == (3, 1, 2, 1)
    assert (c["control_n"], c["control_guess"], c["control_silent"], c["control_abstain"]) == (2, 1, 1, 1)


def test_pick_best_variant_rule():
    rows = [_row(hx.PILOT_ARM.format("clahe"), "name", "ישראל ישראלי"),
            _row(hx.PILOT_ARM.format("lines"), "name", "ישראל ישראלי"),
            _row(hx.PILOT_ARM.format("lines"), "name", "יסמין יוסף")]
    assert hx.pick_best_variant(rows) == "clahe"       # same correct, fewer invented
    assert hx.pick_best_variant([]) == hx.PILOT_VARIANTS[0]   # no signal: the fixed order decides


def test_estimate_labelled_counts_requests():
    prices = {"flash": (1.0, 1.0), "pro": (2.0, 10.0)}
    steps = hx.estimate_labelled(5, 10, ["flash:300:none:default", "flash:300:combo:high", "pro:300:none:high"],
                                 draws=2, pilot=True, prices=prices)
    assert [s[1] for s in steps] == [5, 5 * len(hx.PILOT_VARIANTS), 30, 30, 30]
    assert steps[3][2] > steps[2][2]                   # high thinking costs more
    assert steps[4][2] > steps[3][2]                   # pro costs more
    assert all(s[3] >= s[2] for s in steps)            # the bound is never below the estimate
