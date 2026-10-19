from pathlib import Path

import pytest

from src import form_splitter as fs
from src.file_converter import Page


def _pages(*spec):
    """spec: (source_file, n_pages) tuples -> flat list of Page objects."""
    out = []
    for name, n in spec:
        out += [Page(name, i, Path(f"{name}_{i}.jpg")) for i in range(1, n + 1)]
    return out


def D(new=True, conf=0.95, rot=0, rot_conf=0.95, blank=False, error=None):
    return fs.PageDecision(new, conf, rot, rot_conf, blank, error)


def test_one_page_per_form():
    forms = fs.group_pages(_pages(("a.pdf", 3)), [D(), D(), D()])
    assert [f.page_numbers for f in forms] == [[1], [2], [3]]
    assert [f.index for f in forms] == [1, 2, 3]
    assert all(not f.split_flags for f in forms)


def test_confident_continuation_merges_pages():
    forms = fs.group_pages(_pages(("a.pdf", 4)), [D(), D(new=False), D(), D(new=False)])
    assert [f.page_numbers for f in forms] == [[1, 2], [3, 4]]


def test_uncertain_continuation_splits_and_flags():
    # Merging two patients is worse than splitting one form: split when unsure.
    forms = fs.group_pages(_pages(("a.pdf", 2)), [D(), D(new=False, conf=0.5)])
    assert [f.page_numbers for f in forms] == [[1], [2]]
    assert forms[1].split_flags == [fs.SPLIT_FLAG_UNCERTAIN]


def test_new_file_always_starts_new_form():
    forms = fs.group_pages(_pages(("a.pdf", 1), ("b.jpg", 1)), [D(), D(new=False, conf=1.0)])
    assert [(f.source_file, f.page_numbers) for f in forms] == [("a.pdf", [1]), ("b.jpg", [1])]


def test_api_error_starts_new_flagged_form_and_keeps_page():
    forms = fs.group_pages(_pages(("a.pdf", 3)), [D(), D(error="Claude API error 500"), D(new=False)])
    assert [f.page_numbers for f in forms] == [[1], [2, 3]]
    assert forms[1].split_flags == [fs.SPLIT_FLAG_FAILED]


def test_error_on_first_page_is_flagged():
    forms = fs.group_pages(_pages(("a.pdf", 1)), [D(error="quota")])
    assert forms[0].split_flags == [fs.SPLIT_FLAG_FAILED]


def test_blank_page_stays_with_previous_form():
    forms = fs.group_pages(_pages(("a.pdf", 3)), [D(), D(blank=True, new=True), D()])
    assert [f.page_numbers for f in forms] == [[1, 2], [3]]


def test_no_page_is_ever_dropped():
    pages = _pages(("a.pdf", 5), ("b.pdf", 2))
    decisions = [D(), D(blank=True), D(new=False, conf=0.1), D(error="x"), D(new=False), D(), D(new=False)]
    forms = fs.group_pages(pages, decisions)
    assert sum(len(f.pages) for f in forms) == len(pages)


def test_rotation_applied_only_when_confident():
    forms = fs.group_pages(_pages(("a.pdf", 2)), [D(rot=90, rot_conf=0.9), D(rot=180, rot_conf=0.4)])
    assert [f.rotations for f in forms] == [[90], [0]]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        fs.group_pages(_pages(("a.pdf", 2)), [D()])


def test_batches_respect_files_and_size(monkeypatch):
    monkeypatch.setattr(fs, "SPLIT_BATCH_SIZE", 3)
    pages = _pages(("a.pdf", 7), ("b.pdf", 2))
    assert fs._batches(pages) == [(0, 3), (3, 6), (6, 7), (7, 9)]


def test_parse_decision_clamps_and_defaults():
    d = fs._parse_decision({"confidence": 7, "is_blank": None})
    assert (d.rotate_clockwise, d.confidence, d.rotation_confidence, d.starts_new_form, d.is_blank) == (
        0, 1.0, 0.0, True, False)


# ---- classify_batch / split_forms with a fake model (no API calls) -----------

def _fake_answer(n, **overrides):
    item = {"is_blank": False, "starts_new_form": True, "confidence": 0.9}
    item.update(overrides)
    return {"pages": [dict(item, page_index=i) for i in range(1, n + 1)]}


@pytest.fixture
def fake_pages(tmp_path):
    from PIL import Image

    pages = []
    for name, n in (("a.pdf", 3), ("b.png", 1)):
        for i in range(1, n + 1):
            path = tmp_path / f"{name}_{i}.jpg"
            Image.new("RGB", (60, 80), "white").save(path)
            pages.append(Page(name, i, path))
    return pages


def test_split_forms_with_fake_model(monkeypatch, fake_pages):
    from src import extraction_engine

    calls = []

    def fake_generate(contents, schema, **kwargs):
        if "upright" in schema["properties"]:  # orientation request: one per page
            calls.append("orient")
            return {"upright": "D", "confidence": 0.99}
        n = sum(1 for c in contents if isinstance(c, str) and c.startswith("PAGE "))
        calls.append(n)
        return _fake_answer(n)

    monkeypatch.setattr(extraction_engine, "generate_json", fake_generate)
    forms = fs.split_forms(fake_pages)
    assert calls.count("orient") == 4
    assert sorted(c for c in calls if c != "orient") == [1, 3]  # one split request per file batch
    assert [(f.source_file, f.page_numbers) for f in forms] == [
        ("a.pdf", [1]), ("a.pdf", [2]), ("a.pdf", [3]), ("b.png", [1])]
    assert all(f.rotations == [270] for f in forms)  # "D" = 270 deg clockwise


def test_split_receives_upright_pages(monkeypatch, tmp_path):
    from PIL import Image

    from src import extraction_engine

    path = tmp_path / "p.jpg"
    Image.new("RGB", (60, 80), "white").save(path)  # portrait as scanned
    sizes = []

    def fake_generate(contents, schema, **kwargs):
        if "upright" in schema["properties"]:
            return {"upright": "B", "confidence": 0.99}
        sizes.extend(c for c in contents if not isinstance(c, str))
        return _fake_answer(1)

    real_part = extraction_engine.image_part
    monkeypatch.setattr(extraction_engine, "image_part",
                        lambda img, side: (sizes.append(img.size), real_part(img, side))[1])
    monkeypatch.setattr(extraction_engine, "generate_json", fake_generate)
    fs.split_forms([Page("p.pdf", 1, path)])
    assert (80, 60) in sizes  # the split step saw the page after a 90-degree turn


@pytest.mark.parametrize("answer, expected", [
    ({"upright": "A", "confidence": 0.9}, (0, 0.9)),
    ({"upright": "C", "confidence": 0.95}, (180, 0.95)),
    ({"upright": "Z", "confidence": 0.95}, (0, 0.0)),
    ({}, (0, 0.0)),
])
def test_detect_rotation_maps_candidates(monkeypatch, answer, expected):
    from PIL import Image

    from src import extraction_engine

    monkeypatch.setattr(extraction_engine, "generate_json", lambda c, s, **k: answer)
    assert fs.detect_rotation(Image.new("RGB", (40, 60))) == expected


def test_low_confidence_orientation_is_not_applied(monkeypatch, fake_pages):
    from src import extraction_engine

    def fake_generate(contents, schema, **kwargs):
        if "upright" in schema["properties"]:
            return {"upright": "C", "confidence": 0.4}
        return _fake_answer(sum(1 for c in contents if isinstance(c, str) and c.startswith("PAGE ")))

    monkeypatch.setattr(extraction_engine, "generate_json", fake_generate)
    assert all(f.rotations == [0] for f in fs.split_forms(fake_pages))


def test_incomplete_answer_flags_batch(monkeypatch, fake_pages):
    from src import extraction_engine

    monkeypatch.setattr(extraction_engine, "generate_json",
                        lambda c, s, **k: {"upright": "A", "confidence": 1} if "upright" in s["properties"]
                        else {"pages": []})
    forms = fs.split_forms(fake_pages)
    assert len(forms) == 4
    assert all(f.split_flags == [fs.SPLIT_FLAG_FAILED] for f in forms)


def test_quota_error_flags_forms_without_crashing(monkeypatch, fake_pages):
    from src import extraction_engine

    def boom(*a, **k):
        raise extraction_engine.LLMQuotaError("API quota exhausted")

    monkeypatch.setattr(extraction_engine, "generate_json", boom)
    forms = fs.split_forms(fake_pages)
    assert sum(len(f.pages) for f in forms) == 4
    assert all(fs.SPLIT_FLAG_FAILED in f.split_flags for f in forms)


def test_orientation_uses_small_low_resolution_thumbnails(monkeypatch):
    from PIL import Image

    from src import extraction_engine

    seen = {}

    def fake_generate(contents, schema, **kwargs):
        seen["purpose"] = kwargs.get("purpose")
        seen["sizes"] = [c.data for c in contents if not isinstance(c, str)]
        return {"upright": "A", "confidence": 0.99}

    monkeypatch.setattr(extraction_engine, "generate_json", fake_generate)
    fs.detect_rotation(Image.new("RGB", (2480, 3508), "white"))  # full 300-dpi A4 page
    assert seen["purpose"] == "orientation"                       # low effort, see claude_client
    assert len(seen["sizes"]) == 4
    from io import BytesIO
    for data in seen["sizes"]:
        assert max(Image.open(BytesIO(data)).size) <= fs.ORIENTATION_IMAGE_MAX_SIDE == 512
