"""Handwritten-name reading experiment: name CROPS only, never full pages. Separate from the pipeline.

Test sets
- photo:  data/handwriting_test/labels.csv (file, first_name, last_name[, notes]) + the photos it names.
- sample: name boxes on the 48 sample pages (found once with `locate`, saved as coordinates only).
  Their ground truth is the fictitious placeholder name below - it lives ONLY in this scoring code
  and never in a prompt (a test enforces it).

Commands (API ones need --max-requests, a hard cap on live requests):
  models                      list the models this key can use (API, free)
  locate  --max-requests N    one request per sample page: where are the patient names? (boxes only)
  prepare [--previews DIR] [--arms A]   build crops, write preprocessing previews, estimate cost (no API)
  run --round R --arms A --max-requests N [--limit K] [--fresh] [--pro-model NAME]
                              read every crop with every arm, ONE crop per request
  report --round R [--consensus ARM1,ARM2]   summary of a saved round (no API)

Arms: "<model>:<variant>" with model "flash" (GEMINI_MODEL) or "pro" (--pro-model), and variant one of
VARIANTS, e.g. "flash:none,flash:clahe,pro:none". Every handwritten name goes to every arm - no routing
by model confidence (wrong names came back with high confidence).

Scoring per crop: correct (all words right) | partial (only some of the right words) | invented (a word
that is not in the name) | empty (marked illegible / no name) | error. A reading is "accepted" (not
flagged) when legible with confidence >= CONFIDENCE_THRESHOLD; a wrong accepted reading is a SILENT
ERROR and must stay 0. --consensus accepts a name only when two arms read the same words.

Privacy: readings are saved to data/output/handwriting/ (git-ignored, local) for failure analysis;
the console shows counts only. Crops are sent to the API, nothing else.
"""
import argparse
import csv
import json
import re
import sys
import threading
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src import extraction_engine as ee  # noqa: E402

HW_DIR = config.DATA_DIR / "handwriting_test"
LABELS_PATH = HW_DIR / "labels.csv"
BOXES_PATH = HW_DIR / "sample_name_boxes.json"
OUT_DIR = config.DATA_DIR / "output" / "handwriting"
SAMPLES_DIR = config.DATA_DIR / "samples"
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Ground truth of every handwritten name on the sample pages (fictitious placeholder). SCORING ONLY.
SAMPLE_TRUTH = ("ישראל", "ישראלי")

CROP_MAX_SIDE = 2048
PHOTO_MAX_SIDE = 2048
CATEGORIES = ("correct", "partial", "invented", "empty", "error")
WRONG = frozenset({"partial", "invented"})

# USD per 1M tokens (input, output incl. thinking), Standard paid tier, ai.google.dev/gemini-api/docs/pricing,
# read 2026-10-15. Override with --price MODEL=IN/OUT.
PRICES = {
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-2.5-pro": (1.25, 10.00),
}
# Token estimate per one-crop request (calibrated on the verify_names runs: ~1.1k tokens per high-res
# image, ~2k output tokens incl. thinking). Pro models think more. Replaced by measured values later.
EST_INPUT_TOKENS = 1500
EST_OUTPUT_TOKENS = {"flash": 2000, "pro": 3500}

READ_PROMPT = """This image is a crop from a scanned Israeli medical form. It should show a patient's NAME, usually
handwritten in Hebrew (print or cursive), possibly next to printed labels or inside table lines.

Transcribe ONLY the written name (not the printed labels), letter by letter, words in reading order
(right-to-left), separated by single spaces.
- Hebrew cursive letter shapes differ from print. Read each letter from the strokes you see.
- If you would have to guess even ONE letter, set legible = false and name = null. Never output a
  plausible-looking name that you cannot actually see - an unreadable name is checked by a person,
  an invented one is not.
- If the crop shows no name at all, legible = false and name = null.
- confidence: 0-1, how sure you are of every letter."""

READ_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "legible": {"type": "BOOLEAN"},
        "name": {"type": "STRING", "nullable": True},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["legible", "name", "confidence"],
    "propertyOrdering": ["legible", "name", "confidence"],
}

LOCATE_PROMPT = """This is one page of a scanned Israeli medical form. Find every place where the PATIENT's name
is written (handwritten or printed): name fields with their labels, patient stickers, header lines.
Never the doctor, signer or clerk. Do not transcribe anything.
For each place: box_2d = [ymin, xmin, ymax, xmax] normalized 0-1000, covering the name AND its label(s);
kind = "labeled_fields" | "single_field" | "patient_sticker" | "other"; script = "handwritten" | "printed".
If the patient's name appears nowhere, return an empty list."""

LOCATE_SCHEMA = {
    "type": "OBJECT",
    "properties": {"names": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {"box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                       "kind": {"type": "STRING", "enum": ["labeled_fields", "single_field", "patient_sticker",
                                                           "other"]},
                       "script": {"type": "STRING", "enum": ["handwritten", "printed"]}},
        "required": ["box_2d", "kind", "script"],
        "propertyOrdering": ["box_2d", "kind", "script"]}}},
    "required": ["names"],
}


# ------------------------------------------------------------------ preprocessing variants (OpenCV)

def _gray(a: np.ndarray) -> np.ndarray:
    return a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)


def _upscale(a: np.ndarray) -> np.ndarray:
    return cv2.resize(a, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def _clahe(a: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(_gray(a))


def _adaptive(a: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(_gray(a), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def _denoise(a: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(_gray(a), None, h=10, templateWindowSize=7, searchWindowSize=21)


def _remove_lines(a: np.ndarray) -> np.ndarray:
    """Remove long horizontal / vertical table lines and fill them from their surroundings (inpaint),
    so strokes that cross a line are not cut. A line must be much longer than any letter stroke:
    a quarter of the crop width, or 70% of its height (letters of tall Hebrew letters like ל, ן, ו
    stay well below that, since crops are padded around the name)."""
    g = _gray(a)
    h, w = g.shape
    ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(80, w // 4), 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(60, int(h * 0.7)))))
    mask = cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8))
    return cv2.inpaint(g, mask, 3, cv2.INPAINT_TELEA)


VARIANTS = OrderedDict([
    ("none", lambda a: a),
    ("gray", _gray),
    ("upscale", _upscale),
    ("clahe", _clahe),
    ("adaptive", _adaptive),
    ("denoise", _denoise),
    ("lines", _remove_lines),
    # Initial guess at the best combination; replaced by the winners of round 1.
    ("combo", lambda a: _upscale(_clahe(_remove_lines(a)))),
])


def apply_variant(image: Image.Image, variant: str) -> Image.Image:
    """Return a new image; the input is never modified."""
    out = VARIANTS[variant](np.asarray(image.convert("RGB")).copy())
    return Image.fromarray(out)


# ------------------------------------------------------------------ scoring

_PUNCT_RE = re.compile(r"[\"'`׳״.,:;()\-_/]")


def name_tokens(text: Optional[str]) -> tuple:
    """Order-insensitive comparison key: the words of a name, punctuation removed."""
    return tuple(sorted(_PUNCT_RE.sub(" ", text or "").split()))


def categorize(reading: Optional[dict], truth: tuple) -> str:
    if not reading or reading.get("error"):
        return "error"
    tokens = name_tokens(reading.get("name"))
    if not reading.get("legible") or not tokens:
        return "empty"
    expected = name_tokens(" ".join(truth))
    if tokens == expected:
        return "correct"
    return "partial" if set(tokens) <= set(expected) else "invented"


def accepted(reading: Optional[dict], threshold: float = config.CONFIDENCE_THRESHOLD) -> bool:
    """Not flagged: legible, a name, confidence >= threshold (as the pipeline would accept it)."""
    return bool(reading and not reading.get("error") and reading.get("legible") and name_tokens(reading.get("name"))
                and float(reading.get("confidence") or 0) >= threshold)


def consensus(a: Optional[dict], b: Optional[dict]) -> dict:
    """Combine two independent readings: a name only when both are accepted and read the same words."""
    if accepted(a) and accepted(b) and name_tokens(a["name"]) == name_tokens(b["name"]):
        return {"legible": True, "name": a["name"], "confidence": min(a["confidence"], b["confidence"])}
    return {"legible": False, "name": None, "confidence": 0.0}


def summarize_round(rows: list) -> dict:
    """rows: [{"arm", "set", "category", "accepted"}] -> {(arm, set): counts}. Counts only."""
    out = OrderedDict()
    for r in rows:
        c = out.setdefault((r["arm"], r["set"]), Counter())
        c["n"] += 1
        c[r["category"]] += 1
        if r["category"] in WRONG:
            c["silent_errors" if r["accepted"] else "wrong_flagged"] += 1
        if r["category"] == "correct" and r["accepted"]:
            c["correct_accepted"] += 1
    return out


# ------------------------------------------------------------------ cost

def price_for(model: str, overrides: dict) -> tuple:
    if model in overrides:
        return overrides[model]
    if model in PRICES:
        return PRICES[model]
    raise SystemExit(f"No price known for this model - pass --price {model}=IN/OUT (USD per 1M tokens)")


def cost_usd(prompt_tokens: int, output_tokens: int, price: tuple) -> float:
    return (prompt_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def estimate(n_crops: int, arms: list, models: dict, overrides: dict) -> list:
    """[(arm, requests, usd)] for one round, from the token estimates above."""
    out = []
    for arm in arms:
        alias = arm.split(":")[0]
        usd = n_crops * cost_usd(EST_INPUT_TOKENS, EST_OUTPUT_TOKENS[alias], price_for(models[alias], overrides))
        out.append((arm, n_crops, usd))
    return out


# ------------------------------------------------------------------ request guard

_local = threading.local()


def install_guard(max_requests: int, allowed_stages: tuple, fresh: bool = False) -> dict:
    """Count live requests; refuse any beyond max_requests or from another stage. With fresh=True the
    dev cache is bypassed for the allowed stages (a new draw from the model)."""
    stats = {"live": 0, "blocked": 0, "cap_reached": False}
    lock = threading.Lock()
    original_generate, original_call = ee.generate_json, ee._generate_with_transport_retry
    original_get, original_put = ee._cache_get, ee._cache_put

    def is_experiment() -> bool:
        return getattr(_local, "stage", "").startswith(allowed_stages)

    def generate_json(*args, **kwargs):
        _local.stage = kwargs.get("purpose", "other")
        return original_generate(*args, **kwargs)

    def call(contents, cfg):
        with lock:
            if not is_experiment():
                stats["blocked"] += 1
                raise ee.GeminiError("blocked: not an experiment request")
            if stats["live"] >= max_requests:
                stats["cap_reached"] = True
                raise ee.GeminiQuotaError(f"--max-requests {max_requests} reached")
            stats["live"] += 1
        return original_call(contents, cfg)

    config.DEV_CACHE = True  # re-running a round answers from the cache (free) unless --fresh
    ee.generate_json, ee._generate_with_transport_retry = generate_json, call
    if fresh:
        ee._cache_get = lambda key: None if is_experiment() else original_get(key)
        ee._cache_put = lambda key, data: None if is_experiment() else original_put(key, data)
    return stats


class use_model:
    """Temporarily point config.GEMINI_MODEL at another model (arms run one after another)."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.had = "GEMINI_MODEL" in vars(config)
        self.old = vars(config).get("GEMINI_MODEL")
        config.GEMINI_MODEL = self.name

    def __exit__(self, *exc):
        if self.had:
            config.GEMINI_MODEL = self.old
        else:
            del config.GEMINI_MODEL


# ------------------------------------------------------------------ test sets

def load_labels(path: Path = LABELS_PATH) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("file") or "").strip()]


def photo_crops(labels: list) -> list:
    """[(crop_id, set, truth_tuple, image)] for the photos listed in labels.csv (EXIF orientation applied)."""
    out = []
    for i, row in enumerate(labels, start=1):
        path = HW_DIR / row["file"].strip()
        if path.suffix.lower() not in PHOTO_EXTENSIONS or not path.exists():
            print(f"  photo row {i}: file missing or not an image - skipped")
            continue
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        img.thumbnail((PHOTO_MAX_SIDE, PHOTO_MAX_SIDE), Image.LANCZOS)
        truth = tuple(w for w in (row.get("first_name", "").strip(), row.get("last_name", "").strip()) if w)
        out.append((f"photo{i:03d}", "photo", truth, img))
    return out


def _sample_pages(run_dir: Path) -> list:
    from src.file_converter import convert_files

    return convert_files(sorted(SAMPLES_DIR.glob("*.pdf")), run_dir)


def sample_crops(run_dir: Path) -> list:
    """[(crop_id, set, truth, image)] cut from the saved sample boxes (original pixels, page made upright)."""
    from src.image_preprocessor import fix_rotation, preprocess

    if not BOXES_PATH.exists():
        return []
    saved = json.loads(BOXES_PATH.read_text(encoding="utf-8"))
    pages = {(p.source_file, p.page_number): p for p in _sample_pages(run_dir)}
    file_tags = {f.name: f"f{i}" for i, f in enumerate(sorted(SAMPLES_DIR.glob("*.pdf")), start=1)}
    out = []
    for key, entry in saved.items():
        source, page_number = key.rsplit("|", 1)
        page = pages.get((source, int(page_number)))
        if page is None or not entry["names"]:
            continue
        prepared = preprocess(fix_rotation(page.load(), entry["rotation"]))
        img = prepared.crop_source
        file_tag = file_tags.get(source, "f?")
        for j, loc in enumerate(entry["names"], start=1):
            if loc["script"] != "handwritten":
                continue
            box = ee.padded_pixel_box(img.size, tuple(loc["box"]), ee.NAME_CROP_PAD_X, ee.NAME_CROP_PAD_Y,
                                      ee.NAME_CROP_MIN_PAD_PX)
            if box is not None:
                out.append((f"{file_tag}p{int(page_number):02d}c{j}", "sample", SAMPLE_TRUTH, img.crop(box)))
    return out


# ------------------------------------------------------------------ commands

def cmd_models(args) -> None:
    for m in ee.get_client().models.list():
        if "generateContent" in (getattr(m, "supported_actions", None) or []):
            print(f"  {m.name.removeprefix('models/')}")


def cmd_locate(args) -> None:
    from src.file_converter import cleanup_temp, new_run_dir
    from src.form_splitter import split_forms
    from src.image_preprocessor import preprocess

    stats = install_guard(args.max_requests, ("hw_locate",))
    run_dir = new_run_dir()
    try:
        pages = _sample_pages(run_dir)
        forms = split_forms(pages)  # orientation + split come from the dev cache (blocked otherwise)
        if stats["blocked"]:
            raise SystemExit("Orientation / split were not cached - stopping (no locate request sent).")
        saved = json.loads(BOXES_PATH.read_text(encoding="utf-8")) if BOXES_PATH.exists() else {}

        def locate(form):
            out = []
            for page, rotation, img in zip(form.pages, form.rotations, form.images()):
                key = f"{page.source_file}|{page.page_number}"
                if key in saved:
                    continue
                contents = [LOCATE_PROMPT, ee.image_part(preprocess(img).processed, config.EXTRACTION_IMAGE_MAX_SIDE)]
                try:
                    data = ee.generate_json(contents, LOCATE_SCHEMA, media_resolution="high", purpose="hw_locate")
                except ee.GeminiError as exc:
                    print(f"  locate failed for one page: {type(exc).__name__}")
                    continue
                names = []
                for item in data.get("names") or []:
                    parsed = ee._parse_box({**item, "page": 1}) if isinstance(item, dict) else None
                    if parsed:
                        names.append({"box": list(parsed[1]), "kind": item.get("kind", "other"),
                                      "script": item.get("script", "handwritten")})
                out.append((key, {"rotation": rotation, "names": names[:ee.MAX_NAME_LOCATIONS]}))
            return out

        with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_REQUESTS) as pool:
            for result in pool.map(locate, forms):
                saved.update(result)
        BOXES_PATH.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
        usage = ee.get_usage("hw_locate")
        kinds = Counter(n["script"] for e in saved.values() for n in e["names"])
        print(f"pages with boxes: {sum(1 for e in saved.values() if e['names'])}/{len(saved)}  boxes: {dict(kinds)}")
        print(f"live requests: {stats['live']}  tokens in/out: {usage['prompt_tokens']}/{usage['output_tokens']}")
    finally:
        cleanup_temp(run_dir)


def _all_crops(run_dir: Path, sets: str) -> list:
    crops = []
    if sets in ("all", "sample"):
        crops += sample_crops(run_dir)
    if sets in ("all", "photo"):
        crops += photo_crops(load_labels())
    return crops


def _models(args) -> dict:
    models = {"flash": config.GEMINI_MODEL}
    if args.pro_model:
        models["pro"] = args.pro_model
    return models


def _parse_arms(spec: str, models: dict) -> list:
    arms = [a.strip() for a in spec.split(",") if a.strip()]
    for arm in arms:
        alias, _, variant = arm.partition(":")
        if alias not in models or variant not in VARIANTS:
            raise SystemExit(f"Bad arm {arm!r}: model must be one of {sorted(models)}, variant one of {list(VARIANTS)}")
    return arms


def _prices(args) -> dict:
    out = {}
    for item in args.price or []:
        model, _, value = item.partition("=")
        p_in, _, p_out = value.partition("/")
        out[model] = (float(p_in), float(p_out))
    return out


def cmd_prepare(args) -> None:
    from src.file_converter import cleanup_temp, new_run_dir

    run_dir = new_run_dir()
    try:
        crops = _all_crops(run_dir, args.sets)
        counts = Counter(c[1] for c in crops)
        print(f"crops: {dict(counts)}  (sample boxes file: {'yes' if BOXES_PATH.exists() else 'missing - run locate'})")
        if args.previews:
            args.previews.mkdir(parents=True, exist_ok=True)
            for crop_id, _, _, img in crops:
                for variant in VARIANTS:
                    apply_variant(img, variant).save(args.previews / f"{crop_id}_{variant}.png")
            print(f"previews written: {len(crops) * len(VARIANTS)} (delete the folder afterwards)")
        if args.arms:
            models = _models(args)
            total = 0.0
            for arm, n, usd in estimate(len(crops), _parse_arms(args.arms, models), models, _prices(args)):
                total += usd
                print(f"  estimate {arm:16s} requests={n:<4d} ~${usd:.2f}")
            print(f"  estimate total ~${total:.2f} (token estimate; the run reports the measured cost)")
    finally:
        cleanup_temp(run_dir)


def cmd_run(args) -> None:
    from src.file_converter import cleanup_temp, new_run_dir

    models = _models(args)
    arms = _parse_arms(args.arms, models)
    prices = _prices(args)
    for alias in {a.split(":")[0] for a in arms}:
        price_for(models[alias], prices)  # fail early on an unknown price
    stats = install_guard(args.max_requests, ("hw_read",), fresh=args.fresh)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = new_run_dir()
    try:
        crops = _all_crops(run_dir, args.sets)[: args.limit or None]
        results, costs = [], {}
        for arm in arms:
            alias, _, variant = arm.partition(":")
            purpose = f"hw_read:{arm}"
            ee.reset_usage()

            def read(crop, variant=variant, purpose=purpose):
                crop_id, set_name, truth, img = crop
                contents = [READ_PROMPT, ee.image_part(apply_variant(img, variant), CROP_MAX_SIDE)]
                try:
                    data = ee.generate_json(contents, READ_SCHEMA, media_resolution="high", purpose=purpose)
                    reading = {"legible": bool(data.get("legible")), "name": data.get("name"),
                               "confidence": float(data.get("confidence") or 0)}
                except ee.GeminiError as exc:
                    reading = {"error": type(exc).__name__}
                return {"crop": crop_id, "set": set_name, "arm": arm, "truth": list(truth), **reading,
                        "category": categorize(reading, truth), "accepted": accepted(reading)}

            with use_model(models[alias]), ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_REQUESTS) as pool:
                results += list(pool.map(read, crops))
            usage = ee.get_usage(purpose)
            costs[arm] = {"live_requests": usage["requests"], "prompt_tokens": usage["prompt_tokens"],
                          "output_tokens": usage["output_tokens"],
                          "usd": cost_usd(usage["prompt_tokens"], usage["output_tokens"],
                                          price_for(models[alias], prices))}
            if stats["cap_reached"]:
                print("Request cap reached - stopping after this arm.")
                break
        path = OUT_DIR / f"round_{args.round}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (OUT_DIR / f"round_{args.round}_cost.json").write_text(json.dumps(costs, indent=1), encoding="utf-8")
        print_summary(results, costs)
        print(f"\nlive requests: {stats['live']} (cap {args.max_requests}); readings saved locally (git-ignored)")
    finally:
        cleanup_temp(run_dir)


def print_summary(results: list, costs: Optional[dict] = None) -> None:
    summary = summarize_round(results)
    print(f"\n{'arm':22s} {'set':7s} {'n':>4s} {'correct':>8s} {'SILENT':>7s} {'wrong':>6s} {'empty':>6s} "
          f"{'invented':>9s} {'errors':>7s} {'$/name':>8s}")
    print(f"{'':22s} {'':7s} {'':>4s} {'accepted':>8s} {'ERRORS':>7s} {'flagged':>6s}")
    for (arm, set_name), c in summary.items():
        cost = (costs or {}).get(arm)
        per_name = f"{cost['usd'] / max(cost['live_requests'], 1):.4f}" if cost and cost["live_requests"] else "cached"
        invented = str(c["invented"]) if set_name == "sample" else "-"
        print(f"{arm:22s} {set_name:7s} {c['n']:>4d} {c['correct_accepted']:>8d} {c['silent_errors']:>7d} "
              f"{c['wrong_flagged']:>6d} {c['empty']:>6d} {invented:>9s} {c['error']:>7d} {per_name:>8s}")
    if costs:
        total = sum(v["usd"] for v in costs.values())
        print(f"\nmeasured cost this round: ${total:.2f} (live requests only)")


def cmd_report(args) -> None:
    path = OUT_DIR / f"round_{args.round}.jsonl"
    results = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cost_path = OUT_DIR / f"round_{args.round}_cost.json"
    costs = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else None
    if args.consensus:
        a, b = args.consensus.split(",")
        by = {}
        for r in results:
            by.setdefault(r["crop"], {})[r["arm"]] = r
        for crop, arms in by.items():
            if a in arms and b in arms:
                reading = consensus(arms[a], arms[b])
                results.append({"crop": crop, "set": arms[a]["set"], "arm": f"{a}&{b}",
                                "category": categorize(reading, tuple(arms[a]["truth"])),
                                "accepted": accepted(reading)})
    print_summary(results, costs)


# ------------------------------------------------------------------ labelled forms: DPI x preprocessing x thinking
# The 5 labelled sample forms (ground_truth.csv). Name boxes come from the pipeline's own extraction (dev
# cache, no request). Controls detect guessing: an empty area and handwritten text that is not a name on each
# page - ANY name returned on a control is a guess. Every crop is rendered straight from the sample PDF at
# each DPI, with the pipeline's geometry (rotation + deskew angle), so all arms see the same area.

LABELLED_BOXES_PATH = HW_DIR / "labelled_boxes.json"   # coordinates only (git-ignored folder)
DPIS = (300, 600)
THINKING = ("default", "high")
LABELLED_MODELS = ("flash", "pro")      # flash = GEMINI_MODEL, pro = --pro-model
DEFAULT_PRO_MODEL = "gemini-3.1-pro-preview"
# Pilot rule, fixed before any result is seen: most correct names; tie -> fewest invented; tie -> this order.
PILOT_VARIANTS = ("combo", "clahe", "lines", "upscale", "denoise", "gray", "adaptive")
PILOT_ARM = "flash:300:{}:high"         # the pilot reads the names with each variant in this arm
# Output-token estimates per request (incl. thinking) for the cost estimate; the run reports measured cost.
EST_LABELLED_OUTPUT = {"default": 2000, "high": 6000}
EST_LABELLED_OUTPUT_HIGH_BOUND = 10000
# One name crop per form: the first name location. (Sample page 13 has a second, page-wide location -
# sticker + ID + another name field - that is not a name crop.)
NAMES_PER_PAGE = 1
EMPTY_GRID_STEP = 20         # 0-1000 units
EMPTY_MARGIN = 100           # stay inside the form, away from scan borders
EMPTY_MAX_INK = 0.01         # an "empty" control may hold at most 1% dark pixels (a stray line or speck)

CONTROL_PROMPT = """This is one page of a scanned Israeli medical form. Find places with HANDWRITTEN marks that are NOT any
person's name: handwritten numbers, dates, codes, short notes, ticks or circles. Never return a signature, an
initial, or anything that could be a name of a person. Do not transcribe anything.
For each place: box_2d = [ymin, xmin, ymax, xmax] normalized 0-1000, covering only that handwriting;
content = "number" | "date" | "note" | "mark". At most 4 places. None found => an empty list."""

CONTROL_SCHEMA = {
    "type": "OBJECT",
    "properties": {"texts": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {"box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                       "content": {"type": "STRING", "enum": ["number", "date", "note", "mark"]}},
        "required": ["box_2d", "content"],
        "propertyOrdering": ["box_2d", "content"]}}},
    "required": ["texts"],
}


def parse_arm(arm: str) -> tuple:
    """"pro:300:clahe:high" -> ("pro", 300, "clahe", "high")."""
    parts = arm.split(":")
    if len(parts) != 4:
        raise SystemExit(f"Bad arm {arm!r}: expected MODEL:DPI:VARIANT:THINKING")
    alias, dpi, variant, thinking = parts
    if (alias not in LABELLED_MODELS or not dpi.isdigit() or int(dpi) not in DPIS or variant not in VARIANTS
            or thinking not in THINKING):
        raise SystemExit(f"Bad arm {arm!r}: model in {LABELLED_MODELS}, DPI in {DPIS}, variant in "
                         f"{list(VARIANTS)}, thinking in {THINKING}")
    return alias, int(dpi), variant, thinking


def labelled_models(args) -> dict:
    return {"flash": config.GEMINI_MODEL, "pro": args.pro_model or DEFAULT_PRO_MODEL}


def deskew_angle(steps: list) -> float:
    """The deskew angle preprocess applied ("deskew +1.25" -> 1.25), 0.0 when none."""
    for step in steps:
        if step.startswith("deskew "):
            return float(step.split()[1])
    return 0.0


def _overlaps(a: tuple, b: tuple) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _padded(box: tuple, size: tuple) -> tuple:
    """The 0-1000 area a crop of `box` really shows (crop_target's padding at 300 DPI), unclamped."""
    w, h = size
    y0, x0, y1, x1 = box
    py = max((y1 - y0) * ee.NAME_CROP_PAD_Y, ee.NAME_CROP_MIN_PAD_PX * 1000 / h)
    px = max((x1 - x0) * ee.NAME_CROP_PAD_X, ee.NAME_CROP_MIN_PAD_PX * 1000 / w)
    return (y0 - py, x0 - px, y1 + py, x1 + px)


def find_empty_box(image: Image.Image, size: tuple, avoid: list) -> Optional[tuple]:
    """A box of `size` (height, width in 0-1000 units) whose padded crop is the blankest area inside the
    margins and overlaps none of `avoid`: lowest ink (dark pixels) first, then lowest variation.
    None when even the best area has more than EMPTY_MAX_INK ink."""
    gray = np.asarray(image.convert("L").resize((500, 500)), dtype=np.float32)
    bh, bw = size
    best = None
    for y0 in range(0, 1000 - bh + 1, EMPTY_GRID_STEP):
        for x0 in range(0, 1000 - bw + 1, EMPTY_GRID_STEP):
            box = (y0, x0, y0 + bh, x0 + bw)
            shown = _padded(box, image.size)
            if (min(shown[:2]) < EMPTY_MARGIN or max(shown[2:]) > 1000 - EMPTY_MARGIN
                    or any(_overlaps(shown, a) for a in avoid)):
                continue
            patch = gray[int(shown[0]) // 2:int(shown[2]) // 2, int(shown[1]) // 2:int(shown[3]) // 2]
            score = (float((patch < 200).mean()), float(patch.std()))
            if best is None or score < best[0]:
                best = (score, box)
    return best[1] if best and best[0][0] <= EMPTY_MAX_INK else None


def render_page(source_file: str, page_number: int, dpi: int, rotation: int, angle: float) -> Image.Image:
    """One sample page rendered at `dpi`, with the pipeline's geometry (upright, then deskewed)."""
    from pdf2image import convert_from_path

    from src.image_preprocessor import fix_rotation

    [img] = convert_from_path(str(SAMPLES_DIR / source_file), dpi=dpi, first_page=page_number,
                              last_page=page_number, **config.PDF2IMAGE_KWARGS)
    img = fix_rotation(img.convert("RGB"), rotation)
    if angle:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
    return img


def crop_target(page: Image.Image, box: tuple, dpi: int) -> Optional[Image.Image]:
    """Same padding as the pipeline's name crops; the pixel minimum scales with the DPI (same area)."""
    pixel_box = ee.padded_pixel_box(page.size, tuple(box), ee.NAME_CROP_PAD_X, ee.NAME_CROP_PAD_Y,
                                    int(ee.NAME_CROP_MIN_PAD_PX * dpi / 300))
    return page.crop(pixel_box) if pixel_box else None


def build_labelled_pages(stats: dict) -> list:
    """[{source_file, page, rotation, angle, names, avoid, image300}] for the labelled forms, from the dev cache
    only (split + extraction exactly as evaluate_accuracy ran them). Stops when any of those requests was not
    cached (install_guard blocks it): a blocked orientation would silently leave a page unrotated."""
    sys.path.insert(0, str(ROOT / "evaluation"))
    import evaluate_accuracy as ea

    from src.file_converter import cleanup_temp, convert_files, new_run_dir
    from src.form_splitter import split_forms
    from src.image_preprocessor import preprocess
    from src.validators import list_approved_senders

    rows = ea.load_ground_truth(config.DATA_DIR / "ground_truth" / "ground_truth.csv")
    by_file = OrderedDict()
    for row in rows:
        by_file.setdefault(row["source_file"], []).append(row)
    senders = list_approved_senders(config.SENDERS_PATH)
    out = []
    for source_file, file_rows in by_file.items():
        numbers = [p for r in file_rows for p in ea.row_pages(r)]
        run_dir = new_run_dir()
        try:
            pages = convert_files([(source_file, ea._subset_pdf(SAMPLES_DIR / source_file, numbers))], run_dir)
            for form in split_forms(pages):
                prepared = [preprocess(img) for img in form.images()]
                extraction = ee.extract_form([p.processed for p in prepared], senders)
                if extraction.error:
                    raise SystemExit(f"Extraction not in the dev cache ({extraction.error}) - nothing sent.")
                for i, (page, rotation, prep) in enumerate(zip(form.pages, form.rotations, prepared), start=1):
                    names = [loc["box"] for loc in extraction.name_locations if loc["page"] == i]
                    fields = [loc["box"] for loc in extraction.field_locations if loc["page"] == i]
                    out.append({"source_file": source_file, "page": numbers[page.page_number - 1],
                                "rotation": rotation, "angle": deskew_angle(prep.steps), "names": names,
                                "avoid": names + fields, "image300": prep.crop_source})
        finally:
            cleanup_temp(run_dir)
    if stats["blocked"]:
        raise SystemExit(f"{stats['blocked']} pipeline request(s) were not in the dev cache - stopping.")
    return out


def labelled_targets(pages: list, saved: dict) -> list:
    """[{id, kind, source_file, page, rotation, angle, box}]: names, one empty control per page, and the
    handwritten-text controls found by `labelled-locate` (saved: page key -> [box])."""
    targets = []
    for p in pages:
        key = f"{p['source_file']}|{p['page']}"
        base = {k: p[k] for k in ("source_file", "page", "rotation", "angle")}
        tag = f"p{p['page']:02d}"
        names = p["names"][:NAMES_PER_PAGE]
        for j, box in enumerate(names, start=1):
            targets.append({**base, "id": f"{tag}_name{j}", "kind": "name", "box": list(box)})
        if names:
            height, width = names[0][2] - names[0][0], names[0][3] - names[0][1]
            avoid = [_padded(b, p["image300"].size) for b in p["avoid"]]
            empty = None
            while empty is None and width >= 100:  # a page-wide name box leaves no blank area that wide
                empty = find_empty_box(p["image300"], (height, width), avoid)
                width = int(width * 0.7)
            if empty:
                targets.append({**base, "id": f"{tag}_empty", "kind": "control_empty", "box": list(empty)})
        for j, box in enumerate(saved.get(key, []), start=1):
            targets.append({**base, "id": f"{tag}_text{j}", "kind": "control_text", "box": list(box)})
    return targets


def render_targets(targets: list, dpis=DPIS) -> dict:
    """{(target id, dpi): crop image}, one PDF render per page and DPI."""
    crops, pages = {}, {}
    for t in targets:
        for dpi in dpis:
            key = (t["source_file"], t["page"], dpi)
            if key not in pages:
                pages[key] = render_page(t["source_file"], t["page"], dpi, t["rotation"], t["angle"])
            crop = crop_target(pages[key], t["box"], dpi)
            if crop is not None:
                crops[(t["id"], dpi)] = crop
    return crops


def score_reading(kind: str, reading: dict) -> str:
    """Names: correct / partial / invented / empty / error. Controls: guess (any name) / abstain / error."""
    if kind == "name":
        return categorize(reading, SAMPLE_TRUTH)
    if not reading or reading.get("error"):
        return "error"
    return "guess" if reading.get("legible") and name_tokens(reading.get("name")) else "abstain"


def summarize_arms(rows: list) -> OrderedDict:
    """rows: [{"arm", "kind", "category", "accepted"}] -> {arm: Counter}. Counts only."""
    out = OrderedDict()
    for r in rows:
        c = out.setdefault(r["arm"], Counter())
        group = "name" if r["kind"] == "name" else "control"
        c[f"{group}_n"] += 1
        c[f"{group}_{r['category']}"] += 1
        if r["accepted"] and r["category"] in ("partial", "invented", "guess"):
            c[f"{group}_silent"] += 1  # a wrong name that would NOT be flagged
    return out


def pick_best_variant(rows: list) -> str:
    """The pilot rule (PILOT_VARIANTS, read in PILOT_ARM): most correct names, then fewest invented, then
    the fixed order."""
    summary = summarize_arms(rows)

    def rank(variant: str) -> tuple:
        c = summary.get(PILOT_ARM.format(variant), Counter())
        return (-c["name_correct"], c["name_invented"] + c["name_partial"], PILOT_VARIANTS.index(variant))

    return min(PILOT_VARIANTS, key=rank)


def estimate_labelled(n_names: int, n_controls: int, grid_arms: list, draws: int, pilot: bool,
                      prices: dict) -> list:
    """[(step, requests, usd, usd_high_bound)] from the token estimates. prices: model alias -> (in, out).
    The bound assumes EST_LABELLED_OUTPUT_HIGH_BOUND output tokens for every high-thinking request."""
    def per(alias: str, thinking: str, bound: bool = False) -> float:
        out = EST_LABELLED_OUTPUT_HIGH_BOUND if bound and thinking == "high" else EST_LABELLED_OUTPUT[thinking]
        return cost_usd(EST_INPUT_TOKENS, out, prices[alias])

    locate = 5 * cost_usd(2500, 1500, prices["flash"])
    steps = [("locate text controls", 5, locate, locate)]
    if pilot:
        alias, _, _, thinking = parse_arm(PILOT_ARM.format(PILOT_VARIANTS[0]))
        n = len(PILOT_VARIANTS) * n_names
        steps.append(("pilot: pick the best preprocessing", n, n * per(alias, thinking),
                      n * per(alias, thinking, bound=True)))
    for arm in grid_arms:
        alias, _, _, thinking = parse_arm(arm)
        n = (n_names + n_controls) * draws
        steps.append((arm, n, n * per(alias, thinking), n * per(alias, thinking, bound=True)))
    return steps


def cmd_labelled_prepare(args) -> None:
    stats = install_guard(0, ("none",))  # dev cache only: any request that would reach the API is refused
    pages = build_labelled_pages(stats)
    saved = json.loads(LABELLED_BOXES_PATH.read_text(encoding="utf-8")) if LABELLED_BOXES_PATH.exists() else {}
    targets = labelled_targets(pages, saved)
    kinds = Counter(t["kind"] for t in targets)
    print(f"pages: {len(pages)}  targets: {dict(kinds)}  (text controls file: "
          f"{'yes' if LABELLED_BOXES_PATH.exists() else 'not yet - located in the run'})")
    if args.previews:
        args.previews.mkdir(parents=True, exist_ok=True)
        crops = render_targets(targets)
        for (tid, dpi), img in crops.items():
            img.save(args.previews / f"{tid}_{dpi}.png")
            if tid.split("_")[1].startswith("name") and dpi == 300:
                for variant in PILOT_VARIANTS:
                    apply_variant(img, variant).save(args.previews / f"{tid}_{dpi}_{variant}.png")
        sizes = {f"{tid}@{dpi}": img.size for (tid, dpi), img in crops.items() if "name" in tid}
        print(f"previews written to {args.previews} (delete afterwards); name crop sizes: {sizes}")
    if args.arms:
        n_names = kinds["name"]
        n_controls = kinds["control_empty"] + max(kinds["control_text"], len(pages))  # text controls: ~1/page
        models = labelled_models(args)
        prices = {alias: price_for(model, _prices(args)) for alias, model in models.items()}
        total = bound = 0.0
        for step, n, usd, high in estimate_labelled(n_names, n_controls, args.arms.split(","), args.draws,
                                                    args.pilot, prices):
            total, bound = total + usd, bound + high
            print(f"  estimate {step:36s} requests={n:<4d} ~${usd:.2f}  (bound ${high:.2f})")
        print(f"  estimate total ~${total:.2f}, bound ${bound:.2f}  models: {models}; the run reports measured cost")


def cmd_labelled_locate(args) -> None:
    stats = install_guard(args.max_requests, ("hw_controls",))
    pages = build_labelled_pages(stats)
    saved = json.loads(LABELLED_BOXES_PATH.read_text(encoding="utf-8")) if LABELLED_BOXES_PATH.exists() else {}
    for p in pages:
        key = f"{p['source_file']}|{p['page']}"
        if key in saved:
            continue
        contents = [CONTROL_PROMPT, ee.image_part(p["image300"], config.EXTRACTION_IMAGE_MAX_SIDE)]
        try:
            data = ee.generate_json(contents, CONTROL_SCHEMA, media_resolution="high", purpose="hw_controls")
        except ee.GeminiError as exc:
            print(f"  locate failed for one page: {type(exc).__name__}")
            continue
        boxes = []
        for item in data.get("texts") or []:
            parsed = ee._parse_box({**item, "page": 1}) if isinstance(item, dict) else None
            if parsed and not any(_overlaps(parsed[1], a) for a in p["names"]):
                boxes.append(list(parsed[1]))
        saved[key] = boxes[: args.per_page]
    LABELLED_BOXES_PATH.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    usage = ee.get_usage("hw_controls")
    print(f"text controls per page: {[len(v) for v in saved.values()]}  live requests: {stats['live']}  "
          f"tokens in/out: {usage['prompt_tokens']}/{usage['output_tokens']}")


def cmd_labelled_run(args) -> None:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        parse_arm(arm)
    models = labelled_models(args)
    prices = {alias: price_for(model, _prices(args)) for alias, model in models.items()}
    stats = install_guard(args.max_requests, ("hw_read",))
    original_get, original_put = ee._cache_get, ee._cache_put
    ee._cache_get = lambda key: None if getattr(_local, "fresh", False) else original_get(key)
    ee._cache_put = lambda key, data: None if getattr(_local, "fresh", False) else original_put(key, data)

    saved = json.loads(LABELLED_BOXES_PATH.read_text(encoding="utf-8")) if LABELLED_BOXES_PATH.exists() else {}
    targets = labelled_targets(build_labelled_pages(stats), saved)
    if args.names_only:
        targets = [t for t in targets if t["kind"] == "name"]
    dpis = sorted({parse_arm(a)[1] for a in arms})
    crops = render_targets(targets, dpis)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, costs = [], {}
    for arm in arms:
        alias, dpi, variant, thinking = parse_arm(arm)
        purpose = f"hw_read:{arm}"

        def read(job, dpi=dpi, variant=variant, thinking=thinking, purpose=purpose, arm=arm):
            target, draw = job
            _local.fresh = draw > 1  # later draws bypass the dev cache: a new answer from the model
            contents = [READ_PROMPT, ee.image_part(apply_variant(crops[(target["id"], dpi)], variant), CROP_MAX_SIDE)]
            try:
                data = ee.generate_json(contents, READ_SCHEMA, media_resolution="high", purpose=purpose,
                                        thinking_level=None if thinking == "default" else thinking)
                reading = {"legible": bool(data.get("legible")), "name": data.get("name"),
                           "confidence": float(data.get("confidence") or 0)}
            except ee.GeminiError as exc:
                reading = {"error": type(exc).__name__}
            return {"target": target["id"], "kind": target["kind"], "draw": draw, "arm": arm, **reading,
                    "category": score_reading(target["kind"], reading), "accepted": accepted(reading)}

        jobs = [(t, d) for d in range(1, args.draws + 1) for t in targets if (t["id"], dpi) in crops]
        with use_model(models[alias]), ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_REQUESTS) as pool:
            results += list(pool.map(read, jobs))
        usage = ee.get_usage(purpose)
        costs[arm] = {"model": models[alias], "live_requests": usage["requests"],
                      "prompt_tokens": usage["prompt_tokens"], "output_tokens": usage["output_tokens"],
                      "usd": cost_usd(usage["prompt_tokens"], usage["output_tokens"], prices[alias])}
        if stats["cap_reached"]:
            print("Request cap reached - stopping after this arm.")
            break
    path = OUT_DIR / f"labelled_{args.round}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT_DIR / f"labelled_{args.round}_cost.json").write_text(json.dumps(costs, indent=1), encoding="utf-8")
    print_arms(results, costs)
    if args.pilot:
        print(f"\nbest preprocessing variant (pilot rule): {pick_best_variant(results)}")
    print(f"\nlive requests: {stats['live']} (cap {args.max_requests}); readings saved locally (git-ignored)")


def print_arms(results: list, costs: Optional[dict] = None) -> None:
    summary = summarize_arms(results)
    print(f"\n{'arm':22s} {'names':>5s} {'correct':>7s} {'partial':>7s} {'invent':>6s} {'empty':>5s} {'SILENT':>6s} "
          f"{'ctrl':>4s} {'GUESS':>5s} {'err':>3s} {'out tok/req':>11s} {'$/read':>7s} {'$ arm':>6s}")
    for arm, c in summary.items():
        cost = (costs or {}).get(arm) or {}
        live = cost.get("live_requests") or 0
        per = f"{cost['usd'] / live:.4f}" if live else "cached"
        tok = f"{cost['output_tokens'] // live}" if live else "-"
        print(f"{arm:22s} {c['name_n']:>5d} {c['name_correct']:>7d} {c['name_partial']:>7d} {c['name_invented']:>6d} "
              f"{c['name_empty']:>5d} {c['name_silent']:>6d} {c['control_n']:>4d} {c['control_guess']:>5d} "
              f"{c['name_error'] + c['control_error']:>3d} {tok:>11s} {per:>7s} {cost.get('usd', 0):>6.2f}")
    if costs:
        print(f"\nmeasured cost: ${sum(v['usd'] for v in costs.values()):.2f} (live requests only)")


def cmd_labelled_report(args) -> None:
    path = OUT_DIR / f"labelled_{args.round}.jsonl"
    results = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cost_path = OUT_DIR / f"labelled_{args.round}_cost.json"
    print_arms(results, json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models")
    p = sub.add_parser("locate")
    p.add_argument("--max-requests", type=int, required=True)
    for name in ("prepare", "run"):
        p = sub.add_parser(name)
        p.add_argument("--arms", required=(name == "run"))
        p.add_argument("--pro-model")
        p.add_argument("--price", action="append", help="MODEL=IN/OUT USD per 1M tokens")
        p.add_argument("--sets", choices=("all", "sample", "photo"), default="all")
        if name == "prepare":
            p.add_argument("--previews", type=Path)
        else:
            p.add_argument("--round", required=True)
            p.add_argument("--max-requests", type=int, required=True)
            p.add_argument("--limit", type=int)
            p.add_argument("--fresh", action="store_true", help="bypass the dev cache (a new draw)")
    p = sub.add_parser("report")
    p.add_argument("--round", required=True)
    p.add_argument("--consensus", help="ARM1,ARM2: accept a name only when both arms read the same words")
    p = sub.add_parser("labelled-prepare", help="5 labelled forms: crops, previews, cost estimate (no API)")
    p.add_argument("--previews", type=Path)
    p.add_argument("--arms", help='grid arms "MODEL:DPI:VARIANT:THINKING", e.g. flash:300:none:high')
    p.add_argument("--pro-model", help=f"default {DEFAULT_PRO_MODEL}")
    p.add_argument("--draws", type=int, default=2)
    p.add_argument("--pilot", action="store_true", help="include the pilot that picks the preprocessing")
    p.add_argument("--price", action="append", help="MODEL=IN/OUT USD per 1M tokens")
    p = sub.add_parser("labelled-locate", help="find handwritten non-name text controls (one request per page)")
    p.add_argument("--max-requests", type=int, required=True)
    p.add_argument("--per-page", type=int, default=1)
    p = sub.add_parser("labelled-run")
    p.add_argument("--round", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--draws", type=int, default=2)
    p.add_argument("--max-requests", type=int, required=True)
    p.add_argument("--names-only", action="store_true")
    p.add_argument("--pro-model", help=f"default {DEFAULT_PRO_MODEL}")
    p.add_argument("--pilot", action="store_true", help="report the best variant by the pilot rule")
    p.add_argument("--price", action="append", help="MODEL=IN/OUT USD per 1M tokens")
    p = sub.add_parser("labelled-report")
    p.add_argument("--round", required=True)
    args = parser.parse_args()
    {"models": cmd_models, "locate": cmd_locate, "prepare": cmd_prepare, "run": cmd_run,
     "report": cmd_report, "labelled-prepare": cmd_labelled_prepare, "labelled-locate": cmd_labelled_locate,
     "labelled-run": cmd_labelled_run, "labelled-report": cmd_labelled_report}[args.command](args)


if __name__ == "__main__":
    main()
