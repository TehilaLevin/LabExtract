"""Compare engine output against data/ground_truth/ground_truth_draft.csv, per field.

Usage:
    .venv/Scripts/python evaluation/evaluate_accuracy.py [--ground-truth PATH] [--cache-only]
        [--max-requests N] [--engine-root DIR] [--save-summary OUT.json] [--compare BASE.json]
        [--show-values]

Runs the full pipeline on exactly the (source_file, page) rows listed in the CSV (pages are
taken from data/samples/), then prints per-field accuracy (exact match after normalization),
a per-field breakdown and the list of failed fields. Every field counts, names included (first and
last name are compared order-insensitively: the forms do not write them in a fixed order). An
empty expected value means the correct answer is "no value": it matches only an empty engine
value. Correctness is decided ONLY by the ground truth - passing validation never makes a value
correct. Unverified suggestions (handwritten names) and re-read alternatives never count as
correct values.

Breakdown per field (OUTCOMES): correct_unflagged | correct_flagged (right, still shown for review) |
wrong_flagged (wrong, but shown for review) | WRONG_SILENT (wrong and not flagged - the dangerous case) |
empty_flagged / empty_unflagged (a value was expected, the output is empty) | no_record (the form was
not produced). Only the correct_* outcomes count as correct.

Label UNREADABLE (any case) marks a field a human cannot read with confidence from the image
itself. Then an empty, flagged output is the right answer (correct_flagged); an empty unflagged output
is empty_unflagged; a value is a guess: wrong_flagged when the reviewer sees the flag, wrong_silent only
when it was not flagged. It applies to the whole field (for a list, use it when not every number can be
read).

Optional columns: tags (e.g. "handwritten;multi_central") and split ("tune" / "test"). Results
are also reported per split and per tag; --split / --tag evaluate only the matching rows. Tune
prompts and rules on "tune" rows only; run "test" rows only to measure a finished change.

Comparing two versions on the same forms: save a baseline summary, then --compare it. The
scorer, ground truth, samples and dev cache always come from THIS tree; --engine-root DIR runs
the engine code of another tree (e.g. a git worktree of the baseline commit), so both versions
are scored identically. A summary records a fingerprint of the rows and labels, and the
comparison refuses to compare different ones.

Cost: live API calls are made unless --cache-only is given. --cache-only answers every request
from data/dev_cache/ and NEVER calls the API; uncached requests fail and are counted per stage.
--max-requests N stops the run after N live requests. Use DEV_CACHE=true for live development
runs (unchanged requests, e.g. orientation and splitting, are then answered from the cache).

Privacy: no field value is printed by default - failures show only "mismatch". --show-values
prints expected/got for non-personal fields (numbers, dates, senders); names and ID numbers are
never printed. Saved summaries contain counts, file names, page numbers and hashes only.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import threading
from collections import Counter, OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _argv_value(flag: str) -> Optional[str]:
    args = sys.argv[1:]
    return args[args.index(flag) + 1] if flag in args[:-1] else None


# --engine-root must be resolved before `src` is imported: `src` is the evaluated engine.
ENGINE_ROOT = Path(_argv_value("--engine-root") or ROOT).resolve()
if ENGINE_ROOT != ROOT:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # the other tree has no .env of its own; nothing is printed
sys.path.insert(0, str(ENGINE_ROOT))


def _load_scorer_module(name: str):
    """Scoring helpers always come from THIS tree, whichever engine is evaluated."""
    spec = importlib.util.spec_from_file_location(f"_scorer_{name}", ROOT / "src" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_constants = _load_scorer_module("constants")
_validators = _load_scorer_module("validators")
FIELD_HEADERS, FIELD_KEYS, LIST_FIELDS = _constants.FIELD_HEADERS, _constants.FIELD_KEYS, _constants.LIST_FIELDS

from src import config  # noqa: E402  - the evaluated engine's config

if ENGINE_ROOT != ROOT:
    config.DEV_CACHE_DIR = DATA_DIR / "dev_cache"

DEFAULT_GROUND_TRUTH = DATA_DIR / "ground_truth" / "ground_truth_draft.csv"
SAMPLES_DIR = DATA_DIR / "samples"
SENDERS_PATH = DATA_DIR / "reference" / "approved_senders.json"
PERSONAL_FIELDS = frozenset({"first_name", "last_name", "id_number"})
UNREADABLE_LABEL = "UNREADABLE"
SPLITS = ("tune", "test")
OUTCOMES = ("correct_unflagged", "correct_flagged", "wrong_flagged", "wrong_silent", "empty_flagged",
            "empty_unflagged", "no_record")
CORRECT_OUTCOMES = frozenset({"correct_unflagged", "correct_flagged"})
_NAME_PUNCT_RE = re.compile(r"[\"'`׳״.,:;()\-]")
NAME_FIELDS = ("first_name", "last_name")


# ------------------------------------------------------------------ comparison

def normalize_value(key: str, value) -> object:
    """Comparable form of a field value; empty values normalize to "" / frozenset()."""
    if key in LIST_FIELDS:
        return frozenset(n.replace(" ", "") for n in _validators.normalize_number_list(value))
    text = (value or "").strip()
    if not text:
        return ""
    if key in ("first_name", "last_name"):
        return " ".join(_NAME_PUNCT_RE.sub(" ", text).split())
    if key == "id_number":
        digits = re.sub(r"\D", "", text)
        # A leading zero does not change an Israeli ID (the checksum pads to 9 digits).
        return digits.zfill(9) if 0 < len(digits) <= 9 else digits
    if key == "received_date":
        return _validators.normalize_date(text) or text
    if key == "sending_entity":
        return _validators.match_sending_entity(text, SENDERS_PATH) or text
    return text


def is_unreadable_label(value) -> bool:
    """Ground-truth label for a field a human cannot read with confidence from the image itself."""
    return isinstance(value, str) and value.strip().upper() == UNREADABLE_LABEL


def names_swapped(truth: dict, first, last) -> bool:
    """The two names are right but first / last are swapped. The forms do not write names in a consistent
    order (confirmed by the lab side), so a swapped pair counts as a correct reading."""
    t_first, t_last = (normalize_value(k, truth.get(k)) for k in NAME_FIELDS)
    if is_unreadable_label(truth.get("first_name")) or is_unreadable_label(truth.get("last_name")):
        return False
    got = (normalize_value("first_name", first), normalize_value("last_name", last))
    return bool(t_first and t_last) and got == (t_last, t_first)


def compare_row(truth: dict, values: dict) -> dict:
    """field -> True/False for one form (an UNREADABLE label never matches here: see _outcome).
    First and last name are compared order-insensitively (see names_swapped)."""
    result = {k: (not is_unreadable_label(truth.get(k)))
              and normalize_value(k, truth.get(k)) == normalize_value(k, values.get(k)) for k in FIELD_KEYS}
    if names_swapped(truth, values.get("first_name"), values.get("last_name")):
        result.update({k: True for k in NAME_FIELDS})
    return result


def _outcome(ok: bool, got, flagged: bool, has_record: bool, unreadable: bool = False) -> str:
    """One of OUTCOMES. A flagged field is shown to the reviewer in orange, so a wrong value there is
    caught; only a wrong, unflagged value is silent."""
    if not has_record:
        return "no_record"
    if unreadable:
        # Nobody can read it: empty + flagged is the right answer; a value is a guess.
        if got:
            return "wrong_flagged" if flagged else "wrong_silent"
        return "correct_flagged" if flagged else "empty_unflagged"
    if ok:
        return "correct_flagged" if flagged else "correct_unflagged"
    if not got:
        return "empty_flagged" if flagged else "empty_unflagged"
    return "wrong_flagged" if flagged else "wrong_silent"


def summarize(results: list) -> dict:
    """results: [(truth_row, values_or_None[, suggestions[, flags]])] ->
    {"per_field": {k: (ok, total)}, "breakdown": {k: {outcome: n}}, "suggestions": {k: (correct, wrong)},
     "failures": [...]}.

    A suggestion is judged only where the approved value is empty; it never makes a field correct.
    flags (field -> reason) splits correct / wrong values into flagged and not flagged.
    """
    per_field = OrderedDict((k, [0, 0]) for k in FIELD_KEYS)
    breakdown = OrderedDict((k, OrderedDict((o, 0) for o in OUTCOMES)) for k in FIELD_KEYS)
    sugg = OrderedDict((k, [0, 0]) for k in FIELD_KEYS)
    failures = []
    for item in results:
        truth, values = item[0], item[1]
        suggestions = (item[2] if len(item) > 2 else None) or {}
        flags = (item[3] if len(item) > 3 else None) or {}
        outcome = compare_row(truth, values) if values is not None else {k: False for k in FIELD_KEYS}
        for k, ok in outcome.items():
            got = "" if values is None else values.get(k, "")
            result = _outcome(ok, normalize_value(k, got), k in flags, values is not None,
                              unreadable=is_unreadable_label(truth.get(k)))
            ok = result in CORRECT_OUTCOMES
            per_field[k][1] += 1
            per_field[k][0] += ok
            breakdown[k][result] += 1
            suggestion = suggestions.get(k) if values is not None and not got else None
            suggestion_ok = None
            if suggestion:
                suggestion_ok = normalize_value(k, truth.get(k)) == normalize_value(k, suggestion) or (
                    k in NAME_FIELDS and names_swapped(truth, *(values.get(n) or suggestions.get(n)
                                                                for n in NAME_FIELDS)))
                sugg[k][0 if suggestion_ok else 1] += 1
            if not ok:
                failures.append({"source_file": truth["source_file"], "page": truth["page_numbers"], "field": k,
                                 "expected": truth.get(k, ""), "got": got, "outcome": result,
                                 "suggestion": suggestion or "", "suggestion_ok": suggestion_ok})
    return {"per_field": {k: tuple(v) for k, v in per_field.items()},
            "breakdown": {k: dict(v) for k, v in breakdown.items()},
            "suggestions": {k: tuple(v) for k, v in sugg.items()}, "failures": failures}


# ------------------------------------------------------------------ running the engine

def load_ground_truth(path: Path) -> list:
    """Labelled rows. Optional columns (older files lack them): tags (e.g. "handwritten;multi") and
    split ("tune" / "test"; anything else counts as "unassigned")."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("source_file")]
    for row in rows:
        row["tags"] = row.get("tags") or ""
        row["split"] = row.get("split") or ""
    return rows


def row_tags(row: dict) -> list:
    return [t.strip().lower() for t in re.split(r"[;|,]", row.get("tags") or "") if t.strip()]


def row_split(row: dict) -> str:
    split = (row.get("split") or "").strip().lower()
    return split if split in SPLITS else "unassigned"


def select_rows(rows: list, split: Optional[str] = None, tag: Optional[str] = None) -> list:
    return [r for r in rows if (split is None or row_split(r) == split.lower())
            and (tag is None or tag.lower() in row_tags(r))]


def summarize_groups(results: list) -> dict:
    """{"split": {name: summary}, "tag": {name: summary}} over summarize()-style result items.
    A row with several tags counts in each of its tag groups."""
    by_split, by_tag = OrderedDict(), OrderedDict()
    for item in results:
        by_split.setdefault(row_split(item[0]), []).append(item)
        for tag in row_tags(item[0]) or ["untagged"]:
            by_tag.setdefault(tag, []).append(item)
    return {"split": {k: summarize(v) for k, v in sorted(by_split.items())},
            "tag": {k: summarize(v) for k, v in sorted(by_tag.items())}}


def _subset_pdf(source: Path, pages: list) -> bytes:
    from pdf2image import convert_from_path

    images = []
    for page in pages:
        images += convert_from_path(str(source), dpi=config.PDF_DPI, first_page=page, last_page=page,
                                    **config.PDF2IMAGE_KWARGS)
    buf = BytesIO()
    images[0].save(buf, "PDF", save_all=True, append_images=images[1:], resolution=config.PDF_DPI)
    return buf.getvalue()


def row_pages(row: dict) -> list:
    """All pages of one labelled form, e.g. "4,5" -> [4, 5] (multi-page forms keep every page)."""
    return [int(p) for p in re.split(r"[,;\s]+", row["page_numbers"].strip()) if p]


def run_engine(truth_rows: list, progress=None) -> list:
    """Run the pipeline on the listed pages. Returns [(truth_row, record_or_None)] in CSV order.

    Each file's labelled pages (every page of every row, in row order) form one subset PDF, so the
    splitter still has to find where each form starts. A row is matched to the record that starts
    at its first page; a form split differently than labelled shows up as wrong or missing fields.
    """
    from src.record_builder import process_files

    by_file = OrderedDict()
    for row in truth_rows:
        by_file.setdefault(row["source_file"], []).append(row)
    out = {}
    for source_file, rows in by_file.items():
        pages = [p for r in rows for p in row_pages(r)]
        records = process_files([(source_file, _subset_pdf(SAMPLES_DIR / source_file, pages))], progress)
        # Map each record back to the original page number of its first subset page.
        for rec in records:
            first_subset_page = int(re.split(r"[,;\s]+", rec.metadata["page_numbers"].strip())[0])
            out[(source_file, pages[first_subset_page - 1])] = rec
    return [(r, out.get((r["source_file"], row_pages(r)[0]))) for r in truth_rows]


# ------------------------------------------------------------------ request guard

_local = threading.local()


def new_stats() -> dict:
    return {"cache_misses": 0, "misses_by_stage": Counter(), "live_requests": 0,
            "live_by_stage": Counter(), "budget_exhausted": False}


def install_request_guard(stats: dict, cache_only: bool = False, max_requests: Optional[int] = None) -> None:
    """Count every request that would reach the API, per stage. --cache-only refuses all of them;
    --max-requests refuses them after the budget (as a quota error, so the run stops sending)."""
    from src import extraction_engine as ee

    for key, value in new_stats().items():
        stats.setdefault(key, value)
    original_generate_json, original_call = ee.generate_json, ee._call_model
    lock = threading.Lock()

    def generate_json(*args, **kwargs):
        _local.stage = kwargs.get("purpose", "other")
        return original_generate_json(*args, **kwargs)

    def call(contents, cfg):
        stage = getattr(_local, "stage", "other")
        with lock:
            if cache_only:
                stats["cache_misses"] += 1
                stats["misses_by_stage"][stage] += 1
                refusal = ee.LLMError("not in dev cache (--cache-only: no live API call made)")
            elif max_requests is not None and stats["live_requests"] >= max_requests:
                stats["budget_exhausted"] = True
                refusal = ee.LLMQuotaError(f"--max-requests {max_requests} reached: no further live calls")
            else:
                stats["live_requests"] += 1
                stats["live_by_stage"][stage] += 1
                refusal = None
        if refusal is not None:
            raise refusal
        return original_call(contents, cfg)

    if cache_only:
        config.DEV_CACHE = True
    ee.generate_json = generate_json
    ee._call_model = call


def _block_live_calls(stats: dict) -> None:
    """--cache-only: answer from data/dev_cache/ only; any request that would go to the API fails."""
    install_request_guard(stats, cache_only=True)


# ------------------------------------------------------------------ report

def reread_outcomes(records: list) -> dict:
    """field -> Counter of targeted re-read outcomes (codes only) over the evaluated records."""
    out = OrderedDict()
    for rec in records:
        for key, d in (getattr(rec, "diagnostics", None) or {}).get("fields", {}).items():
            if d.get("reread") not in (None, "n/a"):
                out.setdefault(key, Counter())[d["reread"]] += 1
    return out


def print_report(summary: dict, show_values: bool = False) -> None:
    print("\nPer-field accuracy (exact match after normalization):")
    total_ok = total = 0
    for key, (ok, n) in summary["per_field"].items():
        total_ok, total = total_ok + ok, total + n
        print(f"  {key:20s} {FIELD_HEADERS[key]:16s} {ok}/{n}  ({100 * ok / n:.0f}%)" if n else f"  {key}: -")
    if total:
        print(f"  {'ALL FIELDS':20s} {'':16s} {total_ok}/{total}  ({100 * total_ok / total:.0f}%)")
    if summary.get("breakdown"):
        print("\nBreakdown (correct = right value; flagged = shown for review):")
        top = ("correct", "correct", "wrong", "WRONG", "empty", "empty", "no")
        bottom = ("unflag.", "flagged", "flagged", "SILENT", "flagged", "unflag.", "record")
        print(f"  {'field':20s} " + " ".join(f"{t:>8s}" for t in top))
        print(f"  {'':20s} " + " ".join(f"{t:>8s}" for t in bottom))
        for key, b in summary["breakdown"].items():
            print(f"  {key:20s} " + " ".join(f"{b.get(o, 0):>8d}" for o in OUTCOMES))
    sugg = {k: v for k, v in summary.get("suggestions", {}).items() if sum(v)}
    if sugg:
        print("\nUnverified suggestions (not counted as correct values):")
        for key, (good, bad) in sugg.items():
            print(f"  {key:20s} correct suggestions: {good}   wrong suggestions: {bad}")
    print("\nFailed fields:" if summary["failures"] else "\nNo failed fields.")
    for f in summary["failures"]:
        if f["field"] in PERSONAL_FIELDS or not show_values:
            detail = "mismatch" if (f["expected"] or f["got"]) else ""
            if f.get("suggestion"):
                detail += f" (suggestion {'correct' if f['suggestion_ok'] else 'wrong'})"
        else:
            detail = f"expected={f['expected'] or '—'} got={f['got'] or '—'}"
        if f.get("outcome"):
            detail += f" [{f['outcome']}]"
        print(f"  {f['source_file']} p{f['page']} {f['field']}: {detail}")


def _group_counts(summary: dict) -> dict:
    per_field = summary["per_field"]
    totals = Counter()
    for b in summary["breakdown"].values():
        totals.update(b)
    return {"rows": next(iter(per_field.values()))[1] if per_field else 0,
            "ok": sum(ok for ok, _ in per_field.values()), "total": sum(n for _, n in per_field.values()),
            "silent_errors": totals["wrong_silent"], "missing": totals["empty_flagged"] + totals["empty_unflagged"],
            "per_field": {k: list(v) for k, v in per_field.items()}}


def print_groups(groups: dict) -> None:
    for kind, label in (("split", "Per split"), ("tag", "Per tag (a row counts once per tag)")):
        if not groups.get(kind):
            continue
        print(f"\n{label}:")
        for name, summary in groups[kind].items():
            c = _group_counts(summary)
            fields = " ".join(f"{k.split('_')[0]}={ok}/{n}" for k, (ok, n) in c["per_field"].items())
            print(f"  {name:14s} rows={c['rows']:<3d} all={c['ok']}/{c['total']}  silent_errors={c['silent_errors']}"
                  f"  missing={c['missing']}   {fields}")


def labels_fingerprint(truth_rows: list) -> str:
    """SHA-256 over all labelled rows (not reversible to single values): same labels <=> same hash."""
    h = hashlib.sha256()
    for row in truth_rows:
        h.update(json.dumps([row.get(k, "") for k in ("source_file", "page_numbers", *FIELD_KEYS)],
                            ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


# Claude configuration of a run, recorded in the summary and printed by a comparison.
CONFIG_KEYS = ("CLAUDE_MODEL", "CLAUDE_EFFORT", "HIGHLIGHT_CROPS_ENABLED", "STRICT_VALIDATION", "EVAL_MODE",
               "FIELD_REREAD_ENABLED", "CROP_SOURCE", "NAME_CHECK_MODEL", "NAME_CHECK_EFFORT", "NAME_DICTIONARY_ENABLED",
               "BATCH_NAME_SHARING", "NAME_CROP_ENHANCE")


def run_config() -> dict:
    """The engine settings that change results (absent in older engine trees: "n/a")."""
    return {k.lower(): getattr(config, k, "n/a") for k in CONFIG_KEYS}


def run_info(stats: dict, records: list, cache_only: bool) -> dict:
    """Settings and completeness of one run: codes, counts and settings only."""
    try:
        commit = subprocess.run(["git", "-C", str(ENGINE_ROOT), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ENGINE_ROOT), "status", "--porcelain", "--", "src", "prompts"],
                                    capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unknown", False
    errors = sum(1 for r in records if r is None or getattr(r, "error", None))
    return {
        "engine_commit": commit + ("+uncommitted" if dirty else ""),
        "config": run_config(),
        "field_reread": getattr(config, "FIELD_REREAD_ENABLED", "n/a"),
        "crop_source": getattr(config, "CROP_SOURCE", "n/a"),
        "cache_only": cache_only,
        "cache_misses": dict(stats["misses_by_stage"]),
        "live_requests": dict(stats["live_by_stage"]),
        "budget_exhausted": stats["budget_exhausted"],
        "forms_failed_or_missing": errors,
        "complete": not (stats["cache_misses"] or stats["budget_exhausted"] or errors),
    }


def counts_only(summary: dict, outcomes: Optional[dict] = None, truth_rows: Optional[list] = None,
                info: Optional[dict] = None, groups: Optional[dict] = None) -> dict:
    """What --save-summary writes: counts, (file, page, field) of failures, fingerprints - no values."""
    return {"per_field": {k: list(v) for k, v in summary["per_field"].items()},
            "groups": {kind: {name: _group_counts(s) for name, s in g.items()} for kind, g in (groups or {}).items()},
            "breakdown": summary.get("breakdown", {}),
            "failed": sorted([f["source_file"], str(f["page"]), f["field"]] for f in summary["failures"]),
            "reread": {k: dict(v) for k, v in (outcomes or {}).items()},
            "rows": [[r["source_file"], r["page_numbers"]] for r in truth_rows or []],
            "labels_sha256": labels_fingerprint(truth_rows) if truth_rows else None,
            "run": info or {}}


def print_comparison(baseline: dict, current: dict) -> bool:
    """Print the per-field change between two runs on the same rows and labels - e.g. two Claude
    configurations (crops on/off, model, effort), whose settings are printed. Returns False (and prints
    why) when the runs are not comparable."""
    problems = []
    if baseline.get("labels_sha256") != current.get("labels_sha256") or baseline.get("rows") != current.get("rows"):
        problems.append("different ground-truth rows or labels")
    b_run, c_run = baseline.get("run", {}), current.get("run", {})
    for name, run in (("baseline", b_run), ("current", c_run)):
        if run and not run.get("complete", False):
            problems.append(f"{name} run is incomplete (cache misses, request budget or failed forms)")
    if problems:
        print("\nNOT COMPARABLE: " + "; ".join(problems))
        return False
    print(f"\nComparison: baseline {b_run.get('engine_commit', '?')} -> current {c_run.get('engine_commit', '?')}"
          " (same rows and labels):")
    b_cfg, c_cfg = b_run.get("config", {}), c_run.get("config", {})
    changed = {k: (b_cfg.get(k, "n/a"), c_cfg.get(k, "n/a")) for k in sorted(set(b_cfg) | set(c_cfg))
               if b_cfg.get(k, "n/a") != c_cfg.get(k, "n/a")}
    print("  settings: " + (", ".join(f"{k} {b} -> {c}" for k, (b, c) in changed.items()) or "identical"))
    for key, (ok, n) in current["per_field"].items():
        b_ok, _ = baseline["per_field"].get(key, (0, 0))
        b, c = baseline.get("breakdown", {}).get(key, {}), current.get("breakdown", {}).get(key, {})
        extra = "  ".join(f"{o} {b.get(o, 0)}->{c.get(o, 0)}" for o in OUTCOMES[1:] if b.get(o, 0) or c.get(o, 0))
        print(f"  {key:20s} {b_ok}/{n} -> {ok}/{n} ({ok - b_ok:+d})   {extra}")
    base_failed = {tuple(f) for f in baseline.get("failed", [])}
    now_failed = {tuple(f) for f in current["failed"]}
    for label, items in (("Fixed", base_failed - now_failed), ("New failures", now_failed - base_failed)):
        for source, page, key in sorted(items):
            print(f"  {label}: {source} p{page} {key}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--cache-only", action="store_true", help="never call the API (dev cache only)")
    parser.add_argument("--max-requests", type=int, help="stop after this many live API requests")
    parser.add_argument("--engine-root", type=Path, help="evaluate the engine code of another tree (see doc)")
    parser.add_argument("--save-summary", type=Path, help="write counts (no values) to this JSON file")
    parser.add_argument("--compare", type=Path, help="baseline summary JSON from --save-summary")
    parser.add_argument("--show-values", action="store_true", help="print values of non-personal fields")
    parser.add_argument("--split", choices=SPLITS + ("unassigned",), help="evaluate only rows of this split")
    parser.add_argument("--tag", help="evaluate only rows carrying this tag")
    args = parser.parse_args()
    truth_rows = select_rows(load_ground_truth(args.ground_truth), args.split, args.tag)
    if not truth_rows:
        sys.exit(f"No matching rows in {args.ground_truth}")
    stats = new_stats()
    install_request_guard(stats, cache_only=args.cache_only, max_requests=args.max_requests)
    results = run_engine(truth_rows, progress=lambda s, d, t: print(f"  [{s}] {d}/{t}"))
    records = [rec for _, rec in results]
    items = [(t, rec.values if rec else None, rec.suggestions if rec else None, rec.flags if rec else None)
             for t, rec in results]
    summary, groups = summarize(items), summarize_groups(items)
    print_report(summary, show_values=args.show_values)
    print_groups(groups)
    outcomes = reread_outcomes(records)
    if outcomes:
        print("\nTargeted re-read outcomes (per field):")
        for key, counter in outcomes.items():
            print(f"  {key:20s} " + ", ".join(f"{k}={v}" for k, v in sorted(counter.items())))
    info = run_info(stats, records, args.cache_only)
    print(f"\nEngine: {info['engine_commit']}  " + "  ".join(f"{k}={v}" for k, v in info["config"].items()))
    if args.cache_only:
        print(f"Cache-only run: 0 live API calls; uncached requests per stage: {info['cache_misses'] or 'none'}")
    else:
        from src import extraction_engine as ee

        print(f"Live API requests per stage: {info['live_requests'] or 'none'}"
              f"{'  (BUDGET REACHED)' if info['budget_exhausted'] else ''}")
        total_cost, priced = 0.0, True
        for stage, usage in ee.get_usage_by_purpose().items():
            cost = stage_cost(usage)
            priced = priced and cost is not None
            total_cost += cost or 0.0
            print(f"  tokens {stage:14s} requests={usage['requests']} prompt={usage['prompt_tokens']} "
                  f"(cache read={usage.get('cache_read_tokens', 0)} write={usage.get('cache_write_tokens', 0)}) "
                  f"output={usage['output_tokens']} time={usage.get('seconds', 0):.0f}s"
                  + (f" cost=${cost:.4f}" if cost is not None else ""))
        if priced and records:
            print(f"  total cost ${total_cost:.4f} for {len(records)} form(s): ${total_cost / len(records):.4f} per form")
    if not info["complete"]:
        print("Results are INCOMPLETE (cache misses, request budget or failed forms): do not compare them.")
    current = counts_only(summary, outcomes, truth_rows, info, groups)
    if args.save_summary:
        args.save_summary.write_text(json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.compare:
        print_comparison(json.loads(args.compare.read_text(encoding="utf-8")), current)


def stage_cost(usage: dict) -> Optional[float]:
    """USD for one stage's usage, summed over the models that answered it; None if any price is unknown."""
    from src.claude_client import cost_usd

    models = usage.get("models") or {}
    if len(models) != 1:  # per-model token split is not tracked; price only single-model stages
        return None
    return cost_usd(next(iter(models)), usage)


if __name__ == "__main__":
    main()
