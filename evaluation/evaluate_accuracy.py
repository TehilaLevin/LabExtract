"""Compare engine output against data/ground_truth/ground_truth.csv, per field.

Usage:
    .venv/Scripts/python evaluation/evaluate_accuracy.py [--ground-truth PATH]

Runs the full pipeline on exactly the (source_file, page) rows listed in the CSV (pages are
taken from data/samples/), then prints per-field accuracy (exact match after normalization)
and the list of failed fields. Every field counts, names included. An empty expected value
means the correct answer is "no value": it matches only an empty engine value.
Unverified suggestions (handwritten names) never count as correct values; they are counted
separately as correct or wrong suggestions.

Privacy: personal values (names, ID numbers) are never printed - a failure shows only
"mismatch" for them. Live API calls are made (use DEV_CACHE=true during development).
"""
import argparse
import csv
import re
import sys
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.constants import FIELD_HEADERS, FIELD_KEYS, LIST_FIELDS  # noqa: E402
from src.validators import match_sending_entity, normalize_date, normalize_number_list  # noqa: E402

DEFAULT_GROUND_TRUTH = config.DATA_DIR / "ground_truth" / "ground_truth.csv"
SAMPLES_DIR = config.DATA_DIR / "samples"
PERSONAL_FIELDS = frozenset({"first_name", "last_name", "id_number"})
_NAME_PUNCT_RE = re.compile(r"[\"'`׳״.,:;()\-]")


# ------------------------------------------------------------------ comparison

def normalize_value(key: str, value) -> object:
    """Comparable form of a field value; empty values normalize to "" / frozenset()."""
    if key in LIST_FIELDS:
        return frozenset(n.replace(" ", "") for n in normalize_number_list(value))
    text = (value or "").strip()
    if not text:
        return ""
    if key in ("first_name", "last_name"):
        return " ".join(_NAME_PUNCT_RE.sub(" ", text).split())
    if key == "id_number":
        return re.sub(r"\D", "", text)
    if key == "received_date":
        return normalize_date(text) or text
    if key == "sending_entity":
        return match_sending_entity(text, config.SENDERS_PATH) or text
    return text


def compare_row(truth: dict, values: dict) -> dict:
    """field -> True/False for one form."""
    return {k: normalize_value(k, truth.get(k)) == normalize_value(k, values.get(k)) for k in FIELD_KEYS}


def summarize(results: list) -> dict:
    """results: [(truth_row, values_or_None[, suggestions])] ->
    {"per_field": {k: (ok, total)}, "suggestions": {k: (correct, wrong)}, "failures": [...]}.

    A suggestion is judged only where the approved value is empty; it never makes a field correct.
    """
    per_field = OrderedDict((k, [0, 0]) for k in FIELD_KEYS)
    sugg = OrderedDict((k, [0, 0]) for k in FIELD_KEYS)
    failures = []
    for item in results:
        truth, values = item[0], item[1]
        suggestions = (item[2] if len(item) > 2 else None) or {}
        outcome = compare_row(truth, values) if values is not None else {k: False for k in FIELD_KEYS}
        for k, ok in outcome.items():
            per_field[k][1] += 1
            per_field[k][0] += ok
            suggestion = suggestions.get(k) if values is not None and not values.get(k) else None
            suggestion_ok = None
            if suggestion:
                suggestion_ok = normalize_value(k, truth.get(k)) == normalize_value(k, suggestion)
                sugg[k][0 if suggestion_ok else 1] += 1
            if not ok:
                failures.append({"source_file": truth["source_file"], "page": truth["page_numbers"], "field": k,
                                 "expected": truth.get(k, ""), "got": "" if values is None else values.get(k, ""),
                                 "suggestion": suggestion or "", "suggestion_ok": suggestion_ok})
    return {"per_field": {k: tuple(v) for k, v in per_field.items()},
            "suggestions": {k: tuple(v) for k, v in sugg.items()}, "failures": failures}


# ------------------------------------------------------------------ running the engine

def load_ground_truth(path: Path) -> list:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("source_file")]


def _subset_pdf(source: Path, pages: list) -> bytes:
    from pdf2image import convert_from_path

    images = []
    for page in pages:
        images += convert_from_path(str(source), dpi=config.PDF_DPI, first_page=page, last_page=page,
                                    **config.PDF2IMAGE_KWARGS)
    buf = BytesIO()
    images[0].save(buf, "PDF", save_all=True, append_images=images[1:], resolution=config.PDF_DPI)
    return buf.getvalue()


def run_engine(truth_rows: list, progress=None) -> list:
    """Run the pipeline on the listed pages. Returns [(truth_row, record_or_None)] in CSV order."""
    from src.record_builder import process_files

    by_file = OrderedDict()
    for row in truth_rows:
        by_file.setdefault(row["source_file"], []).append(row)
    out = {}
    for source_file, rows in by_file.items():
        pages = [int(r["page_numbers"].split(",")[0]) for r in rows]
        records = process_files([(source_file, _subset_pdf(SAMPLES_DIR / source_file, pages))], progress)
        # Map each record back to the original page number of its first subset page.
        for rec in records:
            first_subset_page = int(rec.metadata["page_numbers"].split(",")[0])
            out[(source_file, pages[first_subset_page - 1])] = rec
    return [(r, out.get((r["source_file"], int(r["page_numbers"].split(",")[0])))) for r in truth_rows]


# ------------------------------------------------------------------ report

def print_report(summary: dict) -> None:
    print("\nPer-field accuracy (exact match after normalization):")
    total_ok = total = 0
    for key, (ok, n) in summary["per_field"].items():
        total_ok, total = total_ok + ok, total + n
        print(f"  {key:20s} {FIELD_HEADERS[key]:16s} {ok}/{n}  ({100 * ok / n:.0f}%)" if n else f"  {key}: -")
    if total:
        print(f"  {'ALL FIELDS':20s} {'':16s} {total_ok}/{total}  ({100 * total_ok / total:.0f}%)")
    sugg = {k: v for k, v in summary.get("suggestions", {}).items() if sum(v)}
    if sugg:
        print("\nUnverified suggestions (not counted as correct values):")
        for key, (good, bad) in sugg.items():
            print(f"  {key:20s} correct suggestions: {good}   wrong suggestions: {bad}")
    print("\nFailed fields:" if summary["failures"] else "\nNo failed fields.")
    for f in summary["failures"]:
        if f["field"] in PERSONAL_FIELDS:
            detail = "mismatch" if (f["expected"] or f["got"]) else ""
            if f.get("suggestion"):
                detail += f" (suggestion {'correct' if f['suggestion_ok'] else 'wrong'})"
        else:
            detail = f"expected={f['expected'] or '—'} got={f['got'] or '—'}"
        print(f"  {f['source_file']} p{f['page']} {f['field']}: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    args = parser.parse_args()
    truth_rows = load_ground_truth(args.ground_truth)
    if not truth_rows:
        sys.exit(f"No rows in {args.ground_truth}")
    results = run_engine(truth_rows, progress=lambda s, d, t: print(f"  [{s}] {d}/{t}"))
    print_report(summarize([(t, rec.values if rec else None, rec.suggestions if rec else None)
                            for t, rec in results]))


if __name__ == "__main__":
    main()
