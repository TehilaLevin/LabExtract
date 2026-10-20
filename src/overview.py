"""Activity overview: how many processed forms came from each sending entity (pure Python, no Streamlit).

Input is a {sending_entity: form count} mapping taken from the stored records, so it only ever holds the
organization names the engine and reviewer already produced - no patient data. Each entity is classified
with the closed list in approved_senders.json: health fund, hospital, or "other" (a value that is not on
the list). Forms without a sender are counted separately. The list has a city per hospital but no
coordinates, so the overview is a chart, not a map: no location is guessed.
"""
import json
from pathlib import Path
from typing import Optional, Union

from src import config

KIND_HMO = "hmo"
KIND_HOSPITAL = "hospital"
KIND_OTHER = "other"
KIND_UNKNOWN = "unknown"
KIND_LABELS = {KIND_HMO: "קופת חולים", KIND_HOSPITAL: "בית חולים", KIND_OTHER: "גורם אחר (לא ברשימה)",
               KIND_UNKNOWN: "ללא גורם שולח"}
UNKNOWN_NAME = "לא זוהה"


def sender_kinds(senders_path: Union[str, Path, None] = None) -> dict:
    """{canonical name: KIND_HMO | KIND_HOSPITAL} from approved_senders.json."""
    with open(senders_path or config.SENDERS_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    kinds = {h["name"]: KIND_HOSPITAL for h in data.get("hospitals", [])}
    kinds.update({h["name"]: KIND_HMO for h in data.get("hmos", [])})
    return kinds


def breakdown(counts: dict, senders_path: Union[str, Path, None] = None) -> list:
    """Rows {name, kind, kind_label, count, percent} sorted by count (then name), percent of all forms.

    An empty sender is one "לא זוהה" row; a sender not on the approved list keeps its own name as kind
    "other". Counts of the same name are added. No forms -> empty list.
    """
    kinds = sender_kinds(senders_path)
    merged = {}
    for name, count in counts.items():
        name = " ".join(str(name or "").split())
        if count and count > 0:
            merged[name] = merged.get(name, 0) + count
    total = sum(merged.values())
    rows = []
    for name, count in merged.items():
        kind = KIND_UNKNOWN if not name else kinds.get(name, KIND_OTHER)
        rows.append({"name": name or UNKNOWN_NAME, "kind": kind, "kind_label": KIND_LABELS[kind],
                     "count": count, "percent": round(100 * count / total, 1)})
    return sorted(rows, key=lambda r: (-r["count"], r["name"]))


def totals_by_kind(rows: list) -> dict:
    """{kind: {count, percent}} for every kind (zeros included), from `breakdown` rows."""
    total = sum(r["count"] for r in rows)
    out = {}
    for kind in KIND_LABELS:
        n = sum(r["count"] for r in rows if r["kind"] == kind)
        out[kind] = {"count": n, "percent": round(100 * n / total, 1) if total else 0.0}
    return out


def known_sender_count(rows: list) -> int:
    """Distinct entities on the approved list (used by the achievements)."""
    return sum(1 for r in rows if r["kind"] in (KIND_HMO, KIND_HOSPITAL))


def counts_from_values(values) -> dict:
    """{sender: count} from a list of sender strings (the session fallback when the database is off)."""
    out: dict = {}
    for v in values:
        out[v or ""] = out.get(v or "", 0) + 1
    return out
