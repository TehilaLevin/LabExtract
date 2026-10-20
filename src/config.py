"""Central configuration: loads .env, model name, confidence threshold, paths."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value == "your_key_here":
        raise RuntimeError(
            f"Missing {name}. Set it in the .env file (see .env.example)."
        )
    return value


def __getattr__(name: str) -> str:
    # The API key is validated lazily (PEP 562), so modules that never call the API (converters,
    # validators, tests) import without a key, while `config.CLAUDE_API_KEY` raises a clear error.
    if name == "CLAUDE_API_KEY":  # ANTHROPIC_API_KEY (the SDK's own name) is accepted too
        if not os.getenv("CLAUDE_API_KEY", "").strip() and os.getenv("ANTHROPIC_API_KEY", "").strip():
            return _require("ANTHROPIC_API_KEY")
        return _require(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


CONFIDENCE_THRESHOLD = 0.8
POPPLER_PATH = os.getenv("POPPLER_PATH") or None
# Extra kwargs for pdf2image.convert_from_path; empty when Poppler is already on PATH.
PDF2IMAGE_KWARGS = {"poppler_path": POPPLER_PATH} if POPPLER_PATH else {}
PDF_DPI = 300

# Claude (the only model provider): model id and effort ("low" | "medium" | "high" | "xhigh" | "max"). Opus reads handwriting
# best; CLAUDE_MODEL=claude-sonnet-5-5 is about 3x cheaper per form. Every call type uses this model;
# orientation and splitting run at low effort, extraction / name check / re-read at CLAUDE_EFFORT.
CLAUDE_MODEL = (os.getenv("CLAUDE_MODEL") or "claude-opus-5-5").strip()
CLAUDE_EFFORT = (os.getenv("CLAUDE_EFFORT") or "high").strip().lower()
CLAUDE_TIMEOUT_S = 180      # per attempt; thinking at high effort can take a while
CLAUDE_MAX_RETRIES = 4      # SDK retries with exponential backoff: 429, 529, 5xx, connection errors
CLAUDE_MAX_TOKENS = 16000   # output incl. thinking; the JSON answers themselves are ~1-3k tokens

INVALID_JSON_RETRIES = 1    # extra attempts when the model returns unparsable JSON
MAX_PARALLEL_REQUESTS = 4
# Longest side (px) of images sent to the model: enough for handwriting, smaller payload.
# claude_client shrinks further when needed to stay within the per-image limits.
EXTRACTION_IMAGE_MAX_SIDE = 2400
SPLIT_IMAGE_MAX_SIDE = 1400


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    return default if not value else value in ("1", "true", "yes")


# Targeted re-read of flagged number fields from high-resolution crops (at most one extra
# request per form). FIELD_REREAD=false turns it off, e.g. for a baseline evaluation run.
FIELD_REREAD_ENABLED = _env_flag("FIELD_REREAD", True)
# Which page version name / field crops are cut from: "original" (scanned pixels, only the
# deskew geometry applied, so boxes still line up) or "processed" (contrast/brightness fixed,
# the version the full-page pass saw). CROP_SOURCE=processed switches back for comparison.
CROP_SOURCE = os.getenv("CROP_SOURCE", "original").strip().lower()

# Highlight crops: highlighted regions are cut from the 300 DPI page, upscaled and sent with the full page
# in the extraction request, plus a cleaned dark-ink version. Additive: a page without highlighting is
# extracted exactly as before. HIGHLIGHT_CROPS=false turns it off.
HIGHLIGHT_CROPS_ENABLED = _env_flag("HIGHLIGHT_CROPS", True)
# DEV ONLY: save the crops as PNG under data/temp/highlight_debug/ (git-ignored) to inspect them. The
# folder is emptied at the start of every run and by clear_debug_crops().
HIGHLIGHT_DEBUG = _env_flag("HIGHLIGHT_DEBUG", False)

# Extra validation after extraction. STRICT_VALIDATION=false restores the earlier checks (for comparisons).
STRICT_VALIDATION = _env_flag("STRICT_VALIDATION", True)
# Central (GLC) numbers are 6 digits; a different prefix is a warning flag, never a rejection.
CENTRAL_NUMBER_PREFIXES = tuple(p.strip() for p in (os.getenv("CENTRAL_NUMBER_PREFIXES") or "93,94").split(",")
                                if p.strip())
# A received date older than this many days is flagged (the earlier check allowed one year).
RECEIVED_DATE_MAX_AGE_DAYS = int(os.getenv("RECEIVED_DATE_MAX_AGE_DAYS") or 90)
# The received date must be 0..N days after the sample collection date, when the form states one.
RECEIVED_AFTER_COLLECTION_MAX_DAYS = int(os.getenv("RECEIVED_AFTER_COLLECTION_MAX_DAYS") or 14)
# EVALUATION ONLY: skip the "not older than RECEIVED_DATE_MAX_AGE_DAYS before today" check, because the
# sample forms are months old. Every other date check (valid, not in the future, collection gap) stays.
EVAL_MODE = _env_flag("EVAL_MODE", False)

DATA_DIR = BASE_DIR / "data"
# DEV ONLY - never enable in production. Caches model responses on disk (they may
# contain personal data) so repeated development runs on the same images cost nothing.
DEV_CACHE = _env_flag("DEV_CACHE", False)
DEV_CACHE_DIR = DATA_DIR / "dev_cache"
TEMP_DIR = DATA_DIR / "temp"
HIGHLIGHT_DEBUG_DIR = TEMP_DIR / "highlight_debug"
OUTPUT_DIR = DATA_DIR / "output"
# Local SQLite store of reviewed records (src/record_store.py). It holds personal data: git-ignored, local only.
# DB_ENABLED=false turns it off.
DB_ENABLED = _env_flag("DB_ENABLED", True)
DB_PATH = DATA_DIR / "db" / "labextract.db"
REFERENCE_DIR = DATA_DIR / "reference"
SENDERS_PATH = REFERENCE_DIR / "approved_senders.json"
PROMPT_PATH = BASE_DIR / "prompts" / "extraction_prompt.txt"
