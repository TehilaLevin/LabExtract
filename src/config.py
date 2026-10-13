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
    # GEMINI_API_KEY / GEMINI_MODEL are validated lazily (PEP 562), so modules that
    # never call the API (converters, validators, tests) import without a key, while
    # `from src.config import GEMINI_API_KEY` still raises a clear error when missing.
    if name in ("GEMINI_API_KEY", "GEMINI_MODEL"):  # no hardcoded model fallback on purpose
        return _require(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


CONFIDENCE_THRESHOLD = 0.8
POPPLER_PATH = os.getenv("POPPLER_PATH") or None
# Extra kwargs for pdf2image.convert_from_path; empty when Poppler is already on PATH.
PDF2IMAGE_KWARGS = {"poppler_path": POPPLER_PATH} if POPPLER_PATH else {}
PDF_DPI = 300

# Gemini request behaviour.
REQUEST_TIMEOUT_S = 120
MAX_ATTEMPTS = 3            # per request, for transient API errors (429/5xx/timeouts)
INVALID_JSON_RETRIES = 1    # extra attempts when the model returns unparsable JSON
MAX_PARALLEL_REQUESTS = 4
# Longest side (px) of images sent to Gemini: enough for handwriting, smaller payload.
EXTRACTION_IMAGE_MAX_SIDE = 2400
SPLIT_IMAGE_MAX_SIDE = 1400

DATA_DIR = BASE_DIR / "data"
# DEV ONLY - never enable in production. Caches Gemini responses on disk (they may
# contain personal data) so repeated development runs on the same images cost nothing.
DEV_CACHE = os.getenv("DEV_CACHE", "").strip().lower() in ("1", "true", "yes")
DEV_CACHE_DIR = DATA_DIR / "dev_cache"
TEMP_DIR = DATA_DIR / "temp"
OUTPUT_DIR = DATA_DIR / "output"
REFERENCE_DIR = DATA_DIR / "reference"
SENDERS_PATH = REFERENCE_DIR / "approved_senders.json"
PROMPT_PATH = BASE_DIR / "prompts" / "extraction_prompt.txt"
