"""Split a batch of pages into individual forms (one form = one record).

Two Gemini steps:
1. Orientation (one request per page): the page is shown in 4 rotations (A-D) and the
   model picks the upright one. Measured on the 48 sample pages: asking the model for a
   rotation angle directly was right on 1 of 11 rotated pages (at 0.95 "confidence");
   picking among candidates was right on 16/16 per page, but 46/48 when 4 pages were
   batched in one request - so it stays one page per request.
2. Splitting (one request per SPLIT_BATCH_SIZE upright pages of the same file): does
   each page start a new form or continue the previous page?
The grouping itself (group_pages) is pure Python and unit-tested.

Safety rules (an empty flagged field beats a wrong one):
- A new input file always starts a new form (forms never span files).
- "Continuation" is accepted only with confidence >= threshold. Merging two patients
  into one record is worse than splitting one form in two, so when unsure we split
  and flag the form for review.
- If the API call fails, the page starts a new form and is flagged. No page is dropped.
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image

from src import config
from src.file_converter import Page
from src.image_preprocessor import VALID_ROTATIONS, fix_rotation

logger = logging.getLogger(__name__)

SPLIT_FLAG_UNCERTAIN = "split_uncertain"
SPLIT_FLAG_FAILED = "split_failed"

SPLIT_PROMPT = """You are helping a virology lab separate a scanned batch of patient commitment forms
("טופס התחייבות" / referral / lab request forms from Israeli HMOs and hospitals).

You get {n} numbered PAGES of one file, in order (PAGE 1 ... PAGE {n}). If a CONTEXT page is
given, it is the page that came right before PAGE 1 (already classified; do not answer for it).
Return exactly one entry per numbered page, in order, with page_index = its number:

1. is_blank: true only if the page has no form content at all (empty back side).
2. starts_new_form: true if the page is the first page of a new form; false if it continues
   the form on the page right before it (e.g. "page 2 of 2", "עמוד 2", a back side or
   attachment of the same patient's form). A different patient, a different header/logo
   or a fresh form title means a new form. For PAGE 1 without a CONTEXT page, answer true.
3. confidence: 0-1, how sure you are about starts_new_form.

Do not transcribe any text from the pages."""

SPLIT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "pages": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "page_index": {"type": "INTEGER"},
                    "is_blank": {"type": "BOOLEAN"},
                    "starts_new_form": {"type": "BOOLEAN"},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["page_index", "is_blank", "starts_new_form", "confidence"],
                "propertyOrdering": ["page_index", "is_blank", "starts_new_form", "confidence"],
            },
        }
    },
    "required": ["pages"],
}

SPLIT_BATCH_SIZE = 12
_CONTEXT_PAGE_MAX_SIDE = 800

ORIENTATION_PROMPT = """Below are 4 versions (A, B, C, D) of the SAME scanned page, each rotated differently.
Which version shows the main printed text upright: horizontal lines, readable, not upside down
(Hebrew reads right-to-left; Latin letters and digits must not be mirrored or upside down)?
Judge by printed text and table headers, not by handwriting or stamps.
confidence: 0-1, how sure you are. Do not transcribe any text."""

ORIENTATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "upright": {"type": "STRING", "enum": ["A", "B", "C", "D"]},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["upright", "confidence"],
    "propertyOrdering": ["upright", "confidence"],
}
# Candidate letter -> clockwise rotation that produced it.
_CANDIDATES = {"A": 0, "B": 90, "C": 180, "D": 270}
_ORIENTATION_IMAGE_MAX_SIDE = 700


@dataclass
class PageDecision:
    starts_new_form: bool
    confidence: float
    rotate_clockwise: int = 0
    rotation_confidence: float = 0.0
    is_blank: bool = False
    error: Optional[str] = None


@dataclass
class Form:
    index: int                      # 1-based position in the batch
    source_file: str
    pages: list = field(default_factory=list)       # list[Page]
    rotations: list = field(default_factory=list)   # clockwise degrees per page
    split_flags: list = field(default_factory=list)

    @property
    def page_numbers(self) -> list:
        return [p.page_number for p in self.pages]

    def images(self) -> list:
        """Upright page images (rotation from the split step applied)."""
        return [fix_rotation(p.load(), r) for p, r in zip(self.pages, self.rotations)]


def group_pages(pages: list, decisions: list,
                threshold: float = config.CONFIDENCE_THRESHOLD) -> list:
    """Group pages into forms using one PageDecision per page. Never drops a page."""
    if len(pages) != len(decisions):
        raise ValueError("pages and decisions must have the same length")
    forms: list = []
    current: Optional[Form] = None
    for page, decision in zip(pages, decisions):
        # A wrong rotation hurts more than none (the extractor can read sideways pages),
        # so a rotation is applied only when the orientation step is confident.
        rotation = decision.rotate_clockwise if decision.rotate_clockwise in VALID_ROTATIONS else 0
        if decision.rotation_confidence < threshold:
            rotation = 0
        same_file = current is not None and current.source_file == page.source_file
        flag = SPLIT_FLAG_FAILED if decision.error else None
        if not same_file or decision.error:
            new_form = True
        elif decision.is_blank:
            new_form = False  # a blank back side stays with its form; nothing is lost
        elif decision.starts_new_form:
            new_form = True
        elif decision.confidence >= threshold:
            new_form = False
        else:
            new_form, flag = True, SPLIT_FLAG_UNCERTAIN

        if new_form:
            current = Form(index=len(forms) + 1, source_file=page.source_file)
            forms.append(current)
        if flag and flag not in current.split_flags:
            current.split_flags.append(flag)
        current.pages.append(page)
        current.rotations.append(rotation)
    return forms


def _unit(value) -> float:
    """Clamp a model-reported confidence to [0, 1]; anything unparsable is 0."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _parse_decision(data: dict) -> PageDecision:
    return PageDecision(
        starts_new_form=bool(data.get("starts_new_form", True)),
        confidence=_unit(data.get("confidence")),
        is_blank=bool(data.get("is_blank", False)),
    )


def detect_rotation(image: Image.Image) -> tuple:
    """Return (clockwise rotation that makes the page upright, confidence).

    Raises GeminiError (incl. GeminiQuotaError) on API failure.
    """
    from src.extraction_engine import generate_json, image_part

    contents: list = [ORIENTATION_PROMPT]
    for letter, rotation in _CANDIDATES.items():
        contents += [f"Version {letter}:",
                     image_part(fix_rotation(image, rotation), _ORIENTATION_IMAGE_MAX_SIDE)]
    data = generate_json(contents, ORIENTATION_SCHEMA, high_resolution=False)
    letter = data.get("upright")
    if letter not in _CANDIDATES:
        return 0, 0.0
    return _CANDIDATES[letter], _unit(data.get("confidence"))


def classify_batch(images: list, context: Optional[Image.Image] = None) -> list:
    """Ask Gemini about consecutive pages of one file in a single request.

    Returns one PageDecision per image. Raises GeminiError (incl. GeminiQuotaError)
    on API failure, or GeminiError if the answer does not cover every page exactly once.
    """
    from src.extraction_engine import GeminiError, generate_json, image_part

    contents: list = [SPLIT_PROMPT.format(n=len(images))]
    if context is not None:
        contents += ["CONTEXT page (the page before PAGE 1):", image_part(context, _CONTEXT_PAGE_MAX_SIDE)]
    for i, img in enumerate(images, start=1):
        contents += [f"PAGE {i}:", image_part(img, config.SPLIT_IMAGE_MAX_SIDE)]
    data = generate_json(contents, SPLIT_SCHEMA, high_resolution=False)

    by_index = {}
    for item in data.get("pages") or []:
        if isinstance(item, dict) and isinstance(item.get("page_index"), int):
            by_index.setdefault(item["page_index"], item)
    if sorted(by_index) != list(range(1, len(images) + 1)):
        raise GeminiError("Split answer does not cover every page exactly once")
    return [_parse_decision(by_index[i]) for i in range(1, len(images) + 1)]


def _batches(pages: list) -> list:
    """Index ranges of consecutive pages of the same file, at most SPLIT_BATCH_SIZE long."""
    batches, start = [], 0
    for i in range(1, len(pages) + 1):
        file_ends = i == len(pages) or pages[i].source_file != pages[start].source_file
        if file_ends or i - start == SPLIT_BATCH_SIZE:
            batches.append((start, i))
            start = i
    return batches


def split_forms(pages: list, progress: Optional[Callable[[int, int], None]] = None) -> list:
    """Detect each page's orientation, classify the upright pages, group them into forms.

    Failures never stop the run: a failed orientation leaves the page as scanned (the
    extractor can still read it); a failed classification batch turns its pages into
    separate, flagged forms. After a quota error no further requests are sent.
    """
    from src.extraction_engine import GeminiError, GeminiQuotaError

    batches = _batches(pages)
    quota_hit = threading.Event()
    total = len(pages) * 2  # progress: orientation pass + classification pass
    done = [0]
    lock = threading.Lock()

    def tick(n: int) -> None:
        with lock:
            done[0] += n
            if progress:
                progress(done[0], total)

    def orient(i: int) -> tuple:
        if quota_hit.is_set():
            return 0, 0.0
        try:
            return detect_rotation(pages[i].load())
        except GeminiError as exc:
            if isinstance(exc, GeminiQuotaError):
                quota_hit.set()
            logger.warning("Orientation failed for page %d: %s", i + 1, exc)
            return 0, 0.0

    def upright(i: int) -> Image.Image:
        rotation, confidence = rotations[i]
        img = pages[i].load()
        return fix_rotation(img, rotation) if confidence >= config.CONFIDENCE_THRESHOLD else img

    def classify(rng: tuple) -> list:
        start, end = rng
        if quota_hit.is_set():
            return [PageDecision(True, 0.0, error="Gemini API quota exhausted") for _ in range(end - start)]
        context = None
        if start > 0 and pages[start - 1].source_file == pages[start].source_file:
            context = upright(start - 1)
        try:
            return classify_batch([upright(i) for i in range(start, end)], context)
        except GeminiError as exc:
            if isinstance(exc, GeminiQuotaError):
                quota_hit.set()
            logger.warning("Page classification failed for pages %d-%d: %s", start + 1, end, exc)
            return [PageDecision(True, 0.0, error=str(exc)) for _ in range(end - start)]

    rotations: list = [(0, 0.0)] * len(pages)
    decisions: list = [None] * len(pages)
    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_REQUESTS) as pool:
        futures = {pool.submit(orient, i): i for i in range(len(pages))}
        for future in as_completed(futures):
            rotations[futures[future]] = future.result()
            tick(1)
        futures = {pool.submit(classify, rng): rng for rng in batches}
        for future in as_completed(futures):
            start, end = futures[future]
            decisions[start:end] = future.result()
            tick(end - start)

    for decision, (rotation, confidence) in zip(decisions, rotations):
        decision.rotate_clockwise, decision.rotation_confidence = rotation, confidence
    forms = group_pages(pages, decisions)
    rotated = sum(1 for f in forms for r in f.rotations if r)
    logger.info("Split %d page(s) into %d form(s); %d page(s) rotated; %d request(s)",
                len(pages), len(forms), rotated, len(pages) + len(batches))
    return forms
