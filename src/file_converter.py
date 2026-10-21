"""PDF -> page images, image loading (with EXIF orientation fix).

Every page is written once to a per-run folder under data/temp/ and loaded lazily:
a 300-dpi A4 page is ~26 MB in memory, so a 50-page batch must not live in RAM at once.
Call cleanup_temp() at the end of each run (privacy: nothing is kept on disk).
"""
import logging
import shutil
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator, Union

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

from src import config

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"})
# Photos are capped at A4 @ 300 dpi, the same detail level as converted PDF pages.
MAX_IMAGE_SIDE = 3508
_JPEG_QUALITY = 95


@dataclass(frozen=True)
class Page:
    """One page of an input file. Unpacks as (source_file, page_number, image)."""

    source_file: str
    page_number: int  # 1-based
    path: Path

    def load(self) -> Image.Image:
        """Load the page image fully into memory (the file handle is closed)."""
        with Image.open(self.path) as im:
            im.load()
            return im.copy()

    @property
    def image(self) -> Image.Image:
        return self.load()

    def __iter__(self) -> Iterator:
        return iter((self.source_file, self.page_number, self.load()))


# An input is a filesystem path, (filename, bytes), or a Streamlit UploadedFile.
FileInput = Union[str, Path, tuple, object]


def new_run_dir() -> Path:
    run_dir = config.TEMP_DIR / f"run_{uuid.uuid4().hex[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def cleanup_temp(run_dir: Union[Path, None] = None) -> None:
    """Delete one run folder, or everything in data/temp/ except .gitkeep."""
    targets = [run_dir] if run_dir else [p for p in config.TEMP_DIR.iterdir() if p.name != ".gitkeep"]
    for target in targets:
        if target is None or not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)


def _read_input(item: FileInput) -> tuple:
    """Return (filename, path_or_None, bytes_or_None) for any supported input form."""
    if isinstance(item, (str, Path)):
        path = Path(item)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path.name}")
        return path.name, path, None
    if isinstance(item, tuple) and len(item) == 2:
        name, data = item
        return Path(str(name)).name, None, bytes(data)
    if hasattr(item, "name") and hasattr(item, "getvalue"):  # Streamlit UploadedFile
        return Path(item.name).name, None, item.getvalue()
    raise TypeError(f"Unsupported input type: {type(item).__name__}")


def _to_rgb(im: Image.Image) -> Image.Image:
    if im.mode == "RGB":
        return im
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        background = Image.new("RGB", im.size, "white")
        background.paste(im, mask=im.getchannel("A"))
        return background
    return im.convert("RGB")


def _convert_pdf(name: str, path: Union[Path, None], data: Union[bytes, None],
                 run_dir: Path, file_index: int) -> list:
    tmp_pdf = None
    if path is None:
        # pdf2image/Poppler needs a file; it lives in the run folder only until converted.
        tmp_pdf = run_dir / f"input_{file_index}.pdf"
        tmp_pdf.write_bytes(data)
        path = tmp_pdf
    try:
        page_paths = convert_from_path(
            str(path),
            dpi=config.PDF_DPI,
            output_folder=str(run_dir),
            output_file=f"f{file_index:03d}_p",
            fmt="jpeg",
            jpegopt={"quality": _JPEG_QUALITY, "optimize": False, "progressive": False},
            paths_only=True,
            thread_count=2,
            **config.PDF2IMAGE_KWARGS,
        )
    except PDFInfoNotInstalledError as exc:
        raise RuntimeError(
            "Poppler is not installed or not on PATH. Install it or set POPPLER_PATH in .env."
        ) from exc
    except (PDFPageCountError, PDFSyntaxError) as exc:
        raise ValueError(f"Could not read PDF: {name}") from exc
    finally:
        if tmp_pdf is not None:
            tmp_pdf.unlink(missing_ok=True)
    return [Page(name, i, Path(p)) for i, p in enumerate(sorted(page_paths), start=1)]


def _convert_image(name: str, path: Union[Path, None], data: Union[bytes, None],
                   run_dir: Path, file_index: int) -> list:
    """One page per image; a multi-page TIFF (a scanner's batch) gives one page per frame."""
    pages = []
    try:
        with Image.open(path if path is not None else BytesIO(data)) as source:
            for number, frame in enumerate(ImageSequence.Iterator(source), start=1):
                im = ImageOps.exif_transpose(frame)  # phone photos: apply EXIF orientation
                im = _to_rgb(im)
                im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
                out = run_dir / f"f{file_index:03d}_p{number}.jpg"
                im.save(out, "JPEG", quality=_JPEG_QUALITY)
                pages.append(Page(name, number, out))
    except UnidentifiedImageError as exc:
        raise ValueError(f"Could not read image: {name}") from exc
    return pages


def convert_files(files: Union[FileInput, Iterable[FileInput]], run_dir: Union[Path, None] = None) -> list:
    """Convert one or many PDF/PNG/JPG/JPEG/TIFF inputs into a flat list of Page objects.

    Pages keep input order; page_number is 1-based within its source file.
    """
    if isinstance(files, (str, Path, tuple)) or hasattr(files, "getvalue"):
        files = [files]
    run_dir = run_dir or new_run_dir()
    pages: list = []
    for index, item in enumerate(files):
        name, path, data = _read_input(item)
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}' ({name}). Use PDF, PNG, JPG, JPEG or TIFF.")
        convert = _convert_pdf if ext == ".pdf" else _convert_image
        file_pages = convert(name, path, data, run_dir, index)
        logger.info("Converted file #%d (%s): %d page(s)", index + 1, ext, len(file_pages))
        pages.extend(file_pages)
    return pages
