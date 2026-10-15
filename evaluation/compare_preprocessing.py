"""Compare original vs preprocessed page images - offline, no API calls.

Usage:
    .venv/Scripts/python evaluation/compare_preprocessing.py FILE [FILE ...] [--save DIR]

For every page prints the preprocessing steps and simple measurements that show whether
preprocessing may have erased detail:
- mean luminance (original -> processed),
- clipped: share of pixels at pure white (>= 250) - a jump means faint strokes may be gone,
- detail: mean absolute difference between neighbouring pixels (local contrast).
With --save DIR, writes side-by-side PNGs (original | processed) to DIR for visual checks.
Pages are measured as scanned: the orientation fix (an API step) is not applied here, so the
deskew angle of a sideways page can differ from the one the pipeline uses.
Use sanitized / fictitious examples only, and delete DIR afterwards: the images are form pages.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.file_converter import cleanup_temp, convert_files, new_run_dir  # noqa: E402
from src.image_preprocessor import mean_luminance, preprocess  # noqa: E402

_WORK_SIDE = 1600


def measure(image) -> dict:
    gray = image.convert("L")
    gray.thumbnail((_WORK_SIDE, _WORK_SIDE))
    a = np.asarray(gray, dtype=np.float32)
    detail = (np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()) / 2
    return {"mean": mean_luminance(image), "clipped": float((a >= 250).mean()), "detail": float(detail)}


def side_by_side(left, right):
    from PIL import Image

    h = max(left.height, right.height)
    out = Image.new("RGB", (left.width + right.width + 20, h), "white")
    out.paste(left.convert("RGB"), (0, 0))
    out.paste(right.convert("RGB"), (left.width + 20, 0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--save", type=Path, help="folder for side-by-side PNGs (delete it afterwards)")
    args = parser.parse_args()
    run_dir = new_run_dir()
    try:
        pages = convert_files(args.files, run_dir)
        file_number = {p.name: n for n, p in enumerate(args.files, start=1)}  # names are not printed
        for i, page in enumerate(pages, start=1):
            image = page.load()
            result = preprocess(image)
            before, after = measure(image), measure(result.processed)
            print(f"file #{file_number.get(page.source_file, '?')} page {page.page_number}: "
                  f"size={image.size[0]}x{image.size[1]} "
                  f"steps={','.join(result.steps) or 'none'}")
            print(f"    mean {before['mean']:.0f} -> {after['mean']:.0f}   clipped {before['clipped']:.1%} -> "
                  f"{after['clipped']:.1%}   detail {before['detail']:.2f} -> {after['detail']:.2f}")
            if args.save:
                args.save.mkdir(parents=True, exist_ok=True)
                side_by_side(image, result.processed).save(args.save / f"page_{i:03d}.png")
    finally:
        cleanup_temp(run_dir)


if __name__ == "__main__":
    main()
