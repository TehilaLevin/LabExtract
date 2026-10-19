"""Delete development data that can hold personal data: the DEV_CACHE responses (data/dev_cache/) and the
HIGHLIGHT_DEBUG crops (data/temp/highlight_debug/). Both folders are git-ignored.

Run: .venv/Scripts/python -m src.clear_cache
"""
from src.extraction_engine import clear_dev_cache
from src.highlight_crops import clear_debug_crops


def main() -> None:
    print(f"dev cache: {clear_dev_cache()} file(s) deleted")
    print(f"highlight debug crops: {clear_debug_crops()} file(s) deleted")


if __name__ == "__main__":
    main()
