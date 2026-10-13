from io import BytesIO

import pytest
from PIL import Image

from src import config
from src import file_converter as fc


def _poppler_available() -> bool:
    try:
        from pdf2image import pdfinfo_from_path  # noqa: F401
        from pdf2image.exceptions import PDFInfoNotInstalledError

        try:
            pdfinfo_from_path("__missing__.pdf", **config.PDF2IMAGE_KWARGS)
        except PDFInfoNotInstalledError:
            return False
        except Exception:
            return True
    except ImportError:
        return False
    return True


needs_poppler = pytest.mark.skipif(not _poppler_available(), reason="Poppler not installed")


@pytest.fixture(autouse=True)
def temp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path)
    return tmp_path


def _pdf_bytes(n_pages: int) -> bytes:
    pages = [Image.new("RGB", (300, 400), (255, 255 - 40 * i, 255)) for i in range(n_pages)]
    buf = BytesIO()
    pages[0].save(buf, "PDF", save_all=True, append_images=pages[1:], resolution=72)
    return buf.getvalue()


def _jpeg_with_exif_orientation(orientation: int) -> bytes:
    img = Image.new("RGB", (200, 100), "white")
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


@needs_poppler
def test_pdf_pages_in_order_with_source_and_page_number(tmp_path):
    pdf = tmp_path / "batch.pdf"
    pdf.write_bytes(_pdf_bytes(3))
    pages = fc.convert_files(pdf)
    assert [(p.source_file, p.page_number) for p in pages] == [("batch.pdf", i) for i in (1, 2, 3)]
    source, number, image = pages[1]  # unpacks as (source_file, page_number, image)
    assert (source, number) == ("batch.pdf", 2)
    assert image.mode == "RGB"
    # 300 dpi conversion of a 72-dpi 300x400 page -> ~1250x1667 px
    assert image.size[0] > 1000


@needs_poppler
def test_pdf_from_upload_bytes_leaves_no_pdf_on_disk():
    pages = fc.convert_files([("upload.pdf", _pdf_bytes(2))])
    assert len(pages) == 2
    run_dir = pages[0].path.parent
    assert not list(run_dir.glob("*.pdf"))


def test_many_inputs_mixed_types(tmp_path):
    png = tmp_path / "photo.png"
    Image.new("RGBA", (50, 60), (0, 0, 0, 0)).save(png)
    pages = fc.convert_files([png, ("scan.JPG", _jpeg_with_exif_orientation(1))])
    assert [(p.source_file, p.page_number) for p in pages] == [("photo.png", 1), ("scan.JPG", 1)]
    assert pages[0].load().mode == "RGB"  # transparency flattened onto white
    assert pages[0].load().getpixel((0, 0)) == (255, 255, 255)


def test_exif_orientation_is_applied():
    # Orientation 6 = the camera was rotated 90 deg: the upright image is portrait.
    pages = fc.convert_files([("phone.jpg", _jpeg_with_exif_orientation(6))])
    assert pages[0].load().size == (100, 200)


def test_large_photo_is_capped():
    big = Image.new("RGB", (5000, 4000), "white")
    buf = BytesIO()
    big.save(buf, "JPEG")
    pages = fc.convert_files([("big.jpeg", buf.getvalue())])
    assert max(pages[0].load().size) == fc.MAX_IMAGE_SIDE


def test_streamlit_like_upload_object():
    class Upload:
        name = "x.png"

        def getvalue(self):
            buf = BytesIO()
            Image.new("RGB", (10, 10)).save(buf, "PNG")
            return buf.getvalue()

    assert len(fc.convert_files(Upload())) == 1


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        fc.convert_files([("notes.docx", b"abc")])


def test_corrupt_image_raises():
    with pytest.raises(ValueError, match="Could not read image"):
        fc.convert_files([("broken.png", b"not an image")])


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fc.convert_files(tmp_path / "nope.pdf")


def test_cleanup_removes_run_dir_and_keeps_gitkeep(temp_dir):
    (temp_dir / ".gitkeep").write_text("")
    pages = fc.convert_files([("a.png", _jpeg_with_exif_orientation(1))])
    run_dir = pages[0].path.parent
    assert run_dir.exists()
    fc.cleanup_temp()
    assert not run_dir.exists()
    assert (temp_dir / ".gitkeep").exists()
