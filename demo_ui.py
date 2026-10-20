"""UI demo with fictitious data: `streamlit run demo_ui.py`. Makes no API call and reads no real form.

It injects fake review records (with generated form images and crops) into st.session_state and then runs
app.py unchanged, so the real review, summary and export screens can be looked at for free.
"""
import runpy
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from src import record_builder as rb
from src import review_session as rs
from src.constants import FIELD_KEYS
from src.field_crops import PagePreview, SourceCrop

FIELD_BOXES = {  # (ymin, xmin, ymax, xmax) on the 0-1000 grid of the fake page
    "first_name": (150, 100, 220, 450), "last_name": (150, 500, 220, 900),
    "id_number": (260, 100, 330, 500), "central_numbers": (400, 100, 470, 500),
    "received_date": (400, 550, 470, 900), "sending_entity": (60, 100, 120, 900),
    "commitment_numbers": (560, 100, 630, 700),
}
FIELD_TEXT = {
    "first_name": "FIRST NAME: Yisrael", "last_name": "LAST NAME: Yisraeli", "id_number": "ID: 000000018",
    "central_numbers": "CENTRAL: 937309, 937310", "received_date": "RECEIVED: 01/10/2026",
    "sending_entity": "SENDER: Maccabi", "commitment_numbers": "COMMITMENT: 50364186",
}


def _jpeg(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _fake_page() -> Image.Image:
    img = Image.new("RGB", (1000, 1400), "#FBFAF6")
    draw = ImageDraw.Draw(img)
    for key, (y0, x0, y1, x1) in FIELD_BOXES.items():
        box = (x0 * 1, y0 * 1.4, x1 * 1, y1 * 1.4)
        draw.rectangle(box, outline="#999999", width=2)
        draw.text((box[0] + 16, box[1] + 30), FIELD_TEXT[key], fill="black")
    return img


def _record(index: int, flags: dict, **overrides) -> rb.Record:
    page = _fake_page()
    values = {"first_name": "ישראל", "last_name": "ישראלי", "id_number": "000000018",
              "central_numbers": "937309, 937310", "received_date": "01/10/2026", "sending_entity": "מכבי",
              "commitment_numbers": "50364186"}
    values.update(overrides)
    crops = {}
    for key, (y0, x0, y1, x1) in FIELD_BOXES.items():
        pad = 25
        pixel = (max(0, x0 - pad), max(0, round(y0 * 1.4) - pad), min(1000, x1 + pad), round(y1 * 1.4) + pad)
        crops[key] = [SourceCrop(1, (y0, x0, y1, x1), _jpeg(page.crop(pixel)))]
    confidences = {k: (0.4 if k in flags else 0.95) for k in FIELD_KEYS}
    return rb.Record(index, values, dict(flags), confidences,
                     {"source_file": "demo.pdf", "page_numbers": str(index), "form_type": "מכבי – דמו"},
                     field_crops=crops, page_images=[PagePreview(index, _jpeg(page))])


if "reviews" not in st.session_state or st.session_state.reviews is None:
    st.session_state.reviews = rs.new_session([
        _record(1, {}),
        _record(2, {"id_number": "invalid_id", "received_date": "missing"}, id_number="000000019", received_date=""),
        _record(3, {"central_numbers": "low_confidence"}),
    ])
    st.session_state.screen = st.query_params.get("screen", "summary")  # e.g. ?screen=review&pos=1
    st.session_state.pos = int(st.query_params.get("pos", 0))

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
