"""LabExtract - Streamlit app: upload -> extract -> review each form -> export Excel.

Run: streamlit run app.py

Screens (st.session_state.screen): upload, summary, review, export.

Privacy: uploads, results and images live only in this browser session's st.session_state. Nothing is
cached across sessions (no st.cache_data), logged or written to disk, except the pipeline's per-run temp
folder, which process_files always deletes.

Streamlit limits and how they are handled:
- st.data_editor cannot colour single cells. Editing therefore happens on the per-form review screen, one
  bordered card per field; a flagged card gets an orange frame through CSS on its key class
  ("st-key-<key>"). Read-only tables (summary, export preview) are coloured with a pandas Styler in
  st.dataframe, which does support per-cell styles.
- Every interaction reruns the script. Claude is called only from the "Extract" button; the results are
  kept in st.session_state, so reruns never call it again. The field inputs sit in an st.form, so typing
  does not rerun; values are saved into FormReview.edits on submit. Widget state cannot hold the edits,
  because Streamlit drops the state of a widget that was not rendered in the last run (other form shown).
- The pipeline reports progress from worker threads, which cannot touch the page. It runs in a background
  thread and this script polls its progress into st.progress.
- Navigation is a plain session-state value, not st.tabs: tabs cannot be switched from code, and
  "approve form" has to move to the next form.
"""
import base64
import html
import threading
import uuid
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import altair as alt

from src import achievements as achv
from src import archive_assistant as aa
from src import config
from src import overview
from src import record_store
from src import review_session as rs
from src.constants import (
    ALL_HEADERS,
    FIELD_HEADERS,
    FIELD_KEYS,
    FLAG_COLOR,
    FLAG_REASONS,
    LIST_SEPARATOR,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_HEADER,
    REVIEW_STATUS_KEY,
    SPLIT_FLAG_LABELS,
    SUGGESTION_LABEL,
)
from src.excel_exporter import to_xlsx_bytes
from src.field_crops import outline_boxes
from src.claude_client import LLMError, LLMQuotaError, check_configured
from src.validators import list_approved_senders

st.set_page_config(page_title="LabExtract", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

SCREENS = {"upload": "העלאה", "summary": "סיכום", "review": "סקירה", "export": "ייצוא", "database": "מאגר"}
STEP_SCREENS = ("upload", "summary", "review", "export")  # the four steps of one run; "database" stands apart
APPROVED_COLOR = "#1F8A73"  # Sheba teal: approved forms and high confidence
ACCENT_COLOR = "#C8306F"    # Sheba pink: current step, highlights (see .streamlit/config.toml)
PIPELINE_STEPS = (
    ("פיצול לטפסים", "כל עמוד: טופס חדש או המשך של הקודם"),
    ("זיהוי סוג טופס", "קופה או בית חולים, מרשימה סגורה"),
    ("חילוץ שדות", "7 שדות, עם רמת ביטחון ומיקום בטופס"),
    ("אימות ונרמול", "ספרת ביקורת בת״ז, תאריכים, מספרים"),
    ("סקירה וייצוא", "אתם מאשרים – ומורידים Excel"),
)
# Pipeline step shown as active for each progress stage reported by process_files.
STAGE_STEP = {"start": 0, "split": 0, "extract": 2}
# Column widths (px) of the export-shaped tables: all eleven columns fit without scrolling, because
# st.dataframe starts scrolled to the left edge, which would hide the first fields of a right-to-left table.
EXPORT_WIDTHS = {"first_name": 70, "last_name": 72, "id_number": 96, "central_numbers": 76, "received_date": 94,
                 "sending_entity": 68, "commitment_numbers": 80, "source_file": 150, "page_numbers": 62,
                 "form_type": 116, REVIEW_STATUS_KEY: 76}
ASSETS = Path(__file__).resolve().parent / "assets"


def _data_uri(name: str) -> str:
    """A logo from assets/ as a data: URI, so it can sit inside sidebar HTML ("" when the file is missing)."""
    path = ASSETS / name
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" if path.exists() else ""


# Logos: the Central Virology Laboratory at the top of the sidebar; Sheba Medical Center and the Ministry of
# Health in the page footer ("in partnership with"). The look follows the lab's page on the Sheba web site.
BRAND_HTML = f"""
<div class="lx-brand">
  <img class="lx-lab-logo" src="{_data_uri('lab_logo.png')}" alt="המעבדה המרכזית לנגיפים">
  <div><div class="lx-name" dir="ltr">LabExtract</div><div class="lx-sub">המעבדה המרכזית לנגיפים<br>חילוץ טפסי התחייבות</div></div>
</div>
"""
PARTNER_HTML = f"""
<footer class="lx-partner"><span>בשיתוף:</span>
  <div class="lx-partner-logos"><img src="{_data_uri('sheba_logo.png')}" alt="שיבא תל השומר">
  <img src="{_data_uri('moh_logo.png')}" alt="משרד הבריאות"></div></footer>
"""
PRIVACY_NOTE = ('<div class="lx-privacy">הקבצים שהועלו נמחקים בסיום החילוץ. הרשומות נשמרות במחשב הזה בלבד '
                'ואפשר למחוק אותן במסך המאגר. שום נתון אישי לא נכתב ליומן.</div>')
UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg"]
DEV_CACHE_BANNER = ("**מצב פיתוח: DEV_CACHE פעיל.** תשובות המודל (שעלולות לכלול נתונים אישיים) נשמרות בדיסק "
                    "בתיקייה data/dev_cache, כדי שהרצה חוזרת על אותם עמודים לא תעלה כסף. "
                    "לקבצי הדוגמה בלבד – אסור להעלות טפסים אמיתיים במצב זה.")

# RTL throughout. Inputs use unicode-bidi: plaintext so an ID or a date is laid out by its own content.
# st.dataframe draws on a canvas that ignores CSS direction; its columns are ordered right-to-left instead.
CSS = f"""
<style>
.stApp, [data-testid="stSidebar"] {{ direction: rtl; }}
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {{ text-align: right; }}
.stApp input, .stApp textarea {{ unicode-bidi: plaintext; text-align: right; }}
[data-testid="stDataFrame"] {{ direction: ltr; }}
.stApp pre, .stApp code {{ direction: rtl; unicode-bidi: plaintext; text-align: right; }}
div[class*="st-key-cardflag_"] {{
    border: 2px solid {FLAG_COLOR} !important;
    background: rgba(255, 165, 0, 0.10);
}}
.flag-reason {{ color: #9A5800; font-weight: 600; }}  /* dark orange: readable on white (#FFA500 is too light for text) */
.suggestion {{ color: #9A5800; }}

/* Layout and typography. Colours come from the lab and Ministry of Health logos (see .streamlit/config.toml). */
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.5rem; max-width: 1400px; }}
.lx-kicker {{ font-size: 13px; font-weight: 600; color: {ACCENT_COLOR}; margin: 0 0 6px; }}
.lx-h1 {{ font-size: 40px; font-weight: 800; line-height: 1.1; margin: 0 0 10px; padding: 0; color: #1B1A6B; }}
.lx-lead {{ font-size: 16px; color: #44546B; margin: 0; max-width: 760px; }}
.lx-muted {{ font-size: 13px; color: #5F6E82; }}
.lx-section {{ font-size: 12px; font-weight: 500; color: #5B6B80; margin: 18px 0 8px; }}

/* Primary buttons: white text on lab blue. */
[data-testid="stBaseButton-primary"] {{ color: #FFFFFF; font-weight: 600; border: none; }}
[data-testid="stBaseButton-primary"]:disabled {{ background: #ECEBF4; color: #8D8CA8; }}
.st-key-extract button, .st-key-start_review button {{
    min-height: 52px; padding: 0 28px; border-radius: 12px; font-size: 16px;
}}
.st-key-extract button:not(:disabled), .st-key-start_review button {{ box-shadow: 0 0 0 4px rgba(27, 26, 107, 0.15); }}

/* Upload drop zone. Streamlit's built-in English texts are hidden (font-size: 0) and replaced with
   Hebrew through ::before / ::after; the widget itself and its behaviour stay Streamlit's. */
[data-testid="stFileUploaderDropzone"] {{
    border: 2px dashed #E7A3C4; background: #FFF7FB; border-radius: 16px; padding: 36px 24px;
    display: flex; flex-direction: column; align-items: center; gap: 14px; text-align: center;
}}
[data-testid="stFileUploaderDropzone"]::before {{
    content: ""; width: 64px; height: 64px; border-radius: 16px; background: #FBEAF2 center / 32px no-repeat
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' fill='none' stroke='%23C8306F' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M16 21V7M10 12l6-6 6 6'/%3E%3Cpath d='M6 20v4a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-4'/%3E%3C/svg%3E");
}}
[data-testid="stFileUploaderDropzone"]::before {{ order: -2; }}
[data-testid="stFileUploaderDropzoneInstructions"] {{ order: -1; }}
[data-testid="stFileUploaderDropzoneInstructions"] span {{ font-size: 0; }}
[data-testid="stFileUploaderDropzoneInstructions"] > div::before {{
    content: "גררו קבצים לכאן"; display: block; font-size: 20px; font-weight: 600; color: #1B1A6B;
}}
[data-testid="stFileUploaderDropzoneInstructions"] > div::after {{
    content: "PDF · PNG · JPG · JPEG · עד 500MB לקובץ"; display: block; font-size: 12px; color: #5F6E82; margin-top: 6px;
}}
[data-testid="stFileUploaderDropzone"] button p {{ font-size: 0; }}
[data-testid="stFileUploaderDropzone"] button p::after {{ content: "בחירה מהמחשב"; font-size: 14px; }}
.lx-lead {{ margin-bottom: 22px; }}

/* Sidebar: brand, step navigation, batch list. */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 4px; }}
.lx-brand {{ display: flex; align-items: center; gap: 10px; margin-bottom: 40px; }}
.lx-lab-logo {{ width: 56px; height: 56px; flex: none; }}
.lx-partner {{ display: flex; flex-direction: column; gap: 10px; border-top: 1px solid #E3E2EE; padding-top: 18px;
    margin-top: 48px; font-size: 14px; font-weight: 600; color: #1B1A6B; }}
.lx-partner-logos {{ display: flex; align-items: center; gap: 32px; flex-wrap: wrap; }}
.lx-partner-logos img {{ height: 56px; width: auto; max-width: 100%; }}
.lx-partner-logos img + img {{ height: 68px; }}
.lx-name {{ font-weight: 700; font-size: 20px; line-height: 1.2; color: #1B1A6B; }}
.lx-sub {{ font-size: 12px; color: #5B6B80; }}
.lx-privacy {{ font-size: 12px; color: #5F6E82; border-top: 1px solid #DCE4EC; padding-top: 14px; margin-top: 18px; }}
.lx-empty {{ border: 1px dashed #C3CEDA; border-radius: 10px; padding: 16px 12px; font-size: 13px;
    color: #5F6E82; text-align: center; }}
[data-testid="stSidebar"] [class*="st-key-nav_"] button,
[data-testid="stSidebar"] [class*="st-key-queue_"] button {{
    width: 100%; justify-content: flex-start; border: none; background: transparent; color: #2E3E55;
    min-height: 40px; padding: 6px 12px; border-radius: 8px;
}}
[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover,
[data-testid="stSidebar"] [class*="st-key-queue_"] button:hover {{ background: #FBEAF2; color: #1B1A6B; }}
[data-testid="stSidebar"] [class*="st-key-nav_"] button:disabled {{ color: #9AA6B5; background: transparent; }}
[data-testid="stSidebar"] [class*="st-key-nav_"] button::before {{
    content: ""; width: 8px; height: 8px; border-radius: 50%; background: #C3CEDA; margin-inline-end: 10px; flex: none;
}}
[data-testid="stSidebar"] [class*="st-key-nav_"] button > div,
[data-testid="stSidebar"] [class*="st-key-queue_"] button > div {{ flex: 1; min-width: 0; justify-content: flex-start; }}
[data-testid="stSidebar"] [class*="st-key-queue_"] button {{ font-size: 13px; min-height: 36px; }}
[data-testid="stSidebar"] [class*="st-key-queue_"] button::after {{
    content: ""; margin-inline-start: auto; min-width: 22px; height: 22px; border-radius: 999px; font-size: 11px;
    font-weight: 600; display: inline-flex; align-items: center; justify-content: center; background: #FBEAF2;
}}
[data-testid="stSidebar"] [class*="st-key-queue_"] [data-testid="stMarkdownContainer"] p {{
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}

/* Stat tiles and pipeline steps. */
.lx-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 8px 0 20px; }}
.lx-tile {{ background: #FFFFFF; border: 1px solid #DCE4EC; border-radius: 12px; padding: 16px 18px; }}
.lx-tile.flag {{ background: #FFF6E5; border-color: #F5C77A; }}
.lx-tile.ok {{ background: #EEF8F5; border-color: #A9DCCD; }}
.lx-tile-label {{ font-size: 13px; color: #5B6B80; }}
.lx-tile.flag .lx-tile-label {{ color: #9A5800; }}
.lx-tile.ok .lx-tile-label {{ color: {APPROVED_COLOR}; }}
.lx-tile-value {{ font-family: "JetBrains Mono", monospace; font-size: 32px; font-weight: 600; line-height: 1.3; }}
.lx-tile.flag .lx-tile-value {{ color: {FLAG_COLOR}; }}
.lx-tile.ok .lx-tile-value {{ color: {APPROVED_COLOR}; }}
.lx-tile-note {{ font-size: 12px; color: #5F6E82; }}
.lx-tile.flag .lx-tile-note {{ color: #8A6420; }}
.lx-bar {{ height: 6px; background: #D0EDE5; border-radius: 3px; overflow: hidden; margin-top: 8px; }}
.lx-bar > div {{ height: 100%; background: {APPROVED_COLOR}; }}
.lx-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
.lx-step {{ display: flex; flex-direction: column; gap: 6px; border-radius: 12px; padding: 14px;
    background: #FFFFFF; border: 1px solid #DCE4EC; }}
.lx-step.active {{ background: #FBEAF2; border-color: {ACCENT_COLOR}; }}
.lx-step-num {{ width: 24px; height: 24px; border-radius: 999px; display: inline-flex; align-items: center;
    justify-content: center; font-size: 12px; font-weight: 600; background: #FBEAF2; color: #5B6B80; }}
.lx-step.active .lx-step-num, .lx-step.done .lx-step-num {{ background: {ACCENT_COLOR}; color: #FFFFFF; }}
.lx-step-title {{ font-weight: 600; font-size: 14px; }}
.lx-step-text {{ font-size: 12px; color: #5B6B80; line-height: 1.45; }}
.lx-legend {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; background: {FLAG_COLOR};
    vertical-align: middle; margin-inline-end: 4px; }}
/* Review screen: field cards, confidence chips, sticky action bar and form image. */
div[class*="st-key-cardok_"] {{ background: #FFFFFF; }}
div[class*="st-key-cardflag_"] {{ background: #FFF6E5 !important; }}
div[class*="st-key-card"] [data-testid="stImage"] img {{ max-height: 220px; object-fit: contain; background: #FFFFFF;
    border-radius: 8px; }}
.lx-card-head {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
.lx-chip {{ font-size: 11px; font-weight: 600; border-radius: 999px; padding: 2px 8px; white-space: nowrap; }}
.lx-chip.ok {{ color: {APPROVED_COLOR}; background: #E2F4EF; }}
.lx-chip.warn {{ color: #9A5800; background: #FFEBC7; }}
.lx-chip.big {{ font-size: 12px; padding: 3px 10px; border: 1px solid currentColor; }}
.flag-reason {{ font-size: 13px; margin-top: 6px; }}
.lx-title-row {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
.lx-title-row .lx-h1 {{ font-size: 30px; margin: 0; }}
.lx-dim {{ color: #7A889B; font-weight: 500; }}
.lx-mono {{ font-family: "JetBrains Mono", monospace; }}
.lx-fields-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin-bottom: 4px; }}
.lx-approved {{ font-weight: 600; color: {APPROVED_COLOR}; }}
.lx-outline {{ display: inline-block; width: 12px; height: 12px; border: 2px solid {FLAG_COLOR}; border-radius: 3px;
    box-sizing: border-box; vertical-align: middle; margin-inline-end: 6px; }}
.st-key-actionbar {{ position: sticky; bottom: 0; z-index: 5; background: #FFFFFF; border: 1px solid #DCE4EC;
    border-radius: 14px; padding: 12px 14px; margin-top: 8px; }}
.st-key-approve button {{ min-height: 50px; padding: 0 24px; border-radius: 12px; font-size: 16px;
    box-shadow: 0 0 0 4px rgba(27, 26, 107, 0.15); }}
.st-key-save button, .st-key-reopen button {{ min-height: 50px; border-radius: 12px; }}
.st-key-force_approve button {{ background: {FLAG_COLOR}; color: #1A1100; border: none; font-weight: 600; min-height: 44px; }}
[data-testid="stColumn"]:has(.st-key-formimage) {{ position: sticky; top: 16px; align-self: flex-start; }}
/* Export screen. */
.st-key-downloadcard, .st-key-waitingcard {{ background: #FFFFFF; border-radius: 16px; padding: 6px; }}
.st-key-waitingcard {{ background: #FFF6E5; border-color: #F5C77A !important; }}
.lx-file {{ display: flex; align-items: center; gap: 14px; }}
.lx-file-icon {{ width: 52px; height: 52px; border-radius: 12px; flex: none; background: #FBEAF2 center / 26px no-repeat
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 26 26' fill='none' stroke='%23C8306F' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M7 3h9l5 5v15H7z'/%3E%3Cpath d='M16 3v5h5'/%3E%3Cpath d='M10.5 13l5 6M15.5 13l-5 6'/%3E%3C/svg%3E"); }}
.lx-file bdi {{ font-size: 15px; font-weight: 500; }}
.st-key-download button {{ min-height: 56px; border-radius: 12px; font-size: 17px; }}
.st-key-download button:not(:disabled) {{ box-shadow: 0 0 0 4px rgba(27, 26, 107, 0.15); }}
.lx-ready {{ display: flex; flex-direction: column; gap: 10px; background: #EEF8F5; border: 1px solid #A9DCCD;
    border-radius: 16px; padding: 22px; }}
.lx-ready strong {{ font-size: 18px; color: {APPROVED_COLOR}; }}
.lx-ready span {{ font-size: 14px; color: #2F6A5D; }}
.lx-waiting {{ display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; font-size: 14px; }}
.lx-waiting span {{ font-size: 13px; color: #8A6420; }}
.lx-waiting em {{ font-style: normal; font-size: 12px; color: #9A5800; }}
.st-key-waitingcard [data-testid="stAlertContainer"] {{ background: transparent; color: #9A5800; font-weight: 600;
    font-size: 17px; padding: 4px 4px 8px; }}
[class*="st-key-open_"] button {{ border-color: #F5C77A; color: #9A5800; background: transparent; min-height: 36px; }}
/* Sheba look: hero band with a soft pink gradient, decorative rings, round cards and pill buttons. */
.lx-hero {{ position: relative; overflow: hidden; border-radius: 28px; padding: 30px 34px 28px; margin-bottom: 22px;
    background: radial-gradient(circle at 70% 40%, #FFFFFF 0%, #FBEAF2 55%, #F4D3E4 100%); }}
.lx-hero::after {{ content: ""; position: absolute; width: 360px; height: 360px; left: -150px; top: -60px;
    border-radius: 50%; border: 26px solid rgba(217, 63, 135, 0.45); pointer-events: none; }}
.lx-hero::before {{ content: ""; position: absolute; width: 300px; height: 300px; left: -40px; top: -190px;
    border-radius: 50%; border: 18px solid rgba(77, 182, 160, 0.35); pointer-events: none; }}
.lx-hero > * {{ position: relative; z-index: 1; }}
.lx-hero .lx-lead {{ margin-bottom: 0; color: #3A3963; }}
.stApp::before {{ content: ""; position: fixed; width: 520px; height: 520px; left: -260px; bottom: -260px;
    border-radius: 50%; border: 34px solid rgba(217, 63, 135, 0.18); pointer-events: none; z-index: 0; }}
.stApp::after {{ content: ""; position: fixed; width: 420px; height: 420px; left: 120px; bottom: -330px;
    border-radius: 50%; border: 24px solid rgba(77, 182, 160, 0.16); pointer-events: none; z-index: 0; }}
[data-testid="stMain"], [data-testid="stSidebar"] {{ position: relative; z-index: 1; }}
.lx-tile, .lx-step, .lx-ready {{ border-radius: 22px; box-shadow: 0 6px 22px rgba(27, 26, 107, 0.06); }}
div[class*="st-key-card"], .st-key-formimage, .st-key-downloadcard, .st-key-waitingcard, .st-key-actionbar {{
    border-radius: 22px !important; box-shadow: 0 6px 22px rgba(27, 26, 107, 0.06); }}
[data-testid="stBaseButton-secondary"] {{ border-color: #1B1A6B; color: #1B1A6B; }}
.stApp button[data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-tertiary"]) {{ border-radius: 999px !important; }}
/* Archive assistant (database screen). */
.st-key-assistant {{ background: #FFFFFF; border-radius: 22px !important; box-shadow: 0 6px 22px rgba(27, 26, 107, 0.06);
    margin-bottom: 18px; }}
.st-key-assistant_examples, .st-key-assistant_options {{ flex-wrap: wrap; gap: 6px; }}
.st-key-assistant_examples button, .st-key-assistant_options button {{ min-height: 32px; padding: 2px 12px;
    background: #FBEAF2; border-color: #F2C6DB; color: #1B1A6B; }}
.st-key-assistant_examples button p, .st-key-assistant_options button p {{ font-size: 13px; }}
.lx-answer {{ font-size: 20px; font-weight: 700; color: #1B1A6B; margin: 4px 0; }}
.lx-chips {{ display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin: 2px 0 8px; }}
.lx-chip.info {{ color: #1B1A6B; background: #ECEBF4; }}
.lx-note {{ font-size: 13px; color: #9A5800; }}
</style>
"""


# ------------------------------------------------------------------ state

def init_state() -> None:
    defaults = {"screen": "upload", "reviews": None, "pos": 0, "uploader_n": 0, "nonce": 0,
                "pending": None, "notice": None, "run_error": None, "batch_id": None, "notice_error": None,
                "toasts": [], "ach_unlocked": {}, "ach_processed": 0,
                "assistant_pending": None, "assistant_result": None, "assistant_cache": {}}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go(screen: str, pos=None) -> None:
    st.session_state.screen = screen
    if pos is not None:
        st.session_state.pos = pos
    st.session_state.pending = None
    st.session_state.nonce += 1  # fresh widget keys: inputs show the saved values, old selections clear


def clear_all() -> None:
    st.session_state.reviews = None
    st.session_state.uploader_n += 1  # a new uploader key empties the uploader
    st.session_state.run_error = None
    go("upload", 0)


def reviews() -> list:
    return st.session_state.reviews or []


def persist(items: list) -> None:
    """Save forms to the local SQLite store (a failure warns, never loses the in-memory review)."""
    if not config.DB_ENABLED or not items:
        return
    try:
        record_store.save_all(items, st.session_state.batch_id)
    except Exception as exc:  # sqlite3.Error, a locked or read-only file...
        st.session_state.notice_error = f"השמירה במאגר נכשלה ({type(exc).__name__}). הסקירה בזיכרון לא נפגעה."


def config_problem():
    """Hebrew message when the Claude API key or settings are not configured, else None."""
    try:
        check_configured()
    except RuntimeError as exc:
        return f"ההגדרות חסרות: {exc}"
    return None


def run_error_message(exc: Exception) -> str:
    """Hebrew explanation of a run that stopped. Provider error messages never contain form content."""
    if isinstance(exc, LLMQuotaError):
        return f"הריצה נעצרה – השירות דחה את הבקשות (מכסה, מפתח API או שם מודל). {exc}"
    if isinstance(exc, LLMError):
        return f"שגיאה בקריאה למודל. {exc}"
    return f"{type(exc).__name__}: {exc}"


# ------------------------------------------------------------------ extraction (background thread)

def steps_html(active: int = -1) -> str:
    """The five pipeline cards on the upload screen; `active` lights the running step (-1: none)."""
    cards = []
    for i, (title, text) in enumerate(PIPELINE_STEPS):
        state = "active" if i == active else ("done" if 0 <= i < active else "")
        num = "✓" if state == "done" else str(i + 1)
        cards.append(f'<div class="lx-step {state}"><span class="lx-step-num">{num}</span>'
                     f'<span class="lx-step-title">{title}</span><span class="lx-step-text">{text}</span></div>')
    return f'<div class="lx-section">מה קורה בחילוץ</div><div class="lx-steps">{"".join(cards)}</div>'


def run_extraction(files: list, progress_slot, steps_slot) -> None:
    from src.record_builder import process_files

    payload = [(f.name, f.getvalue()) for f in files]
    progress = {"stage": "start", "done": 0, "total": 0}
    outcome = {}
    lock = threading.Lock()

    def report(stage: str, done: int, total: int) -> None:
        with lock:
            progress.update(stage=stage, done=done, total=total)

    def target() -> None:
        try:
            outcome["records"] = process_files(payload, report)
        except Exception as exc:  # shown to the user; LLMError messages never contain form content
            outcome["error"] = run_error_message(exc)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    bar = progress_slot.progress(0.0, text="ממיר קבצים לתמונות…")
    shown_step = None
    while worker.is_alive():
        with lock:
            stage, done, total = progress["stage"], progress["done"], progress["total"]
        step = STAGE_STEP.get(stage, 0)
        if step != shown_step:
            steps_slot.markdown(steps_html(step), unsafe_allow_html=True)
            shown_step = step
        if stage == "split" and total:
            bar.progress(done / total, text=f"מזהה כיוון דפים ומפצל לטפסים… ({done}/{total})")
        elif stage == "extract" and total:
            bar.progress(done / total, text=f"מחלץ שדות: טופס {done} מתוך {total}")
        time.sleep(0.25)
    worker.join()
    bar.empty()
    if "error" in outcome:
        st.session_state.run_error = outcome["error"]
        return
    st.session_state.reviews = rs.new_session(outcome["records"])
    st.session_state.batch_id = uuid.uuid4().hex
    persist(st.session_state.reviews)
    award(new_forms=sum(1 for r in st.session_state.reviews if not r.record.error), batch_finished=True)
    st.session_state.run_error = None
    go("summary", 0)
    st.rerun()


# ------------------------------------------------------------------ achievements

def award(new_forms: int = 0, batch_finished: bool = False, hour=None) -> None:
    """Re-check the achievements after an event and queue a toast for each one unlocked for the first time.

    Unlocked ids are persisted (SQLite, or session_state when DB_ENABLED=false), so a rerun or a refresh
    never announces one twice. A failure here never interrupts the review.
    """
    items = reviews()
    try:
        if config.DB_ENABLED:
            total = achv.add_processed(new_forms) if new_forms else achv.processed_total()
            counts = record_store.sender_counts()
        else:
            st.session_state.ach_processed += new_forms
            total = st.session_state.ach_processed
            counts = overview.counts_from_values(r.value("sending_entity") for r in items)
        stats = achv.Stats(processed_total=total, batch_size=len(items),
                           batch_approved=len(items) - len(rs.unapproved(items)), batch_finished=batch_finished,
                           known_senders=overview.known_sender_count(overview.breakdown(counts)), hour=hour)
        earned = achv.evaluate(stats)
        if config.DB_ENABLED:
            new = achv.unlock(earned)
        else:
            known = st.session_state.ach_unlocked
            new = [a.id for a in achv.ACHIEVEMENTS if a.id in earned and a.id not in known]
            known.update({i: "" for i in new})
    except Exception:  # sqlite3.Error etc.: achievements are decoration, never block the work
        return
    st.session_state.toasts += new


def show_toasts() -> None:
    queued, st.session_state.toasts = st.session_state.toasts, []
    for id_ in queued:
        st.toast(f"הישג חדש: {achv.BY_ID[id_].title}", icon=":material/workspace_premium:")


def achievements_panel() -> None:
    try:
        done = achv.unlocked() if config.DB_ENABLED else dict(st.session_state.ach_unlocked)
    except Exception:
        return
    with st.expander(f"הישגים ({len(done)}/{len(achv.ACHIEVEMENTS)})", icon=":material/workspace_premium:"):
        for a in achv.ACHIEVEMENTS:
            if a.id in done:
                when = f" · {done[a.id][:10]}" if done[a.id] else ""
                st.markdown(f'<div style="margin-bottom:8px"><strong style="color:{APPROVED_COLOR}">✓ {a.title}</strong>'
                            f'<div class="lx-muted">{a.description}{when}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="margin-bottom:8px;opacity:.55"><strong>{"הישג מוסתר" if a.hidden else a.title}'
                            f'</strong><div class="lx-muted">{"???" if a.hidden else a.description}</div></div>',
                            unsafe_allow_html=True)


# ------------------------------------------------------------------ activity overview

OVERVIEW_COLORS = {overview.KIND_LABELS[overview.KIND_HMO]: APPROVED_COLOR,
                   overview.KIND_LABELS[overview.KIND_HOSPITAL]: "#7AA2F7",
                   overview.KIND_LABELS[overview.KIND_OTHER]: "#93A0BC",
                   overview.KIND_LABELS[overview.KIND_UNKNOWN]: "#5B6685"}


def overview_section(counts: dict) -> None:
    """Forms per sending entity, from the stored records: tiles by kind and a hover bar chart.

    Hospitals have a city but no coordinates in the approved list, so this is a chart, not a map (no
    location is guessed). Orange is not used: it is reserved for fields that need review.
    """
    st.markdown('<div class="lx-fields-head" style="margin-top:8px"><strong style="font-size:18px">'
                'סקירת פעילות לפי גורם שולח (טפסים מאושרים)</strong></div>', unsafe_allow_html=True)
    rows = overview.breakdown(counts)
    if not rows:
        st.markdown('<div class="lx-empty">עדיין אין נתונים להצגה.</div>', unsafe_allow_html=True)
        return
    kinds = overview.totals_by_kind(rows)
    tiles = "".join(
        f'<div class="lx-tile"><div class="lx-tile-label">{overview.KIND_LABELS[k]}</div>'
        f'<div class="lx-tile-value">{kinds[k]["count"]}'
        f'<span class="lx-muted" style="font-size:14px"> · {kinds[k]["percent"]}%</span></div></div>'
        for k in overview.KIND_LABELS if kinds[k]["count"])
    st.markdown(f'<div class="lx-tiles">{tiles}</div>', unsafe_allow_html=True)
    frame = pd.DataFrame(rows)
    total = int(frame["count"].sum())
    st.markdown(f'<div class="lx-muted">סה״כ {total} טפסים מאושרים; האחוזים מחושבים מתוך סכום זה.</div>',
                unsafe_allow_html=True)
    hover = alt.selection_point(fields=["name"], on="pointerover", clear="pointerout")
    base = alt.Chart(frame).encode(
        y=alt.Y("name:N", sort="-x", title=None),
        x=alt.X("count:Q", title="מספר טפסים", axis=alt.Axis(tickMinStep=1, format="d")))
    bars = base.mark_bar(cornerRadiusEnd=4, size=16).encode(
        color=alt.Color("kind_label:N", title="סוג",
                        scale=alt.Scale(domain=list(OVERVIEW_COLORS), range=list(OVERVIEW_COLORS.values()))),
        opacity=alt.condition(hover, alt.value(1), alt.value(0.6)),
        tooltip=[alt.Tooltip("name:N", title="גורם"), alt.Tooltip("kind_label:N", title="סוג"),
                 alt.Tooltip("count:Q", title="טפסים"), alt.Tooltip("percent:Q", title="אחוז מהסך", format=".1f")])
    labels = base.mark_text(align="left", dx=4).encode(text=alt.Text("count:Q", format="d"))
    chart = (bars + labels).add_params(hover).properties(height=max(120, 28 * len(frame)))
    st.altair_chart(chart, width="stretch")
    with st.expander("תצוגת טבלה"):
        table = frame.rename(columns={"name": "גורם", "kind_label": "סוג", "count": "טפסים", "percent": "אחוז"})
        st.dataframe(table[["גורם", "סוג", "טפסים", "אחוז"]], hide_index=True)


# ------------------------------------------------------------------ screens

def hero(kicker: str, title: str, lead: str) -> None:
    """Hero band at the top of a screen (soft pink gradient with a ring, as on the lab's Sheba web page)."""
    st.markdown(f'<div class="lx-hero"><p class="lx-kicker">{kicker}</p>'
                f'<h1 class="lx-h1">{title}</h1><p class="lx-lead">{lead}</p></div>', unsafe_allow_html=True)


def screen_header(step: int, name: str, title: str, lead: str) -> None:
    hero(f"שלב {step} מתוך 4 · {name}", title, lead)


def screen_upload() -> None:
    screen_header(1, "העלאה", "קליטת טפסי התחייבות",
                  "גררו את כל הטפסים של היום – סריקות, צילומי טלפון או PDF מרובה עמודים. המערכת מפצלת לטפסים, "
                  "מזהה את הגורם השולח ומחלצת את השדות. כל שדה לא ודאי יסומן בכתום לבדיקה שלכם.")
    problem = config_problem()
    if problem:
        st.error(problem, icon=":material/error:")
    files = st.file_uploader("קבצים", type=UPLOAD_TYPES, accept_multiple_files=True,
                             key=f"uploader_{st.session_state.uploader_n}", label_visibility="collapsed")
    label = "חלץ נתונים"
    if reviews():
        st.warning("חילוץ חדש יחליף את התוצאות הקיימות, כולל תיקונים ואישורים שלא יוצאו. כדאי לייצא קודם.",
                   icon=":material/warning:")
        label = "חלץ והחלף את התוצאות"
    if problem:
        hint = "יש להשלים את ההגדרות"
    elif not files:
        hint = "בחרו קבצים כדי להתחיל"
    else:
        hint = f"{len(files)} קבצים · הזמן משתנה לפי מספר העמודים"
    button_col, hint_col = st.columns([1, 3], vertical_alignment="center")  # RTL: the button is on the right
    clicked = button_col.button(label, type="primary", disabled=not files or bool(problem), key="extract",
                                icon=":material/bolt:")
    hint_col.markdown(f'<span class="lx-muted">{hint}</span>', unsafe_allow_html=True)
    progress_slot, steps_slot = st.empty(), st.empty()
    steps_slot.markdown(steps_html(), unsafe_allow_html=True)
    if clicked:
        run_extraction(files, progress_slot, steps_slot)
    if st.session_state.run_error:
        st.error(f"החילוץ נכשל: {st.session_state.run_error}")


def _rtl_columns(df: pd.DataFrame) -> pd.DataFrame:
    """st.dataframe is left-to-right: reverse the columns so the table reads from the right."""
    return df[list(reversed(df.columns))]


def screen_summary() -> None:
    items = reviews()
    flagged = sum(len(r.record.flags) for r in items)
    approved = len(items) - len(rs.unapproved(items))
    screen_header(2, "סיכום", "החילוץ הסתיים",
                  f"{len(items)} טפסים עובדו, {flagged} שדות סומנו לבדיקה, {approved} טפסים אושרו.")
    nxt = rs.next_unapproved(items, -1)
    if nxt is not None and st.button("התחל סקירה מהטופס הראשון שלא אושר", type="primary", key="start_review",
                                     icon=":material/arrow_back:", icon_position="right"):
        go("review", nxt)
        st.rerun()

    df = rs.summary_frame(items)
    files = df["source_file"].nunique()
    pages = sum(len(str(p).split(LIST_SEPARATOR)) for p in df["page_numbers"] if str(p))
    flagged_forms = int((df["flagged_fields"] > 0).sum())
    warnings = int(df["form_warnings"].sum())
    warning_forms = int((df["form_warnings"] > 0).sum())
    share = round(100 * approved / len(items)) if items else 0
    st.markdown(f"""
<div class="lx-tiles">
  <div class="lx-tile"><div class="lx-tile-label">טפסים</div><div class="lx-tile-value">{len(items)}</div>
    <div class="lx-tile-note">מתוך {files} קבצים · {pages} עמודים</div></div>
  <div class="lx-tile flag"><div class="lx-tile-label">שדות לבדיקה</div><div class="lx-tile-value">{flagged}</div>
    <div class="lx-tile-note">ב־{flagged_forms} טפסים</div></div>
  <div class="lx-tile flag"><div class="lx-tile-label">אזהרות טופס</div><div class="lx-tile-value">{warnings}</div>
    <div class="lx-tile-note">ב־{warning_forms} טפסים · פיצול או חילוץ</div></div>
  <div class="lx-tile ok"><div class="lx-tile-label">אושרו</div>
    <div class="lx-tile-value">{approved}<span style="font-size:18px;color:#4E9C8A"> / {len(items)}</span></div>
    <div class="lx-bar"><div style="width:{share}%"></div></div></div>
</div>""", unsafe_allow_html=True)

    # One row per commitment with every field (the lab's record), orange on flagged fields of forms not yet
    # approved. A form with several commitment numbers repeats on several rows; a click opens its form.
    rows, highlight = rs.export_frame(items, with_form_index=True)
    form_of_row = rows.pop(rs.FORM_INDEX_KEY).tolist()
    rows.insert(0, "position", [pos + 1 for pos in form_of_row])
    headers = {**ALL_HEADERS, REVIEW_STATUS_KEY: REVIEW_STATUS_HEADER, "position": "טופס"}
    view = _rtl_columns(rows.rename(columns=headers))

    def colour(frame: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
        status = frame.columns.get_loc(REVIEW_STATUS_HEADER)
        for pos in range(len(frame)):
            if frame.iat[pos, status] == REVIEW_STATUS_APPROVED:
                styles.iat[pos, status] = f"color: {APPROVED_COLOR}; font-weight: 600"
        for pos, key in highlight:  # dark text on orange: readable on the dark theme
            styles.iloc[pos, frame.columns.get_loc(headers[key])] = (f"background-color: {FLAG_COLOR}; "
                                                                      "color: #1A1100; font-weight: 600")
        return styles

    st.markdown('<div class="lx-section" style="font-size:18px;font-weight:600;color:#1B1A6B">כל הרשומות</div>'
                f'<span class="lx-muted">{len(rows)} רשומות (שורה לכל מספר התחייבות) · לחצו על שורה כדי לפתוח '
                'את הטופס · <span class="lx-legend"></span>דורש בדיקה</span>', unsafe_allow_html=True)
    widths = {headers[key]: st.column_config.Column(width=w) for key, w in EXPORT_WIDTHS.items()}
    event = st.dataframe(view.style.apply(colour, axis=None), hide_index=True, on_select="rerun",
                         selection_mode="single-row", column_config=widths,
                         key=f"summary_{st.session_state.nonce}")
    if event.selection.rows:
        go("review", form_of_row[event.selection.rows[0]])
        st.rerun()


def _confidence_chip(value: float) -> str:
    """The extractor reports high / medium / low (constants.CONFIDENCE_LABELS); only high passes the threshold."""
    if value >= config.CONFIDENCE_THRESHOLD:
        return '<span class="lx-chip ok">ביטחון גבוה</span>'
    return f'<span class="lx-chip warn">ביטחון {"בינוני" if value >= 0.5 else "נמוך"}</span>'


def _field_card(review: rs.FormReview, key: str, senders: list, nonce: int) -> object:
    """One field card inside the form. Returns the input's current value."""
    rec = review.record
    flag = review.open_flags().get(key)
    card_key = f"{'cardflag' if flag else 'cardok'}_{review.form_index}_{key}_{nonce}"
    with st.container(border=True, key=card_key):
        chip = _confidence_chip(rec.confidences[key]) if key in rec.confidences and not rec.error else ""
        title = f'<div class="lx-card-head"><strong>{FIELD_HEADERS[key]}</strong>{chip}</div>'
        if flag:
            title += f'<div class="flag-reason">⚠ {FLAG_REASONS.get(flag, flag)}</div>'
        st.markdown(title, unsafe_allow_html=True)

        # Crop on one side, the editable value on the other: the reviewer compares them at a glance.
        crop_col, input_col = st.columns([1, 1], gap="medium", vertical_alignment="center")
        crops = rec.field_crops.get(key, [])
        with crop_col:
            for crop in crops:
                caption = None
                if len(rec.page_images) > 1 and crop.page <= len(rec.page_images):
                    caption = f"עמוד {rec.page_images[crop.page - 1].page_number}"
                st.image(crop.jpeg, caption=caption, width="stretch")
            if not crops and not rec.error:
                st.caption("אין חיתוך מקור: המודל לא סימן היכן הערך מופיע בטופס.")

        with input_col:
            input_key = f"in_{review.form_index}_{key}_{nonce}"
            current = review.value(key)
            if key == "sending_entity":
                options = [""] + senders + ([current] if current and current not in senders else [])
                value = st.selectbox(FIELD_HEADERS[key], options, index=options.index(current), key=input_key,
                                     label_visibility="collapsed", format_func=lambda o: o or "— לא נבחר —")
            else:
                value = st.text_input(FIELD_HEADERS[key], value=current, key=input_key,
                                      label_visibility="collapsed",
                                      placeholder="להקלדה מתוך התמונה" if rec.error else "ריק – להשלמה")

        suggestion = rec.suggestions.get(key)
        if suggestion and not review.value(key):
            st.markdown(f'<span class="suggestion">{SUGGESTION_LABEL}</span> · העתיקו לשדה רק אם הקריאה נכונה',
                        unsafe_allow_html=True)
            st.code(suggestion, language=None)  # st.code has a copy button
        details = []
        alternatives = rec.alternatives.get(key)
        if alternatives and flag:
            details.append("קריאות נוספות של המנוע: " + " | ".join(alternatives))
        if rec.notes.get(key):
            details.append(f"מקור לפי המודל: {rec.notes[key]}")
        raw_text = rec.raw_texts.get(key)
        if raw_text and raw_text != review.value(key):
            # A reading aid, never a value: it may be partial ("?" = a character the model could not read).
            details.append(f"טקסט כפי שנקרא (לא ערך מאושר, ? = תו לא קריא): {raw_text}")
        if details:
            st.caption(" · ".join(details))
    return value


def _warning_lines(warnings: dict) -> str:
    return "\n".join(f"- {FIELD_HEADERS[k]}: {FLAG_REASONS.get(code, code)}" for k, code in warnings.items())


def _after_approval(pos: int, empty: list) -> None:
    award(hour=datetime.now().hour)
    items = reviews()
    note = f"טופס {pos + 1} אושר."
    if empty:
        note += " שדות ריקים: " + ", ".join(FIELD_HEADERS[k] for k in empty) + "."
    st.session_state.notice = note
    nxt = rs.next_unapproved(items, pos)
    if nxt is None:
        st.session_state.notice = note + " כל הטפסים אושרו – אפשר לייצא."
        go("export")
    else:
        go("review", nxt)


def _page_with_flags(review: rs.FormReview, page: int, jpeg: bytes) -> tuple:
    """(preview with the open-flag boxes of this page outlined in orange, names of the outlined fields)."""
    boxes, names = [], []
    for key in review.open_flags():
        located = [c.box for c in review.record.field_crops.get(key, []) if c.page == page]
        if located:
            boxes += [b for b in located if b not in boxes]
            names.append(FIELD_HEADERS[key])
    return outline_boxes(jpeg, boxes), names


def screen_review(senders: list) -> None:
    items = reviews()
    pos = min(st.session_state.pos, len(items) - 1)
    review = items[pos]
    rec = review.record
    nonce = st.session_state.nonce
    meta = rec.metadata

    head_col, nav_col = st.columns([3, 1], vertical_alignment="bottom")
    with head_col:
        chip = ('<span class="lx-chip ok big">אושר</span>' if review.approved
                else '<span class="lx-chip warn big">ממתין לאישור</span>')
        st.markdown(
            f'<p class="lx-kicker">שלב 3 מתוך 4 · סקירה</p>'
            f'<div class="lx-title-row"><h1 class="lx-h1">טופס {pos + 1} <span class="lx-dim">/ {len(items)}</span>'
            f' · {meta.get("form_type", "") or "לא מזוהה"}</h1>{chip}</div>'
            f'<p class="lx-muted">קובץ <bdi class="lx-mono">{meta.get("source_file", "")}</bdi>'
            f' · עמודים <bdi dir="ltr">{meta.get("page_numbers", "")}</bdi></p>', unsafe_allow_html=True)
    with nav_col:
        prev_col, next_col = st.columns(2)
        if prev_col.button("הקודם", disabled=pos == 0, width="stretch", key="prev",
                          icon=":material/chevron_left:"):  # Material icons mirror in RTL: points right
            go("review", pos - 1)
            st.rerun()
        if next_col.button("הבא", disabled=pos == len(items) - 1, width="stretch", key="next",
                           icon=":material/chevron_right:"):
            go("review", pos + 1)
            st.rerun()

    if rec.error:
        # The error text holds only status codes / error types, never form content.
        st.error("**החילוץ האוטומטי נכשל בטופס זה – יש להקליד את השדות ידנית מתוך התמונה.**\n\n"
                 f"פרטי השגיאה: {rec.error}", icon=":material/error:")
    for code in rec.split_flags:
        st.warning(SPLIT_FLAG_LABELS.get(code, code), icon=":material/warning:")

    fields_col, image_col = st.columns([6, 5], gap="large")  # RTL: the first column is on the right
    with image_col, st.container(border=True, key="formimage"):
        pages = rec.page_images
        if pages:
            index = 0
            if len(pages) > 1:
                numbers = [p.page_number for p in pages]
                chosen = st.radio("עמוד", numbers, horizontal=True, key=f"page_{rec.form_index}_{nonce}",
                                  format_func=lambda n: f"עמוד {n}", label_visibility="collapsed")
                index = numbers.index(chosen)
            image, outlined = _page_with_flags(review, index + 1, pages[index].jpeg)
            st.image(image, width="stretch")
            if outlined:
                st.markdown('<span class="lx-muted"><span class="lx-outline"></span>מסומנים בכתום על הטופס: '
                            + ", ".join(outlined) + "</span>", unsafe_allow_html=True)
        else:
            st.info("אין תמונה זמינה לטופס זה.")

    with fields_col:
        open_count = len(review.open_flags())
        st.markdown('<div class="lx-fields-head"><strong>שדות שחולצו</strong><span class="lx-muted">'
                    + (f"{open_count} מתוך {len(FIELD_KEYS)} שדות לבדיקה" if open_count else
                       ("כל השדות אושרו" if review.approved else "אין שדות מסומנים"))
                    + "</span></div>", unsafe_allow_html=True)
        # enter_to_submit=False: Enter in a field must never approve a form by accident.
        with st.form(key=f"fields_{rec.form_index}_{nonce}", border=False, enter_to_submit=False):
            entered = {}
            swap_clicked = False
            for key in FIELD_KEYS:  # one full-width card per field, so the crop is large enough to read
                entered[key] = _field_card(review, key, senders, nonce)
                if key == "last_name":
                    # The forms do not write first / last name in a fixed order: one click fixes a swapped pair.
                    swap_clicked = st.form_submit_button("החלף בין שם פרטי לשם משפחה", key="swap_names",
                                                         icon=":material/swap_horiz:")
            approve_clicked = save_clicked = reopen_clicked = False
            with st.container(key="actionbar", horizontal=True, vertical_alignment="center"):
                if review.approved:
                    st.markdown('<span class="lx-approved">✓ הטופס אושר</span>', unsafe_allow_html=True)
                    reopen_clicked = st.form_submit_button("בטל אישור", key="reopen")
                else:
                    approve_clicked = st.form_submit_button("אשר טופס ועבור לבא", type="primary", key="approve",
                                                            icon=":material/check:")
                    save_clicked = st.form_submit_button("שמור בלי לאשר", key="save")

        if approve_clicked:
            result = rs.approve(review, entered)
            persist([review])
            if result.approved:
                _after_approval(pos, result.empty)
            else:
                st.session_state.pending = {"pos": pos, "warnings": result.warnings}
                st.session_state.nonce += 1
            st.rerun()
        if swap_clicked:
            rs.save_edits(review, rs.swap_names(entered))
            rs.reopen(review)
            persist([review])
            st.session_state.notice = "השם הפרטי ושם המשפחה הוחלפו ונשמרו. הטופס עדיין לא אושר."
            st.session_state.nonce += 1
            st.rerun()
        if save_clicked:
            rs.save_edits(review, entered)
            rs.reopen(review)
            persist([review])
            st.session_state.notice = "התיקונים נשמרו. הטופס עדיין לא אושר."
            st.session_state.nonce += 1
            st.rerun()
        if reopen_clicked:
            rs.reopen(review)
            persist([review])
            st.session_state.nonce += 1
            st.rerun()

        pending = st.session_state.pending
        if pending and pending["pos"] == pos and not review.approved:
            st.warning("**בדיקת התקינות נכשלה** – תקנו ואשרו שוב, או אשרו בכל זאת אם הערך נכון "
                       "(למשל מספר דרכון):\n" + _warning_lines(pending["warnings"]), icon=":material/warning:")
            if st.button("אשר בכל זאת", key="force_approve"):
                result = rs.approve(review, {}, force=True)
                persist([review])
                _after_approval(pos, result.empty)
                st.rerun()


def screen_export() -> None:
    items = reviews()
    df, highlight = rs.export_frame(items)
    waiting = rs.unapproved(items)
    file_name = f"labextract_{date.today():%Y-%m-%d}.xlsx"
    screen_header(4, "ייצוא", "ייצוא לאקסל" if waiting else "הקובץ מוכן להורדה",
                  "שורה לכל מספר התחייבות, עמודה לכל שדה, כותרות בעברית וגיליון מימין לשמאל. "
                  "שדות שעדיין לא נבדקו מסומנים בכתום גם בקובץ.")

    download_col, status_col = st.columns(2, gap="medium")  # RTL: the download card is on the right
    with download_col, st.container(border=True, key="downloadcard"):
        st.markdown(f'<div class="lx-file"><div class="lx-file-icon"></div><div><bdi class="lx-mono">{file_name}</bdi>'
                    f'<div class="lx-muted">{len(df)} שורות · {len(df.columns)} עמודות · גיליון מימין לשמאל</div>'
                    '</div></div>', unsafe_allow_html=True)
        allowed = not waiting
        if waiting:
            allowed = st.checkbox("אני מבין/ה – ייצא בכל זאת (השדות שלא נבדקו יסומנו בכתום)", key="export_anyway")
        st.download_button("הורד קובץ Excel", data=to_xlsx_bytes(df, highlight), file_name=file_name,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary", disabled=not allowed, on_click="ignore", key="download",
                           icon=":material/download:", width="stretch")
        st.markdown('<span class="lx-muted">הקובץ נוצר בזיכרון ויורד ישירות למחשב שלכם – הוא לא נשמר בשרת.</span>',
                    unsafe_allow_html=True)

    with status_col:
        if not waiting:
            st.markdown(f'<div class="lx-ready"><strong>✓ כל {len(items)} הטפסים אושרו</strong>'
                        '<span>כל ערך בקובץ נבדק ואושר על ידכם. שדות שהושארו ריקים מופיעים ריקים – בלי ניחוש.</span>'
                        '</div>', unsafe_allow_html=True)
        else:
            with st.container(border=True, key="waitingcard"):
                st.warning(f"{len(waiting)} טפסים עדיין לא אושרו", icon=":material/warning:")
                for r in waiting:
                    pos = items.index(r)
                    meta = r.record.metadata
                    flags = len(r.open_flags())
                    text_col, open_col = st.columns([5, 1], vertical_alignment="center")
                    text_col.markdown(
                        f'<div class="lx-waiting"><strong>טופס {pos + 1}</strong> <span>{meta.get("form_type", "")} · '
                        f'<bdi>{meta.get("source_file", "")}</bdi></span> <em>'
                        + (f"{flags} שדות לבדיקה" if flags else "ללא סימונים") + "</em></div>",
                        unsafe_allow_html=True)
                    open_col.button("פתח", key=f"open_{pos}", on_click=go, args=("review", pos), width="stretch")

    headers = {**ALL_HEADERS, REVIEW_STATUS_KEY: REVIEW_STATUS_HEADER}
    view = _rtl_columns(df.rename(columns=headers))

    def colour(frame: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
        status = frame.columns.get_loc(REVIEW_STATUS_HEADER)
        for pos in range(len(frame)):
            if frame.iat[pos, status] == REVIEW_STATUS_APPROVED:
                styles.iat[pos, status] = f"color: {APPROVED_COLOR}; font-weight: 600"
        for pos, key in highlight:  # dark text on orange: readable on the dark theme
            styles.iloc[pos, frame.columns.get_loc(headers[key])] = (f"background-color: {FLAG_COLOR}; "
                                                                      "color: #1A1100; font-weight: 600")
        return styles

    st.markdown('<div class="lx-fields-head" style="margin-top:20px"><strong style="font-size:18px">'
                'תצוגה מקדימה של הקובץ</strong><span class="lx-muted"><span class="lx-legend"></span>'
                'שדה מסומן בטופס שלא אושר</span></div>', unsafe_allow_html=True)
    widths = {headers[key]: st.column_config.Column(width=w) for key, w in EXPORT_WIDTHS.items()}
    st.dataframe(view.style.apply(colour, axis=None), hide_index=True, column_config=widths)


def open_stored(batch_id: str, form_index: int) -> None:
    """Load a saved run into this window and open one of its forms on the review screen."""
    loaded = record_store.load_batch(batch_id)
    if not loaded:
        st.session_state.notice_error = "לא ניתן לפתוח את הרשומה: היא נשמרה לפני שהמאגר שמר גם את התמונות."
        return
    if not any(r.record.field_crops for r in loaded):
        st.session_state.notice_error = ("הרשומה נפתחה בלי תמונות: התמונות לא נשמרו במאגר או נמחקו. "
                                         "אפשר לערוך ולאשר רק לפי הערכים.")
    st.session_state.reviews = loaded
    st.session_state.batch_id = batch_id
    st.session_state.uploader_n += 1
    go("review", next((i for i, r in enumerate(loaded) if r.form_index == form_index), 0))


ASSISTANT_MAX_ROWS = 200  # rows drawn in the results table; the answer still counts all of them


def _ask(question: str) -> None:
    """Button callback: queue a question. It is interpreted in the script body, under a spinner."""
    st.session_state.assistant_q = question
    st.session_state.assistant_pending = question


def _ask_typed() -> None:
    _ask(st.session_state.get("assistant_q", ""))


def archive_assistant_section(present_senders: list) -> None:
    """Questions in Hebrew about the stored records (src/archive_assistant.py).

    Fixed rules read the usual questions; Claude is asked only about a question they do not understand, with
    the question text alone, at most once per distinct question (kept in session_state, so reruns never call
    it again). The answer itself is always computed from the database rows, never by the model.
    """
    with st.container(border=True, key="assistant"):
        st.markdown('<div class="lx-fields-head"><strong style="font-size:18px">עוזר המאגר</strong>'
                    '<span class="lx-muted">שאלה בעברית · התשובה מחושבת מהרשומות השמורות</span></div>',
                    unsafe_allow_html=True)
        with st.container(horizontal=True, key="assistant_examples"):
            for i, example in enumerate(aa.EXAMPLES):
                st.button(example, key=f"ask_ex_{i}", on_click=_ask, args=(example,))
        with st.form("assistant_form", border=False):
            text_col, button_col = st.columns([5, 1], vertical_alignment="bottom")  # RTL: the input on the right
            text_col.text_input("שאלה", key="assistant_q", max_chars=aa.MAX_QUESTION_CHARS,
                                placeholder="למשל: כמה טפסים ממכבי ממתינים לבדיקה?", label_visibility="collapsed")
            button_col.form_submit_button("שאל", type="primary", on_click=_ask_typed, icon=":material/search:",
                                          width="stretch")
        st.markdown('<span class="lx-muted">שאלות שהכללים הקבועים לא מבינים נשלחות ל-Claude כטקסט בלבד, בלי רשומות. '
                    'אל תקלידו שמות או מספרי זהות.</span>', unsafe_allow_html=True)

        question = st.session_state.assistant_pending
        if question is not None:
            st.session_state.assistant_pending = None
            key = " ".join(question.split())
            result = st.session_state.assistant_cache.get(key)
            if result is None:
                with st.spinner("מפרש את השאלה…"):
                    result = aa.interpret(question, present_senders=present_senders)
                if getattr(result, "source", "") == "model":  # rules are free to rerun; model answers are not
                    st.session_state.assistant_cache[key] = result
            st.session_state.assistant_result = result
        if st.session_state.assistant_result is not None:
            _assistant_answer(st.session_state.assistant_result)


def _assistant_answer(result) -> None:
    if isinstance(result, aa.Clarification):
        st.info(result.message, icon=":material/help:")
        if result.options:
            with st.container(horizontal=True, key="assistant_options"):
                for i, (label, question) in enumerate(result.options):
                    st.button(label, key=f"ask_opt_{i}", on_click=_ask, args=(question,))
        return
    try:  # re-run on every render: the answer always reflects the database as it is now
        answer = aa.run_query(result, record_store.archive_rows())
    except Exception as exc:
        st.error(f"החיפוש במאגר נכשל ({type(exc).__name__}).")
        return
    origin = ("פוענח בעזרת Claude – ודאו שהפירוש מתאים לשאלה" if result.source == "model"
              else "פוענח לפי כללים קבועים")
    chips = "".join(f'<span class="lx-chip info">{html.escape(c)}</span>' for c in aa.describe(result))
    st.markdown(f'<div class="lx-answer">{html.escape(answer.message)}</div>'
                f'<div class="lx-chips"><span class="lx-muted">הבנתי:</span>{chips}'
                f'<span class="lx-muted">· {origin}</span></div>'
                + "".join(f'<div class="lx-note">⚠ {html.escape(n)}</div>' for n in answer.notes),
                unsafe_allow_html=True)
    if answer.groups:
        table = pd.DataFrame(answer.groups).rename(columns={"label": aa.GROUP_LABELS[result.group_by],
                                                            "count": "טפסים", "percent": "אחוז"})
        st.dataframe(_rtl_columns(table), hide_index=True)
        return
    if not answer.rows:
        return
    shown = answer.rows[:ASSISTANT_MAX_ROWS]
    view = _rtl_columns(pd.DataFrame(aa.display_rows(aa.Answer(answer.message, answer.count, rows=shown))))
    st.markdown('<span class="lx-muted">לחצו על שורה כדי לפתוח את הטופס במסך הסקירה · שמות לא מוצגים ות״ז מוסתרת'
                + (f" · מוצגות {len(shown)} השורות הראשונות" if len(shown) < len(answer.rows) else "")
                + "</span>", unsafe_allow_html=True)
    event = st.dataframe(view, hide_index=True, on_select="rerun", selection_mode="single-row",
                         key=f"assistant_rows_{st.session_state.nonce}")
    if event.selection.rows:
        row = shown[event.selection.rows[0]]
        open_stored(row["batch_id"], int(row["form_index"]))
        st.rerun()


def screen_database() -> None:
    hero("מאגר · שמירה מקומית", "מאגר הרשומות",
         "כל טופס שחולץ נשמר כאן – ממתין לאישור או מאושר – וזמין גם אחרי סגירת הדפדפן. "
         "חפשו, סננו, פתחו רשומה לתיקון ואישור – ליד כל שדה תופיע התמונה החתוכה מטופס ההתחייבות.")
    if not config.DB_ENABLED:
        st.info("המאגר כבוי (DB_ENABLED=false).")
        return
    try:
        total = record_store.counts()
        known_senders = record_store.senders()
    except Exception as exc:
        st.error(f"לא ניתן לפתוח את המאגר ({type(exc).__name__}).")
        return
    st.markdown(f"""
<div class="lx-tiles">
  <div class="lx-tile"><div class="lx-tile-label">טפסים במאגר</div><div class="lx-tile-value">{total["total"]}</div></div>
  <div class="lx-tile ok"><div class="lx-tile-label">אושרו</div><div class="lx-tile-value">{total["approved"]}</div></div>
  <div class="lx-tile flag"><div class="lx-tile-label">ממתינים</div><div class="lx-tile-value">{total["pending"]}</div></div>
</div>""", unsafe_allow_html=True)

    if config.ARCHIVE_ASSISTANT_ENABLED:
        archive_assistant_section(known_senders)

    try:
        overview_section(record_store.sender_counts(approved_only=True))
    except Exception as exc:
        st.warning(f"הסקירה לא זמינה ({type(exc).__name__}).")

    text_col, status_col, sender_col = st.columns([3, 2, 2])
    text = text_col.text_input("חיפוש חופשי", placeholder="שם, ת״ז, מספר מרכזי, מספר התחייבות…", key="db_text")
    status_label = status_col.selectbox("סטטוס", ["הכל", "אושר", "לא אושר"], key="db_status")
    sender = sender_col.selectbox("גורם שולח", ["הכל", *known_senders], key="db_sender")
    status = {"אושר": record_store.STATUS_APPROVED, "לא אושר": record_store.STATUS_PENDING}.get(status_label)
    stored = record_store.search(text, status, None if sender == "הכל" else sender)
    labels = {row.id: f"רשומה {row.id} · {row.first_name} {row.last_name} · {row.central_numbers} · "
                      f"{row.source_file} עמ׳ {row.page_numbers}" for row in stored.itertuples()}
    if stored.empty:
        st.markdown('<div class="lx-empty">אין רשומות שמתאימות לחיפוש.</div>', unsafe_allow_html=True)
    else:
        pick_col, open_col = st.columns([4, 1], vertical_alignment="bottom")  # RTL: the picker on the right
        chosen = pick_col.selectbox("פתיחת רשומה לתיקון ואישור", list(labels), format_func=labels.get,
                                    key="db_pick")
        if open_col.button("פתח", key="db_open", type="primary", icon=":material/open_in_new:", width="stretch"):
            row = stored.loc[stored["id"] == chosen].iloc[0]
            open_stored(row["batch_id"], int(row["form_index"]))
            st.rerun()
        frame, highlight = record_store.export_frame(stored)
        headers = {**ALL_HEADERS, REVIEW_STATUS_KEY: REVIEW_STATUS_HEADER}
        view = _rtl_columns(frame.rename(columns=headers))

        def colour(df):  # same orange rule as the review: an open flag of a form not yet approved
            cols = list(df.columns)  # the view's (right-to-left) order, not `frame`'s
            out = pd.DataFrame("", index=df.index, columns=df.columns)
            for row, key in highlight:
                out.iloc[row, cols.index(headers[key])] = (f"background-color: {FLAG_COLOR}; color: #1A1100; "
                                                           "font-weight: 600")
            return out

        widths = {headers[key]: st.column_config.Column(width=w) for key, w in EXPORT_WIDTHS.items()}
        st.dataframe(view.style.apply(colour, axis=None), hide_index=True, column_config=widths)
        st.download_button("הורד את התוצאות כ-Excel", data=to_xlsx_bytes(frame, highlight),
                           file_name=f"labextract_db_{date.today():%Y-%m-%d}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           on_click="ignore", key="db_download", icon=":material/download:")

    with st.expander("מחיקה (המאגר מכיל פרטים אישיים)"):
        st.markdown('<span class="lx-muted">מוחק לצמיתות מהדיסק. אין שחזור, ואין מחיקה אוטומטית.</span>',
                    unsafe_allow_html=True)
        if not stored.empty:
            del_pick = st.selectbox("רשומה למחיקה", list(labels), format_func=labels.get, key="db_del_pick")
            del_ok = st.checkbox("אני מבין/ה – למחוק את הרשומה הזו עם התמונות שלה", key="db_confirm_delete_one")
            if st.button("מחק רשומה", disabled=not del_ok, key="db_delete_one", icon=":material/delete:"):
                record_store.delete([int(del_pick)])
                st.session_state.notice = "הרשומה נמחקה מהמאגר."
                st.rerun()
        img_ok = st.checkbox("אני מבין/ה – למחוק את כל התמונות (הערכים נשארים)", key="db_confirm_delete_images")
        if st.button("מחק תמונות בלבד", disabled=not img_ok, key="db_delete_images", icon=":material/hide_image:"):
            n = record_store.delete_images()
            st.session_state.notice = f"נמחקו {n} תמונות מהמאגר. הערכים נשמרו."
            st.rerun()
        confirm = st.checkbox("אני מבין/ה – למחוק את כל הרשומות במאגר", key="db_confirm_delete")
        if st.button("מחק את כל המאגר", disabled=not confirm, key="db_delete_all", icon=":material/delete:"):
            n = record_store.delete_all()
            st.session_state.notice = f"נמחקו {n} רשומות מהמאגר."
            st.rerun()


# ------------------------------------------------------------------ main

def _sidebar_css(items: list, screen: str, pos: int) -> str:
    """Per-item CSS for the sidebar: the current step and form, step dots and the batch-list badges.

    Streamlit buttons take only a text label, so state (current, approved, number of open flags) is drawn
    with CSS on each button's key class ("st-key-<key>").
    """
    order = list(STEP_SCREENS)
    rules = [f'[data-testid="stSidebar"] div.st-key-nav_{screen} button {{ background: #FBEAF2; color: #1B1A6B; '
             f'font-weight: 600; }}',
             f'[data-testid="stSidebar"] div.st-key-nav_{screen} button::before {{ background: transparent; '
             f'border: 2px solid {ACCENT_COLOR}; box-sizing: border-box; }}']
    if items and screen in STEP_SCREENS:
        rules += [f'[data-testid="stSidebar"] div.st-key-nav_{s} button::before {{ background: {ACCENT_COLOR}; }}'
                  for s in order[:order.index(screen)]]
    for i, r in enumerate(items):
        sel = f'[data-testid="stSidebar"] div.st-key-queue_{i} button'
        issues = len(r.open_flags()) + len(r.record.split_flags) + bool(r.record.error)
        if r.approved:
            rules.append(f'{sel}::after {{ content: "✓"; background: #E2F4EF; color: {APPROVED_COLOR}; }}')
        elif issues:
            rules.append(f'{sel}::after {{ content: "{issues}"; background: #FFEBC7; color: #9A5800; }}')
        if screen == "review" and i == pos:
            rules.append(f'{sel} {{ background: #FBEAF2; color: #1B1A6B; box-shadow: inset 0 0 0 1px #E9A6C6; }}')
    return "<style>" + "\n".join(rules) + "</style>"


def sidebar(has_results: bool) -> None:
    items = reviews()
    with st.sidebar:
        st.markdown(BRAND_HTML + _sidebar_css(items, st.session_state.screen, st.session_state.pos),
                    unsafe_allow_html=True)
        for screen, label in SCREENS.items():
            st.button(label, key=f"nav_{screen}", type="tertiary", disabled=not has_results and screen not in ("upload", "database"),
                      on_click=go, args=(screen,), width="stretch")
        achievements_panel()
        approved = len(items) - len(rs.unapproved(items))
        st.markdown(f'<div class="lx-section" style="display:flex;justify-content:space-between">'
                    f'<span>טפסים שהועלו</span><span dir="ltr">{approved} / {len(items)}</span></div>'
                    if items else '<div class="lx-section">טפסים שהועלו</div>', unsafe_allow_html=True)
        if not items:
            st.markdown('<div class="lx-empty">עדיין אין טפסים.<br>הטפסים יופיעו כאן אחרי החילוץ.</div>'
                        + PRIVACY_NOTE, unsafe_allow_html=True)
            return
        for i, r in enumerate(items):
            st.button(f"{i + 1} {r.record.metadata.get('form_type', '') or 'לא מזוהה'}", key=f"queue_{i}",
                      type="tertiary", on_click=go, args=("review", i), width="stretch")
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        if st.button("נקה הכל", help="מנקה את המסך. הרשומות נשארות שמורות במאגר", key="clear_all", icon=":material/delete:",
                     width="stretch"):
            clear_all()
            st.rerun()


def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    if config.DEV_CACHE:
        st.error(DEV_CACHE_BANNER, icon="⚠️")

    has_results = bool(reviews())
    if not has_results and st.session_state.screen not in ("upload", "database"):
        st.session_state.screen = "upload"
    sidebar(has_results)

    show_toasts()
    if st.session_state.notice:
        st.success(st.session_state.notice)
        st.session_state.notice = None

    if st.session_state.notice_error:
        st.warning(st.session_state.notice_error)
        st.session_state.notice_error = None

    screen = st.session_state.screen
    if screen == "database":
        screen_database()
    elif screen == "summary":
        screen_summary()
    elif screen == "review":
        screen_review(list_approved_senders(config.SENDERS_PATH))
    elif screen == "export":
        screen_export()
    else:
        screen_upload()
    st.markdown(PARTNER_HTML, unsafe_allow_html=True)  # page footer on every screen


main()
