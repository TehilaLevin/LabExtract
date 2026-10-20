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
import threading
import time
from datetime import date

import pandas as pd
import streamlit as st

from src import config
from src import review_session as rs
from src.constants import (
    ALL_HEADERS,
    FIELD_HEADERS,
    FIELD_KEYS,
    FLAG_COLOR,
    FLAG_REASONS,
    LIST_SEPARATOR,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_HEADER,
    REVIEW_STATUS_KEY,
    SPLIT_FLAG_LABELS,
    SUGGESTION_LABEL,
)
from src.excel_exporter import to_xlsx_bytes
from src.claude_client import LLMError, LLMQuotaError, check_configured
from src.validators import list_approved_senders

st.set_page_config(page_title="LabExtract", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

SCREENS = {"upload": "העלאה", "summary": "סיכום", "review": "סקירה", "export": "ייצוא"}
APPROVED_COLOR = "#5EEAD4"  # mint: actions and approvals ("control room" design, see .streamlit/config.toml)
PIPELINE_STEPS = (
    ("פיצול לטפסים", "כל עמוד: טופס חדש או המשך של הקודם"),
    ("זיהוי סוג טופס", "קופה או בית חולים, מרשימה סגורה"),
    ("חילוץ שדות", "7 שדות, עם רמת ביטחון ומיקום בטופס"),
    ("אימות ונרמול", "ספרת ביקורת בת״ז, תאריכים, מספרים"),
    ("סקירה וייצוא", "אתם מאשרים – ומורידים Excel"),
)
# Pipeline step shown as active for each progress stage reported by process_files.
STAGE_STEP = {"start": 0, "split": 0, "extract": 2}
BRAND_HTML = """
<div class="lx-brand">
  <div class="lx-logo"><svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true" fill="none" stroke="#06221E"
    stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h12v12H4z"/><path d="M7 8h6M7 11h6M7 14h3"/></svg></div>
  <div><div class="lx-name" dir="ltr">LabExtract</div><div class="lx-sub">חילוץ טפסי התחייבות</div></div>
</div>
"""
PRIVACY_NOTE = ('<div class="lx-privacy">הקבצים נשמרים רק בזיכרון של החלון הזה ונמחקים בסיום. '
                'שום נתון אישי לא נכתב ליומן.</div>')
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
.flag-reason {{ color: {FLAG_COLOR}; font-weight: 600; }}
.suggestion {{ color: {FLAG_COLOR}; }}

/* Layout and typography of the "control room" design. Colours and fonts come from the theme. */
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.5rem; max-width: 1400px; }}
.lx-kicker {{ font-size: 13px; font-weight: 600; color: {APPROVED_COLOR}; margin: 0 0 6px; }}
.lx-h1 {{ font-size: 34px; font-weight: 700; line-height: 1.15; margin: 0 0 10px; padding: 0; }}
.lx-lead {{ font-size: 16px; color: #B4BFD6; margin: 0; max-width: 760px; }}
.lx-muted {{ font-size: 13px; color: #8390AC; }}
.lx-section {{ font-size: 12px; font-weight: 500; color: #93A0BC; margin: 18px 0 8px; }}

/* Primary buttons: dark text on mint. */
[data-testid="stBaseButton-primary"] {{ color: #06221E; font-weight: 600; border: none; }}
[data-testid="stBaseButton-primary"]:disabled {{ background: #172036; color: #5B6782; }}
.st-key-extract button, .st-key-start_review button {{
    min-height: 52px; padding: 0 28px; border-radius: 12px; font-size: 16px;
}}
.st-key-extract button:not(:disabled), .st-key-start_review button {{ box-shadow: 0 0 0 4px rgba(94, 234, 212, 0.15); }}

/* Upload drop zone. Streamlit's built-in English texts are hidden (font-size: 0) and replaced with
   Hebrew through ::before / ::after; the widget itself and its behaviour stay Streamlit's. */
[data-testid="stFileUploaderDropzone"] {{
    border: 2px dashed #2F5F5A; background: #0C1A1F; border-radius: 16px; padding: 36px 24px;
    display: flex; flex-direction: column; align-items: center; gap: 14px; text-align: center;
}}
[data-testid="stFileUploaderDropzone"]::before {{
    content: ""; width: 64px; height: 64px; border-radius: 16px; background: #0D2A27 center / 32px no-repeat
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' fill='none' stroke='%235EEAD4' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M16 21V7M10 12l6-6 6 6'/%3E%3Cpath d='M6 20v4a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-4'/%3E%3C/svg%3E");
}}
[data-testid="stFileUploaderDropzone"]::before {{ order: -2; }}
[data-testid="stFileUploaderDropzoneInstructions"] {{ order: -1; }}
[data-testid="stFileUploaderDropzoneInstructions"] span {{ font-size: 0; }}
[data-testid="stFileUploaderDropzoneInstructions"] > div::before {{
    content: "גררו קבצים לכאן"; display: block; font-size: 20px; font-weight: 600; color: #E8EDF7;
}}
[data-testid="stFileUploaderDropzoneInstructions"] > div::after {{
    content: "PDF · PNG · JPG · JPEG · עד 500MB לקובץ"; display: block; font-size: 12px; color: #8390AC; margin-top: 6px;
}}
[data-testid="stFileUploaderDropzone"] button p {{ font-size: 0; }}
[data-testid="stFileUploaderDropzone"] button p::after {{ content: "בחירה מהמחשב"; font-size: 14px; }}
.lx-lead {{ margin-bottom: 22px; }}

/* Sidebar: brand, step navigation, batch list. */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 4px; }}
.lx-brand {{ display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }}
.lx-logo {{ width: 34px; height: 34px; border-radius: 9px; background: {APPROVED_COLOR};
    display: flex; align-items: center; justify-content: center; }}
.lx-name {{ font-weight: 700; font-size: 19px; line-height: 1.2; }}
.lx-sub {{ font-size: 12px; color: #93A0BC; }}
.lx-privacy {{ font-size: 12px; color: #8390AC; border-top: 1px solid #1E2944; padding-top: 14px; margin-top: 18px; }}
.lx-empty {{ border: 1px dashed #243049; border-radius: 10px; padding: 16px 12px; font-size: 13px;
    color: #8390AC; text-align: center; }}
[data-testid="stSidebar"] [class*="st-key-nav_"] button,
[data-testid="stSidebar"] [class*="st-key-queue_"] button {{
    width: 100%; justify-content: flex-start; border: none; background: transparent; color: #C7D0E2;
    min-height: 40px; padding: 6px 12px; border-radius: 8px;
}}
[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover,
[data-testid="stSidebar"] [class*="st-key-queue_"] button:hover {{ background: #172036; color: #E8EDF7; }}
[data-testid="stSidebar"] [class*="st-key-nav_"] button:disabled {{ color: #5B6782; background: transparent; }}
[data-testid="stSidebar"] [class*="st-key-nav_"] button::before {{
    content: ""; width: 8px; height: 8px; border-radius: 50%; background: #243049; margin-inline-end: 10px; flex: none;
}}
[data-testid="stSidebar"] [class*="st-key-nav_"] button > div,
[data-testid="stSidebar"] [class*="st-key-queue_"] button > div {{ flex: 1; min-width: 0; justify-content: flex-start; }}
[data-testid="stSidebar"] [class*="st-key-queue_"] button {{ font-size: 13px; min-height: 36px; }}
[data-testid="stSidebar"] [class*="st-key-queue_"] button::after {{
    content: ""; margin-inline-start: auto; min-width: 22px; height: 22px; border-radius: 999px; font-size: 11px;
    font-weight: 600; display: inline-flex; align-items: center; justify-content: center; background: #172036;
}}
[data-testid="stSidebar"] [class*="st-key-queue_"] [data-testid="stMarkdownContainer"] p {{
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}

/* Stat tiles and pipeline steps. */
.lx-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 8px 0 20px; }}
.lx-tile {{ background: #111829; border: 1px solid #1E2944; border-radius: 12px; padding: 16px 18px; }}
.lx-tile.flag {{ background: #1A1508; border-color: #5C3D00; }}
.lx-tile.ok {{ background: #0D1F1D; border-color: #1C4A44; }}
.lx-tile-label {{ font-size: 13px; color: #93A0BC; }}
.lx-tile.flag .lx-tile-label {{ color: #FFC15C; }}
.lx-tile.ok .lx-tile-label {{ color: {APPROVED_COLOR}; }}
.lx-tile-value {{ font-family: "JetBrains Mono", monospace; font-size: 32px; font-weight: 600; line-height: 1.3; }}
.lx-tile.flag .lx-tile-value {{ color: {FLAG_COLOR}; }}
.lx-tile.ok .lx-tile-value {{ color: {APPROVED_COLOR}; }}
.lx-tile-note {{ font-size: 12px; color: #8390AC; }}
.lx-tile.flag .lx-tile-note {{ color: #C9A86A; }}
.lx-bar {{ height: 6px; background: #12302C; border-radius: 3px; overflow: hidden; margin-top: 8px; }}
.lx-bar > div {{ height: 100%; background: {APPROVED_COLOR}; }}
.lx-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
.lx-step {{ display: flex; flex-direction: column; gap: 6px; border-radius: 12px; padding: 14px;
    background: #111829; border: 1px solid #1E2944; }}
.lx-step.active {{ background: #0D1F1D; border-color: {APPROVED_COLOR}; }}
.lx-step-num {{ width: 24px; height: 24px; border-radius: 999px; display: inline-flex; align-items: center;
    justify-content: center; font-size: 12px; font-weight: 600; background: #172036; color: #93A0BC; }}
.lx-step.active .lx-step-num, .lx-step.done .lx-step-num {{ background: {APPROVED_COLOR}; color: #06221E; }}
.lx-step-title {{ font-weight: 600; font-size: 14px; }}
.lx-step-text {{ font-size: 12px; color: #93A0BC; line-height: 1.45; }}
.lx-principle {{ font-size: 13px; color: #8390AC; margin-top: 24px; }}
.lx-principle span {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; background: {FLAG_COLOR};
    margin-inline-end: 8px; }}
.lx-legend {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; background: {FLAG_COLOR};
    vertical-align: middle; margin-inline-end: 4px; }}
</style>
"""


# ------------------------------------------------------------------ state

def init_state() -> None:
    defaults = {"screen": "upload", "reviews": None, "pos": 0, "uploader_n": 0, "nonce": 0,
                "pending": None, "notice": None, "run_error": None}
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
    st.session_state.run_error = None
    go("summary", 0)
    st.rerun()


# ------------------------------------------------------------------ screens

def screen_header(step: int, name: str, title: str, lead: str) -> None:
    st.markdown(f'<p class="lx-kicker">שלב {step} מתוך 4 · {name}</p><h1 class="lx-h1">{title}</h1>'
                f'<p class="lx-lead">{lead}</p>', unsafe_allow_html=True)


def screen_upload() -> None:
    screen_header(1, "העלאה", "קליטת אצווה של טפסי התחייבות",
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
    st.markdown('<div class="lx-principle"><span></span>העיקרון: שדה ריק ומסומן עדיף תמיד על ערך מנוחש.</div>',
                unsafe_allow_html=True)


def _rtl_columns(df: pd.DataFrame) -> pd.DataFrame:
    """st.dataframe is left-to-right: reverse the columns so the table reads from the right."""
    return df[list(reversed(df.columns))]


def screen_summary() -> None:
    items = reviews()
    flagged = sum(len(r.record.flags) for r in items)
    approved = len(items) - len(rs.unapproved(items))
    head_col, action_col = st.columns([3, 2], vertical_alignment="bottom")
    with head_col:
        screen_header(2, "סיכום", "החילוץ הסתיים",
                      f"{len(items)} טפסים עובדו, {flagged} שדות סומנו לבדיקה, {approved} טפסים אושרו.")
    nxt = rs.next_unapproved(items, -1)
    with action_col:
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
    <div class="lx-tile-value">{approved}<span style="font-size:18px;color:#3E8F84"> / {len(items)}</span></div>
    <div class="lx-bar"><div style="width:{share}%"></div></div></div>
</div>""", unsafe_allow_html=True)

    df.insert(0, "position", range(1, len(df) + 1))
    headers = {"position": "#", "source_file": ALL_HEADERS["source_file"], "page_numbers": ALL_HEADERS["page_numbers"],
               "form_type": ALL_HEADERS["form_type"], "flagged_fields": "שדות לבדיקה",
               "form_warnings": "אזהרות טופס", REVIEW_STATUS_KEY: REVIEW_STATUS_HEADER}
    view = _rtl_columns(df.drop(columns=["form_index"]).rename(columns=headers))

    def colour(row):
        done = row[REVIEW_STATUS_HEADER] == REVIEW_STATUS_APPROVED
        out = []
        for col in row.index:
            hot = not done and ((col in ("שדות לבדיקה", "אזהרות טופס") and row[col] > 0)
                                or col == REVIEW_STATUS_HEADER)
            if hot:  # dark text on orange: readable on the dark theme
                out.append(f"background-color: {FLAG_COLOR}; color: #1A1100; font-weight: 600")
            elif done and col == REVIEW_STATUS_HEADER:
                out.append(f"color: {APPROVED_COLOR}; font-weight: 600")
            else:
                out.append("")
        return out

    st.markdown('<div class="lx-section" style="font-size:18px;font-weight:600;color:#E8EDF7">כל הטפסים</div>'
                '<span class="lx-muted">לחצו על שורה כדי לפתוח את הטופס · <span class="lx-legend"></span>'
                'דורש בדיקה</span>', unsafe_allow_html=True)
    event = st.dataframe(view.style.apply(colour, axis=1), hide_index=True, on_select="rerun",
                         selection_mode="single-row", key=f"summary_{st.session_state.nonce}")
    if event.selection.rows:
        go("review", event.selection.rows[0])
        st.rerun()


def _field_card(review: rs.FormReview, key: str, senders: list, nonce: int) -> object:
    """One field card inside the form. Returns the input's current value."""
    rec = review.record
    flag = review.open_flags().get(key)
    card_key = f"{'cardflag' if flag else 'cardok'}_{review.form_index}_{key}_{nonce}"
    with st.container(border=True, key=card_key):
        title = f"**{FIELD_HEADERS[key]}**"
        if flag:
            title += f' &nbsp; <span class="flag-reason">⚠ {FLAG_REASONS.get(flag, flag)}</span>'
        st.markdown(title, unsafe_allow_html=True)

        crops = rec.field_crops.get(key, [])
        for crop in crops:
            caption = f"עמוד {crop.page}" if len(rec.page_images) > 1 else None
            st.image(crop.jpeg, caption=caption, width="stretch")
        if not crops and not rec.error:
            st.caption("אין חיתוך מקור: המודל לא סימן היכן הערך מופיע בטופס.")

        input_key = f"in_{review.form_index}_{key}_{nonce}"
        current = review.value(key)
        if key == "sending_entity":
            options = [""] + senders + ([current] if current and current not in senders else [])
            value = st.selectbox(FIELD_HEADERS[key], options, index=options.index(current), key=input_key,
                                 label_visibility="collapsed", format_func=lambda o: o or "— לא נבחר —")
        else:
            value = st.text_input(FIELD_HEADERS[key], value=current, key=input_key, label_visibility="collapsed")

        suggestion = rec.suggestions.get(key)
        if suggestion and not review.value(key):
            st.markdown(f'<span class="suggestion">{SUGGESTION_LABEL}:</span> (העתיקו לשדה רק אם הקריאה נכונה)',
                        unsafe_allow_html=True)
            st.code(suggestion, language=None)
        alternatives = rec.alternatives.get(key)
        if alternatives and flag:
            st.caption("קריאות נוספות של המנוע: " + " | ".join(alternatives))
        if key in rec.confidences and not rec.error:
            st.caption(f"ביטחון המודל: {rec.confidences[key]:.0%}")
        if rec.notes.get(key):
            st.caption(f"מקור לפי המודל: {rec.notes[key]}")
        raw_text = rec.raw_texts.get(key)
        if raw_text and raw_text != review.value(key):
            # A reading aid, never a value: it may be partial ("?" = a character the model could not read).
            st.caption(f"טקסט כפי שנקרא (לא ערך מאושר, ? = תו לא קריא): {raw_text}")
    return value


def _warning_lines(warnings: dict) -> str:
    return "\n".join(f"- {FIELD_HEADERS[k]}: {FLAG_REASONS.get(code, code)}" for k, code in warnings.items())


def _after_approval(pos: int, empty: list) -> None:
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


def screen_review(senders: list) -> None:
    items = reviews()
    pos = min(st.session_state.pos, len(items) - 1)
    review = items[pos]
    rec = review.record
    nonce = st.session_state.nonce

    nav = st.columns([1, 1, 1, 3])
    if nav[0].button("→ הקודם", disabled=pos == 0, width="stretch", key="prev"):
        go("review", pos - 1)
        st.rerun()
    if nav[1].button("הבא ←", disabled=pos == len(items) - 1, width="stretch", key="next"):
        go("review", pos + 1)
        st.rerun()
    if nav[2].button("חזרה לסיכום", width="stretch", key="to_summary"):
        go("summary")
        st.rerun()

    status = "✅ אושר" if review.approved else "⏳ ממתין לאישור"
    st.subheader(f"טופס {pos + 1} מתוך {len(items)} · {status}")
    st.caption(f"{review.label()} · {rec.metadata.get('form_type', '')}")
    if rec.error:
        # The error text holds only status codes / error types, never form content.
        st.error("החילוץ האוטומטי נכשל בטופס זה – יש להקליד את השדות ידנית מתוך התמונה.\n\n"
                 f"פרטי השגיאה: {rec.error}")
    for code in rec.split_flags:
        st.warning(SPLIT_FLAG_LABELS.get(code, code))

    fields_col, image_col = st.columns([6, 5], gap="large")  # RTL: the first column is on the right
    with image_col:
        pages = rec.page_images
        if pages:
            shown = pages[0]
            if len(pages) > 1:
                numbers = [p.page_number for p in pages]
                chosen = st.radio("עמוד", numbers, horizontal=True, key=f"page_{rec.form_index}_{nonce}",
                                  format_func=lambda n: f"עמוד {n}")
                shown = pages[numbers.index(chosen)]
            st.image(shown.jpeg, width="stretch")
        else:
            st.info("אין תמונה זמינה לטופס זה.")

    with fields_col:
        # enter_to_submit=False: Enter in a field must never approve a form by accident.
        with st.form(key=f"fields_{rec.form_index}_{nonce}", border=False, enter_to_submit=False):
            entered = {key: _field_card(review, key, senders, nonce) for key in FIELD_KEYS}
            b1, b2 = st.columns(2)
            approve_clicked = b1.form_submit_button("אשר טופס ועבור לבא", type="primary", width="stretch",
                                                    key="approve")
            save_clicked = b2.form_submit_button("שמור בלי לאשר", width="stretch", key="save")

        if approve_clicked:
            result = rs.approve(review, entered)
            if result.approved:
                _after_approval(pos, result.empty)
            else:
                st.session_state.pending = {"pos": pos, "warnings": result.warnings}
                st.session_state.nonce += 1
            st.rerun()
        if save_clicked:
            rs.save_edits(review, entered)
            rs.reopen(review)
            st.session_state.notice = "התיקונים נשמרו. הטופס עדיין לא אושר."
            st.session_state.nonce += 1
            st.rerun()

        pending = st.session_state.pending
        if pending and pending["pos"] == pos and not review.approved:
            st.warning("בדיקת התקינות נכשלה בשדות הבאים. תקנו ואשרו שוב, או אשרו בכל זאת אם הערך נכון "
                       "(למשל מספר דרכון):\n" + _warning_lines(pending["warnings"]))
            if st.button("אשר בכל זאת", type="primary", key="force_approve"):
                result = rs.approve(review, {}, force=True)
                _after_approval(pos, result.empty)
                st.rerun()
        if review.approved and st.button("בטל אישור", key="reopen"):
            rs.reopen(review)
            st.session_state.nonce += 1
            st.rerun()


def screen_export() -> None:
    items = reviews()
    df, highlight = rs.export_frame(items)
    waiting = rs.unapproved(items)
    st.subheader("ייצוא לאקסל")
    st.write("שורה לכל טופס. תאים כתומים: שדות שסומנו לבדיקה בטפסים שעדיין לא אושרו.")

    headers = {**ALL_HEADERS, REVIEW_STATUS_KEY: REVIEW_STATUS_HEADER}
    view = _rtl_columns(df.rename(columns=headers))

    def colour(frame: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
        for pos, key in highlight:
            styles.iloc[pos, frame.columns.get_loc(headers[key])] = f"background-color: {FLAG_COLOR}"
        return styles

    st.dataframe(view.style.apply(colour, axis=None), hide_index=True)

    allowed = not waiting
    if waiting:
        st.warning(f"{len(waiting)} טפסים עדיין לא אושרו:\n"
                   + "\n".join(f"- טופס {items.index(r) + 1}: {r.label()}" for r in waiting))
        allowed = st.checkbox("אני מבין/ה – ייצא בכל זאת (השדות שלא נבדקו יסומנו בכתום)", key="export_anyway")
    st.download_button("הורד קובץ Excel", data=to_xlsx_bytes(df, highlight),
                       file_name=f"labextract_{date.today():%Y-%m-%d}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary", disabled=not allowed, on_click="ignore", key="download")


# ------------------------------------------------------------------ main

def _sidebar_css(items: list, screen: str, pos: int) -> str:
    """Per-item CSS for the sidebar: the current step and form, step dots and the batch-list badges.

    Streamlit buttons take only a text label, so state (current, approved, number of open flags) is drawn
    with CSS on each button's key class ("st-key-<key>").
    """
    order = list(SCREENS)
    rules = [f'[data-testid="stSidebar"] div.st-key-nav_{screen} button {{ background: #172036; color: #E8EDF7; '
             f'font-weight: 600; }}',
             f'[data-testid="stSidebar"] div.st-key-nav_{screen} button::before {{ background: transparent; '
             f'border: 2px solid {APPROVED_COLOR}; box-sizing: border-box; }}']
    if items:
        rules += [f'[data-testid="stSidebar"] div.st-key-nav_{s} button::before {{ background: {APPROVED_COLOR}; }}'
                  for s in order[:order.index(screen)]]
    for i, r in enumerate(items):
        sel = f'[data-testid="stSidebar"] div.st-key-queue_{i} button'
        issues = len(r.open_flags()) + len(r.record.split_flags) + bool(r.record.error)
        if r.approved:
            rules.append(f'{sel}::after {{ content: "✓"; background: #0D2A27; color: {APPROVED_COLOR}; }}')
        elif issues:
            rules.append(f'{sel}::after {{ content: "{issues}"; background: #2A1E05; color: #FFC15C; }}')
        if screen == "review" and i == pos:
            rules.append(f'{sel} {{ background: #172036; color: #E8EDF7; box-shadow: inset 0 0 0 1px #2A3654; }}')
    return "<style>" + "\n".join(rules) + "</style>"


def sidebar(has_results: bool) -> None:
    items = reviews()
    with st.sidebar:
        st.markdown(BRAND_HTML + _sidebar_css(items, st.session_state.screen, st.session_state.pos),
                    unsafe_allow_html=True)
        for screen, label in SCREENS.items():
            st.button(label, key=f"nav_{screen}", type="tertiary", disabled=not has_results and screen != "upload",
                      on_click=go, args=(screen,), width="stretch")
        approved = len(items) - len(rs.unapproved(items))
        st.markdown(f'<div class="lx-section" style="display:flex;justify-content:space-between">'
                    f'<span>טפסים באצווה</span><span dir="ltr">{approved} / {len(items)}</span></div>'
                    if items else '<div class="lx-section">טפסים באצווה</div>', unsafe_allow_html=True)
        if not items:
            st.markdown('<div class="lx-empty">עדיין אין טפסים.<br>הטפסים יופיעו כאן אחרי החילוץ.</div>'
                        + PRIVACY_NOTE, unsafe_allow_html=True)
            return
        for i, r in enumerate(items):
            st.button(f"{i + 1} {r.record.metadata.get('form_type', '') or 'לא מזוהה'}", key=f"queue_{i}",
                      type="tertiary", on_click=go, args=("review", i), width="stretch")
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        if st.button("נקה הכל", help="מוחק את כל התוצאות מהזיכרון", key="clear_all", icon=":material/delete:",
                     width="stretch"):
            clear_all()
            st.rerun()


def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    if config.DEV_CACHE:
        st.error(DEV_CACHE_BANNER, icon="⚠️")

    has_results = bool(reviews())
    if not has_results:
        st.session_state.screen = "upload"
    sidebar(has_results)

    if st.session_state.notice:
        st.success(st.session_state.notice)
        st.session_state.notice = None

    screen = st.session_state.screen
    if screen == "summary":
        screen_summary()
    elif screen == "review":
        screen_review(list_approved_senders(config.SENDERS_PATH))
    elif screen == "export":
        screen_export()
    else:
        screen_upload()


main()
