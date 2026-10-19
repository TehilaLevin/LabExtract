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
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_HEADER,
    REVIEW_STATUS_KEY,
    SPLIT_FLAG_LABELS,
    SUGGESTION_LABEL,
)
from src.excel_exporter import to_xlsx_bytes
from src.claude_client import LLMError, LLMQuotaError, check_configured
from src.validators import list_approved_senders

st.set_page_config(page_title="LabExtract", page_icon="🧪", layout="wide")

SCREENS = {"upload": "העלאה", "summary": "סיכום", "review": "סקירת טפסים", "export": "ייצוא"}
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

def run_extraction(files: list) -> None:
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
    bar = st.progress(0.0, text="ממיר קבצים לתמונות…")
    while worker.is_alive():
        with lock:
            stage, done, total = progress["stage"], progress["done"], progress["total"]
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

def screen_upload() -> None:
    st.subheader("העלאת טפסים")
    st.write("גררו לכאן קבצי PDF סרוקים או צילומי טלפון (PNG / JPG). קובץ אחד יכול להכיל טפסים רבים.")
    problem = config_problem()
    if problem:
        st.error(problem)
    files = st.file_uploader("קבצים", type=UPLOAD_TYPES, accept_multiple_files=True,
                             key=f"uploader_{st.session_state.uploader_n}", label_visibility="collapsed")
    label = "חלץ"
    if reviews():
        st.warning("חילוץ חדש יחליף את התוצאות הקיימות, כולל תיקונים ואישורים שלא יוצאו.")
        label = "חלץ והחלף את התוצאות"
    if st.button(label, type="primary", disabled=not files or bool(problem), key="extract"):
        run_extraction(files)
    if st.session_state.run_error:
        st.error(f"החילוץ נכשל: {st.session_state.run_error}")


def _rtl_columns(df: pd.DataFrame) -> pd.DataFrame:
    """st.dataframe is left-to-right: reverse the columns so the table reads from the right."""
    return df[list(reversed(df.columns))]


def screen_summary() -> None:
    items = reviews()
    flagged = sum(len(r.record.flags) for r in items)
    approved = len(items) - len(rs.unapproved(items))
    st.subheader("סיכום")
    st.markdown(f"**{len(items)} טפסים עובדו, {flagged} שדות סומנו לבדיקה, {approved} טפסים אושרו.**")

    df = rs.summary_frame(items)
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
            out.append(f"background-color: {FLAG_COLOR}" if hot else "")
        return out

    st.caption("לחצו על שורה כדי לפתוח את הטופס.")
    event = st.dataframe(view.style.apply(colour, axis=1), hide_index=True, on_select="rerun",
                         selection_mode="single-row", key=f"summary_{st.session_state.nonce}")
    if event.selection.rows:
        go("review", event.selection.rows[0])
        st.rerun()
    nxt = rs.next_unapproved(items, -1)
    if nxt is not None and st.button("התחל סקירה מהטופס הראשון שלא אושר", type="primary", key="start_review"):
        go("review", nxt)
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

def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("LabExtract")
    st.caption("חילוץ נתונים מטפסי התחייבות · שדה ריק ומסומן עדיף תמיד על ערך מנוחש")
    if config.DEV_CACHE:
        st.error(DEV_CACHE_BANNER, icon="⚠️")

    has_results = bool(reviews())
    if not has_results:
        st.session_state.screen = "upload"
    top = st.columns([4, 1])
    with top[0]:
        options = list(SCREENS) if has_results else ["upload"]
        st.session_state.nav = st.session_state.screen  # set before the widget is created
        st.radio("מסך", options, format_func=SCREENS.get, horizontal=True, key="nav",
                 label_visibility="collapsed", on_change=lambda: go(st.session_state.nav))
    with top[1]:
        if has_results and st.button("נקה הכל", help="מוחק את כל התוצאות מהזיכרון", key="clear_all"):
            clear_all()
            st.rerun()

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
