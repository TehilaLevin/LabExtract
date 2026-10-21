"""Archive assistant: rules, model-output validation, execution on stored rows, and the app section.
Uses the temp DB from conftest. No test calls the API: the model is a fake `generate` or fails the test."""
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import archive_assistant as aa
from src import config
from src import extraction_engine as ee
from src import record_builder as rb
from src import record_store as store
from src import review_session as rs
from src.claude_client import LLMError, alias_quoted_enums, to_json_schema
from src.constants import FIELD_KEYS
from src.validators import list_approved_senders

TODAY = date(2026, 10, 21)
VALID_ID = "000000018"
SENDERS = list_approved_senders()
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _review(index, sender="מכבי", approved=False, flags=None, **values):
    base = {"first_name": "ישראל", "last_name": "ישראלי", "id_number": VALID_ID, "central_numbers": "937309",
            "received_date": "21/10/2026", "sending_entity": sender, "commitment_numbers": "50364186"}
    base.update(values)
    record = rb.Record(index, base, dict(flags or {}), {k: 0.9 for k in FIELD_KEYS},
                       {"source_file": "a.pdf", "page_numbers": str(index), "form_type": "טופס"})
    return rs.FormReview(record, approved=approved)


def _store(*reviews):
    store.save_all(list(reviews), "b1")
    return store.archive_rows()


def _ask(question, rows=None, **kwargs):
    """Rules only (use_model=False unless given): the query, executed on the stored rows."""
    kwargs.setdefault("use_model", False)
    query = aa.interpret(question, today=TODAY, **kwargs)
    assert isinstance(query, aa.Query), query
    return query, aa.run_query(query, store.archive_rows() if rows is None else rows)


def _no_model(*args, **kwargs):
    raise AssertionError("the model must not be called")


def _model_json(**overrides):
    data = {"intent": "count", "status": "any", "sender": None, "sender_kind": "any", "missing_field": None,
            "flagged_only": False, "date_field": "none", "date_from": None, "date_to": None, "group_by": "none",
            "limit": None, "clarification": None}
    data.update(overrides)
    return data


# ------------------------------------------------------------------ counting and filtering

def test_count_pending_uses_the_stored_status():
    _store(_review(1), _review(2), _review(3, approved=True))
    query, answer = _ask("כמה טפסים ממתינים לבדיקה?")
    assert (query.intent, query.status, query.source) == ("count", store.STATUS_PENDING, "rules")
    assert answer.count == 2 and "2" in answer.message
    _, answer = _ask("כמה טפסים אושרו?")
    assert answer.count == 1


def test_list_by_approved_sender_and_status():
    _store(_review(1, "מכבי"), _review(2, "כללית"), _review(3, "מכבי", approved=True), _review(4, ""))
    query, answer = _ask("הצג טפסים ממכבי שממתינים לבדיקה")
    assert (query.intent, query.sender, query.status) == ("list", "מכבי", store.STATUS_PENDING)
    assert [r["sending_entity"] for r in answer.rows] == ["מכבי"]
    assert any("בלי גורם שולח" in n for n in answer.notes)  # the form without a sender is reported, not hidden


def test_sender_includes_its_campuses_and_reports_off_list_senders():
    _store(_review(1, "רבין"), _review(2, "בילינסון"), _review(3, "שיבא"), _review(4, "מרפאה פרטית"))
    _, answer = _ask("כמה טפסים מרבין?")
    assert answer.count == 2
    assert any("אינו ברשימה המאושרת" in n for n in answer.notes)
    _, answer = _ask("כמה טפסים מבתי חולים?")
    assert answer.count == 3


def test_unknown_status_is_reported_never_silently_dropped():
    _store(_review(1), _review(2, approved=True))
    with closing(sqlite3.connect(config.DB_PATH)) as conn, conn:
        conn.execute("UPDATE forms SET status = 'archived' WHERE form_index = 1")
    _, answer = _ask("כמה טפסים ממתינים לבדיקה?")
    assert answer.count == 0 and any("סטטוס לא ידוע" in n for n in answer.notes)
    _, answer = _ask("כמה טפסים לפי סטטוס?")
    assert {g["label"]: g["count"] for g in answer.groups} == {"אושר": 1, aa.UNKNOWN_STATUS_LABEL: 1}


def test_received_date_filter_reports_forms_without_a_valid_date():
    _store(_review(1), _review(2, received_date="20/10/2026"), _review(3, received_date=""),
           _review(4, received_date="32/13/2026"))
    query, answer = _ask("כמה טפסים התקבלו היום?")
    assert (query.date_field, query.date_from, query.date_to) == ("received_date", TODAY, TODAY)
    assert answer.count == 1 and any("2 טפסים ללא תאריך קבלה תקין" in n for n in answer.notes)
    _, answer = _ask("הצג טפסים שהתקבלו בין 19/10/2026 ל-20/10/2026")
    assert answer.count == 1


def test_intake_and_approval_dates_come_from_the_database():
    _store(_review(1), _review(2, approved=True))
    today = date.today()  # created_at / approved_at are written with the real clock
    query = aa.interpret("כמה טפסים נקלטו היום?", today=today, use_model=False)
    assert query.date_field == "created_at" and aa.run_query(query, store.archive_rows()).count == 2
    query = aa.interpret("כמה טפסים אושרו היום?", today=today, use_model=False)
    assert (query.status, query.date_field) == (store.STATUS_APPROVED, "approved_at")
    assert aa.run_query(query, store.archive_rows()).count == 1
    query = aa.interpret("כמה טפסים אושרו אתמול?", today=today, use_model=False)
    assert query.date_from == today - timedelta(days=1) and aa.run_query(query, store.archive_rows()).count == 0


def test_missing_field_and_flagged_forms():
    _store(_review(1, commitment_numbers=""), _review(2), _review(3, flags={"id_number": "invalid_id"}))
    query, answer = _ask("אילו טפסים חסר בהם מספר התחייבות?")
    assert query.missing_field == "commitment_numbers" and answer.count == 1
    _, answer = _ask("טפסים ללא גורם שולח")
    assert answer.count == 0 and "לא נמצאו" in answer.message
    _, answer = _ask("אילו טפסים עם שדות לבדיקה?")
    assert answer.count == 1


def test_group_by_sender_shows_unknown_and_off_list_senders():
    _store(_review(1, "מכבי"), _review(2, "מכבי"), _review(3, ""), _review(4, "מרפאה פרטית"))
    query, answer = _ask("כמה טפסים מכל גורם שולח?")
    assert query.group_by == "sending_entity"
    assert {g["label"]: g["count"] for g in answer.groups} == {"מכבי": 2, aa.NO_SENDER_LABEL: 1, "מרפאה פרטית": 1}
    _, answer = _ask("כמה טפסים לפי סוג גורם?")
    assert {g["label"] for g in answer.groups} == {"קופת חולים", "ללא גורם שולח", "גורם אחר (לא ברשימה)"}


def test_latest_forms_limit():
    _store(*[_review(i) for i in range(1, 8)])
    query, answer = _ask("הצג את 5 הטפסים האחרונים")
    assert query.limit == 5 and answer.count == 7 and len(answer.rows) == 5


def test_empty_database_and_empty_result():
    query = aa.interpret("כמה טפסים ממתינים לבדיקה?", today=TODAY, use_model=False)
    assert "המאגר ריק" in aa.run_query(query, store.archive_rows()).message
    _store(_review(1, approved=True))
    _, answer = _ask("כמה טפסים ממתינים לבדיקה?")
    assert answer.count == 0 and answer.rows == [] and "לא נמצאו" in answer.message


def test_results_never_show_names_and_mask_the_id():
    _store(_review(1, id_number="123456782"))
    _, answer = _ask("הצג טפסים ממכבי")
    row = aa.display_rows(answer)[0]
    assert row[aa.RESULT_HEADERS["id_number"]] == "••••••782"
    assert "ישראל" not in str(row.values()) and not {"שם פרטי", "שם משפחה"} & set(row)
    assert aa.mask_id("") == "" and aa.mask_id("1234") == "••••"


# ------------------------------------------------------------------ ambiguity

def test_a_certain_hospital_asks_which_one_from_the_database():
    result = aa.interpret("הצג טפסים שנשלחו מבית חולים מסוים", present_senders=["מכבי", "שיבא", "רמב\"ם"],
                          today=TODAY, use_model=False)
    assert isinstance(result, aa.Clarification)
    assert [label for label, _ in result.options] == ["רמב\"ם", "שיבא"]  # hospitals only, no health fund
    follow_up = aa.interpret(result.options[1][1], today=TODAY, use_model=False)
    assert isinstance(follow_up, aa.Query) and follow_up.sender == "שיבא"


@pytest.mark.parametrize("question", ["אילו טפסים חסר בהם שם?", "כמה טפסים מהיום", "כמה טפסים התקבלו ב-12/10",
                                      "כמה טפסים לפי סוג?", "כמה טפסים ממכבי ומכללית?",
                                      "אילו טפסים חסר בהם מספר מרכזי ותאריך קבלה?"])
def test_ambiguous_questions_ask_instead_of_guessing(question):
    result = aa.interpret(question, today=TODAY, use_model=True, generate=_no_model)
    assert isinstance(result, aa.Clarification)


def test_date_clarification_options_are_answerable():
    result = aa.interpret("כמה טפסים מהיום", today=TODAY, use_model=False)
    fields = [aa.interpret(q, today=TODAY, use_model=False).date_field for _, q in result.options]
    assert fields == ["received_date", "created_at"]


def test_unknown_question_without_model_asks_to_rephrase():
    result = aa.interpret("מה מזג האוויר מחר", today=TODAY, use_model=False)
    assert isinstance(result, aa.Clarification) and "מזג" in result.message and result.options


# ------------------------------------------------------------------ model: question text only, checked output

def test_model_gets_only_the_question_and_its_answer_is_checked():
    _store(_review(1, "שיבא"), _review(2, "שיבא", approved=True))
    sent = {}

    def fake_generate(contents, schema, **kwargs):
        sent.update(contents=contents, schema=schema, **kwargs)
        return _model_json(intent="list", sender="שיבא", status="pending")

    result = aa.interpret("תן לי בבקשה רשימה של מה שעוד פתוח משיבא", today=TODAY, use_model=True,
                          generate=fake_generate)
    assert isinstance(result, aa.Query) and result.source == "model"
    assert sent["contents"] == ["Question: תן לי בבקשה רשימה של מה שעוד פתוח משיבא"]
    assert sent["purpose"] == "archive_query" and sent["thinking_level"] == "minimal"
    assert VALID_ID not in str(sent) and "ישראלי" not in str(sent)  # no record content
    assert aa.run_query(result, store.archive_rows()).count == 1


def test_question_with_an_identifier_is_never_sent():
    result = aa.interpret('מה מספר ההתחייבות של ת"ז 000000018', today=TODAY, use_model=True, generate=_no_model)
    assert result == aa.Clarification(aa.VALUE_SEARCH_MESSAGE)
    result = aa.interpret("הצג טפסים עם מספר מרכזי 937 309", today=TODAY, use_model=True, generate=_no_model)
    assert result.message == aa.VALUE_SEARCH_MESSAGE


@pytest.mark.parametrize("bad", [
    {"extra": "x"},                                             # unknown key
    {"intent": "delete"},                                       # unsupported intent
    {"intent": "sql"},
    {"sender": "x' OR 1=1 --"},                                 # not on the approved list
    {"sender": "__import__('os').system('dir')"},
    {"missing_field": "record_json"},                           # not a field
    {"status": "deleted"},
    {"group_by": "id_number", "intent": "group_by"},            # grouping by an identifier is not allowed
    {"date_field": "received_date", "date_from": "2026-13-01", "date_to": "2026-10-21"},
    {"date_field": "received_date", "date_from": None, "date_to": None},
    {"date_field": "received_date", "date_from": "2026-10-21", "date_to": "2026-10-01"},
    {"date_field": "received_date", "date_from": "21/10/2026", "date_to": "2026-10-21"},
    {"limit": 10 ** 6}, {"limit": True}, {"limit": "5"},
    {"flagged_only": "yes"},
    {"intent": "group_by", "group_by": "none"},
    {"sender": "שיבא", "sender_kind": "hmo"},
])
def test_invalid_model_output_is_rejected(bad):
    data = _model_json(**{k: v for k, v in bad.items() if k != "extra"})
    if "extra" in bad:
        data["extra"] = bad["extra"]
    with pytest.raises(aa.QueryError):
        aa.validate_model_output(data, SENDERS)
    result = aa.interpret("שאלה שהכללים לא מבינים", today=TODAY, use_model=True, generate=lambda *a, **k: data)
    assert isinstance(result, aa.Clarification) and "לא בוצע חיפוש" in result.message


def test_missing_key_or_non_object_is_rejected():
    data = _model_json()
    del data["limit"]
    for bad in (data, [], "count", None):
        with pytest.raises(aa.QueryError):
            aa.validate_model_output(bad, SENDERS)


def test_model_clarification_and_service_failure():
    result = aa.interpret("שאלה עמומה", today=TODAY, use_model=True,
                          generate=lambda *a, **k: _model_json(intent="ambiguous", clarification="לאיזה תאריך הכוונה?"))
    assert result == aa.Clarification("לאיזה תאריך הכוונה?", source="model")

    def fail(*a, **k):
        raise LLMError("Claude API error 500")

    result = aa.interpret("שאלה עמומה", today=TODAY, use_model=True, generate=fail)
    assert isinstance(result, aa.Clarification) and "נכשל" in result.message and result.source == "rules"


def test_check_query_rejects_values_outside_the_allowlists():
    for bad in (aa.Query("drop"), aa.Query("count", sender="'; DROP TABLE forms; --"),
                aa.Query("count", date_field="received_date", date_from="2026-10-01", date_to="2026-10-02"),
                aa.Query("count", date_from=TODAY), aa.Query("list", limit=0), aa.Query("count", source="user"),
                aa.Query("group_by", group_by="first_name"), aa.Query("count", missing_field="__class__")):
        with pytest.raises(aa.QueryError):
            aa.check_query(bad, SENDERS)
        with pytest.raises(aa.QueryError):
            aa.run_query(bad, [])


def test_model_schema_converts_for_structured_outputs():
    schema, aliases = alias_quoted_enums(to_json_schema(aa.model_schema(SENDERS)))
    assert set(schema["required"]) == set(aa.MODEL_KEYS) and schema["additionalProperties"] is False
    assert "רמב\"ם" in aliases.values()  # quoted names go through the existing alias mechanism


def test_example_questions_are_all_answered_by_the_rules():
    present = ["מכבי", "שיבא"]
    for question in aa.EXAMPLES:
        result, leftovers = aa.parse_question(question, present, TODAY)
        assert result is not None and leftovers == [], question


# ------------------------------------------------------------------ side chat in the app (no model calls)

@pytest.fixture
def no_model_calls(monkeypatch):
    monkeypatch.setattr(ee, "generate_json", _no_model)


def _app(screen="database"):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["screen"] = screen
    return at.run()


def _text(at):
    return " ".join(m.value for m in at.markdown)


def test_chat_button_is_on_every_screen_and_opens_the_panel(no_model_calls):
    at = _app("upload")
    assert not at.exception and at.button(key="chat_toggle").label == "עוזר המאגר"
    assert not at.chat_input  # closed by default
    at.button(key="chat_toggle").click().run()
    assert not at.exception and at.chat_input and at.button(key="chat_toggle").label == "סגירת העוזר"


def test_chat_example_answers_from_the_database_and_opens_a_record(no_model_calls):
    _store(_review(1), _review(2), _review(3, approved=True))
    at = _app()
    at.button(key="chat_toggle").click().run()
    at.button(key="ask_ex_0").click().run()
    assert not at.exception and "נמצאו 2 טפסים התואמים לשאלה." in _text(at)
    rows = [b for b in at.button if (b.key or "").startswith("chat_row_0_")]
    assert len(rows) == 2 and all(VALID_ID not in b.label and "ישראל" not in b.label for b in rows)
    rows[0].click().run()
    assert not at.exception and at.session_state["screen"] == "review"
    assert at.button(key="chat_toggle").label == "סגירת העוזר"  # the chat stays open beside the review


def test_chat_clarification_buttons_and_typed_question(no_model_calls):
    _store(_review(1, "שיבא"), _review(2, "מכבי"))
    at = _app()
    at.button(key="chat_toggle").click().run()
    at.button(key="ask_ex_4").click().run()  # "a certain hospital"
    assert at.button(key="chat_opt_0_0").label == "שיבא"
    at.button(key="chat_opt_0_0").click().run()
    assert "נמצא טופס אחד התואם לשאלה." in _text(at)
    at.chat_input(key="chat_input").set_value("כמה טפסים מכל גורם שולח?").run()
    assert not at.exception and "בחלוקה ל-2 קבוצות" in _text(at) and len(at.session_state["chat"]) == 3
    at.button(key="chat_clear").click().run()
    assert at.session_state["chat"] == []


def test_chat_can_be_turned_off(monkeypatch, no_model_calls):
    monkeypatch.setattr(config, "ARCHIVE_ASSISTANT_ENABLED", False)
    at = _app()
    assert not at.exception and "chat_toggle" not in [b.key for b in at.button]


def test_chat_names_its_data_source_and_recomputes_after_a_change(no_model_calls):
    _store(_review(1), _review(2))
    at = _app()
    at.button(key="chat_toggle").click().run()
    assert "מאגר חלופי: test.db · 2 טפסים" in _text(at)  # the tests' DB_PATH is not the usual database file
    at.button(key="ask_ex_0").click().run()
    assert "נמצאו 2 טפסים התואמים לשאלה." in _text(at) and "חושב מ-2 הטפסים שבמאגר" in _text(at)
    assert "chat_refresh_0" not in [b.key for b in at.button]
    store.save_all([_review(3)], "b2")  # the database changes after the answer was computed
    at.run()
    assert "המאגר השתנה מאז" in _text(at)
    at.button(key="chat_refresh_0").click().run()
    assert "נמצאו 3 טפסים התואמים לשאלה." in _text(at) and "chat_refresh_0" not in [b.key for b in at.button]


def test_last_change_moves_on_save_and_delete():
    empty = store.last_change()
    assert empty == (0, None)
    _store(_review(1))
    saved = store.last_change()
    assert saved[0] == 1 and saved[1]
    store.delete_all()
    assert store.last_change() == (0, None)
