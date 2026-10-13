import json
from datetime import date

import pytest

from src import constants
from src.validators import (
    format_number_list,
    list_approved_senders,
    match_sending_entity,
    check_received_date,
    normalize_date,
    normalize_number_list,
    validate_israeli_id,
)


# ---- Israeli ID -------------------------------------------------------------

@pytest.mark.parametrize("id_str", ["000000018", "18", "123456782", "039337423", "000000026"])
def test_valid_ids(id_str):
    assert validate_israeli_id(id_str) is True


@pytest.mark.parametrize(
    "id_str",
    ["123456789", "000000000", "", "abc", "12345678a", "1234567890", "12 3456782", None, 123456782],
)
def test_invalid_ids(id_str):
    assert validate_israeli_id(id_str) is False


def test_id_is_padded_to_nine_digits():
    assert validate_israeli_id("18") == validate_israeli_id("000000018")


# ---- sending entity ---------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("כללית", "כללית"),
        ("שירותי בריאות כללית", "כללית"),
        ("קופת חולים כללית", "כללית"),
        ('קופ"ח מכבי', "מכבי"),
        ("  מכבי   שירותי בריאות ", "מכבי"),
        ("Maccabi", "מכבי"),
        ("מאוחדת", "מאוחדת"),
        ("קופת חולים לאומית", "לאומית"),
        ("לאומית", "לאומית"),
        ("בית חולים סורוקה", "סורוקה"),
        ('בי"ח שיבא', "שיבא"),
        ("שיבא תל השומר", "שיבא"),
        ("רמב״ם", 'רמב"ם'),
        ("רמבם", 'רמב"ם'),
        ("הדסה עין-כרם", "הדסה עין כרם"),
        ("בית חולים הדסה עין כרם ירושלים", "הדסה עין כרם"),
    ],
)
def test_match_sending_entity_known(raw, expected):
    assert match_sending_entity(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        None,
        "   ",
        "בית חולים",          # prefix only
        "הדסה",               # two campuses: ambiguous on purpose
        "קליניקה פרטית",      # not in the closed list
        "כללית מכבי",         # matches two senders
    ],
)
def test_match_sending_entity_never_guesses(raw):
    assert match_sending_entity(raw) is None


def test_every_canonical_name_matches_itself():
    for name in list_approved_senders():
        assert match_sending_entity(name) == name, name


def test_no_alias_is_shared_between_two_senders():
    from src.validators import DEFAULT_SENDERS_PATH, _load_senders

    alias_map, _ = _load_senders(str(DEFAULT_SENDERS_PATH))
    shared = {a: n for a, n in alias_map.items() if len(n) > 1}
    assert shared == {}


def test_custom_senders_file(tmp_path):
    path = tmp_path / "senders.json"
    path.write_text(
        json.dumps({"hmos": [{"name": "בדיקה", "aliases": ["בדיקה א"]}], "hospitals": []}),
        encoding="utf-8",
    )
    assert match_sending_entity("בדיקה א", senders_path=path) == "בדיקה"
    assert match_sending_entity("כללית", senders_path=path) is None


# ---- dates ------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2025-03-15", "15/03/2025"),
        ("15/03/2025", "15/03/2025"),
        ("15.03.25", "15/03/2025"),         # 2-digit year means 20YY
        ("25/1/25", "25/01/2025"),
        ("5/3/26", "05/03/2026"),
        ("03/04/2025", "03/04/2025"),       # always day-first: 3 April, not 4 March
        ("12/11/2025", "12/11/2025"),
        ("01.02.25", "01/02/2025"),
        ("31/12/2024", "31/12/2024"),
        (" 15/03/2025 ", "15/03/2025"),
        ("29/02/2024", "29/02/2024"),       # leap year
    ],
)
def test_normalize_date_ok(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", None, "abc", "31/02/2025", "29/02/2025", "32/01/2025", "15/13/2025", "13/25/2025",
     "2025-13-01", "15/03/202", "15/03", "1/1/1/1", 20250315, "15/0?/2025", "15/03/20.."],
)
def test_normalize_date_invalid(raw):
    assert normalize_date(raw) is None


TODAY = date(2026, 6, 15)


def test_recent_date_is_not_flagged():
    assert check_received_date("10/06/2026", TODAY) == ("10/06/2026", False, None)
    assert check_received_date("15/06/2026", TODAY).needs_review is False  # today


def test_day_month_never_swapped_in_review_check():
    # 03/04/2026 is 3 April 2026 (past, fine); a month-first reading would be future.
    assert check_received_date("03/04/2026", TODAY) == ("03/04/2026", False, None)


def test_future_date_is_flagged():
    assert check_received_date("16/06/2026", TODAY) == ("16/06/2026", True, "future")


def test_more_than_one_year_old_is_flagged():
    assert check_received_date("14/06/2025", TODAY) == ("14/06/2025", True, "too_old")
    assert check_received_date("15/06/2025", TODAY).needs_review is False  # exactly 1 year


def test_old_boundary_on_leap_day():
    assert check_received_date("28/02/2025", date(2026, 2, 28)).needs_review is False
    assert check_received_date("27/02/2025", date(2026, 2, 28)).reason == "too_old"
    assert check_received_date("28/02/2023", date(2024, 2, 29)).needs_review is False  # 29 Feb -> 28 Feb
    assert check_received_date("27/02/2023", date(2024, 2, 29)).reason == "too_old"


@pytest.mark.parametrize("raw", ["31/02/2026", "15/13/2026"])
def test_invalid_date_is_flagged_without_value(raw):
    assert check_received_date(raw, TODAY) == (None, True, "invalid")


@pytest.mark.parametrize("raw", [None, "", "15/0?/2026", "15/03/", "??/??/2026", "ללא תאריך"])
def test_unreadable_or_cut_date_is_flagged(raw):
    assert check_received_date(raw, TODAY) == (None, True, "unreadable")


def test_check_received_date_defaults_to_real_today():
    assert check_received_date(date.today().strftime("%d/%m/%Y")).needs_review is False

# ---- multiple central / commitment numbers ------------------------------------

def test_list_fields_are_central_and_commitment():
    assert constants.LIST_FIELDS == {"central_numbers", "commitment_numbers"}


def test_normalize_number_list_keeps_all_numbers_in_order():
    assert normalize_number_list(["1234", "5678", "9012"]) == ["1234", "5678", "9012"]


def test_normalize_number_list_splits_text_and_dedupes():
    assert normalize_number_list("1234, 5678;1234\n9012") == ["1234", "5678", "9012"]


@pytest.mark.parametrize("empty", [None, "", "  ", [], [None, ""], ",,;"])
def test_normalize_number_list_empty(empty):
    assert normalize_number_list(empty) == []


def test_single_number_stays_single():
    assert normalize_number_list("4455") == ["4455"]
    assert normalize_number_list(["4455"]) == ["4455"]


def test_format_number_list_is_comma_separated():
    assert format_number_list(["1", "2", "3"]) == "1, 2, 3"
    assert format_number_list(None) == ""


# ---- constants --------------------------------------------------------------

def test_constants_cover_all_required_fields():
    assert set(constants.FIELD_HEADERS) == {
        "first_name", "last_name", "id_number", "central_numbers",
        "received_date", "sending_entity", "commitment_numbers",
    }
    assert set(constants.METADATA_HEADERS) == {"source_file", "page_numbers", "form_type"}
    assert constants.FIELD_HEADERS["central_numbers"] == "מספר מרכזי"
    assert len(set(constants.ALL_HEADERS.values())) == len(constants.ALL_HEADERS)
