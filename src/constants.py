"""Field definitions and Hebrew column headers shared by the whole pipeline."""

# Extracted fields (English key -> Hebrew Excel header), in output order.
FIELD_HEADERS = {
    "first_name": "שם פרטי",
    "last_name": "שם משפחה",
    "id_number": "מספר תעודת זהות",
    "central_numbers": "מספר מרכזי",
    "received_date": "תאריך קבלה",
    "sending_entity": "גורם שולח",
    "commitment_numbers": "מספר התחייבות",
}

# Metadata columns (not extracted by the model; filled by the pipeline).
METADATA_HEADERS = {
    "source_file": "קובץ מקור",
    "page_numbers": "מספרי עמודים",
    "form_type": "סוג טופס",
}

FIELD_KEYS = tuple(FIELD_HEADERS)
METADATA_KEYS = tuple(METADATA_HEADERS)
ALL_HEADERS = {**FIELD_HEADERS, **METADATA_HEADERS}

# Fields that may hold several values (a single test can have several numbers).
LIST_FIELDS = frozenset({"central_numbers", "commitment_numbers"})

# Separator used when a list field is displayed / exported as text.
LIST_SEPARATOR = ", "

# Review-table helper column.
NEEDS_REVIEW_KEY = "needs_review"
NEEDS_REVIEW_HEADER = "דורש בדיקה"

# Colour used to highlight flagged cells.
FLAG_COLOR = "#FFA500"

# Where each field is found on the form (also encoded in the extraction prompt).
FIELD_HINTS = {
    "central_numbers": 'Usually on a sticker next to the word "מרכזי", sometimes handwritten. Keep ALL numbers.',
    "received_date": 'From a "התקבל" stamp, or handwritten below the central number.',
    "commitment_numbers": 'May be labeled "דרישה"; in Leumit forms labeled "דגימה". Keep ALL numbers.',
    "sending_entity": "HMO or hospital that sent the form; must come from approved_senders.json.",
}

# Form types the extractor may report (id -> Hebrew label). Hospital / other-entity
# forms are labelled with the sending entity at display time.
FORM_TYPES = {
    "meuhedet_commitment": "מאוחדת – התחייבות",
    "clalit_form17": "כללית – טופס 17",
    "clalit_lab_report": "כללית – דוח מעבדה מרכזית",
    "maccabi_commitment": "מכבי – התחייבות",
    "maccabi_lab_request": "מכבי – מעבדה שולחת",
    "leumit_worklist": "לאומית – רשימת עבודה",
    "sheba_system_printout": "שיבא – הדפסת מערכת",
    "hospital_form": "טופס בית חולים",
    "other_entity_form": "גורם אחר",
    "unknown": "לא מזוהה",
}
# Form types whose display label is completed with the sending entity name.
ENTITY_LABELLED_FORM_TYPES = frozenset({"hospital_form", "other_entity_form"})

# Where the received date was read. Only these two sources are accepted (never guess
# from admission / print / sampling dates).
RECEIVED_DATE_SOURCES = ("received_stamp", "handwritten_near_central", "none")
ACCEPTED_RECEIVED_DATE_SOURCES = frozenset({"received_stamp", "handwritten_near_central"})

# Review flag reasons (field -> reason code), shown to the user in Hebrew.
FLAG_REASONS = {
    "missing": "חסר",
    "low_confidence": "ביטחון נמוך",
    "invalid_id": "ת\"ז לא תקינה",
    "invalid_date": "תאריך לא תקין",
    "date_out_of_range": "תאריך חריג",
    "date_source_rejected": "מקור תאריך לא מאושר",
    "sender_not_in_list": "לא ברשימה המאושרת",
    "invalid_format": "פורמט לא תקין",
    "name_order_unclear": "סדר שם לא ודאי",
    "name_mismatch": "קריאה חוזרת שונה",
    "name_conflict": "סתירה בין מופעי השם",
    "name_unverified": "שם בכתב יד לא אומת",
    "commitment_is_id": "מספר זהה לת\"ז הוסר",
    "extraction_failed": "החילוץ נכשל",
    "unreadable": "קיים בטופס אך לא קריא",
    "reread_conflict": "קריאה חוזרת סותרת",
    "reread_replaced": "הוחלף בקריאה חוזרת – לבדיקה",
    "reread_single_reading": "נקרא רק בקריאה חוזרת – לבדיקה",
    "central_prefix": "קידומת מספר מרכזי לא צפויה – לבדיקה",
    "date_before_collection": "תאריך הקבלה מוקדם מתאריך הלקיחה",
    "date_long_after_collection": "תאריך הקבלה רחוק מתאריך הלקיחה (מעל 14 יום)",
    "id_batch_mismatch": "ת\"ז שונה מזו של אותו מטופל בטופס אחר",
}

# Form-level split warnings (form_splitter.SPLIT_FLAG_*), shown on the review screen.
SPLIT_FLAG_LABELS = {
    "split_uncertain": "לא ודאי שהטופס מתחיל בעמוד זה – בדקו שהעמודים שייכים לאותו טופס",
    "split_failed": "פיצול העמודים לטפסים נכשל – בדקו שהעמודים שייכים לאותו טופס",
}

# Review status column in the export (one value per form).
REVIEW_STATUS_KEY = "review_status"
REVIEW_STATUS_HEADER = "סטטוס בדיקה"
REVIEW_STATUS_APPROVED = "אושר"
REVIEW_STATUS_PENDING = "לא אושר"

# Per-field reading status reported by the extractor: lets "not on the form" (missing)
# be told apart from "on the form but not legible" (unreadable).
FIELD_STATUSES = ("read", "unreadable", "not_present")

# Claude reports field confidence as a label (LLM_PROVIDER=claude); mapped onto the numeric scale
# the rest of the pipeline uses. Only "high" reaches CONFIDENCE_THRESHOLD (0.8): medium and low
# are always flagged for review.
CONFIDENCE_LABELS = {"high": 0.95, "medium": 0.65, "low": 0.3}

# Fields that get a targeted, high-resolution second reading when flagged (names have
# their own verification).
REREAD_FIELDS = ("id_number", "central_numbers", "received_date", "commitment_numbers")
# Fields whose location the extractor reports in field_locations (up to MAX_FIELD_LOCATIONS boxes each):
# the re-read fields plus the sender, so the review screen can show where every value was read.
# Names are located separately (name_locations).
LOCATED_FIELDS = (*REREAD_FIELDS, "sending_entity")
MAX_FIELD_LOCATIONS = {"id_number": 2, "central_numbers": 4, "received_date": 2, "commitment_numbers": 3,
                       "sending_entity": 1}

# How the first/last name split was decided. Names are flagged for review only when
# the order is genuinely unclear ("unclear"), not merely because the name sat in one field.
NAME_ORDER_BASES = ("separate_fields", "form_convention", "unclear")

# Known name orders for single-field names, rendered into the extraction prompt.
# (where the name appears, order). Anything not listed here is "unclear".
# Validate / extend these against data/ground_truth/ground_truth.csv.
NAME_ORDER_CONVENTIONS = (
    ('sheba_system_printout: the patient name in the header line', "LAST name first, then first name"),
    ('leumit_worklist: the field "שם ומשפחה" (literally "name and family")', "FIRST name first, then last name"),
    ('A printed hospital admission/patient sticker ("מדבקת מטופל") where "שם:" appears together with '
     '"מ.ז." / "אב:" / "מקרה:" / "קופ\\"ח:"', "LAST name first, then first name"),
)

# How each name was written on the form (reported by the extractor per field).
NAME_SCRIPTS = ("printed", "handwritten", "unknown")
# Where a patient name appears (the extractor reports up to MAX_NAME_LOCATIONS boxes).
NAME_LOCATION_KINDS = ("labeled_fields", "single_field", "patient_sticker", "other")
MAX_NAME_LOCATIONS = 3

# Shown next to an unverified reading of a handwritten name. A suggestion is NEVER an
# approved value: it is kept apart from Record.values and is never exported as one.
SUGGESTION_LABEL = "הצעה – לא מאומת"
