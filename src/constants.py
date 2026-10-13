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
    "extraction_failed": "החילוץ נכשל",
}
