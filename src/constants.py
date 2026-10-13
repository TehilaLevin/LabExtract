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
