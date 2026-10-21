# LabExtract

Extracts patient commitment-form data for the Central Virology Laboratory (Israeli Ministry of Health):
batch upload → split into forms → identify form type → extract fields → validate → review → Excel export.

> Full setup instructions and pipeline description will be added here.

## Running the app

```bash
.venv/Scripts/activate          # Windows; macOS / Linux: source .venv/bin/activate
streamlit run app.py            # opens http://localhost:8501
```

`.env` must hold `CLAUDE_API_KEY` (or `ANTHROPIC_API_KEY`; see `.env.example`); PDFs need Poppler.

## Model: Claude

Claude (`claude-opus-5-5` by default) is the only model provider. Every model call (orientation, splitting,
extraction, name check, field re-read) goes through `src/claude_client.py`.

| Variable | Default | Meaning |
|---|---|---|
| `CLAUDE_API_KEY` (or `ANTHROPIC_API_KEY`) | – | Anthropic API key |
| `CLAUDE_MODEL` | `claude-opus-5-5` | used for every call type; `claude-sonnet-5-5` is about 3x cheaper but reads handwriting worse |
| `CLAUDE_EFFORT` | `high` | `low` / `medium` / `high` / `xhigh` / `max`; orientation and splitting always run at `low` |
| `NAME_CHECK_MODEL` | `CLAUDE_MODEL` | model for the handwritten-name crops only (small request); `claude-fable-5-1` is stronger but requires 30-day data retention at Anthropic |
| `NAME_CHECK_EFFORT` | `CLAUDE_EFFORT` | effort for the name crops only |
| `NAME_CROP_ENHANCE` | `true` | name crops: small ones enlarged, plus a grey view with the yellow highlighter removed (image processing only) |
| `NAME_DICTIONARY` | `true` | correct name suggestions to a known Israeli name by look-alike letters (ז/צ, ר/ד, a dropped ו/י...) |
| `BATCH_NAME_SHARING` | `false` | forms with the same valid ID share their names as a suggestion (off: the samples share one placeholder ID) |
| `HIGHLIGHT_CROPS` | `true` | zoomed crops of highlighted regions sent with the page (see below) |
| `HIGHLIGHT_DEBUG` | `false` | dev only: save those crops to `data/temp/highlight_debug/` |
| `STRICT_VALIDATION` | `true` | the extra checks below; `false` restores the earlier ones (for comparisons) |
| `CENTRAL_NUMBER_PREFIXES` | `93,94` | expected central-number prefixes (a mismatch is a warning flag) |
| `RECEIVED_DATE_MAX_AGE_DAYS` | `90` | older received dates are flagged |
| `RECEIVED_AFTER_COLLECTION_MAX_DAYS` | `14` | the received date must be 0..N days after the collection date |
| `EVAL_MODE` | `false` | evaluation only: skip the age-from-today check (the sample forms are months old) |

Extraction and the targeted re-read run with thinking (`thinking: {"type": "adaptive"}`, the documented mode on
current models) at `CLAUDE_EFFORT`; the logs show how many output tokens went to thinking.

Images are sent as base64 JPEG and downscaled to the API limits (2576 px long edge, 4784 visual tokens, 10 MB;
2000 px when a request holds more than 20 images). Answers are forced into the JSON schema with structured outputs
(`output_config.format`); the compact schemas in the code are converted automatically. Claude rejects
`temperature`, so determinism comes from the schema and the effort level. Each field's confidence is `high` /
`medium` / `low` plus a short `source_note` (where on the form the value was read - never the value): only `high`
passes the 0.8 threshold, so `medium` and `low` are always flagged. The note is shown on the review card. The
extraction prompt is cached by the API (cache reads cost 10% of input). The SDK retries 429, 529, 5xx and
connection errors with exponential backoff (`CLAUDE_MAX_RETRIES`), with a 180 s timeout per attempt; a rate
limit, missing credit, rejected key or unknown model stops the run with a clear message. Logs hold only token
counts and timing per call. Refused requests fall back to another Claude model server-side (`fallbacks: "default"`).
Each single-value field (names, ID, date, sender) also gets `raw_text` - the text exactly as written, even partial
(`?` = an illegible character). It is shown on the review card as a reading aid and is never used as a value.
(On the list fields it made the compiled schema too large for the API.)

### Highlight crops (`src/highlight_crops.py`)

Highlighter marks (yellow, green, cyan) are found with an HSV mask, morphology and a minimum area. Regions close
to each other are merged (a highlighted label and its value, a stamp heading and its date) unless the merged crop
would be too large. Each region is cut from the 300 DPI page with padding, enlarged 2x and sent in the same
request as the full page, labelled "Crop k of n (position on the page)", together with a cleaned "ink only"
version: highlighter colours removed, dark ink of any colour kept (blue or black pen, print, grey stamps), specks
dropped. When almost no ink is left the cleaned version is not sent. At most 9 regions per form (19 images with
the page), those with blue handwriting first. A page without highlighting is extracted exactly as before.

In the samples the highlighting marks the fields to extract - but also decoys (visit, intake and form dates). The
prompt therefore says highlighting never tells which field a value belongs to: every field is identified from
its printed label and position on the full page. Whether real lab forms carry highlighting is unknown, so a gain
measured on the samples may not carry over.

### Validation after extraction (`STRICT_VALIDATION`)

- Central number: exactly 6 digits; a prefix outside `CENTRAL_NUMBER_PREFIXES` is a warning flag.
- Received date: a valid date, not in the future, at most `RECEIVED_DATE_MAX_AGE_DAYS` old (skipped with
  `EVAL_MODE`), and 0..`RECEIVED_AFTER_COLLECTION_MAX_DAYS` days after the sample collection date when the
  form states one (`collection_date`; a time after the date is ignored, a date without a year is never used;
  never exported).
- ID: Israeli check digit (up to 9 digits after removing separators); the same patient (identical first + last
  name) must carry the same ID on every form of the batch (`id_batch_mismatch`).
- A field failing a check gets one targeted re-read from its high-resolution crop; if it still fails it stays
  flagged (orange in the review screen and in the Excel export). Values are never changed silently.

1. **Upload** – drag in PDFs / phone photos (many forms per file are fine) and press "חלץ". A progress bar
   follows the split and then each form. The model is called only here.
2. **Summary** – one row per form: form type, number of fields to review, form warnings, status. Click a row.
3. **Review** – the form image on one side; on the other, one card per field with the value, the image crop the
   value was read from, an edit box, and the reason for any flag (orange). Unverified handwritten names are never
   filled in: the reading is shown as "הצעה – לא מאומת" for the reviewer to copy in only if it is right. The sender
   is chosen from the closed list. "אשר טופס ועבור לבא" re-validates the form: a value that still fails validation
   (ID check digit, date, sender) needs an explicit "אשר בכל זאת"; empty fields are reported, never blocking.
4. **Export** – Excel with Hebrew headers, right-to-left sheet, **one row per commitment number** (single-value fields repeat on each row; all central numbers
   of the form share one cell; a form without a commitment number still gives one row, flagged) and a review-status column.
   Flagged cells of forms that were not approved are orange; exporting while forms are unapproved shows them
   and needs a confirmation.

5. **Database ("מאגר")** – every form is saved to a local SQLite file (`data/db/labextract.db`, `src/record_store.py`)
   when extracted (pending) and updated on every save / approval, so records survive closing the browser. The screen
   searches all fields, filters by status and sender, downloads the result as Excel (open flags of unapproved forms
   in orange) and opens a saved form on the review screen ("פתח").

   | What | Stored | Removed |
   |---|---|---|
   | Field values, metadata, status, reason codes of open flags, the reviewer's edits | always | manual delete only |
   | Field crops (JPEG, quality 85, long side up to 1200 px, a few tens to ~200 KB each) | yes, so a crop sits next to every field, also after approval and when a saved form is reopened | with the record |
   | Page previews (JPEG, quality 85, up to 1600 px, a few hundred KB per page) | only while the form is **not approved** | **deleted when the form is approved** |
   | Suggestions (plain-text readings of handwritten names and number candidates, **unverified personal data**), alternatives, notes | only while the form is not approved | **stripped from the record when the form is approved** |
   | Raw model readings (`raw_texts`) | **never** | – |
   | The uploaded files | **never** | – |

   Settings and buttons: `DB_STORE_CROPS=false` stores no image at all (saved forms then open without images, with a
   notice); `DB_ENABLED=false` turns the database off. On the database screen, "מחק רשומה" deletes one record with its
   images, "מחק תמונות בלבד" deletes every image and keeps the values, and "מחק את כל המאגר" deletes everything; each
   needs a confirmation. Limitations: **nothing is deleted automatically**, and the file is **not encrypted at rest**.

### Archive assistant (side chat)

"עוזר המאגר" is a side chat on every screen: a floating button in the bottom-left corner opens a chat panel that
answers questions in Hebrew about the stored records (`src/archive_assistant.py`), e.g. "כמה טפסים
ממתינים לבדיקה?", "הצג טפסים ממכבי שממתינים לבדיקה", "אילו טפסים חסר בהם מספר התחייבות?", "כמה טפסים התקבלו היום?",
"כמה טפסים מכל גורם שולח?". A question becomes a small, closed query: count / list / breakdown, filtered by review
status, one approved sender (with its campuses) or health funds / hospitals, one empty field, open flags, and one
date range on the received date ("התקבל" stamp), the intake date or the approval date. "Waiting for review" is the
app's own status `pending`. The query runs in Python on `record_store.archive_rows()`; forms that a filter cannot judge
(no valid date, no sender, unknown status) are listed in a note, never dropped silently. Ambiguous questions ("a
certain hospital", a date without saying which date, "שם" without first / last) get a clarification with buttons.
Results never show names, and the ID is masked to its last 3 digits; a click on a result opens the form on the review
screen while the chat stays open beside it. The conversation lives in the browser session's memory only.

- **Rules first, no API call:** fixed rules understand the example questions and their usual variants.
- **Claude only as a fallback,** for a question the rules do not fully understand: only the question text and the
  closed list of approved sender names are sent (never a record), with a closed JSON schema at low effort (about one
  cent per question on Opus), at most once per distinct question in a session. The answer is checked against
  allowlists of intents, fields, statuses, senders, dates and limits before anything runs; the model never produces
  code or SQL, and a question with 5+ digits (an ID or a number) is never sent.
- `ARCHIVE_ASSISTANT=false` hides the assistant; `ARCHIVE_ASSISTANT_MODEL=false` keeps it rules-only.
- Limitations: one sender, one empty field and one date range per question; no search by a value (use the free
  search); a first name typed into a question cannot be detected and would reach the model if the rules fail.

Privacy: uploads live only in the browser session's memory (`st.session_state`), never in a shared cache, a log
or on disk; the pipeline's temp folder is deleted after every run. The one exception
is the database above: it holds patient data on disk (git-ignored, local only), so keep the folder on an
access-controlled machine and use "מחק את כל המאגר" before sharing the project. "נקה הכל" drops
everything. `.streamlit/config.toml` serves the app on `localhost` only and turns off Streamlit's usage statistics;
open it to other lab computers only deliberately, on a trusted network.

Streamlit limits: `st.data_editor` cannot colour single cells, so editing happens per form in field cards (orange
via CSS) and the read-only tables use a pandas Styler. Field inputs sit in an `st.form` (Enter never submits), so
typing does not rerun the script; saved values are kept in `src/review_session.py`, which is pure Python and tested.

## Known limitations

- **Handwritten names in the sample files.** In the sample files all patient names are handwritten, because the
  samples were anonymized by writing a fictitious name over the original. Real forms usually carry printed names
  (patient stickers, system printouts), which the engine reads reliably: on the samples, printed text (central and
  commitment numbers, stamps, senders) was read correctly, but printed *names* could not be measured because the
  samples contain none – confirm on real forms.
- Handwritten Hebrew cursive names are often misread. Safeguards: each name area is cropped and re-read at high
  resolution, and all readings of every place the name appears must agree. Otherwise the name field is left empty and
  flagged, and the best reading is shown only as an orange "suggestion – unverified" (never an approved value).
  The crop pass always returns its best reading with a count of uncertain letters; when the full page read no
  name, the crops alone approve it only if every crop agrees, no letter is uncertain, confidence >= 0.9 and the
  first / last split comes from labels or a known form convention (`NAME_AUTO_APPROVE_CONFIDENCE`).
- Name dictionary (`data/reference/israeli_names.json`, `src/name_dictionary.py`): a suggested name word that is
  not a known name is corrected to the single known name within two look-alike edits (seen: "אצולאי" -> "אזולאי").
  Only the suggestion changes; the model's reading stays visible as an alternative. The list is a placeholder
  written for development - replace it with the CBS (הלמ"ס) lists of first names and surnames.
- First / last name order is not consistent on the forms: the evaluation compares the two names order-insensitively,
  and the review screen has a "swap first / last name" button.
- Unreadable numbers: the targeted re-read lists candidate readings (one per way of reading an unclear digit).
  Code keeps only candidates that pass validation (ID check digit, 6-digit central number); a single one is shown as
  a suggestion, several as alternatives. The field stays empty and flagged until the reviewer confirms.
- Claude declines to read Hebrew cursive often: such fields stay empty and flagged (never guessed). On the 15
  draft-labelled forms Opus 5.5 with thinking (no crops) got 69/105 fields right with 0 silent errors; names were
  the weak field (1/15).
- Earlier baseline: Gemini 3.6 Flash scored 29/35 on 5 forms; replaced by Claude for accuracy and privacy.

## What the lab needs in order to adopt it

- **A retention and automatic-deletion policy.** The database (`data/db/labextract.db`) keeps patient data until
  someone deletes it manually with "מחק את כל המאגר"; there is no automatic deletion. Real use requires the lab to
  define how long records may be kept and to automate their deletion.
- **Disk encryption.** SQLite is not encrypted at rest; the machine's disk must be encrypted and access to the
  folder restricted.
- An Anthropic API key and an agreement that form images may be sent to the model provider.
- Verification of the approved-senders list and a ground-truth set labelled on real forms.
- Replacing the local file with the lab's own system (API or database) for production.

## Development cache (`DEV_CACHE`) – development only

To save API cost while developing, model responses can be cached on disk, keyed by a hash of the
model and settings, prompt, schema and images. It is **off by default** and must **never be enabled in production**.

- Enable for a development session only: `DEV_CACHE=true` in `.env` (or as an environment variable).
- Cached responses are stored in `data/dev_cache/`. The folder is git-ignored.
- **Cached responses contain extracted personal data (names, ID numbers).**

**Before sharing, zipping or submitting the project, delete the cache:**

```bash
.venv/Scripts/python -m src.clear_cache    # dev cache + highlight debug crops
```

(or delete the `data/dev_cache/` folder). Also make sure `DEV_CACHE` is not set to `true` in `.env`.

## Targeted re-read of number fields

The full-page pass sees each page downscaled to 2400 px, so small digits on GLC stickers, stamps and
handwriting get few pixels. The extractor therefore also reports *where* the ID, central numbers, received
date and commitment numbers are (`field_locations`) and whether an empty field is `unreadable` or
`not_present`. When one of these fields is flagged (missing, unreadable, low confidence, failed validation)
and has a location, it is cropped from the full-resolution page (original pixels, deskew geometry only) and
re-read in **one extra request per form at most**. The crop reading never sees the full-page value.

The crop reading is one more reading, not the truth. Deterministic rules (see `record_builder.py`) decide:
agreement keeps the value (only for the ID, where checksum + agreement may clear a low-confidence flag);
a missing value is filled only from a valid, confident crop reading and stays flagged; a format/range failure
of a central/commitment number or date may be replaced by a valid crop reading (still flagged); an ID that was
read is never replaced (a failed checksum does not prove a misread: passports, fictitious sample IDs);
two valid but different readings empty the field (`reread_conflict`). Other readings are kept in memory as
`Record.alternatives` for the reviewer. `FIELD_REREAD=false` turns the step off; `CROP_SOURCE=processed`
crops from the preprocessed page instead of the original.

Diagnostics (`record_builder.diagnostic_lines`) explain every flag with codes only: field status, confidence,
number of boxes, re-read outcome, image size and preprocessing steps - never values or file names in logs.

## Evaluation

```bash
# Baseline: the engine of another commit, scored by THIS scorer, no API calls (dev cache only).
git worktree add ../labextract-baseline main
.venv/Scripts/python evaluation/evaluate_accuracy.py --engine-root ../labextract-baseline --cache-only \
    --save-summary data/output/baseline_summary.json
# Candidate, live (costs money; at most N requests), compared with the baseline:
DEV_CACHE=true .venv/Scripts/python evaluation/evaluate_accuracy.py --max-requests 30 \
    --save-summary data/output/candidate_summary.json --compare data/output/baseline_summary.json
# Original vs preprocessed pages, offline:
.venv/Scripts/python evaluation/compare_preprocessing.py FILE.pdf [--save data/temp/compare]
```

The report splits every field into correct_unflagged / correct_flagged / wrong_flagged /
**wrong_silent** (wrong and not flagged - the number that matters) / empty_flagged / empty_unflagged.
A value on a field labelled UNREADABLE is a guess: wrong_flagged when flagged, wrong_silent only when not. Correctness comes only from the ground truth, never from
validation. A comparison is refused when rows, labels or model differ, or when a run is incomplete.
Two Claude configurations (crops on/off, model, effort) are compared the same way; the comparison prints
the settings that changed:

```bash
# Baseline without crops, then with crops, on the same rows:
HIGHLIGHT_CROPS=false DEV_CACHE=true EVAL_MODE=true .venv/Scripts/python evaluation/evaluate_accuracy.py \
    --ground-truth data/ground_truth/ground_truth_draft.csv --tag paid5 --save-summary data/output/no_crops.json
DEV_CACHE=true EVAL_MODE=true .venv/Scripts/python evaluation/evaluate_accuracy.py \
    --ground-truth data/ground_truth/ground_truth_draft.csv --tag paid5 --max-requests 60 \
    --compare data/output/no_crops.json
```

Live runs print tokens (incl. prompt-cache reads/writes), time and cost per stage and per form.

No values are printed by default (`--show-values` prints non-personal fields only). The ground truth has only
12 rows, so the results are preliminary.

### Labeling ground truth

`data/ground_truth/labeling_worksheet.csv` lists every sample page with a *suggested* form type, tags, split and
priority (no field values - verify every suggestion against the image). For each page you label, add a row to
`data/ground_truth/ground_truth_draft.csv`:

- `page_numbers`: all pages of the form (`4,5` for a two-page form). List fields: numbers separated by `;`.
- An empty cell means "this field is not on the form". `UNREADABLE` means "it is there, but a person cannot read
  it with confidence from the image": then only an empty, flagged output scores as correct, and any value counts
  as a silent error (a guess). Use it for the whole field (a list with one unreadable number is `UNREADABLE`).
- `received_date` only from the "התקבל" stamp or handwritten next to the central number - otherwise empty.
  `sending_entity` as its canonical name in `approved_senders.json`. ID digits exactly as written.
- `tags`: `;`-separated (vocabulary in the worksheet), `split`: `tune` or `test`.
- Label from the image at full zoom, never from engine output; ideally two people label independently and
  resolve differences.

Avoiding leakage: change prompts, rules and form hints only while looking at `tune` rows
(`--split tune`); run `--split test` only to measure a finished change, and never adjust anything to fix a
specific test failure. The prompt was written while looking at all sample pages, so even the sample `test`
split is optimistic - an honest number needs new, unseen forms. Meuhedet, Leumit and Maccabi lab-request forms
occur only once in the samples (already labelled, `tune`), and the samples hold no real multi-page form and no
phone photo: the test set needs new forms of these kinds.
