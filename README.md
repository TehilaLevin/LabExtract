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
4. **Export** – Excel with Hebrew headers, right-to-left sheet, one row per form and a review-status column.
   Flagged cells of forms that were not approved are orange; exporting while forms are unapproved shows them
   and needs a confirmation.

Privacy: uploads, results, crops and page previews live only in the browser session's memory (`st.session_state`),
never in a shared cache, a log or on disk; the pipeline's temp folder is deleted after every run. "נקה הכל" drops
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
- Claude declines to read Hebrew cursive often: such fields stay empty and flagged (never guessed). On the 15
  draft-labelled forms Opus 5.5 with thinking (no crops) got 69/105 fields right with 0 silent errors; names were
  the weak field (1/15).
- Earlier baseline: Gemini 3.6 Flash scored 29/35 on 5 forms; replaced by Claude for accuracy and privacy.

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
5 rows, so the results are preliminary.

### Labeling ground truth

`data/ground_truth/labeling_worksheet.csv` lists every sample page with a *suggested* form type, tags, split and
priority (no field values - verify every suggestion against the image). For each page you label, add a row to
`data/ground_truth/ground_truth.csv`:

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
