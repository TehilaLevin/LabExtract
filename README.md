# LabExtract

Extracts patient commitment-form data for the Central Virology Laboratory (Israeli Ministry of Health):
batch upload → split into forms → identify form type → extract fields → validate → review → Excel export.

> Full setup instructions and pipeline description will be added here.

## Running the app

```bash
.venv/Scripts/activate          # Windows; macOS / Linux: source .venv/bin/activate
streamlit run app.py            # opens http://localhost:8501
```

`.env` must hold `GEMINI_API_KEY` and `GEMINI_MODEL` (see `.env.example`); PDFs need Poppler.

1. **Upload** – drag in PDFs / phone photos (many forms per file are fine) and press "חלץ". A progress bar
   follows the split and then each form. Gemini is called only here.
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

## Development cache (`DEV_CACHE`) – development only

To save API cost while developing, Gemini responses can be cached on disk, keyed by a hash of the
model, prompt, schema and images. It is **off by default** and must **never be enabled in production**.

- Enable for a development session only: `DEV_CACHE=true` in `.env` (or as an environment variable).
- Cached responses are stored in `data/dev_cache/`. The folder is git-ignored.
- **Cached responses contain extracted personal data (names, ID numbers).**

**Before sharing, zipping or submitting the project, delete the cache:**

```bash
.venv/Scripts/python -c "from src.extraction_engine import clear_dev_cache; print(clear_dev_cache(), 'files deleted')"
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

The report splits every field into correct / correct but flagged / missing / wrong but flagged /
**wrong and not flagged** (silent errors). Correctness comes only from the ground truth, never from
validation. A comparison is refused when rows, labels or model differ, or when a run is incomplete.

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
