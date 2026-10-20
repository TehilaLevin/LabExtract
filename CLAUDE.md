You are setting up a new Python project called "LabExtract" in the current VS Code workspace.

## Context
This is a hackathon project (Open Code AI Hackathon 2026, in partnership with the Central Virology Laboratory, Israeli Ministry of Health).
The lab receives samples daily from HMOs (health funds) and hospitals. Each sample arrives with a commitment form ("טופס התחייבות"): printed, scanned, photographed with a phone, or handwritten. Today a lab worker types everything into Excel manually.
LabExtract replaces this: the user drags in a batch of forms (e.g. 30 forms: scanned PDFs + phone photos). The engine splits them into individual forms, identifies the form type (which HMO/hospital), extracts the fields, normalizes them, assigns a confidence score per field, and shows a review table where uncertain fields are highlighted in orange. The user fixes them and exports a ready Excel file.
Core principle: an empty, flagged field is ALWAYS better than a guessed value. The engine must never guess.

IMPORTANT: Always reply to me in Hebrew. Keep all code, file names, folder names, identifiers, comments and docstrings in English.

## Step 0 – Check the environment first
- Detect my OS (Windows / macOS / Linux) and Python version (require 3.10+). Tell me what you found before installing anything.
- pdf2image requires Poppler as a system dependency. Check if it is installed. If not, explain in Hebrew exactly how to install it for my OS (on Windows: download Poppler, add its /bin folder to PATH, or set POPPLER_PATH in .env). Do not silently skip this.

## Step 1 – Virtual environment and dependencies
- Create a virtual environment named `.venv` in the project root.
- Create `requirements.txt` with:
  anthropic, streamlit, pandas, openpyxl, python-dotenv, pdf2image, pillow, pytest
- Install inside the venv, then verify each package imports successfully.
- Configure VS Code to use the venv interpreter via `.vscode/settings.json`.

## Step 2 – Project structure
Create exactly this structure:

labextract/
├── .venv/
├── .vscode/settings.json
├── .env                        # CLAUDE_API_KEY=your_key_here (placeholder only)
├── .env.example                # same keys, no values; safe to commit
├── .gitignore                  # .env, .venv/, __pycache__/, *.pyc, data/samples/, data/temp/, data/output/
├── requirements.txt
├── README.md                   # setup, run instructions, approach, limitations
├── app.py                      # Streamlit entry point
├── src/
│   ├── __init__.py
│   ├── config.py               # loads .env, model name, confidence threshold, paths
│   ├── file_converter.py       # PDF -> page images, image loading
│   ├── image_preprocessor.py   # deskew, contrast/brightness fix for dark or skewed scans
│   ├── form_splitter.py        # splits a batch into individual forms
│   ├── extraction_engine.py    # prompt, JSON schema, per-field confidence (calls src/claude_client.py)
│   ├── validators.py           # Israeli ID checksum, sender matching, date normalization
│   ├── record_builder.py       # merges extraction + validation into review-ready records
│   ├── excel_exporter.py       # DataFrame -> .xlsx bytes
│   └── constants.py            # field definitions, Hebrew column headers
├── prompts/
│   └── extraction_prompt.txt   # the prompt sent to Claude (contains all field rules below)
├── data/
│   ├── reference/
│   │   └── approved_senders.json  # closed list of HMOs + hospitals
│   ├── samples/                # the two sample PDFs ("התחייבויות - דוגמאות" parts 1 & 2) – git-ignored
│   ├── ground_truth/
│   │   └── ground_truth.csv    # manually labeled answers for 10-15 forms (template with headers)
│   ├── temp/                   # converted page images – deleted after each run
│   └── output/                 # exported Excel files
├── evaluation/
│   └── evaluate_accuracy.py    # compares engine output vs ground_truth.csv, per-field accuracy report
└── tests/
    ├── __init__.py
    ├── test_validators.py
    ├── test_form_splitter.py
    └── test_file_converter.py

Add a .gitkeep to empty folders.

## Step 3 – Field rules (must be encoded in the prompt AND in constants.py)
Output fields (English keys → Hebrew Excel headers):
- first_name → שם פרטי
- last_name → שם משפחה
- id_number → מספר תעודת זהות
- central_numbers → מספר מרכזי
- received_date → תאריך קבלה
- sending_entity → גורם שולח
- commitment_numbers → מספר התחייבות
Plus metadata columns: source_file, page_numbers, form_type.

Where to find each field:
- central_numbers: usually on a sticker with the word "מרכזי" next to it, sometimes handwritten. A single test may have SEVERAL central numbers – keep ALL of them (list).
- received_date: from a "התקבל" stamp, or handwritten below the central number.
- commitment_numbers: may also be labeled "דרישה", and in Leumit (לאומית) forms it is labeled "דגימה". A single test may have SEVERAL – keep ALL of them (list).
- sending_entity: the HMO or hospital that sent the form. Must be chosen from the closed list in approved_senders.json – never free text.

## Step 4 – Module requirements

### config.py
- Load CLAUDE_API_KEY with python-dotenv. Raise a clear error if missing.
- CLAUDE_MODEL = the model set in CLAUDE_MODEL in .env (default claude-opus-5-5; Claude is the only model provider), CONFIDENCE_THRESHOLD = 0.8, optional POPPLER_PATH.

### file_converter.py
- Accept one file or many: PDF, PNG, JPG, JPEG.
- Convert every PDF page to a PIL Image (dpi=300). Return a list of (source_file, page_number, image).
- Fix phone photo orientation using EXIF.

### image_preprocessor.py
- Light preprocessing with Pillow: auto-contrast, brightness correction for dark scans, basic deskew. Keep the original too, in case preprocessing hurts readability.

### form_splitter.py
- A batch PDF may contain many forms. Determine where each form begins (ask Claude per page: new form or continuation of the previous one?). Group pages into forms. Output: one form = one record.

### extraction_engine.py
- For each form, send its image(s) to Claude with prompts/extraction_prompt.txt.
- First identify form_type (e.g. Clalit, Maccabi, Meuhedet, Leumit, or a specific hospital), then extract fields.
- Enforce structured JSON output (response_mime_type="application/json" + response_schema).
- Each field: { "value": string | list[string] | null, "confidence": number 0-1 }.
- The prompt must say: never guess; if unclear, cut off, or missing → value null with low confidence. Preserve Hebrew as-is. Return all central numbers and all commitment numbers found.
- Handle API errors and invalid JSON gracefully.

### validators.py (pure Python)
- validate_israeli_id(id_str): pad to 9 digits, weighted checksum (weights 1,2 alternating; sum digits of products > 9), return bool.
- match_sending_entity(raw_text): normalize (strip spaces, quotes, geresh/gershayim, common prefixes like "בית חולים", "בי\"ח", "קופת חולים") and match against approved_senders.json, including known aliases. Return canonical name or None.
- normalize_date(date_str): parse DD/MM/YYYY, DD.MM.YY, D/M/YY, YYYY-MM-DD, etc. Return "DD/MM/YYYY" or None. Never swap day/month silently – if ambiguous, flag it.

### record_builder.py
- Merge extraction + validation. Flag a field for review if: missing, failed validation, or confidence < threshold.
- Lists (central/commitment numbers) are displayed as comma-separated text.

### data/reference/approved_senders.json
- HMOs: כללית, מכבי, מאוחדת, לאומית (with aliases, e.g. "שירותי בריאות כללית").
- Hospitals: populate the general hospitals in Israel that you know, each with aliases. Add a clear note in the README that this list must be verified against the Ministry of Health list of medical institutions (gov.il) or the Hebrew Wikipedia page "בתי חולים בישראל".

### app.py (Streamlit)
- Title "LabExtract", RTL-friendly layout (inject CSS for direction: rtl).
- Drag-and-drop uploader (multiple files).
- "Extract" button with a progress bar (per form).
- Summary line: X forms processed, Y fields flagged for review.
- Phase 4 decision (approved by the user): instead of one st.data_editor table, a summary table (one row per form, styled with pandas Styler, click to open) and a per-form review screen: form image beside one card per field (value, source crop, edit box, flag reason). Results live in st.session_state, so reruns never call Claude.
- Highlight in orange (#FFA500) every flagged field (CSS on the field card; Styler in read-only tables). Explain the approach in a code comment.
- "Approve form" moves to the next form; a value that fails validation needs an explicit "approve anyway".
- "Download Excel" button from the reviewed data; allowed once all forms are approved, or with a warning listing the unapproved ones.

### excel_exporter.py
- DataFrame → in-memory .xlsx (BytesIO) with openpyxl: one row per commitment, one column per field, Hebrew headers, auto column width, right-to-left sheet, orange fill on cells that were flagged and not yet corrected (= flagged fields of forms that were not approved).

### evaluation/evaluate_accuracy.py
- Run the engine on the forms listed in ground_truth.csv and print per-field accuracy (exact match after normalization), plus a list of failed fields, so we can report honestly which fields work well and which are still hard.

## Step 5 – Privacy (mandatory)
These forms represent real patient data in production. Therefore:
- Send to the API only what is needed (the form images), nothing else.
- Do not log personal data (names, ID numbers) to the console or log files.
- Delete files in data/temp/ after each run. Do not save uploads to disk beyond what processing needs.

Local database (approved by the user, replaces the old rule "nothing personal is stored"):
- The app keeps a local SQLite file, data/db/labextract.db (git-ignored), so records survive closing the browser.
- STORED: the 7 field values as exported, the form details (source file, page numbers, form type), the review
  status and the reason codes of open flags (codes only).
- ALSO STORED (approved by the user, so a saved form can be opened again with the cropped image next to every
  field): the engine's record (values, flags, confidences, suggestions, alternatives, notes, raw text), the
  reviewer's edits, and the field crops and page previews (images table). Rows and their images are deleted together.
- NEVER STORED: the uploaded files themselves (only the crops and page previews derived from them).
- There is NO automatic deletion. The only deletion is the manual "מחק את כל המאגר" button (with a confirmation
  checkbox). Do not add automatic deletion unless the user asks.
- SQLite is NOT encrypted at rest: it relies on disk encryption and access control of the machine.
- Still no personal data in logs or the console; the database file is never committed, and tests use a temp DB.
- Never commit sample files, output files or keys.

Exception – chat reports on the sample files only:
- The sample forms in data/samples/ contain fictitious data. When you report results on these sample files to me in the chat, show full names and full ID numbers (no masking), together with the PDF file name and page number, so I can verify them against the forms.
- This exception applies ONLY to your chat reports about the sample files. It does not change any privacy rule in the app or the code: still no logging of personal data to the console or log files, temp files are deleted after each run, nothing personal is written to disk beyond what processing needs, and samples/outputs/keys are never committed.

## Step 6 – Tests and finish
- Unit tests: ID checksum (valid + invalid), date normalization (incl. ambiguous dates), sender matching with aliases, multiple central/commitment numbers handling. Run pytest and show results.
- Never write a real API key anywhere.
- Run `streamlit run app.py` once to confirm the app starts.
- README.md must include: setup, how to run, the pipeline (input → split → form type → extraction → normalization → confidence → review → export), known limitations, and what the lab would need to adopt it.
- Summarize in Hebrew: what was installed, the final folder tree, what I still need to do manually (API key, put the two sample PDFs in data/samples/, verify the hospitals list, fill ground_truth.csv, Poppler if missing), and how to run the app and the evaluation.