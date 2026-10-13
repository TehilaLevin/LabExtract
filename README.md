# LabExtract

Extracts patient commitment-form data for the Central Virology Laboratory (Israeli Ministry of Health):
batch upload → split into forms → identify form type → extract fields → validate → review → Excel export.

> Full setup, run instructions, pipeline description and limitations will be added here.

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
