# src Layout

## Production pipeline scripts (kept in `src`)
These are used by `run_pipeline.sh`, `scheduler.sh`, `jobs.conf`, or imported by those stage scripts.

### Actual production execution order
- Scheduled at `11:50` daily: `src/downloader/newspaper_downloader.py`
- Scheduled at `23:30` daily: `scripts/run_pipeline.py --force`, which executes:
  - `src/pipeline/stage01_pdf_to_images.py`
  - `src/pipeline/stage02_block_detection.py`
  - `src/pipeline/stage03_block_refiner.py`
  - `src/pipeline/stage04_ocr.py`
  - `src/pipeline/stage05_translation.py`
  - `src/pipeline/stage06_batch_builder.py`
  - `src/pipeline/stage07_llm_extraction.py --no-hybrid`
  - `src/pipeline/stage08_post_processing.py`
  - `src/pipeline/stage09_dynamic_resumes.py`
  - `src/pipeline/stage09_local_filter.py`
  - `src/pipeline/stage09_llm_filter.py`
  - `src/pipeline/stage09_shortlist.py`
  - `src/pipeline/stage10_notification.py`
  - `src/pipeline/stage11_cleanup.py`

`run_pipeline.sh` writes stage logs to `logs/` and completion markers to `run_state/*.done`.

### Production support modules
- `smart_block_detector_b.py` (imported by stage 2)
- `gemini_multikey_9_3_helper_script.py` (imported by stage 9-4)

## Research / experimental scripts
All non-production scripts are stored in `src/research/`.

These include tests, benchmarks, backups, prototypes, and helper experiments that are not part of the scheduler/runner execution chain.

If a script is needed in production later, move it back to `src` and verify scheduler/runner references.
