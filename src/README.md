# src Layout

## Production pipeline scripts (kept in `src`)
These are used by `run_pipeline.sh`, `scheduler.sh`, `jobs.conf`, or imported by those stage scripts.

### Actual production execution order
- Scheduled at `11:50` daily: `12_auto_download_pdfs.py` (downloader job)
- Scheduled at `23:30` daily: `run_pipeline.sh --force`, which executes:
  - `1_pdf_to_images.py`
  - `2_run_smart_detector_batch.py`
  - `3_block_refiner.py`
  - `4_gpu_multilang_easyocr_working_slow_but_accurate.py`
  - `5_argos_translate_batch.py`
  - `6_create_batches_for_ollama.py`
  - `7_final_ollama_pipeline.py --no-hybrid`
  - `8_post_processing.py`
  - `9-1_dynamic_resumes_full.py`
  - `9-2_local_filter.py`
  - `9-4_llm_search.py`
  - `9-5_generate_shortlist_latest.py`
  - `10_notify_shortlist_telegram.py`
  - `11_cleanup_data.py`

`run_pipeline.sh` writes stage logs to `logs/` and completion markers to `run_state/*.done`.

### Production support modules
- `smart_block_detector_b.py` (imported by stage 2)
- `gemini_multikey_9_3_helper_script.py` (imported by stage 9-4)

## Research / experimental scripts
All non-production scripts are stored in `src/research/`.

These include tests, benchmarks, backups, prototypes, and helper experiments that are not part of the scheduler/runner execution chain.

If a script is needed in production later, move it back to `src` and verify scheduler/runner references.
