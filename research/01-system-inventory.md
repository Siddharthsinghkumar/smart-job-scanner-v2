# 01 — System Inventory

Audit date: 2026-07-07. Scope: static inspection of the working tree at `/home/sidd/project/smart-job-scanner-v2` (git `main`, single commit `f51f606`). Remote: `github.com:Siddharthsinghkumar/smart-job-scanner-v2`.

## 1. Runtime and language inventory

| Component | Detail | Evidence |
|---|---|---|
| Language | Python 3 (health check requires >= 3.10; CI pins 3.11) | `scripts/health_check.py:41`, `.github/workflows/tests.yml:17` |
| Shell | Bash (orchestration + scheduling) | `scripts/run_pipeline.sh`, `scripts/scheduler.sh` |
| Interpreter expected at runtime | `./4_env/bin/python` (project-local venv) | `scripts/run_pipeline.sh:40`, `Makefile`, `scripts/jobs.conf` |
| **Interpreter actually present** | **`4_env/` does not exist in this checkout.** System python is 3.12.3 at `/usr/bin/python3`. `data/` also does not exist and no scheduler process is running — the system is currently non-runnable/dormant on this machine, which is the exact absolute path hardcoded in `jobs.conf`. | `ls` verification 2026-07-07; `scripts/jobs.conf:1` |
| GPU stack | torch 2.4.1+cu121, EasyOCR (GPU reader, FP16), pynvml monitoring, sized for a 3050 Ti (`GPU_MEMORY_LIMIT_MB = 3800`) | `requirements.txt`, `src/pipeline/stage04_ocr.py:54` |
| Local LLM | Ollama serving model `openhermes` on `localhost:11434`; the stage itself starts/stops the server | `src/pipeline/stage07_llm_extraction.py:149-151,887-915` |
| Cloud LLM | Google Gemini (`gemini-2.5-flash`) via `google-generativeai`, multi-key rotation | `src/llm/gemini_client.py:62-67` |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` for job/resume similarity | `src/pipeline/stage09_local_filter.py:115` |
| Translation | Argos Translate hi→en (CPU) | `src/pipeline/stage05_translation.py:26,66-67` |
| OCR aux | Tesseract via pytesseract (block refiner filtering only) | `src/pipeline/stage03_block_refiner.py:6,18` |
| Scraping | Selenium (Chrome, headless=new optional, JS-disabled mode), BeautifulSoup, requests+urllib3 Retry, Telethon (Telegram MTProto) | `src/downloader/newspaper_downloader.py` |
| Code volume | ~12,600 lines of Python/bash in production paths; largest files: downloader 2,821, stage04 1,589, stage07 1,423, gemini_client 1,220, stage09_llm_filter 967 | `wc -l` run 2026-07-07 |

## 2. Key scripts and shell entrypoints

### Orchestration layer
- `scripts/scheduler.sh` (227 lines) — homemade daemon: parses `scripts/jobs.conf`, computes next daily run times, 1-second polling loop, backgrounds jobs with output to timestamped log files, plus "time-aware startup" catch-up logic hardcoding 11:50/23:30.
- `scripts/jobs.conf` — 2 jobs: `auto_download_pdfs` daily 11:50 (runs the downloader), `force_pipeline` daily 23:30 (runs `run_pipeline.sh --force`). Commands use absolute paths into this checkout.
- `scripts/run_pipeline.sh` (444 lines) — sequential 14-stage runner: `.done` markers in `run_state/`, per-stage retry with exponential backoff (10s→300s, MAX_RETRIES=5, 0=unlimited), per-stage stdout/stderr/combined logs, structured JSONL log and `data/pipeline_metrics.json` updates via embedded Python heredocs, `--dry-run` validation mode, auto-answers `y` to stage11's interactive prompt.
- `src/pipeline/pipeline_runner.py` (230 lines) — a second, parallel Python implementation of the same runner (drifted: no backoff sleep between retries, different metrics keys). Not referenced by scheduler or Makefile.
- `src/pipeline/stage09_resume_matching.py` — a third composite runner for the four stage-09 sub-steps; not referenced by the stage list.
- `Makefile` — `run`, `dry-run`, `test`, `health`; all depend on `./4_env/`.

### Pipeline stages (executed order per `run_pipeline.sh:58-73`)
1. `stage01_pdf_to_images.py` — PDF→PNG (PyMuPDF, pdf2image fallback, ProcessPool+ThreadPool)
2. `stage02_block_detection.py` — connected-component block detection (multiprocessing Pool → `src/vision/block_detector.py`)
3. `stage03_block_refiner.py` — column split + tesseract-based keep/skip filtering (ProcessPool)
4. `stage04_ocr.py` — GPU-dominant EasyOCR with homemade GPU-thread/CPU-process queue system
5. `stage05_translation.py` — Argos hi→en line-by-line
6. `stage06_batch_builder.py` — page text → ≤1000-word batches with 50-word overlap
7. `stage07_llm_extraction.py --no-hybrid` — Ollama job extraction, SQLite+checkpoint resume
8. `stage08_post_processing.py` — O(n²) SequenceMatcher dedup of extracted job lines
9. `stage09_dynamic_resumes.py` — PDF resumes + GitHub repo scrape → 3 resume texts
10. `stage09_local_filter.py` — MiniLM cosine similarity shortlisting (threshold 0.3)
11. `stage09_llm_filter.py` — Gemini per-job analysis → SQLite + JSON
12. `stage09_shortlist.py` — parse Gemini verdicts → markdown shortlist + `sent_jobs` history DB
13. `stage10_notification.py` — Telegram Bot API sends
14. `stage11_cleanup.py` — deletes all intermediates + raw/processed PDFs, prunes logs >7 days

Legacy numbered scripts (`src/1_pdf_to_images.py` … `src/12_auto_download_pdfs.py`) are 3-line shims delegating to the stage modules via `src/_compat_run_module.py`.

### Ingestion
- `src/downloader/newspaper_downloader.py` (2,821 lines) — scheduled at 11:50; scrapes epaperwave.com listing pages with Selenium, resolves Google Drive / direct-PDF links (three separate GDrive download implementations), falls back to a Telethon scan of a Telegram channel; validates PDFs (header/EOF/min size); writes `data/raw_pdfs/`.

## 3. SQLite databases

| Database | Owner code | Tables | Status |
|---|---|---|---|
| `progress.db` (repo root) | none — zero references in `src/`/`scripts/` | `pages`, `backlog`, `worker_stats` | **Orphaned** (legacy of an older OCR version; 29 rows, last written Nov 2025) |
| `data/processing_state.db` | stage07 | `processed_files` | Live; WAL + 30s timeout |
| `data/llm_results/llm_job_analysis.db` | stage09_llm_filter | `job_analysis` (18 cols, 5 indexes) | Live |
| `data/shortlist_history.db` | stage09_shortlist (writer), stage10 (reader) | `sent_jobs` | Live; cross-stage contract |
| `data/gemini_interactions.db` | gemini_client audit log | `request_logs` | Live; duplicated by JSONL log |

All access is raw `sqlite3` stdlib; no ORM, no migrations, DDL inline at call sites. (Full query surface in `sqlite-query-surface.csv`.)

## 4. External APIs and services

| Service | Used by | Protocol | Notes |
|---|---|---|---|
| Ollama (`localhost:11434/api/generate`) | stage07 | sync `requests` + urllib3 Retry, 240s timeout | stage starts/stops the server itself, incl. `pkill -f ollama` |
| Google Gemini | stage09_llm_filter via `src/llm/gemini_client.py` | sync SDK (`google-generativeai`) | multi-key rotation, homemade token-bucket + 20/day/key quota, `sleep(1.2)` per call |
| Telegram Bot API | stage10 | sync `requests` POST, 15s timeout, 3 retries | live test message every run |
| Telegram MTProto (Telethon) | downloader phase 3 | async wrapped in `asyncio.run` | session file `telegram_session_newspaper_wave_10.session` at repo root |
| GitHub REST (PyGithub) | stage09_dynamic_resumes | sync | hardcoded usernames `downl2160`, `Siddharthsinghkumar`; optional `AIML_GITHUB_TOKEN` |
| epaperwave.com (+ Google Drive) | downloader | Selenium + requests | 12 of 12 web sources are this one aggregator |

(Full matrix in `external-api-matrix.csv`.)

## 5. Scheduling / orchestration model

- **Scheduler**: `scheduler.sh` is a hand-rolled cron+anacron replacement — a foreground bash daemon that must be started manually (no systemd unit, no supervisor, nothing in the repo starts it). It infers "did job X already run today" from log-file mtimes (`find -newermt`), backgrounds jobs with `&`, and never inspects their exit codes beyond appending a line to the job's log.
- **Pipeline**: strictly sequential stages, file-based `.done` completion markers keyed by sanitized command string; nightly run always uses `--force` (deletes markers), so markers only matter for manual resume.
- **Retry**: per-stage subprocess retry in the runner (exp backoff); per-request retries inside stage07/stage09/stage10/gemini_client — but most stage-internal failures do not surface as non-zero exit codes, so the runner-level retry rarely triggers on real errors (see 03-…-audit, F-002).

## 6. State / caching / storage mechanisms

At least nine distinct state mechanisms coexist:

1. `run_state/*.done` — stage completion (runner)
2. `data/progress/<paper>_progress.json` — per-block OCR progress (stage04, rewritten per block)
3. `cache_blocks/*.npy` — preprocessed-image cache, path-hash keyed, never evicted (629 MB on disk now)
4. `data/processing_state.db` + `data/processing_checkpoint.txt` — stage07 dual resume
5. `data/llm_results/progress.json` — stage09_llm_filter index-based resume
6. `data/shortlist_history.db` — dedup across days
7. `data/telegram/last_telegram_sent.txt` — notification watermark
8. `data/raw_pdfs/.download_state.json` — GDrive skip list (24h TTL)
9. `data/.key_usage.json`, `data/.key_validation_cache.json` — Gemini quota/validation state
10. `data/pipeline_metrics.json` + `logs/pipeline_YYYYMMDD.jsonl` — metrics/log (counters provably stuck at 0, see F-006)

Inter-stage IPC is entirely directory-convention based: `raw_pdfs → pdf2img → job_blocks_smart → job_blocks_refined → block_texts/page_texts → all_eng_text → batch_inputs → batch_output + Jobs_found_final → shortlisted_jobs_json → llm_results → shortlists → Telegram`.

## 7. Deployment assumptions

- Single Linux machine with NVIDIA GPU (3050 Ti class), CUDA 12.1 torch wheels, Chrome + chromedriver, Ollama installed, Tesseract installed, Argos hi→en package installed.
- Project must live at `/home/sidd/project/smart-job-scanner-v2` (absolute paths in `jobs.conf`) and stages must run with CWD = project root (relative `Path("data/...")` everywhere).
- Secrets: `.env` (bot token, chat id, API id/hash, Gemini keys), `configs/gemini_config.json` (real keys on disk), `token_telegram/Jobs_sidd_bot.txt`, Telethon `.session` at repo root. All are gitignored (verified against `git ls-files`), but all sit in plaintext in the working tree.
- No containerization, no supervision, no remote logging/alerting. CI (GitHub Actions) runs structural pytest only.

## 8. Inventory-level anomalies worth flagging immediately

- **The production venv is gone** (`4_env/` absent) → `make run/test/health`, `run_pipeline.sh`, and both `jobs.conf` commands fail at interpreter lookup on this machine. Classification: confirmed issue, critical for operations (F-001).
- `data/` tree absent; latest logs are from 2026-06-13; scheduler not running → system dormant ≥3 weeks.
- Repo root contains ~30 loose experiment logs, `1.png`, `tmp_p14.png`/`tmp_p15.png` (~6 MB each), `cuda-keyring_1.0-1_all.deb`, and 888 files under `logs/`.
- `requirements.txt` declares `torchaudio` (never imported; only plausibly needed if EasyOCR pulled it, which it does not) and `pandas` is imported once and never used (`src/pipeline/stage09_shortlist.py:4`).
