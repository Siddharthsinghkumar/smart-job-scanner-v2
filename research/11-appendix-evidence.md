# 11 — Appendix: Evidence

Audit performed 2026-07-07/08 by static inspection plus read-only shell verification. No pipeline code was executed; no network calls were made; no files outside `research/` were modified.

## 1. File-by-file evidence base

### Read in full
| File | Lines | Key extracted evidence |
|---|---|---|
| `scripts/scheduler.sh` | 227 | log-mtime state inference (119-135); backgrounded jobs with swallowed exit codes (68-73, 210-215); hardcoded catch-up times (111-113); `((i++))` under `set -eo pipefail` (29-40); unguarded `compute_next` at boot (54-57) |
| `scripts/run_pipeline.sh` | 444 | stage array (58-73); `.done` naming via `safe_name` (82-85); retry/backoff (53-56, 358-437); metrics/JSONL heredocs with `processed_items="0"` at all call sites (348-349, 371, 414-430); `printf 'y\n'` piped to stage11 (373-381); venv die (319-321); dry-run validators (284-313, note config validation falls back to system python 41-44) |
| `scripts/health_check.py` | 179 | env/preflight checks only; correct exit semantics (175) |
| `scripts/jobs.conf` | 3 | absolute paths into this checkout; 11:50 downloader; 23:30 `run_pipeline.sh --force` |
| `src/pipeline/pipeline_runner.py` | 230 | no sleep in retry loop (198-221); drifted metrics schema (89-99); second `_safe_name` implementation (69-71) |
| `src/pipeline/stage01_pdf_to_images.py` | 246 | empty-input exit 0 (227-229); PDF moved despite failed pages (196-205); unlocked `nonlocal` counters (145-174) |
| `src/pipeline/stage02_block_detection.py` | 111 | debug-image skip marker (42-47); failures counted, exit 0 (53-55, 99) |
| `src/pipeline/stage03_block_refiner.py` | 362 | `DEBUG_SAMPLE_RATE = 0.0` (17); module-level tesseract warm-up (31); exceptions → `skipped_error` (244-251) |
| `src/pipeline/stage04_ocr.py` | 1589 | atomic write helpers (64-76); per-block `save_progress` from writer thread (353-372) and main-process drain (955-957); strict resume wipe (788-825); failed-block permanent skip (839-842); write-only `task_backlog.txt` (1034-1043); duplicated OOM blocks (556-634 vs 673-750); cache keyed on path hash (196-199); exit-0-always (`extract_all_folders` returns None on all paths) |
| `src/pipeline/stage05_translation.py` | 191 | per-line translate calls (136-149); missing-package return → exit 0 (115-117) |
| `src/pipeline/stage06_batch_builder.py` | 137 | prefix split on `"_p"` (33-35); page regex (28-31) |
| `src/pipeline/stage07_llm_extraction.py` | 1423 | multiplicative retries (67-105 + 800-880); `pkill -f ollama` (913); model-not-ready → exit 0 (1365-1376); `--force` disables DB writes (662-671); checkpoint rglob-per-file (699-713); WAL init (637-654) |
| `src/pipeline/stage08_post_processing.py` | 130 | O(n²) SequenceMatcher dedup (35-63) |
| `src/pipeline/stage09_dynamic_resumes.py` | 236 | hardcoded usernames (27-28); module-level GitHub auth (70); non-expiring cache (78-87) |
| `src/pipeline/stage09_local_filter.py` | 184 | threshold default 0.3 (16); full regeneration of shortlist JSON (176-178) |
| `src/pipeline/stage09_llm_filter.py` | 967 | index-based resume (819-830, 854-857); caller-side extra `sleep(1.2)` (504-506); zero-keys/missing-input returns → exit 0 (731-733, 797-799); validation-cache deletion at start (753-754); DDL + REPLACE (221-300) |
| `src/pipeline/stage09_resume_matching.py` | 36 | unused composite runner |
| `src/pipeline/stage09_shortlist.py` | 322 | verdict substring heuristics (122-160); pre-send `log_job` (180-186); latest-JSON-by-mtime (271-275); unused `pandas` import (4) |
| `src/pipeline/stage10_notification.py` | 533 | module-level `SystemExit` on missing env (38-41); test message (131-141); unescaped website (415); watermark advance `jobs[-1][5]` (504-507); no `retry_after` (187-211); positional tuples (391, 483, 506) |
| `src/pipeline/stage11_cleanup.py` | 270 | CLEAN_DIRS incl. raw_pdfs/processed_pdfs (51-66); `input()` confirm (249); `*.log`-only log pruning (168-186) |
| `src/llm/gemini_client.py` | 1220 | global `genai.configure` in worker (507); post-call `sleep(1.2)` (523); per-request `_save_key_usage` under lock (533-538); quota constants (56-59); per-request SQLite connect+DDL (588-623); threaded `run_batch` (677-775) |
| `src/utils/*` (4 files), `src/llm/llm_utils.py`, `src/llm/ollama_runner.py`, `src/ocr/*` (2 facades), `src/vision/block_detector.py`, `src/_compat_run_module.py`, all 14 numbered shims | ~350 total | zero importers for the four utility modules (grep-verified); facades import from stage modules |
| `tests/` (5 files) | ~400 | structural assertions only; placeholder-env fixture existence (test_imports.py:41-56) |
| `Makefile`, `pytest.ini`, `.github/workflows/tests.yml`, `.env.example`, `.gitignore`, `README.md`, `src/README.md`, `configs/pipeline_config.json` | — | full reads; src/README.md documents pre-refactor numbered scripts as "actual production execution order" |

### Read partially (targeted)
| File | Coverage | Basis for claims |
|---|---|---|
| `src/downloader/newspaper_downloader.py` (2,821 lines) | Lines 1–1600 in full; complete `def`-map of all 40 functions; tail 2600–2821 in full (Telethon flow + `download_all_optimized` + `main`) | All downloader findings cite read regions. The three GDrive download implementations (1709, 2086, plus inline preprocessor paths) were identified from the function map and surrounding reads; their internal line-by-line behavior beyond the read ranges is characterized from signatures, call sites, and the read epaperwave/preprocess paths. |
| `configs/gemini_config.json`, `configs/newspaper_config.json` | Structure read with all key/token/hash values redacted before display | headless:false, disable_javascript:true, Chrome-91 UA, 12 epaperwave web_sources, telegram channel id, real-key presence (lengths only — no secret values were read into this report) |
| `docs/*.md`, `research/backend_review.md`, `research/frontend_api_review.md` | Headers/openings | Context only (VLM roadmap intent; prior review existence) |

## 2. Command outputs used (read-only)

All executed 2026-07-07 unless noted; full outputs in the session transcript.

1. `find` file inventories (full tree; code/config/db subset) — established layout, shims, orphan files.
2. `wc -l` over all source — line counts quoted in reports.
3. `sqlite3 progress.db '.schema'` + row counts — orphan DB characterization (pages=29, backlog=0, worker_stats=2; last write Nov 2025 per `file` header + mtime).
4. `git log --oneline`, `git status --porcelain`, `git ls-files`, `git remote -v` — single commit f51f606; only example configs tracked; origin = github.com/Siddharthsinghkumar/smart-job-scanner-v2.
5. `ls -la` repo root (twice) — symlinked root configs; loose logs/PNGs/.deb; `telegram_session_newspaper_wave_10.session` present.
6. Redacting JSON printer over the three configs — config evidence without exposing secrets.
7. `grep` sweeps: sqlite3 usage; threading/multiprocessing/asyncio; `requests.(get|post)`; `progress.db` references (none in src); `task_backlog` (single write site, no reader); `state_manager|logging_utils|file_utils|llm_utils` importers (none); dependency-import mapping per package; `pd\.` in stage09_shortlist (none).
8. `du -sh cache_blocks logs data run_state artifacts` → 629M / 11M / (data absent) / 267M / 6.1M; `ls logs/ | wc -l` → 888.
9. `ls -d 4_env venv .venv` → none exist; `ls data/` → does not exist; `pgrep -af 'scheduler.sh|run_pipeline'` → not running.
10. `timeout 280 ./4_env/bin/python -m pytest -q` → **failed: no such file or directory** (direct evidence for F-001; also why no test run could be performed).

## 3. Finding-to-evidence mapping

Every finding in `findings.json` carries `location` (file:line) and `evidence` fields; report sections cite the same references inline. Cross-index:

| Finding | Primary evidence artifact |
|---|---|
| F-001 | Command outputs #9, #10; run_pipeline.sh:319-321; jobs.conf |
| F-002/F-003 | Stage return-path reads (§1 table rows for stages 01/02/04/05/07/09) |
| F-004/F-005/F-038/F-039 | scheduler.sh + run_pipeline.sh full reads |
| F-006 | grep of all `update_metrics`/`append_structured_log` call sites (4th arg literal "0") |
| F-007 | pipeline_runner.py full read diffed against bash runner |
| F-008–F-012 | stage10/stage09_shortlist full reads |
| F-013 | stage11 full read + runner pipe |
| F-014–F-017 | stage04 full read; grep #7 (task_backlog) |
| F-018/F-019 | stage07 full read |
| F-020 | stage09_llm_filter + stage09_local_filter reads |
| F-021/F-022 | gemini_client full read |
| F-024/F-025 | sqlite-query-surface.csv (compiled from reads) |
| F-026–F-028 | downloader reads + redacted config + scheduler exit-code handling |
| F-029/F-045 | git ls-files vs on-disk ls; .gitignore |
| F-030/F-031/F-044 | cross-stage state inventory (01-inventory §6) |
| F-032/F-033/F-042 | import-time code cites; grep #7 |
| F-034/F-035/F-043 | du/ls measurements #8 |
| F-036 | absence findings + dormancy evidence #9 |
| F-037/H-05 | tests/ + workflow reads; failed local pytest #10 |
| F-040/F-041 | stage03/stage08 reads |

## 4. Unresolved questions

1. **Where did `4_env/` go, and is another machine running this system?** `jobs.conf` points at this checkout, suggesting no — but only Sid can confirm no second deployment exists.
2. **Is GitHub Actions CI currently green?** (H-05; not checked — offline audit.)
3. **Actual nightly wall-clock durations per stage** — no recent successful run logs exist to measure; throughput claims in 05-review are structural, not measured.
4. **Ollama VRAM headroom for `num_parallel=2`** (H-03) — requires GPU runtime.
5. **epaperwave wp-json/RSS availability** (H-04) — requires a network probe.
6. **Real-world frequency of Telegram send failures** (bounds the practical impact of F-008) — stage10 logs from past runs could answer this; the surviving logs predate the current code.
7. **Whether `run_state/` experiment directories (267 MB, tune_v1..v15 etc.) are still needed** — they look like YOLO-era residue but deleting is Sid's call.

## 5. Limitations

- **Static audit of a dormant system.** No stage was executed (venv absent; GPU pipeline; live credentials). All "confirmed issue" classifications rest on code-path logic that is unambiguous on inspection; anything requiring runtime observation is explicitly classified `hypothesis` (H-01…H-05).
- **Downloader internals partially sampled** (~60% by line; 100% by function map) — see §1. Findings there are structural (duplication, cascade shape, config) and do not depend on the unread bodies.
- **No network verification** of epaperwave/Drive/Telegram behavior; scraping-fragility claims derive from code shape and the on-disk failure artifacts (`logs/failed_batches/`, `logs/raw_responses/`, debug-dump machinery).
- **No secret values were read or reproduced**; config evidence used length/type redaction.
- **Line numbers** reference the working tree at commit f51f606 (plus untracked `docs/`); future edits will shift them.
