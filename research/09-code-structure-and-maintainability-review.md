# 09 — Code Structure and Maintainability Review

## 1. Script sprawl review

- **Production Python:** ~12,600 lines. Four files carry 55% of it: downloader (2,821), stage04 (1,589), stage07 (1,423), gemini_client (1,220). Each of the four is a single file mixing infrastructure (process pools, HTTP sessions, rate limiters, service management) with business logic (what is a job ad, which resume matches).
- **Three orchestrator implementations** (bash runner, `pipeline_runner.py`, `stage09_resume_matching.py`) — two are unused-but-maintained (F-007).
- **Shim layer:** 14 numbered scripts (`src/1_…` to `src/12_…`, plus `smart_block_detector_b.py`, `gemini_multikey_9_3_helper_script.py`) delegate to the new modules via `_compat_run_module.py`. Legitimate transitional scaffolding — but nothing schedules their removal, and `src/README.md` still documents the *old* numbered scripts as "the actual production execution order", contradicting `run_pipeline.sh`. Docs and code disagree about what production runs.
- **Dead scaffolding (F-033, low, confirmed):** `src/utils/state_manager.py`, `src/utils/logging_utils.py`, `src/utils/file_utils.py`, `src/llm/llm_utils.py` have **zero importers** (verified by grep). They are the skeleton of a shared-utilities layer that the stages never adopted — every stage still hand-rolls its own logging/config/save-json. The facades `src/ocr/easyocr_engine.py` and `src/llm/ollama_runner.py` re-export from stage modules (inverted dependency: the "library" imports the "script").
- **`research/` (36 scripts) is fine as an archive** but lives in the same tracked tree with names like `4_cpu_multilang_ocr_fast_but_shit.py` and `?_block_detector.py` (a filename starting with `?`), which pollute tooling and globbing.
- **Repo root hygiene:** ~30 loose experiment logs, `1.png`, two 6 MB `tmp_p*.png`, `cuda-keyring_*.deb`, an orphaned `progress.db`, a committed `gemini_interactions_log.json`, and a live Telethon `.session` file. None break anything; together they make the root unreadable and risk accidental commits (the JSON interactions log *is* tracked).

## 2. Naming consistency

- Stage modules are consistently named (`stageNN_purpose.py`) — genuinely good refactor output. The five `stage09_*` files share a number because they were one historical stage; acceptable, but `stage09_llm_filter` vs `stage09_local_filter` vs the unused `stage09_resume_matching` forces readers to open files to know the order (the order lives only in the bash array).
- Data directory names mix conventions: `Jobs_found_final` (capitalized), `job_blocks_smart`, `all_eng_text`, `pdf2img`. Filename-embedded metadata differs per stage (space vs underscore in newspaper names — `stage06:35` normalizes, stage04 doesn't).
- Config keys and constants: thresholds live as magic numbers in code (`0.3` similarity default, `0.37/0.34` priority cutoffs in `stage09_shortlist:229-234`, `0.65` fuzzy ratio, `20` daily quota, `3800` MB VRAM) rather than in `configs/pipeline_config.json`, which currently holds only the stage list and schedule — the one config file that exists is barely used as configuration.

## 3. Package structure quality

- `src/` has an `__init__.py` tree and clean subpackage names (`pipeline`, `llm`, `ocr`, `vision`, `utils`, `downloader`) — the *shape* of a package is right.
- But there is **no packaging metadata** (no `pyproject.toml`/`setup.py`), so imports depend on CWD and `sys.path` hacks (`stage02:20`, `stage09_llm_filter:24`). Stages are executed as file paths by the runner, not as modules (`-m`), which is why every stage uses relative `Path("data/…")` and only works from the project root (F-032, medium, confirmed).
- Python version ambiguity: venv-pinned (missing) vs system 3.12 vs CI 3.11; `datetime.utcnow()` (deprecated 3.12) is used in stage04/gemini_client, and sqlite3 default datetime adapters (deprecated 3.12) in stage07 — the code will emit deprecation noise the moment it runs on the system interpreter.

## 4. Separation of business logic vs orchestration

The recurring anti-pattern: **module-level side effects**. Importing a stage module (which the test suite does) executes: directory creation, log-file creation, Tesseract warm-up (`stage03:31` — runs OCR at import), CUDA environment mutation (`stage04:126-136`), GitHub client construction and network auth (`stage09_dynamic_resumes:70`), `.env` loading and **`raise SystemExit` on missing env** (`stage10:38-41`). Consequences:

- Stages cannot be imported for unit testing without elaborate env mocking (the test suite's placeholder-env fixture exists precisely to work around this, `tests/test_imports.py:41-56`).
- Business logic (block filtering rules, job-keyword lists, prompt templates, resume-choice keywords) is welded to process concerns (queues, sessions, signal handlers) in the same file and often the same function.
- Infrastructure concerns leak upward: stage07 manages a system service; stage11 knows every stage's directories; stage10 embeds matching logic.

## 5. Modularization opportunities

Concrete target structure (no rewrite — extraction refactors):

```
src/
  core/            config.py (typed config incl. thresholds), paths.py (single source of data-dir layout),
                   logging.py (one setup, run-id aware), storage.py (SQLite layer, 04-review §7)
  ingestion/       fetchers/ (epaperwave, telegram, direct), drive.py (ONE GDrive impl), validate.py
  extraction/      pdf_images.py, block_detect.py, refine.py, ocr/ (engine + worker pool), translate.py
  analysis/        batch.py, ollama_extract.py (client only — no service mgmt), dedupe.py,
                   embed_filter.py, gemini/ (client, rate_limit), shortlist.py
  delivery/        telegram_format.py, telegram_send.py (outbox worker)
  cli.py           scanner status / run / retry-failed (08-review §5)
```

Rules that matter more than the layout: no module-level side effects (everything behind `main()`/factory functions), stages runnable as `python -m src.…` from anywhere, one logging setup, one storage module, thresholds in config.

## 6. Refactor targets (ranked by risk-reduction per effort)

1. **Delete dead code** (F-033): unused utils, `pipeline_runner.py` *or* the bash runner (keep one), `stage09_resume_matching.py`, orphaned `progress.db`, root clutter. Zero behavioral risk, immediate clarity. Fix `src/README.md` to describe reality.
2. **Exit-code truthfulness pass** across stages 01–09 (ties to F-002; ~20 lines total, transforms orchestration semantics).
3. **`core/storage.py`** consolidating the four DBs' access (04-review §7) — removes positional-tuple coupling (stage10) and triple DDL.
4. **stage10 outbox split** (07-review §6) — small file, high-value delivery-loss fix.
5. **Downloader decomposition** (06-review §6.5) — biggest file, but do it *fetcher-by-fetcher* opportunistically as sources break, not big-bang; prioritize the single-GDrive-implementation merge.
6. **stage04** — explicitly *defer*: the VLM/OCR roadmap (`docs/V2_MAIN_ROADMAP_VLM_OCR.md`) plans to replace this stage's approach; only extract the resume/progress logic (which survives any engine swap) and leave the executor until the engine decision lands.
7. **Add `pyproject.toml`**, run stages with `-m`, replace `Path("data/…")` with `core/paths.py` anchored at a `SCANNER_HOME` env var — unlocks running from anywhere and eventual relocation.

## 7. Testing and CI reality check (F-037, medium, confirmed)

- The suite is structural: imports succeed, files exist, configs have keys, bash arrays parse (`tests/test_runner_scripts.py` even parses `run_pipeline.sh` to validate script references — clever, but it validates *wiring*, never behavior).
- Zero tests cover: job-line parsing, shortlist verdict extraction, Telegram formatting/escaping, watermark logic, dedup, date heuristics — i.e., none of the logic this audit found bugs in. Each confirmed bug above (F-008, F-009, F-020) is a cheap regression test once fixed.
- CI installs the full cu121 torch stack (~GBs) to run import tests — slow and fragile against wheel availability. Splitting `requirements.txt` (runtime-GPU) from a slim CI set, or guarding heavy imports, would cut CI time drastically.
- `make test` is currently broken on the production machine (no `4_env`) — the validation loop and the production loop are both down (F-001).

## 8. Findings referenced

F-001, F-002, F-007, F-032, F-033, F-034, F-037, F-042 (hardcoded personal usernames/thresholds), F-045 (positive: secrets hygiene in git), F-047 (positive: dry-run/CI/health-check exist), plus the module-level side-effect pattern (rolled into F-032). Details in `findings.json`.
