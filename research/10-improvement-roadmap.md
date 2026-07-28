# 10 — Improvement Roadmap

Sequenced for one maintainer. Each item names the findings it retires. Effort scale: S (<½ day), M (½–2 days), L (multi-day).

## Phase 0 — Resurrection (the system is currently down)

| # | Action | Effort | Retires | Expected impact |
|---|---|---|---|---|
| 0.1 | Recreate `4_env` from `requirements.txt` (or repoint `VENV_PYTHON`/Makefile/jobs.conf at a new venv); verify `make dry-run`, `make test`, `make health` pass | S | F-001 | System becomes runnable at all |
| 0.2 | Decide scheduler start mechanism **now**: two systemd `--user` units + timers (`Persistent=true`) replacing scheduler.sh, with `OnFailure=` unit that sends a Telegram message | M | F-004, F-038, large part of F-036 | Survives reboots, single-instance, journald logs, failure alert — retires the entire homemade scheduler |
| 0.3 | Set `headless: true` in `newspaper_config.json` (or confirm a display exists at 11:50) and restore `max_preprocess_workers=3` | S | part of F-027 | Downloader works unattended and ~3× faster preprocess |

## Phase 1 — Quick wins: make failure loud (order matters; do before any orchestrator)

| # | Action | Effort | Retires | Expected impact |
|---|---|---|---|---|
| 1.1 | **Exit-code truthfulness pass**: every stage exits non-zero on zero-yield or failure-ratio breach; stage01 asserts `raw_pdfs` non-empty; stage07 exits non-zero when the model never readies | M | F-002, F-003 (with 1.2), F-031 partially | The single highest-leverage change in this audit: retries start working, `.done` stops lying, silent no-op nights become impossible |
| 1.2 | **Telegram failure + digest hook** in the runner: on stage failure or pipeline zero-yield, sendMessage to admin chat; nightly one-line digest | S | F-028, F-036 (alerting half), F-003 | Every failure found here becomes a push notification; uses the bot you already run |
| 1.3 | **Guard stage11**: refuse to delete `raw_pdfs`/`processed_pdfs` unless the run produced ≥1 page-text and ≥1 LLM response; replace piped `y` with explicit `--yes`; archive-then-prune instead of unlink | S | F-013 | No more destroying the day's inputs (and debug evidence) after a failed run |
| 1.4 | **Fix stage10 delivery loss**: per-job `sent_at` column (or notifications table); query `WHERE sent_at IS NULL`; escape the website URL; drop the daily test message; honor `retry_after` on 429 | M | F-008, F-009, F-010, F-011, F-012 | Notifications become exactly-once-ish and recoverable — this is the product's output path |
| 1.5 | **Fix or delete the fake metrics**: either emit real counts from stages (echo `PROCESSED=n` convention parsed by the runner) or delete the counters entirely | S | F-006 | Stops false telemetry (a standing don't-want) |
| 1.6 | Delete dead code and clutter: unused utils, `pipeline_runner.py` or bash runner (keep one), `stage09_resume_matching.py`, `progress.db`, root logs/PNGs/.deb; untrack `gemini_interactions_log.json`; correct `src/README.md` | S | F-007, F-033, F-034 partially | Smaller true surface; docs match reality |
| 1.7 | Remove the write-only `task_backlog.txt` path (fail loudly or block until queued — the GPU queue holds 1024, overflow means something is wrong) | S | F-016 | No silent OCR work loss |
| 1.8 | Add `flock` to run_pipeline.sh (until an orchestrator owns this) | S | F-005 | No overlapping pipeline runs |
| 1.9 | Regression tests for each Phase-1 bug fix (watermark, escaping, exit codes, zero-yield) — they are all cheap unit tests now | M | F-037 partially | The fixed bugs stay fixed |

## Phase 2 — Medium-term: state, data layer, ops UX (1–3 weeks of part-time work)

| # | Action | Effort | Retires | Expected impact |
|---|---|---|---|---|
| 2.1 | **`core/storage.py`** + consolidate the 4 SQLite DBs into one `scanner.db` with `user_version` migrations; typed row accessors; kill positional tuples | M | F-024, F-025, remainder of F-012 | One inspectable DB; schema can evolve |
| 2.2 | **Run-state model**: `runs`/`stage_runs` tables written by the runner; replace `.done` markers; add boundary manifests (count + run_id per data dir) | M | F-030, F-031, F-044 | Empty-vs-failed becomes distinguishable; manual runs stop no-op'ing silently |
| 2.3 | **`scanner` operator CLI**: status / run one stage / retry-failed (ocr, analysis, notifications) / tail | M | F-036 (inspection half), F-014 & Gemini-error re-drive | Operations without SSH archaeology; the three permanent-loss queues get replay |
| 2.4 | **Gemini path cleanup**: fix `genai.configure` race (per-call lock or `google-genai` client instances); single sleep; drive 2 keys concurrently under the existing bucket; stop deleting the validation cache at startup | M | F-021, F-022, part of F-023 | ~2× analysis throughput, correct key accounting |
| 2.5 | **stage07 stops managing Ollama**: systemd unit for `ollama serve`; remove `pkill -f ollama`; stage just checks readiness and fails honestly | S | F-018 | No more killing unrelated processes; service supervision where it belongs |
| 2.6 | **stage09_llm_filter resume by job_id** (query its own SQLite) instead of list index; delete progress.json | S | F-020 | Correct resume after interrupted analysis |
| 2.7 | Logging consolidation: one `core/logging.py`, run-id in every record, per-run directory `logs/<run_id>/`, extend cleanup to failed_batches/raw_responses; cache eviction for `cache_blocks` (age-based) + disk-free preflight | M | F-034, F-035, F-043 | Findable logs, bounded disk |
| 2.8 | `pyproject.toml`, run stages as modules, `core/paths.py` with `SCANNER_HOME` | M | F-032 | Runnable from anywhere; unblocks relocation and testing |
| 2.9 | Downloader hardening pass (not rewrite): merge 3 GDrive impls into 1; per-layer match telemetry; expected-vs-actual source-count alarm via the Phase-1 Telegram hook; GitHub-skills cache TTL | M/L | F-027 partially, F-028, stage09 cache staleness | Degradation becomes visible before it reaches zero |

## Phase 3 — Major structural changes (sequence after Phases 1–2 prove out)

| # | Action | Effort | Retires | Notes |
|---|---|---|---|---|
| 3.1 | **Adopt Prefect** (or Dagster if partitioned assets appeal — see 02-review §8): stages become tasks calling existing `main()`s; schedules replace systemd timers (or keep timers triggering `prefect deployment run`); UI gives run history + retry-from-task | L | F-004/F-005/F-007/F-044 remainders; F-036 (dashboard half) | Only after 1.1 — an orchestrator over lying exit codes is paint over rot. Migration caution: keep the bash runner working until N green Prefect runs; cut over jobs.conf last |
| 3.2 | **Source strategy shift**: Telegram-first ingestion; evaluate official/e-paper/RSS/NCS-portal sources for 3–4 papers; keep the Selenium cascade only as the residual path; Playwright port *only* for whatever scraping remains | L | F-026, most of F-027 | Biggest reliability lever for the product; also shrinks the largest file organically |
| 3.3 | **Package restructure** per 09-review §5 (extraction refactors, no behavior change), with unit tests landing alongside each extraction | L | F-032 remainder, module-side-effect pattern | Do incrementally, one subpackage per sitting |
| 3.4 | **stage04 replacement** per the existing VLM/OCR roadmap doc | L | F-014, F-015, F-017 | Deliberately deferred: don't refactor an executor you plan to delete; port only the resume/progress semantics (fixed per 2.2) into the new engine |

## Sequencing rationale

1. Phase 0 first because nothing else is testable on a dead system.
2. Phase 1 makes failures visible *before* any architectural work — every later migration relies on trusting exit codes and alerts to know the migration itself works.
3. Phase 2 builds the state/data foundation that Phase 3's orchestrator will sit on (and is valuable even if Phase 3 never happens).
4. Phase 3 items are independent of each other and can be reordered by pain: if scraping breaks first, do 3.2 before 3.1.

## Migration cautions

- **Never run old and new schedulers simultaneously** (double-fire). Disable scheduler.sh the same commit that enables timers.
- **DB consolidation (2.1)**: copy tables into `scanner.db` with a one-shot script, keep the old files read-only for two weeks, then archive. Back up before the first write (per standing rules: backups before irreversible DB steps).
- **Exit-code pass (1.1)** will surface failures that were always happening silently — expect the first week to be noisy; that noise is the point. Tune failure-ratio thresholds per stage rather than reverting.
- **Prefect adoption**: pin the version; run `prefect server` as a systemd unit; keep task bodies as thin wrappers so the stages remain runnable standalone (escape hatch if the orchestrator misbehaves).
- Keep all secrets exactly where they are (env/.gitignored files) through every migration; never move keys into orchestrator config.

## Expected impact summary

- Phase 0+1 (≈ one focused week): system runs again, and **cannot fail silently** — the two critical findings (F-001, F-002/F-003) plus the notification-loss bug are gone. ~80% of the audit's risk mass retires here.
- Phase 2 (2–3 weeks part-time): operations stop requiring SSH forensics; state/data layers stop being nine fragmented mechanisms; throughput roughly doubles in the LLM/downloader segments.
- Phase 3 (opportunistic): homemade infra fully replaced by boring tools; source risk diversified; codebase converges to a testable package.
