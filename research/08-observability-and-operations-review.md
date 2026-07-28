# 08 — Observability and Operations Review

Per the mandate, the absence of an internal health interface is evaluated as an operability defect, not a missing convenience.

## 1. Logging quality

**Volume without structure.** 888 files currently in `logs/`. Every stage configures logging independently with its own filename scheme, format, and console policy:

- `logging.basicConfig(filename=…timestamped…)` per run: stages 01, 02, 03, 06, 07, 08 → a new file per stage per run.
- Root-logger surgery (clear handlers, add file+console): stages 04, 05, 09_llm_filter, 09_local_filter — these *replace root handlers*, so when stages share a process (they don't today, but the facade modules `src/ocr/easyocr_engine.py` and `src/llm/ollama_runner.py` invite it) they'd clobber each other.
- RotatingFileHandler used in exactly three places (gemini_client, downloader, stage09_llm_filter); everything else grows unbounded until stage11's 7-day `*.log` purge — which does **not** cover `logs/failed_batches/*.json`, `logs/raw_responses/*.txt`, or `logs/tuning*/` (all present and accumulating since 2025).
- No run ID appears in any log line. Correlating "last night's stage04 failure" across `easyocr_gpu_dominant_<ts>.log`, the runner's `*.combined.log`, and `pipeline_YYYYMMDD.jsonl` is manual timestamp archaeology.
- The one structured log (`logs/pipeline_YYYYMMDD.jsonl`, written by the runner) records only stage-level start/success/failure/skipped with `processed_items` **always 0** (F-006) — the single machine-readable signal carries no throughput information.

**Positive:** log content quality inside stages is generally good — stage04 and the downloader log decisions, not just errors; per-stage `.out/.err/.combined` capture in the runner is genuinely useful.

## 2. Health visibility

- `scripts/health_check.py` is a well-built **preflight** check (python version, CUDA, ollama binary, env vars, config keys, dir permissions) — but it is (a) manual-only (`make health`; nothing schedules it), (b) about *environment*, not *pipeline state*: it cannot answer "did last night's run work", "how many jobs were sent this week", or "is the scheduler alive".
- **Nothing monitors the scheduler.** Its liveness is only observable via `pgrep`. Current reality proves the gap: the scheduler is not running, the venv it depends on is deleted, and nothing flagged either — the system has been silently down since ~June 13 (F-001/F-036, confirmed by log mtimes and missing `data/`).
- `data/pipeline_metrics.json` is the only artifact that *looks* like a health surface, and its four business counters are provably frozen at zero (F-006). This is worse than no metrics: it invites false confidence.
- No disk-space checks anywhere despite image-heavy intermediates and a 629 MB never-evicted cache (F-035, F-043).

## 3. Ability to inspect failures

Walking the realistic questions an operator asks:

| Question | Today's answer |
|---|---|
| Did last night's pipeline run? | SSH; `ls -lt run_state/*.done`; cross-check log mtimes; remember that `.done` exists even for no-op runs (F-002) |
| Why did stage X fail? | Find the right 2–4 files among 888 in `logs/`; for stage04 also open `data/progress/<paper>_progress.json`; for stage07 query `processing_state.db` by hand |
| How many jobs were found/sent? | `sqlite3 data/shortlist_history.db 'select count(*)…'` + grep stage10 log — the metrics file lies (F-006) |
| Which papers failed to download? | grep `newspaper_downloader.log` for "Failed preprocessing"; debug HTML dumps get deleted nightly by stage11 (F-013 interaction) |
| Which sends failed? | grep the telegram log — and know that they will never be retried (F-008) |

Every row requires SSH + tribal knowledge. There is no status command, no dashboard, no API.

## 4. Admin / operator experience

- **Manual retry capability is nearly absent.** `run_pipeline.sh --force` re-runs *everything* (hours of GPU time) — there is no "re-run stage07 only" or "re-run paper X" through the orchestration layer. Operators must invoke stage scripts directly with correct CWD, and stage-internal flags (`--fresh`, `--force`, `--newspaper`, `--force-reset`, `--no-cache`) differ per stage with overlapping-but-different semantics (stage07's `--force` also *disables DB writes*, `stage07:662-671` — a re-run flag that silently changes persistence behavior).
- **Failed-item re-drive doesn't exist:** OCR blocks marked failed are never retried (F-014); Gemini error rows are terminal (04-review §4); failed Telegram sends are skipped (F-008). Three permanent-loss queues with no replay tooling.
- **Destructive ops are casual:** stage11's confirmation is auto-piped; `pkill -f ollama` can kill unrelated processes (F-018); no backup exists for any SQLite DB before nightly writes.

## 5. Manual retry capability — what "good" looks like here

Minimum viable operator toolkit (a single `scanner` CLI, argparse, ~a day of work):

- `scanner status` — last run per stage (from a run-state table), counts, scheduler liveness, disk free, DB row counts.
- `scanner run <stage> [--paper X] [--date D]` — one stage, correct env/CWD handled for you.
- `scanner retry-failed [ocr|analysis|notifications]` — re-drives the three permanent-loss queues.
- `scanner tail <stage>` — opens the latest relevant log.

## 6. Internal API / dashboard opportunities

Given one operator and one machine, ranked by value-per-effort:

1. **Telegram as the ops channel (near-zero effort, highest value).** The system already ships a bot. A failure hook in the runner (`on non-zero stage exit or zero-yield: sendMessage to admin chat`) plus a nightly one-line digest ("14 papers, 212 pages, 38 jobs, 5 sent, 0 failures") converts every silent failure found in this audit into a push notification. This should precede any dashboard.
2. **`scanner status` CLI** as in §5, backed by a consolidated run-state SQLite table (03-review §8.2).
3. **Prefect/Dagster UI** (02-review §8): adopting an orchestrator brings run history, per-task logs, retry-from-task, and schedule visibility as built-ins — this is the realistic "dashboard", not a custom web app.
4. **A custom FastAPI dashboard is not recommended** at this scale: it would be one more homemade system to operate, which is the pattern this audit repeatedly flags. Only revisit if multiple users need read access.

## 7. Findings referenced

F-001, F-002, F-003, F-006, F-008, F-013, F-014, F-018, F-028, F-034 (log sprawl/hygiene), F-035, F-036 (no health interface — high severity per mandate), F-043 (no disk checks). Positives: health_check.py quality, per-stage log capture, actionable error hints in stage10. Details in `findings.json`.
