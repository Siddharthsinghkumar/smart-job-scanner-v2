# 02 — Orchestration and Topology Review

This review applies the orchestration mandate: bash acting as scheduler/orchestrator/retry-manager/state-coordinator is treated as an owned liability, not a neutral choice.

## 1. What operational responsibilities the bash layer is carrying

`scheduler.sh` + `run_pipeline.sh` together implement, by hand:

| Responsibility | Where | Normally provided by |
|---|---|---|
| Cron scheduling (daily at HH:MM) | `scheduler.sh:43-57,165-227` | cron / systemd timers / orchestrator scheduler |
| Anacron catch-up ("machine was off at trigger time") | `scheduler.sh:107-159` | anacron / orchestrator backfill |
| Job launching + log capture | `scheduler.sh:60-93,201-225` | supervisor / orchestrator task runner |
| DAG execution (linear, 14 nodes) | `run_pipeline.sh:58-73,339-439` | orchestrator DAG engine |
| Completion state (`.done` markers) | `run_pipeline.sh:341-356,413` | orchestrator run/task state DB |
| Retry with exponential backoff | `run_pipeline.sh:53-56,358-437` | orchestrator retry policy |
| Metrics + structured logging | `run_pipeline.sh:167-282` (Python heredocs) | orchestrator UI/metadata |
| Interactive-prompt auto-confirmation for a destructive stage | `run_pipeline.sh:373-381` (`printf 'y\n' \|`) | shouldn't exist at all |

That is a full workflow-orchestrator feature set re-implemented in ~670 lines of bash, minus the parts that make orchestrators safe: locking, run identity, exit-status truthfulness, visibility, and alerting.

## 2. Scheduler review (`scripts/scheduler.sh`)

**Severity: high. Confidence: high. Classification: confirmed issue (design), with several probable-issue mechanics.**

- **State inference from log filenames.** "Did the downloader run today?" is answered by `find "$LOG_DIR" -name 'auto_download_pdfs_*.log' -newermt ...` (`scheduler.sh:119-135`). Touching, rotating, or deleting log files changes scheduling behavior. A run that started and crashed 1 second in still counts as "ran".
- **Exit codes swallowed.** Jobs run as `bash -lc "$cmd" >> "$logfile" 2>&1 || echo "Command exited non-zero" >> "$logfile"` in a backgrounded subshell (`scheduler.sh:68-73,210-215`). Failure is a text line in a file nobody reads. The scheduler never re-runs, never alerts, never records failure state.
- **No locking / no single-instance guard.** Two scheduler instances (e.g., started manually after a reboot while an old one survives) double-fire every job. Nothing prevents the 23:30 pipeline from overlapping a still-running manual pipeline either (see §5).
- **No supervision of the scheduler itself.** There is no systemd unit, no `nohup` wrapper, nothing in the repo that starts it. Verified: no scheduler process is currently running, and the pipeline has been dormant since ~2026-06-13. A scheduler whose liveness is invisible is the root operability gap here.
- **Fragile bash mechanics** (probable issue, medium confidence they bite in practice):
  - `set -eo pipefail` with `((i++))` in the config loop (`scheduler.sh:29-40`) returns exit 1 on the first iteration; the script only survives because the `|| true` after `done` disables `-e` for the whole compound. One refactor away from a boot crash.
  - `compute_next "$idx"` at boot (`scheduler.sh:54-57`) is *not* `|| true`-guarded: a malformed time in `jobs.conf` kills the daemon at startup under `set -e`.
  - Time math relies on `date -d "today 11:50"` / `+1 day` string round-trips (`scheduler.sh:76-92,218-224`); DST-less India makes this mostly safe, but suspend/resume during the 1–3600 s sleeps shifts firing, and a job scheduled during a suspend window fires immediately on resume (all instances at once).
  - Startup catch-up hardcodes `11:50`/`23:30` (`scheduler.sh:111-113`), duplicating `jobs.conf`. Changing a job time in one place and not the other silently diverges.

## 3. Pipeline orchestration review (`scripts/run_pipeline.sh`)

**The core defect is not in the runner — it's that the runner's contract (exit code = truth) is violated by nearly every stage.**

- **Retry/backoff design is reasonable on paper**: 10s initial, ×2 factor, 300s cap, MAX_RETRIES=5 default, stage-scoped logs, JSONL events (`run_pipeline.sh:53-56,358-437`). Positive finding.
- **But stages exit 0 on failure** (F-002, critical, confirmed):
  - stage01: no PDFs found → print + `return` → exit 0 (`stage01:227-229`); per-page render failures only counted (`stage01:172-174`).
  - stage02: failed pages counted and printed, exit 0 always (`stage02:53-55,99`).
  - stage04: OCR block failures recorded as `status: failed` in progress JSON, exit 0 (`stage04:369-371`).
  - stage07: **Ollama never becomes ready → error logged → falls into `finally`, exits 0, `.done` gets written** (`stage07:1365-1376`). A full extraction no-op is recorded as success.
  - stage09_llm_filter: missing shortlist file, zero working API keys, exhausted quota — all `return` from `main()` → exit 0 (`stage09_llm_filter:731-733,797-799,873-878`).
  - The result: the runner's retry loop and `MAX_RETRIES` almost never engage on real failures; `.done` markers certify runs that produced nothing; and the pipeline happily proceeds to stage11, which deletes the inputs (see 03-audit F-013).
- **Metrics are fake** (F-006, high, confirmed): every call site passes `processed_items="0"` (`run_pipeline.sh:348-349,414-415,419-420`), so `pages_processed`, `jobs_detected`, `jobs_filtered`, `jobs_sent_to_telegram` in `data/pipeline_metrics.json` can never move. The counters look like telemetry and measure nothing except `stages_completed` and accumulated wall time.
- **`MAX_RETRIES=0` documented as "unlimited"** (`run_pipeline.sh:332`) is an intentional infinite loop; combined with a stage that fails fast, backoff caps at 300s and the pipeline spins forever with no alert.
- **`rc=$?` after process substitutions** (`run_pipeline.sh:373-407`): the tee writers are asynchronous, so a stage's final log lines can land after the SUCCESS/FAILURE line, and on very fast failures may be lost. Hypothesis — confirm by forcing a stage to exit non-zero instantly and diffing `.err.log` vs `.combined.log` ordering.
- **No lock file.** A manual `./scripts/run_pipeline.sh` overlapping the 23:30 scheduled run races `.done` markers, `pipeline_metrics.json` (read-modify-write, no lock — last writer wins), the stage04 progress JSONs, and the SQLite DBs (F-005, high, confirmed absence).
- **Hard dependency on missing venv**: `die "virtualenv python not found at: $VENV_PYTHON"` (`run_pipeline.sh:319-321`) is currently the *first* failure every invocation on this machine hits (F-001, critical, confirmed). Note the dry-run path partially masks this: `validate_configs` falls back to system python (`run_pipeline.sh:41-44`), so `--dry-run` can pass on a machine where the real run cannot start.

## 4. Duplicate orchestrators (drift already happened)

**Severity: medium. Confidence: high. Classification: confirmed issue.**

Three implementations of "run the stages":
1. `scripts/run_pipeline.sh` (production, per `jobs.conf`).
2. `src/pipeline/pipeline_runner.py` — parallel Python port. Already drifted: retries have **no sleep at all** (`pipeline_runner.py:198-221`; `--max-retries 0` = tight infinite loop), metrics keys differ (`pipeline_runner.py:89-99` lacks `pipeline_runtime_seconds` and the `jobs_detected` stage07 hook present in bash at `run_pipeline.sh:273-274`), and there is no per-stage log capture.
3. `src/pipeline/stage09_resume_matching.py` — a composite runner for the stage-09 quartet that nothing invokes (`pipeline_config.json` lists the four sub-stages individually).

Every future behavior change must be made 2–3 times or the paths diverge further. Recommendation: delete two of them (keep exactly one entrypoint) regardless of any orchestrator decision.

## 5. DAG / stage dependency mapping

The DAG is a pure chain with one fan-in quirk:

```
downloader (11:50) ──(filesystem: data/raw_pdfs)──► stage01 ► stage02 ► stage03 ► stage04 ► stage05 ► stage06 ► stage07 ► stage08
                                                                                                       stage09_dynamic_resumes ┐
                                                                                                (independent of 01-08, needs   ├► stage09_local_filter ► stage09_llm_filter ► stage09_shortlist ► stage10 ► stage11
                                                                                                 resumes/ PDFs + GitHub)      ┘
```

Observations:
- The downloader→stage01 edge is **temporal only** (11:50 vs 23:30). Nothing verifies PDFs arrived; stage01 treats an empty `raw_pdfs/` as success (F-003). A failed morning download produces a silent no-op night.
- `stage09_dynamic_resumes` has no dependency on stages 01–08 and could run any time (or be cached weekly); today it costs a GitHub API scrape per night in the critical path.
- No stage declares inputs/outputs; the DAG exists only as an ordered bash array. There is no way to run "stage04 for newspaper X only" through the orchestration layer — operators must invoke stage scripts manually with correct CWD.

## 6. Restart behavior

- **Machine reboot:** scheduler does not auto-start (no unit file) → nothing runs until a human starts it (confirmed current state). When started, the catch-up block fires missed jobs immediately — good idea, but based on log-mtime inference and hardcoded times.
- **Pipeline crash mid-run:** `.done` markers make completed stages skippable on manual re-run without `--force` — this is the one resume path that mostly works. But markers carry no run/date identity (`run_pipeline.sh:82-85,341`): after last night's successful run, today's manual `run_pipeline.sh` (no `--force`) silently skips all 14 stages and reports success. Changing a stage's CLI args changes the marker name and orphans the old marker (F-044).
- **Stage crash mid-work:** stage-internal resume quality varies wildly — stage04 wipes everything unless the previous run was perfect (03-audit F-014); stage07 resumes well via SQLite; stage09_llm_filter resumes by list index into a regenerated file (03-audit F-020).

## 7. Observability of pipeline state

To answer "what is the pipeline doing right now / what happened last night", an operator must: SSH in; `pgrep` for the scheduler; list `run_state/*.done` mtimes; tail 3–5 of the 888 files in `logs/`; know that `data/pipeline_metrics.json` counters are fake; and query 4 SQLite DBs by hand. There is no run ID connecting any of these. Full treatment in `08-observability-and-operations-review.md`.

## 8. Replacement options

The workload is: 2 scheduled jobs/day, a 14-node linear DAG, one machine, GPU-bound stages, file artifacts. Evaluation against that reality:

| Option | Fit | Notes |
|---|---|---|
| **Prefect 3 (recommended)** | Best fit | Stages become `@task`-decorated functions calling existing `main()`s; retries/backoff per task; state stored in SQLite; local UI dashboard (`prefect server start`) gives run history, per-task logs, manual retry-from-failed-task, and schedule management — directly replacing scheduler.sh, run_pipeline.sh, .done markers, and the fake metrics file. Runs fine on one machine, no infra beyond `pip install prefect`. Failure hooks → Telegram notification closes the silent-failure gap. |
| **Dagster** | Good, heavier | Asset-based model fits the file-artifact pipeline conceptually well (each `data/` directory becomes an asset with materialization metadata — solves "empty vs failed" elegantly). More concepts to learn; daemon + web server always-on. Choose it if the plan is to grow into data-quality checks and backfills per newspaper/date partition (partitioned assets by publication date are a natural fit). |
| **Airflow** | Poor fit | Scheduler+webserver+metadata DB overhead, worker model, and DAG-file ergonomics are all oversized for one machine and one user. Not recommended. |
| **systemd timers + a thinner runner (minimum viable fix)** | Acceptable floor | Replace scheduler.sh with two `.timer`+`.service` units (`Persistent=true` gives anacron semantics for free, journald gives logs, `OnFailure=` gives an alert hook, systemd gives single-instance locking). Keep a *fixed* run_pipeline.sh. This removes the worst scheduler risks without adopting an orchestrator, but does nothing for DAG visibility, per-stage retry UX, or run history. |

**Recommendation:** systemd timers immediately (near-zero effort, kills the daemon-liveness and catch-up problems), Prefect as the medium-term orchestrator. Do not adopt any of them before fixing stage exit codes — an orchestrator scheduling stages that lie about success just paints the same failure green in a nicer UI.

## 9. Findings referenced in this section

F-001 (venv missing, critical), F-002 (stages exit 0 on failure, critical), F-003 (silent end-to-end no-op, critical), F-004 (homemade scheduler, high), F-005 (no locking, high), F-006 (fake metrics, high), F-007 (duplicate orchestrators, medium), F-038 (scheduler bash fragility, medium), F-039 (MAX_RETRIES=0 / no-sleep retry drift, low), F-044 (.done naming/scoping, medium). Full details in `findings.json`.
