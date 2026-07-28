# 00 — Executive Summary

**System:** Smart Job Scanner v2 — a nightly OCR+LLM ETL pipeline that scrapes newspaper PDFs, extracts job ads, matches them against resumes, and sends Telegram alerts.
**Audit:** static code + configuration + on-disk state inspection, 2026-07-07/08. 53 findings (3 critical, 14 high, 22 medium, 8 low, 6 informational/positive; 5 explicit hypotheses). Full registry: `findings.json`; prioritized plan: `10-improvement-roadmap.md`.

## Overall health assessment

**The system is currently down, and its design guarantees it goes down silently.** The production virtualenv (`4_env/`) no longer exists, `data/` is absent, the scheduler isn't running, and the last activity is ~2026-06-13 — a multi-week outage that nothing detected (F-001). That outage is not bad luck; it is the architecture's signature failure mode reproduced at full scale: bash scripts hand-implement a scheduler, orchestrator, retry manager, and state coordinator, while the stages they supervise exit 0 on almost every real failure (F-002) — so retries never fire, success markers certify empty runs, the metrics file's business counters are hardwired to zero (F-006), and the nightly cleanup then deletes the inputs and the forensic evidence (F-013).

Beneath the orchestration problem, the code itself is uneven rather than bad: there are genuinely well-handled hard problems (WAL SQLite in stage07, atomic writes, PDF validation, multi-key rotation — F-045…F-048) sitting inside ~12,600 lines where the four biggest files (downloader 2,821; OCR 1,589; Ollama extraction 1,423; Gemini client 1,220) each embed homemade infrastructure that established tools provide.

## Top strengths

1. **Secrets discipline in git is correct** — only example configs tracked; verified against `git ls-files` (F-045).
2. **Crash-consistent writes where they matter most** — tempfile+`os.replace` for OCR outputs and PDF downloads (F-046).
3. **Validation scaffolding exists** — dry-run mode, JSON health preflight, CI, wiring tests, per-stage log capture (F-047).
4. **Pockets of correct hard-problem engineering** — WAL+lock SQLite, adaptive token windows, key rotation with cooldowns, PDF header/EOF validation, dedup history DB (F-048).
5. **The refactor direction was right** — clean `stageNN_*` module layout, shims for compatibility, configs/env separation.

## Top risks

1. **F-001 (critical):** venv missing → nothing can run; the outage is live and undetected.
2. **F-002/F-003 (critical):** stages lie about success → silent no-op nights are structurally possible end to end (empty download → green pipeline → cleanup).
3. **F-008 (high):** the Telegram watermark advances past failed sends — the product's output messages get permanently lost (compounded by an unescaped URL field, F-009, that *causes* such failures).
4. **F-013 (high):** cleanup deletes raw PDFs and debug artifacts even after failed runs — unrecoverable data loss.
5. **F-026/F-027 (high):** 12 of 12 web sources depend on one unofficial aggregator through a 2,821-line heuristic cascade shipped with debug settings (`workers=1`, `headless:false`).
6. **F-024/F-030 (high):** four SQLite DBs plus nine fragmented state mechanisms, no migrations, no run identity.

## Top urgent actions (Roadmap Phase 0–1, ≈ one focused week)

1. Rebuild the venv; replace scheduler.sh with two systemd timers (`Persistent=true`) with an `OnFailure` Telegram alert (0.1–0.2).
2. **Exit-code truthfulness pass** across all stages + zero-yield assertions (1.1) — the single highest-leverage change in this audit.
3. **Telegram failure/digest hook** in the runner — the system already owns a bot; use it to report on itself (1.2).
4. Gate stage11 deletion on verified yield; kill the piped-`y` confirmation (1.3).
5. Fix notification delivery: per-job `sent_at`, escape the URL, drop the daily test message, honor `retry_after` (1.4).
6. Delete the fake metrics counters or make them real (1.5); add `flock` to the runner (1.8).

## Biggest reliability concerns

The failure-signal chain is broken at every link: stages swallow errors → runner marks success → scheduler ignores exit codes anyway → no alerting channel exists → cleanup destroys the evidence. Secondary reliability debt: OCR blocks that fail once are never retried and queue overflow is written to a file nothing reads (F-014/F-016); Gemini error rows are terminal; failed Telegram sends are unrecoverable (F-008). Every loss queue is permanent and none has replay tooling.

## Biggest maintainability concerns

Three parallel orchestrator implementations (one already behaviorally drifted, F-007); a utility layer with zero importers next to stages that each hand-roll logging/config/state (F-033); module-level side effects (network auth, OCR warm-up, `SystemExit`) that make stages untestable without the placeholder-env fixture the tests had to grow (F-032); duplicated 70-line OOM blocks inside an ~800-line homemade GPU/CPU executor (F-017); and a test suite that validates wiring but none of the logic where this audit found its bugs (F-037). Documentation (`src/README.md`) describes the pre-refactor system as production.

## Highest-value modernization opportunities

1. **systemd timers now, Prefect after Phase 1** — replaces the homemade scheduler/orchestrator/state layer with run history, per-task retries, and a real UI; Dagster is the alternative if per-newspaper/date partitioned assets appeal (02-review §8). Adopting either *before* fixing exit codes would just paint failures green in a nicer dashboard.
2. **One `scanner.db` + `core/storage.py`** with versioned migrations — collapses four DBs, three DDL copies, and positional-tuple coupling (04-review §7).
3. **Telegram-first ingestion + official/RSS/portal sources** for even a subset of papers — shrinks the single-aggregator risk and most of the scraping cascade (06-review §6); Playwright only for the residue.
4. **Operator CLI (`scanner status / run / retry-failed`)** — converts SSH-and-grep operations into commands, and gives the three permanent-loss queues a replay path (08-review §5).
5. **Targeted concurrency fixes, no async rewrite** — restore downloader workers, single Gemini sleep with 2-key concurrency, batched Argos calls (05-review §7).

**Bottom line:** the refactor bought a clean module layout, but the operational core — scheduling, state, failure signaling — is still homemade and currently proven non-viable by its own silent outage. Phase 0+1 of the roadmap (≈ a week) retires roughly 80% of the risk mass; the orchestrator and source-diversification work can then proceed on a system that tells the truth about itself.
