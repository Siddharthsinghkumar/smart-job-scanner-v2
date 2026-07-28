# 04 — Data and SQLite Audit

## 1. SQLite usage review

Five databases; four live, one orphaned. All accessed with raw stdlib `sqlite3`, connection-per-operation in most sites, no shared data-access module.

| DB | Writers | Readers | Journal mode | Connection pattern |
|---|---|---|---|---|
| `data/processing_state.db` | stage07 (`mark_file_processed`, `stage07:657-683`) | stage07 (`get_processed_files`, `stage07:685-696`) | WAL + `synchronous=NORMAL`, timeout=30 (`stage07:637-644`) | open/commit/close **per file processed**, serialized by a `threading.Lock` |
| `data/llm_results/llm_job_analysis.db` | stage09_llm_filter (`save_to_db`, `stage09_llm_filter:262-300`) | ad-hoc (research scripts) | default (rollback journal) | one connection per run — the best-behaved DB here |
| `data/shortlist_history.db` | stage09_shortlist (`log_job`, `stage09_shortlist:57-75`) | stage09_shortlist + stage10 (`stage10:243-311`) | default | open per run (09) / per query-function (10) |
| `data/gemini_interactions.db` | gemini_client `_log_to_sqlite` (`gemini_client:588-623`) | nobody in production | default | **open + CREATE TABLE + insert + close per request** |
| `progress.db` (repo root) | nobody | nobody | — | orphaned artifact of the pre-refactor OCR script (schema: pages/backlog/worker_stats; last write Nov 2025) |

## 2. Schema and migration concerns

**Severity: medium-high. Confidence: high. Classification: confirmed issue (F-024).**

- **DDL lives inline at call sites** and runs on every start (`CREATE TABLE IF NOT EXISTS` in `stage07:645-654`, `stage09_llm_filter:225-256`, `stage09_shortlist:38-48`, `gemini_client:593-605` — the last one on *every logged request*).
- **No migration story at all.** `CREATE TABLE IF NOT EXISTS` never alters an existing table: adding a column to `sent_jobs` requires manual `ALTER TABLE` on the production file or the code and DB silently diverge (INSERTs would start failing — or worse, positional readers like stage10's `job[5]` would misread).
- **No schema versioning, no PRAGMA user_version**, no startup check that the on-disk schema matches expectations. stage10 does check that `sent_jobs` exists (`stage10:253-258`) — the only defensive schema code in the repo.
- **Type discipline:** timestamps stored as ISO-8601 TEXT (fine for SQLite, comparisons work lexically) but three different formats coexist: `datetime.now().isoformat()` (naive local), `datetime.utcnow().isoformat()+"Z"` (gemini_client:612), and `datetime.now()` passed directly as a parameter (stage07:681 — sqlite3 default adapter, deprecated in Python 3.12). Cross-DB time comparison is therefore unreliable, and the 3.12 adapter deprecation will start emitting warnings on the current system interpreter.

## 3. Raw SQL sprawl and query duplication (F-025, medium, confirmed)

- `init_db()` exists three times (stage07 `init_database`, stage09_llm_filter `init_db`, stage09_shortlist `init_db`) with the same open/DDL/commit boilerplate.
- `sent_jobs` is accessed by two stages with hand-written SQL and **positional tuple unpacking** (`job_id, job_text, similarity, score, processed_at, shortlisted_at = job_data`, `stage10:391`; `job[5]` at `stage10:506`). Any column reorder is a silent data corruption for the notifier.
- The two stage10 query functions `get_new_jobs`/`get_all_jobs` (`stage10:243-311`) are copy-paste duplicates differing by one WHERE clause.
- Full inventory in `sqlite-query-surface.csv` (23 query sites).

## 4. Integrity and consistency risks

- **`INSERT OR REPLACE` used as upsert everywhere** (`stage07:677-681`, `stage09_llm_filter:266-273`, `stage09_shortlist:60-64`). REPLACE deletes-then-inserts: it resets `processed_at` defaults, would fire ON DELETE triggers if any existed, and in `sent_jobs` can rewrite `shortlisted_at` for an existing job — which then re-triggers stage10's `shortlisted_at > watermark` query, i.e., **a re-shortlisted duplicate can be re-notified** despite the dedup design intent. (Currently mitigated only because `is_new_job` filters before `log_job`, `stage09_shortlist:180-186` — the REPLACE branch is dead in the happy path but live if that guard ever changes.)
- **Semantic lie in `sent_jobs`:** rows are inserted at *shortlist* time, before any Telegram send (`stage09_shortlist:183` "Log immediately to prevent future duplicates"). If stage10 never runs (config error → its module-level `SystemExit`, `stage10:38-41`), the jobs are permanently recorded as shortlisted and will never be notified. Dedup-by-history and delivery-tracking are conflated in one table (F-012, medium, confirmed).
- **Error rows are terminal in `job_analysis`:** a job analyzed during a Gemini outage is stored with `status='error'` and, because shortlisting dedup keys on job_id upstream, is never re-analyzed. There is no re-drive query or tool.
- **No foreign keys** (each DB is a single table, acceptable) and **no NOT NULL constraints** on fields code assumes present (e.g., `shortlisted_at` — `stage10:403` does `datetime.fromisoformat(shortlisted_at)` and would crash the whole notification run on one NULL row).
- **`data/pipeline_metrics.json`** is a read-modify-write JSON with no lock from two implementations (bash heredoc + pipeline_runner.py) — concurrent runs corrupt or clobber it (contributes to F-005).

## 5. Concurrency / write-lock risks

- stage07 is the only writer that enables WAL and takes an app-level lock; its DB is safe in-process. However `get_processed_files` takes the same `db_write_lock` (`stage07:685-688`) so "read lock to avoid racing writes" serializes reads too — harmless at this scale, just unnecessary.
- gemini_client's per-request connect/CREATE/insert/close under default journal mode is the likeliest "database is locked" source if `run_batch` ever runs threaded (workers default up to 16, `gemini_client:182`): concurrent short-lived writers on a rollback-journal DB with default 5s busy timeout. **Hypothesis (medium confidence):** would surface as `sqlite3.OperationalError: database is locked` under `generate_batch`; confirm with a 16-thread batch against a throwaway DB. Production path today is sequential, so latent.
- Cross-process risk: an overlapping manual + scheduled pipeline (no lock, F-005) can have two stage09_shortlist processes writing `sent_jobs` simultaneously; default journal mode serializes but the second writer's `is_new_job` check may have read pre-insert state → duplicate notifications. Hypothesis; requires overlap to trigger.

## 6. The good parts (positive findings)

- WAL + busy-timeout + app-level lock in stage07 shows awareness of exactly the right issues (F-048).
- `job_analysis` has sensible indexes (status, similarity, resume, api_key_label, tokens — `stage09_llm_filter:252-256`) and a UNIQUE(job_id) constraint.
- Parameterized queries everywhere; no string-interpolated SQL was found anywhere in the production code (no injection surface).
- History-DB dedup (`sent_jobs`) is the right idea for a daily re-scanning pipeline — it just needs the sent/shortlisted split.

## 7. ORM / data-layer opportunities

A full ORM is oversized for four single-table databases. Recommended shape, in order of value:

1. **One `src/storage.py` module** owning: connection factory (WAL, busy_timeout, detect_types), all DDL with a `user_version`-based migration ladder, and typed accessor functions (`record_processed_file`, `record_analysis`, `record_shortlisted`, `mark_sent`, `unsent_jobs_since`). Every current call site becomes a one-line call. This removes the sprawl without adding a dependency. Use `dataclasses` or `NamedTuple` rows to kill positional unpacking.
2. **Consolidate databases.** There is no reason for four files: one `data/scanner.db` with tables `processed_files`, `job_analysis`, `shortlist` (+ a new `notifications` table with `sent_at`, `attempts`, `last_error`) simplifies backup, inspection, and cross-table queries ("show me jobs analyzed but never sent" becomes one JOIN — today it's impossible without joining across files by hand).
3. **If/when an ORM is wanted:** SQLAlchemy Core (not ORM) + Alembic is the natural step, and both Prefect and Dagster metadata layers coexist with it fine. SQLModel is an option if pydantic models get introduced for the job records. Not a near-term need.
4. **Delete `progress.db`** (or archive it out of the repo root) and remove `gemini_interactions.db`-vs-JSONL double logging — pick the DB, drop `gemini_interactions_log.json` (69 KB of it is already committed to git at the repo root, which also should not be tracked).

## 8. Findings referenced

F-005, F-012, F-024, F-025, plus positives F-048. Query-site inventory: `sqlite-query-surface.csv`.
