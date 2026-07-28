# 03 — Pipeline Stage and State Management Audit

## 1. Stage boundary analysis

Every stage boundary is an implicit filesystem contract: "stage N writes files matching pattern P into directory D; stage N+1 globs D." No manifests, no counts, no schemas, no run identity crosses any boundary.

| Boundary | Contract | Fragility |
|---|---|---|
| downloader → 01 | `data/raw_pdfs/*.pdf` | Temporal only (11:50 vs 23:30); empty dir = "success" |
| 01 → 02 | `data/pdf2img/<pdf-stem>/<stem>_pN.png` | Page numbering parsed back out of filenames with `_p(\d+)` regexes in stages 02/04/06 (`stage04:296-299,1317-1322`, `stage06:28-35`) |
| 02 → 03 | `data/job_blocks_smart/<folder>/*_blockN.png` + `debug_pN.png` used as the skip marker (`stage02:42-47`) | Deleting debug images (they look disposable) forces reprocessing |
| 03 → 04 | `data/job_blocks_refined/<folder>/*.png` | Refiner drops blocks silently by design (tiny/graphic/short-text filters) — no record of what was discarded except sampled debug images at rate **0.0** (`stage03:17`), i.e., none |
| 04 → 05 | `data/page_texts/<paper>/<paper>_pN_text.txt` | Empty page = file absent; stage05 can't distinguish "no text on page" from "OCR died" |
| 05 → 06 → 07 | `data/all_eng_text/`, `data/batch_inputs/<prefix>/*_pN_batch_M.txt` | Filename-embedded metadata again (`extract_prefix` splits on `"_p"` — a newspaper name containing `_p` breaks page attribution, `stage06:33-35`) |
| 07 → 08 | `data/Jobs_found_final/<paper>/*_jobs.txt` (lines starting `"- "`) | Job structure is a prompt-format convention; parser is `line.strip().startswith("- ")` (`stage07:1296-1301`) |
| 08 → 09 | `*_jobs_all.txt` numbered lines | |
| 09-chain | `shortlisted_jobs.json` → `llm_job_analysis_*.json` → `sent_jobs` table | `stage09_shortlist` picks "the latest JSON by mtime" (`stage09_shortlist:271-275`) — an old export lying around gets re-analyzed |
| 10 → user | Telegram | watermark file (see 07-review) |

**Finding F-031 (high, confirmed): downstream stages cannot distinguish "upstream produced nothing" from "upstream failed".** Combined with exit-0-on-failure (F-002) this yields the system's worst failure mode: a broken morning download or a dead Ollama produces a complete, green, `.done`-marked pipeline night that sends nothing and then deletes the evidence (F-013).

## 2. File-based IPC review

- **Naming as schema.** Page numbers, newspaper identity, batch indexes, and block indexes all live in filenames and are re-parsed with regexes in at least four stages. Newspaper names with spaces are handled inconsistently (`stage06:35` replaces spaces; stage04 folder names retain them — see actual dirs like `run_state/detections/TH- Delhi 07-04_p10.png.json`).
- **Good practice present:** stage04 and the downloader write via `tempfile` + `os.replace` (atomic on same filesystem) — `stage04:64-76`, `downloader:62-84,203-218`. Positive finding F-046.
- **Bad practice adjacent:** stage05/06/07/08/09 write outputs with plain `open(...,"w")` — a crash mid-write leaves truncated files that downstream will happily consume (e.g., a truncated `shortlisted_jobs.json` makes stage09_llm_filter exit "cleanly" on JSONDecodeError → exit 0, `stage09_llm_filter:743-745`).

## 3. `.done` marker review (F-044, medium, confirmed)

- Marker name = sanitized command string incl. arguments (`run_pipeline.sh:82-85`): `src_pipeline_stage07_llm_extraction.py_--no-hybrid.done`. Removing `--no-hybrid` creates a *new* marker and re-runs the stage while the stale marker lingers forever.
- Markers carry no date/run ID. Content is just a timestamp line. After any successful run, a marker-respecting invocation is a full 14-stage silent no-op; the system only works because the nightly job always passes `--force`. That means the markers' only real function is intra-day crash resume — while imposing a permanent trap for manual operation.
- Python runner uses a slightly different sanitizer (`pipeline_runner.py:69-71` vs `run_pipeline.sh:82-85` sed) — same input, believed-same output for current commands, but two implementations of a state-key function is asking for a mismatch.

**Alternative:** run-scoped state (`run_state/<run_id>/stageNN.json` with status, counts, duration, input/output manifests) — or let an orchestrator own task state entirely and delete the marker system.

## 4. Partial failure behavior, stage by stage

| Stage | On partial failure | On total failure | Retry-able? |
|---|---|---|---|
| downloader | per-source errors collected, summary printed; exit 1 only if *zero* files downloaded (`downloader:2819-2821`) | exit 1 (scheduler ignores it) | re-run skips validated per-day files — good |
| 01 | failed pages counted, logged, lost | "no PDFs" → exit 0 | re-run reprocesses moved PDFs? No — PDFs are *moved* to `processed_pdfs/` even if pages failed (`stage01:196-205`), so failed pages are **unrecoverable** without manual move-back |
| 02 | failed pages logged; debug-marker skip means retry only reprocesses pages without markers | exit 0 | partially |
| 03 | exceptions become `skipped_error` counters (`stage03:244-251`) | exit 0 | no skip-marker → re-runs everything |
| 04 | failed blocks marked `failed` and **never retried** (`stage04:839-842`); any missing output at start → full wipe & redo for that paper (`stage04:788-825`) | exit 0 | all-or-nothing (F-014, medium) |
| 05 | per-file try/except, failure logged & skipped | exit 0 (no packages found → return) | full re-run, overwrites |
| 07 | per-file: empty response → not marked → retried next run (good); errors appended to list that only affects the printed summary | model never ready → exit 0 (F-002) | via SQLite, good |
| 09_llm | per-job error rows saved with status "error" — **never re-attempted** (job_id is UNIQUE, INSERT OR REPLACE only fires if the job reappears in a new shortlist) | key exhaustion → break + exit 0 | index-resume unreliable (F-020) |
| 10 | failed send logged; watermark advances past it → **permanently lost** (F-008, high) | validation failure → exit 1 (the one stage that fails honestly — and its module-level env check `stage10:38-41` even fails at *import*) | no |
| 11 | per-item try/except, prints warnings | exit 0 | destructive, not idempotent in the ways that matter |

## 5. Coupling between stages

- **Temporal coupling:** downloader/pipeline linked only by clock times.
- **Schema coupling via prompts:** stage08/09/10 parse the *shape of LLM output text* ("- " prefixes, "match score: N/5", "recommended" substrings — `stage09_shortlist:122-160`). A prompt tweak in stage07 or a model swap silently changes extraction yield with no test coverage.
- **DB coupling:** stage09_shortlist writes `sent_jobs`; stage10 reads it positionally (`job[5]`, `stage10:483,506`). Adding a column breaks the notifier.
- **Cleanup coupling:** stage11's KEEP/CLEAN lists (`stage11:18-66`) hardcode knowledge of every other stage's directories. New stage output dirs default to *deleted* (loose-file sweep at `stage11:153-166`) unless someone remembers to add them to KEEP_ITEMS.

## 6. State-mechanism inventory and specific defects

Nine mechanisms (inventoried in 01-…-inventory §6). Specific audited defects:

- **F-015 (medium, confirmed): stage04 progress JSON has two concurrent writers.** The writer thread (`result_writer_from_queue`, `stage04:290-411`) does load-once-then-`save_progress` per result, while the main process's meta-drain also does `load_progress`/`save_progress` (`stage04:955-957`). Both hold independent in-memory copies of the whole dict; interleaved saves lose entries. Atomic file replace prevents corruption but not lost updates. Also O(n) full-file rewrite per block ⇒ O(n²) I/O for large papers.
- **F-019 (medium, confirmed): stage07 keeps three resume systems** (SQLite `processed_files`, `processing_checkpoint.txt` "next file", and `--force`/`--fresh` modifiers) with different semantics. `save_checkpoint_for_next` re-globs the entire input tree on every processed file (`stage07:699-713`) — O(n²) and wrong if files appear/disappear between runs.
- **F-020 (medium, confirmed): stage09_llm_filter resumes by integer index** (`progress["last_processed"] + 1`, `stage09_llm_filter:823-824`) into `shortlisted_jobs.json` — a file that stage09_local_filter regenerates (different content/order) every run. A resume after a re-shortlist analyzes the wrong jobs and skips others. The job_id-keyed SQLite table right next to it is the correct resume source and is not used for resume.
- **F-016 (high, confirmed): `cache_blocks/task_backlog.txt` is a write-only file.** GPU-queue overflow tasks are "persisted so they aren't lost" (`stage04:1034-1043`) and no code path ever reads the file back (verified by grep across the repo). Overflowed blocks are lost silently; the log even claims they were saved.
- **F-035 (medium, confirmed): `cache_blocks/` never evicted** — keyed by blake2b of the absolute source path (`stage04:196-199`); daily folders have date-stamped names, so entries are never reused across days and never deleted (stage11 cleans only `data/`; cache_blocks is at repo root). Currently 629 MB and growing with every run. `--no-cache` clears it but nothing invokes that flag.
- **F-013 (high, confirmed): stage11 deletes `raw_pdfs/` and `processed_pdfs/` unconditionally** (`stage11:62-64`) once reached — and it is reached whenever prior stages exit 0, which per F-002 includes fully-failed nights. Source links on epaperwave/Drive rot quickly; the day's inputs are then unrecoverable. The `input()` confirmation ("This cannot be undone!", `stage11:249`) is rubber-stamped by the orchestrator (`run_pipeline.sh:373-381`) — a destructive-action guard that guards nothing.

## 7. Multi-machine / scale limitations

- Everything assumes one machine, one CWD, one filesystem: relative `Path("data/...")` in stages 01–11, absolute `/home/sidd/...` in `jobs.conf`, GPU singleton in stage04, `localhost:11434` in stage07 (a `--gpu-endpoint/--cpu-endpoint` remote-Ollama option exists but production passes `--no-hybrid`).
- SQLite DBs and JSON state files preclude concurrent writers across hosts; `.done` markers are local-only.
- Scaling need is currently modest (a dozen papers/day) — the realistic scale problem is not horizontal but *temporal*: the nightly window is serialized (see 05-review). Verdict: multi-machine is not a near-term requirement; do not architect for it, but stop hard-coding paths so the pipeline can at least be relocated (it currently cannot even be re-run on its own machine, F-001).

## 8. Safer state-management alternatives

Ordered from least to most invasive:

1. **Truthful exit codes + input assertions (do first, tiny):** each stage exits non-zero when it processed nothing or its failure ratio exceeds a threshold; each stage asserts its input dir is non-empty unless a `--allow-empty` flag is passed. This alone converts the silent no-op night into a loud one and makes the existing retry loop meaningful.
2. **A single run-state SQLite DB** (one `runs` table + one `stage_runs` table with run_id, stage, status, counts, started/finished) written by the runner and readable by a status command. Replaces `.done` markers, `pipeline_metrics.json`, and the JSONL for state purposes; keeps file artifacts as-is.
3. **Manifests at boundaries:** stage N writes `data/<dir>/_manifest.json` (count, run_id, produced_at); stage N+1 refuses to run against a manifest from a different run_id unless forced. Cheap, kills the empty-vs-failed ambiguity.
4. **Orchestrator-owned state (Prefect/Dagster):** subsumes 2 and most of 3; Dagster's asset materializations are exactly the manifest concept. See 02-review §8 for selection.
5. **Cleanup safety:** stage11 becomes non-interactive with explicit `--yes`, refuses to delete `raw_pdfs` unless the run-state DB shows stage04+stage07 processed ≥1 item, and archives (moves to a dated cold dir pruned after N days) instead of unlinking. Delete-after-verify, not delete-after-exit-0.

## 9. Findings referenced

F-002, F-003, F-013, F-014, F-015, F-016, F-019, F-020, F-030 (state fragmentation), F-031, F-035, F-044, F-046 (positive). Details in `findings.json`.
