# 05 — External APIs and Concurrency Audit

## 1. Interaction review per service

### Ollama (stage07)
- Sync `requests.Session` with urllib3 `Retry(total=5, backoff_factor=1, status_forcelist=[429,5xx])` mounted on thread-local sessions (`stage07:67-101`), plus a *manual* retry loop of 3 with linear sleep in `query_model_safe` (`stage07:800-880`) — retries are therefore multiplicative (up to ~15 HTTP attempts per prompt, and urllib3 retries **POSTs**, so a slow-but-successful generation can be paid for twice).
- 240s request timeout (`stage07:105`); adaptive `num_predict` windows by content size (`stage07:769-791`) — sensible cost control.
- **The stage owns the server lifecycle**: `start_ollama_serve()` Popens `ollama serve` in a new process group; on connection errors it force-restarts via `shutdown_ollama(force=True)` → `subprocess.run(["pkill", "-f", "ollama"])` (`stage07:887-915`). `pkill -f ollama` kills *any* process whose command line contains "ollama" — including a concurrent pipeline run, an interactive session, or an unrelated script. Service lifecycle belongs to systemd, not to a pipeline stage. **F-018, medium-high, confirmed.**
- If the model never becomes ready, the stage logs an error and **exits 0** (`stage07:1365-1376`) — see F-002.

### Gemini (`src/llm/gemini_client.py`, driven by stage09_llm_filter)
- Sync `google-generativeai` SDK. Per-request timeout 90s via `request_options` (`gemini_client:511-519`).
- **Homemade rate-limit engine** (F-022, medium, confirmed complexity/tradeoff): token bucket (BURST=5, refill 15/min), daily quota 20/key, 60s cooldown-on-429, all persisted to `data/.key_usage.json` **with a file write inside the lock on every successful request** (`gemini_client:533-538`). It works, but it re-implements what `tenacity` + a 30-line quota dict would do, and the constants are hardcoded (free-tier assumptions baked into code).
- A **mandatory `time.sleep(1.2)` after every successful call** (`gemini_client:523`) plus the caller's own `time.sleep(1.2)` between jobs (`stage09_llm_filter:504-506`) — the delay is paid twice, ~2.4s of pure sleep per analyzed job.
- **Thread-safety defect (F-021, medium, confirmed-latent):** `genai.configure(api_key=...)` mutates process-global SDK state (`gemini_client:507`). `run_batch` submits `_process_single_request` to a ThreadPoolExecutor with up to 16 workers (`gemini_client:677-693`): thread B's `configure(key2)` can land between thread A's `configure(key1)` and A's `generate_content`, so A's request is issued (and its quota/cooldown accounted) against the wrong key. The production path (`generate()` → sequential) never triggers it, but `generate_batch` is a public method and the CLI `--batch` path uses it. Fix: per-thread `genai.GenerativeModel` construction guarded by a configure-lock around configure+call, or migrate to the `google-genai` client class which takes the key per-client instance.
- Key-validation cache and usage state are two more dot-files; stage09_llm_filter *deletes* the validation cache at startup (`stage09_llm_filter:753-754`) — forcing daily revalidation that the cache was built to avoid.

### Telegram Bot API (stage10)
Covered in depth in 07-review. Summary: sync `requests.post`, 15s timeout, 3 retries with 3/6/12s backoff, no `retry_after` parsing, fixed 1.5s inter-message sleep.

### Telethon (downloader phase 3)
- Async client correctly awaited, but wrapped in `asyncio.run(asyncio.wait_for(..., 180))` inside a sync method (`downloader:2674-2683`), with a `RuntimeError` fallback to `get_event_loop().run_until_complete` (dead code on 3.10+ in a fresh thread; harmless).
- Scans last 100 messages, fuzzy-matches filenames at 0.65 ratio — see 06-review.

### GitHub (stage09_dynamic_resumes)
- Sync PyGithub, `time.sleep(0.5)` per repo, README fetch per repo (`stage09_dynamic_resumes:94-124`). Unauthenticated fallback = 60 req/hr, easily exhausted by two accounts' repo scans; failure degrades to empty skills text silently. Cache exists (`github_skills_<user>.json`) but is **never expired** — once written, GitHub is never fetched again unless the cache file is manually deleted (`stage09_dynamic_resumes:78-87`). "Dynamic resumes" are static after day one. Classification: confirmed issue, low severity.

## 2. Sync vs async analysis

Everything user-facing is synchronous, and that is mostly the right call for this system: the workload is a nightly batch on one machine where the bottlenecks are GPU OCR and LLM generation, not I/O multiplexing. Async would buy real throughput only in three places:

1. **stage09_llm_filter** — N independent Gemini calls executed strictly sequentially with 2.4s of sleep each. With 2 keys at 15 RPM each, a compliant concurrent scheduler (async or thread pool with a real rate limiter) is ~2× faster and removes the artificial sleeps.
2. **Downloader preprocessing** — designed for parallel drivers but shipped with `max_preprocess_workers=1  # Reduced to 1 for debugging` (`downloader:2806`). Restoring 3–4 is the cheapest speedup in the whole system (each source burns 10s `INITIAL_PAGE_WAIT` + Cloudflare waits serially today).
3. **stage10** — irrelevant at current volumes; leave sync.

**Recommendation:** do not migrate to asyncio. Fix the two throttles above with existing thread pools + a shared rate limiter. An async rewrite of 12.6k lines is cost without payoff.

## 3. Thread usage review

| Site | Mechanism | Assessment |
|---|---|---|
| stage01 | ProcessPoolExecutor (PDFs) × ThreadPoolExecutor (pages) | Reasonable; `nonlocal` counters mutated from threads without a lock (`stage01:145-174`) — counts can undercount, cosmetic only |
| stage02 | multiprocessing.Pool imap_unordered | Fine |
| stage03 | ProcessPoolExecutor with tesseract warm-up initializer | Fine |
| **stage04** | **1 GPU thread + 4 spawn CPU processes + writer thread + monitor thread + 3 queues (local Queue, spawn Queue ×2) + Manager dict/lock + signal handlers + global mutable registries** | **F-017, high, confirmed.** ~800 lines of homemade executor. Two ~70-line OOM-spill blocks are byte-similar copy-paste (`stage04:556-634` vs `673-750`). Failure modes found: overflow → write-only backlog file (F-016); quick-drain vs writer-thread progress race (F-015); `results_queue` maxsize 32 while the drain re-enqueues into the same bounded queue it is draining (`stage04:958-964`) — under load the re-enqueue can block/timeout and drop a result. The design (queue + spillover + poison pills) is what `concurrent.futures` + a two-priority work list would give in ~100 lines. |
| stage07 | ThreadPoolExecutor for file prefetch (fine) + 2-thread hybrid endpoint dispatch (disabled in prod via `--no-hybrid`) | `db_write_lock` correctly serializes SQLite writes; global `args` accessed from worker threads is initialized before threads start — OK but fragile pattern |
| gemini_client | ThreadPoolExecutor run_batch | F-021 race above; overall/individual future timeouts handled carefully — but note `future.cancel()` cannot stop an in-flight SDK call, so "timed-out" work keeps consuming a key slot in the background |
| downloader | driver pool via `queue.Queue` + ThreadPoolExecutor (preprocess) | Pool pattern is sound; pool size 1 in production nullifies it |

## 4. Blocking behavior

- The entire nightly run is one serialized chain: GPU OCR (hours-scale for a dozen 20-30-page papers) → translation (CPU, line-by-line — each line is a separate Argos `translate()` call, `stage05:136-149`) → Ollama (240s-timeout calls, sequential batches) → Gemini (sleep-gated sequential) → Telegram (1.5s/message).
- Every sleep found is a fixed cost regardless of conditions: `INITIAL_PAGE_WAIT=10` per page-navigation ×2 events per source, `sleep(1.2)`×2 per Gemini job, `sleep(1.5)` per Telegram message, `sleep(0.5)` per GitHub repo, scheduler's 1s poll.
- Nothing enforces a deadline on the pipeline as a whole: with `MAX_RETRIES=0` or a hung Selenium/Ollama call at exactly the timeout boundary, the 23:30 run can still be alive at 11:50 and overlap the next downloader (no locking, F-005).

## 5. Timeout / retry inventory

| Call | Timeout | Retry | Gap |
|---|---|---|---|
| Ollama generate | 240s | urllib3 5× + manual 3× | multiplicative; POST retried |
| Ollama warm-up poll | 3s ×60 attempts exp backoff | — | OK |
| Gemini generate | 90s SDK | manual 3× exp backoff + key rotation | OK-ish; `time.sleep` on cooldown *inside* `_get_next_key` of the fallback client blocks the only thread |
| Telegram send | 15s | 3× exp backoff | no `retry_after`; watermark advances past failures (F-008) |
| Telegram getMe/test | 10s | none | one-shot; failure aborts stage (correctly non-zero) |
| Telethon batch | 180s overall | none | one-shot per night |
| GitHub | PyGithub defaults (10s) | none | silent empty-result degradation |
| requests HEAD/GET in downloader | 8–15s | session Retry 3× | OK |
| Selenium page load | 120s | none per-navigation; heuristic cascade acts as retry | acceptable |

## 6. Throughput bottlenecks (ranked)

1. **stage04 GPU OCR** — intrinsic; mitigations are model-level (the roadmap doc `docs/V2_MAIN_ROADMAP_VLM_OCR.md` already targets a VLM redesign) — out of audit scope but noted.
2. **stage07 Ollama** — sequential batches of ≤2 files, ≤5000 chars (`stage07:1054`); no pipelining of the next prompt while one generates. An Ollama server with `num_parallel=2` + 2 in-flight requests would roughly halve wall time; verify VRAM headroom first (hypothesis).
3. **stage05 translation** — per-line Argos calls; batching lines per file into one call (Argos handles multi-line text) would cut Python/marshalling overhead substantially. Low risk.
4. **stage09_llm_filter** — double sleep + sequential (see §2).
5. **Downloader** — worker pool of 1 + fixed waits (see §2).
6. **stage08 dedup** — O(n²) SequenceMatcher over all job lines per issue (`stage08:35-63`); fine at dozens of jobs, degrades quadratically. Pre-bucketing by normalized prefix or rapidfuzz would future-proof it. F-041, low.

## 7. Async migration opportunities — concrete verdict

- **Do not** rewrite to asyncio.
- **Do**: restore downloader parallelism (1 line), remove the duplicated Gemini sleep and drive 2 keys concurrently with the existing token bucket (small change in `stage09_llm_filter`/`gemini_client`), batch Argos calls per file, and consider Ollama `num_parallel` after a VRAM check.
- **Do**: extract stage04's executor into a boring `ProcessPoolExecutor`-based design when stage04 is next touched (it is scheduled for replacement by the VLM roadmap anyway — do not gold-plate it before that decision is made).

## 8. Findings referenced

F-002, F-005, F-008, F-015, F-016, F-017, F-018, F-021, F-022, F-023 (sync-serialization tradeoff), F-041; positives: adaptive token windows, thread-local sessions, per-future timeout handling. Details in `findings.json` and `external-api-matrix.csv`.
