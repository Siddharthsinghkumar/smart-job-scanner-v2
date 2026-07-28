# 06 — Scraping Reliability Audit

Subject: `src/downloader/newspaper_downloader.py` (2,821 lines, single class `CompleteNewspaperDownloader`).

## 1. Scraper architecture review

Three-phase design (`download_all_optimized`, `downloader:2688-2802`):
1. **Preprocess (parallel-capable):** Selenium visits each source's listing page, locates today's edition by date-text heuristics, resolves a download URL (Google Drive ID or direct PDF).
2. **Download (batched serial):** requests-based streaming download with GDrive confirm-token handling; Selenium-driven browser-download fallback for scan-warning pages.
3. **Telegram fallback:** Telethon scans a channel's last 100 messages for PDFs fuzzy-matching configured newspaper names.

Structurally sound ideas: driver pool, PDF validation (header + `%%EOF` + min size, `downloader:248-272`), atomic streaming writes (`downloader:62-84`), per-day already-downloaded detection with invalid-file deletion (`downloader:359-385`), 24h skip-state for scan-protected Drive files (`downloader:222-246`). These are positive findings — the failure-handling *inside* one run is genuinely better than the rest of the pipeline.

## 2. Fragility inventory

**F-026 (high, confirmed): single-aggregator dependency.** All 12 `web_sources` in `configs/newspaper_config.json` point at `epaperwave.com` — an unofficial aggregator that redistributes newspaper PDFs via Google Drive links. One site redesign, one Cloudflare policy change, one takedown, and web ingestion drops to zero simultaneously. The Telegram channel fallback is likewise a single unofficial channel (`newspaper_wave_10`). This is a structural availability risk (and carries copyright/ToS exposure worth acknowledging: the sources are not licensed feeds).

**F-027 (high, confirmed): heuristic cascade depth.** Finding "today's paper" involves, in order: 20 date-format variants (`downloader:657-707`), regex date search in page text, month-header proximity walks (`downloader:906-961`), fuzzy newspaper-name matching at 0.65 SequenceMatcher ratio (`downloader:94-104`), anchor keyword scans ('edition', 'download', 'epaper'…), `data-*`/onclick/inline-`<script>` regex mining for PDF/Drive URLs (`downloader:1036-1061`), iframe-src inspection, and HEAD-request content-type probing. Each layer was evidently added when the previous broke. The cascade has no telemetry about *which* layer matched, so when yield degrades (wrong edition, yesterday's paper, wrong city), nothing flags it — a fuzzy match at 0.65 will happily download "IE Chandigarh" when "IE Delhi" was wanted.

**Three separate Google Drive download implementations** coexist: `download_gdrive_file_via_selenium` (~380 lines), `download_gdrive_file` (requests-based confirm-token dance, ~270 lines), and inline resolution paths in the two preprocessors (all four visible in the function map: `downloader:1709,2086,833,1237`). Each handles the confirm-token/`resourcekey`/scan-warning flow slightly differently; a Drive-side change means fixing it three times.

**Anti-bot arms race:** `navigator.webdriver` hiding, `AutomationControlled` disabling, UA pinned to Chrome 91 (2021-era — itself a fingerprint anomaly against a current chromedriver), Cloudflare interstitial polling by *page-text substrings* ('checking your browser', 'just a moment', `downloader:457-472`). These countermeasures decay silently; the config's pinned UA is stale evidence of exactly that.

**Shipped debug configuration:** `main()` constructs the downloader with `max_preprocess_workers=1  # Reduced to 1 for debugging` (`downloader:2806`) — production runs serialized with 10s `INITIAL_PAGE_WAIT` per navigation. Also `headless: false` in the live config (`configs/newspaper_config.json` settings) means the scheduled 11:50 run **requires a display**; on a headless boot or locked X session, every Selenium driver fails to start. Classification: confirmed issue, medium — likely a contributor to silent zero-download days.

## 3. Target-site dependency risks

| Dependency | Failure mode | Blast radius | Current detection |
|---|---|---|---|
| epaperwave.com layout/date format | date heuristics find nothing → "Date not found" errors | all 12 web sources | log line + debug HTML dump only |
| epaperwave Cloudflare posture | interstitial never clears → partial DOM scraping | all web sources | debug log only |
| Google Drive confirm/scan flow | token dance fails → skip-state marks file 24h | per file | log only |
| Telegram channel deletion/rename | `get_entity` fails | fallback path | log only |
| Chrome/chromedriver version drift | driver creation fails | all Selenium paths | log only; downloader continues with zero drivers and every source returns "No driver available" |
| Display availability (headless:false) | driver creation fails | same | same |

The common row: **every detection is a log line in `newspaper_downloader.log`** that nobody is alerted about, followed at 23:30 by a pipeline that treats an empty `raw_pdfs/` as success (F-003).

## 4. Resilience gaps

1. **No yield alarm.** `main()` returns success if ≥1 file downloaded (`downloader:2819-2821`) — 1 of 14 papers is "success". There is no expected-vs-actual source count check, no Telegram ping saying "downloaded 3/14 today", despite a working bot token in the same repo.
2. **Exit status discarded** by scheduler.sh regardless (02-review §2).
3. **No per-source retry across the day.** One attempt at 11:50; a source that publishes late is missed until tomorrow. The 24h GDrive skip-state can even mask a *transient* scan-warning failure for a full day.
4. **`_move_latest_browser_download`** claims "the newest file in the download dir" as the wanted PDF (`downloader:387-455`) — any concurrent download (or a second worker, if parallelism is ever restored) can be mis-attributed to the wrong source name. Latent race, medium confidence.
5. **Wrong-edition risk:** fuzzy matching + variant lists means the validator checks *that* a PDF arrived, not that it is the right paper/city/date. Filenames from Telegram are trusted after a 0.65 fuzzy match.

## 5. Extraction failure modes (observed downstream)

`logs/failed_batches/` holds 60+ error-batch JSONs from a GPU worker, and `logs/raw_responses/` holds ~45 unparseable-LLM-response dumps — evidence that OCR-batch failures and JSON-parse failures are recurring operational events, saved for forensics that no tooling ever aggregates. The debug-HTML dumps (`save_debug_page`) similarly accumulate in `data/raw_pdfs/debug/` until stage11 deletes the whole tree nightly — i.e., the forensic artifacts for a failed day are destroyed the same night by design (interaction of F-013 with debugging needs).

## 6. Alternatives

Ranked by leverage:

1. **Official/licensed feeds where they exist (highest leverage, partial coverage).** Several Indian dailies expose official e-paper portals with subscriptions and stable per-edition URLs; employment-news sources (the actual product target is job ads) have official channels (e.g., Employment News/Rozgar Samachar e-edition). Even converting 3–4 of 12 sources to stable authenticated fetches removes the deepest heuristics for those and de-risks the aggregator. Also worth evaluating: government job portals/RSS (NCS, state PSC feeds) as a *parallel* structured source that bypasses OCR entirely for a large slice of the job universe — likely the single biggest reliability win available to the product, since scraped-newspaper OCR→LLM is the fragile path end to end.
2. **Telegram-first instead of Telegram-fallback.** The Telethon path is dramatically simpler and more robust than the Selenium cascade (no Cloudflare, no Drive tokens, filename metadata included). Inverting priority — scan 2–3 known channels first, fall back to web scraping only for missing papers — would shrink the risky code path to a remainder.
3. **Playwright over Selenium** for whatever scraping remains: auto-waiting locators remove most fixed sleeps, built-in download API removes `_move_latest_browser_download`, context-level JS toggles replace Chrome pref hacks, and headless operation is first-class (fixes the display dependency). Migration cost is moderate (the heuristics port unchanged); do it only after the cascade is slimmed, not as a like-for-like port of all 2,800 lines.
4. **RSS/JSON where available:** epaperwave itself is WordPress (`/download-the-…/` permalinks); its `wp-json`/feed endpoints may expose post lists more stably than the rendered HTML. Ten-minute experiment; would replace the date-hunting heuristics with structured post queries. (Unverified hypothesis — test `https://epaperwave.com/wp-json/wp/v2/posts?search=<paper>`.)
5. **Structural refactor regardless of driver:** split the class into fetchers (per source type), resolvers (Drive/direct), validators, and a yield reporter; one Drive implementation; per-layer match telemetry ("source X matched via layer 3-fuzzy") so degradation is visible before it hits zero.

## 7. Findings referenced

F-003, F-013 (evidence destruction interaction), F-026, F-027, F-028 (invisible failures), F-029 (session/token files on disk — `telegram_session_newspaper_wave_10.session` at repo root, `token_telegram/`, real keys in `configs/gemini_config.json`; all gitignored, none encrypted), F-036. Positives: PDF validation, atomic writes, daily skip-state, driver pooling. Details in `findings.json`.
