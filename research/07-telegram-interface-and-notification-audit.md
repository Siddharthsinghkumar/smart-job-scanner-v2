# 07 — Telegram Interface and Notification Audit

Subject: `src/pipeline/stage10_notification.py` (533 lines) plus its upstream contract with `stage09_shortlist.py`.

## 1. HTML formatting and safety review

- Two escapers exist: `escape_html` (correct — `html.escape`, `stage10:90-94`) and `escape_html_codeblock` (`stage10:96-111`) which additionally converts `- . ( ) + = |` to numeric entities. That extra escaping is MarkdownV2-style thinking carried into HTML mode — Telegram HTML only requires `& < >`; the numeric entities are accepted by Telegram but make logged messages unreadable and add no safety. Informational.
- **F-009 (medium, high confidence, confirmed issue): the extracted website URL is interpolated raw.** `f"<b>🌐</b> {parsed['website']}\n"` (`stage10:415`) — `parsed['website']` comes from a permissive regex over OCR'd job text (`WEBSITE_PATTERN`, `stage10:47`). A URL containing `&` (extremely common in query strings) or a stray `<` from OCR noise produces invalid HTML → Telegram 400 "can't parse entities" → all 3 retries burn → message lost (and per F-008 below, lost *permanently*). Every other field is escaped; this one was missed.
- `apply_date` (line 414) interpolates `extract_apply_date()` output unescaped too — lower risk (regex-constrained to date-ish text) but same pattern.
- Messages are built by f-string concatenation inline in `format_job_message` — no length guard against Telegram's 4096-char limit. Job text is truncated to 160 chars so overflow is unlikely today; a template change could silently break it. Informational.

## 2. Notification coupling review

- **Upstream coupling:** stage10 reads the `sent_jobs` table stage09_shortlist wrote, by **positional tuple index** (`job_id, job_text, similarity, score, processed_at, shortlisted_at = job_data`, `stage10:391`; `jobs[-1][5]`, `stage10:506`). Schema evolution breaks the notifier silently (F-025 overlap).
- **Semantic coupling:** the table is named `sent_jobs` and is written **at shortlist time, before any send** (`stage09_shortlist:180-186`, comment: "Log immediately to prevent future duplicates"). Delivery state and dedup history are conflated (F-012, medium, confirmed). If stage10 fails hard (its module-level env validation `raise SystemExit` at `stage10:38-41` fires on a missing `.env` — *at import*, which under the runner means 5 futile retries then pipeline abort before cleanup), those jobs are recorded forever and never delivered.
- **Business logic embedded in the notifier:** `choose_resume()` and `suggest_improvement()` (`stage10:353-379`) re-implement keyword-based matching that stage09 already did with embeddings and Gemini — the notification layer second-guesses the analysis layer with cruder rules. Refactor target: these belong upstream (or nowhere), not in message formatting.

## 3. Delivery reliability — the watermark bug

**F-008 (high, high confidence, confirmed issue): failed sends are permanently skipped.**

Flow (`stage10:479-511`):
1. Jobs are fetched `WHERE shortlisted_at > watermark ORDER BY shortlisted_at ASC`.
2. Each send failure only increments `failed_sends`.
3. After the loop, if `successful_sends > 0`, the watermark is set to `jobs[-1][5]` — the timestamp of the **last job in the list regardless of whether it (or any middle job) failed**.

Concrete failure: 10 new jobs; job 4 hits the unescaped-URL 400 (F-009) or a 429 window; jobs 5–10 succeed; watermark = job 10's timestamp → job 4 can never be selected again. There is no `--resend-failed`; `--force` resends *everything ever* (`get_all_jobs`), which is the spam alternative, not a recovery tool.

Correct design: track per-job delivery (a `notifications` table with status/attempts), advance the watermark only through a contiguous prefix of successes, or simply mark rows `sent_at` individually and query `WHERE sent_at IS NULL`.

## 4. Backlog and spam / rate-limit risks

- **Backlog burst:** after N days of downloader-or-pipeline failure followed by a fix, all accumulated jobs sail out in one run (send loop has no cap). With a 1.5s spacing this is compliant for a single chat, but the daily-summary claims and the user experience degrade; a `--max-per-run` with rollover would be kinder.
- **F-011 (medium, confirmed): no 429 handling.** Telegram returns `retry_after` in 429 bodies; `send_telegram_message` treats 429 like any error and retries at 3/6/12s (`stage10:187-211`) — for a long `retry_after` all retries land inside the penalty window and the message fails (then F-008 loses it). Single-chat volume makes 429s rare; the failure interaction is what elevates this.
- **F-010 (low, confirmed): a live test message ("🔧 Bot configuration test - please ignore") is sent to the real chat on every run** (`stage10:131-141`) — daily noise; `getMe` alone validates the token, and a `sendChatAction` would validate the chat without a visible message.
- Fixed `time.sleep(1.5)` between sends is fine for one chat (limit ~1 msg/s per chat) — informational.

## 5. Templating quality

- Message construction is inline f-strings with hardcoded emoji layout; the "employer / deadline / website" fields come from regex heuristics over OCR text (`parse_job_text`, `stage10:314-351`) with defaults "Unknown Organization"/"Not specified" — expect a large fraction of messages to carry defaults (regexes require English suffixes like "Limited|Corporation|…"; OCR'd Hindi-translated ads often won't match). No template file, no separation of data extraction from presentation, no snapshot tests of rendered messages.
- The daily summary message (`format_summary_message`) reports "Jobs scanned: {total_processed}" where `total_processed=len(jobs)` — that is *new jobs fetched*, not scanned; minor but it is another metric that reads as more than it is (echoes F-006's pattern).

## 6. Decoupling recommendations

1. **Split dedup from delivery.** Keep `sent_jobs` as `shortlist_history` (dedup), add a `notifications` table (`job_id, formatted_at, sent_at, attempts, last_error`). stage09_shortlist inserts pending notifications; stage10 becomes a delivery worker: `SELECT … WHERE sent_at IS NULL`, send, mark. Retry-by-default falls out for free; the watermark file disappears (F-008 fixed structurally).
2. **Move `choose_resume`/`suggest_improvement` upstream** into stage09 output (or drop them), so stage10 only formats and sends.
3. **Escape everything through one function**, hyperlink the website via `<a href="...">` with `html.escape` on the href, and add a 4096 length clamp.
4. **Report pipeline health through the same bot.** The system already owns a Telegram bot; a 3-line "pipeline failed at stage04, 0 pages OCR'd" message on failure is the cheapest possible alerting channel and directly addresses the silent-failure findings (F-003, F-028, F-036).

## 7. Queueing recommendations

A message broker is unnecessary at this volume. The `notifications` table above *is* the queue (SQLite as outbox): durable, inspectable with one query, replayable with one UPDATE. If volumes ever grow to many chats/channels, `python-telegram-bot`'s built-in rate-limiter or a tiny token bucket in the delivery worker covers per-chat limits; revisit real queues only if multi-process senders appear.

## 8. Findings referenced

F-008, F-009, F-010, F-011, F-012, plus interactions with F-002/F-003/F-006. Positives: HTML parse mode chosen deliberately over Markdown (more robust), config validation with actionable operator hints (`stage10:148-161` is genuinely good error UX), dry-run mode, per-send retry with backoff. Details in `findings.json`.
