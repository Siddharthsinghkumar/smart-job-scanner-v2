# Frontend & API Project Review

## PROJECT CONTEXT

**Repository / workspace type:**
Custom monolith / script-based ETL pipeline (Headless)

**Frontend Framework(s):**
None (Headless). The "Frontend" UI is entirely abstracted to **Telegram** via Bot messages formatted in HTML.

**API / Web Server Framework(s):**
None. There is no internal API being served (no FastAPI, Flask, Django, Express, etc.). The system operates purely via local cron-like scheduling and direct file/database IO.

**Key third-party APIs consumed:**
- **Telegram Bot API:** Used for the presentation layer (sending alerts/jobs to users).
- **Google Gemini API / Ollama API:** Used for the data transformation layer (LLM extraction and filtering).

---

## Priority concerns (Frontend / Interface Level):
API Resilience, Rate-Limiting, UX Formatting (Telegram), Scraping Reliability.

## Known suspicious areas & Candid Feedback:

1. **No Internal API / Dashboard (The "Black Box" Problem):**
   - **The Problem:** Because the system is headless and heavily bash-scripted, there is no way for a user or admin to check pipeline health, view failed extractions, or manually trigger a retry without SSH-ing into the server and reading log files.
   - **Alternative:** Expose a lightweight internal API using **FastAPI** coupled with a basic **React/Vite** or **Streamlit** dashboard. This API could wrap the SQLite databases (`progress.db`, `shortlist_history.db`) to provide a visual health-check, metrics overview, and manual triggering capabilities.

2. **Telegram as the Sole Presentation Layer:**
   - **The Problem:** The Telegram bot integration (`stage10_notification.py`) manually strings together HTML chunks and escapes characters. If a job description is malformed, it can break the Telegram HTML parser and fail to send, or look terrible. Additionally, if the system generates a massive backlog of jobs, it will spam the Telegram channel, potentially hitting rate limits.
   - **Alternative:** Implement a message queue (like **Redis/Celery**) specifically for the Telegram notifier to handle rate limiting gracefully (e.g., max 20 messages per minute). Also, consider using a templating engine like **Jinja2** for the Telegram messages rather than raw f-strings and manual `html.escape()` for cleaner UI code.

3. **Scraping as an "API" (Brittle Ingestion):**
   - **The Problem:** `newspaper_downloader.py` relies heavily on Selenium, BeautifulSoup, and massive regex parsing to find PDFs. Web scrapers are notoriously brittle API substitutes; any DOM change on the target site breaks the ingestion pipeline.
   - **Alternative:** Where possible, seek out RSS feeds or official API endpoints for the newspapers. If scraping is the only option, abstract the scraping logic into a highly resilient microservice (perhaps using **Playwright** instead of Selenium, as it handles modern SPAs better and is generally more reliable).

4. **Synchronous External API Calls (LLMs):**
   - **The Problem:** The pipeline makes synchronous HTTP calls to local Ollama and external Gemini APIs. If these APIs stall, the entire pipeline thread blocks. 
   - **Alternative:** Move API integrations to an async architecture (e.g., using `asyncio` and `aiohttp` or `httpx`) to maximize throughput, especially when batching prompts to Gemini or parallelizing Ollama requests.

**Summary:** 
The system is entirely headless, relying on third-party APIs for processing and Telegram for its UI. While this is lean, it suffers from a lack of internal observability. Introducing a lightweight FastAPI + Streamlit dashboard would massively improve maintainability. Furthermore, the Telegram notification logic should be decoupled and rate-limited to avoid spamming the UI or crashing on malformed HTML strings.
