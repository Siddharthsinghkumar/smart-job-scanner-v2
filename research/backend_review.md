# Backend Project Review

## PROJECT CONTEXT

**Repository / workspace type:**
Custom monolith / script-based ETL pipeline

**Primary backend language(s):**
Python 3

**Primary backend framework(s):**
None. It's a collection of custom scripts glued together rather than using an established web or data framework.

**Database(s):**
SQLite (e.g., `progress.db`, `data/shortlist_history.db`)

**ORM / query tool:**
Raw SQL via the standard library `sqlite3`. No ORM or query builder is in use.

**Queue / background job system:**
Homegrown. A custom bash loop (`scheduler.sh` parsing `jobs.conf`) acts as a cron replacement, and `scripts/run_pipeline.sh` orchestrates the 14-stage ETL process with its own retry/backoff logic.

**Caching:**
File-system based (`data/batch_inputs/`, `run_state/*.done` markers) and SQLite databases. 

**Deployment target:**
Local server / bare metal (likely a machine with a GPU, given CUDA/Torch and local Ollama requirements).

**Package manager(s):**
`pip` (managed via `requirements.txt` and `requirements-dev.txt`)

**Key third-party services:**
- Telegram API (Telethon / Requests) for notifications and potentially data retrieval.
- Google Gemini API (LLM).
- Local Ollama (OpenHermes) for offline GPU extraction.
- Various scraping targets (Selenium/BeautifulSoup).

---

## Priority concerns:
Maintainability, Reliability, Scaling, and Architecture.

## Known suspicious areas & Candid Feedback:

1. **Reinventing the Orchestrator/Scheduler Wheel:**
   - **The Problem:** The pipeline relies heavily on `scheduler.sh` (a custom daemon parsing `jobs.conf`) and `scripts/run_pipeline.sh` (a massive 400+ line bash script managing 14 python stages, `.done` state files, and retry logic). This is a classic case of reinventing the wheel. It is brittle, hard to monitor, and prone to silent failures or edge cases (e.g., time-zone issues, process zombie-ing).
   - **Alternative:** **Prefect**, **Dagster**, or **Apache Airflow**. These Python-based orchestration frameworks handle DAGs, retries, state management, observability, and scheduling out-of-the-box. Since the pipeline is purely Python, moving to Prefect/Dagster would drastically simplify orchestration.

2. **File-based IPC & Brittle State Management:**
   - **The Problem:** Passing data between 14 consecutive scripts via the filesystem (`data/batch_inputs/`, `.done` files) tightly couples the execution environment and makes scaling to multiple machines or handling partial failures extremely messy.
   - **Alternative:** An orchestrator (mentioned above) passing data references natively, or using a real message queue like **Redis + Celery** / **RQ** if you need concurrent task processing (e.g., chunking OCR or LLM requests in parallel).

3. **Raw SQLite / Lack of Data Access Layer:**
   - **The Problem:** `sqlite3` connects and executes raw SQL queries scattered randomly throughout pipeline stages (e.g., in `stage07_llm_extraction.py`, `stage09_llm_filter.py`, `stage10_notification.py`). This makes migrations, schema updates, and query optimization very difficult.
   - **Alternative:** Introduce a lightweight ORM like **SQLAlchemy** or **Peewee**. If you want modern type-hinting, **SQLModel** or **Prisma (Python)** would make database interactions safer and centralize the schema definitions.

4. **Codebase Structure & Naming:**
   - **The Problem:** The `src/pipeline` directory is a mix of numbered procedural scripts (`stage01_pdf_to_images.py`, `9-1_dynamic_resumes_full.py`, `7_final_ollama_pipeline.py`). There is very little object-oriented design or functional composability; it reads like a series of Jupyter notebooks exported to `.py`.
   - **Alternative:** Restructure into a proper Python package. Define classes/functions for extract, transform, and load operations. Separate the core business logic (e.g., parsing, LLM prompting) from the execution/orchestration layer.

5. **Concurrency vs. Thread-safety:**
   - **The Problem:** In scripts like `stage07_llm_extraction.py`, there is custom threading and thread-local session management being manually handled for HTTP requests. It's complex and prone to bugs.
   - **Alternative:** Rely on **asyncio** with `aiohttp` or `httpx` for IO-bound concurrent requests, which is safer and scales much better than manually wrangling threads.

**Summary:** The project works as a monolithic script-based pipeline, but it's hitting the limits of bash-driven orchestration. Modernizing the scheduling and state management with a Python-native data orchestrator (Prefect/Dagster) and wrapping the database layer in an ORM will drastically improve reliability and developer experience.
