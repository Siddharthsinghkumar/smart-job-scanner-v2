What you've used (evidenced by resume + actual code imports)

  Document & Vision:
  PyMuPDF, pdfplumber, PyPDF2, pdf2image, EasyOCR, Tesseract, PaddleOCR,
  LayoutParser, Detectron2, YOLOv8/Ultralytics, Donut, OpenCV, Pillow,
  Selenium, Telethon, BeautifulSoup, argostranslate

  LLM & AI Frameworks:
  Ollama (OpenHermes), llama.cpp, LangChain, LangGraph, LlamaIndex,
  Transformers, Sentence-Transformers, OpenAI API, Google GenerativeAI
  (Gemini), Anthropic

  Data Libraries:
  NumPy, Pandas, SciPy, PyTorch, scikit-learn

  Backend & Storage:
  FastAPI, Starlette, Uvicorn, Django, Flask, PostgreSQL
  (SQLAlchemy/psycopg2), SQLite, Redis, MongoDB, Supabase

  Infra:
  Docker, Kubernetes/k3s, Helm, Terraform, GitHub Actions, Prometheus,
  Grafana, Sentry

  Concurrency:
  asyncio, ProcessPoolExecutor, ThreadPoolExecutor, multiprocessing

  CLI/Utils:
  Rich, tqdm, tenacity, psutil, pynvml, RapidFuzz

  Conceptual exposure:
  Kafka (producer/consumer design pattern), Ansible (framework extension)

  ───
  What would actually benefit the smart-job-scanner-v2 refactoring

  High impact, low effort

  ┌──────────────────────┬──────────────────────┬──────────────────────┐
  │ Tool                 │ Replaces             │ Why                  │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ surya (OCR+layout)   │ EasyOCR + custom     │ Handles multi-column │
  │                      │ block detection      │ newspapers natively, │
  │                      │                      │ modern model. Could  │
  │                      │                      │ collapse stages      │
  │                      │                      │ 02+03+04 into one    │
  │                      │                      │ call                 │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ instructor           │ Manual JSON parsing  │ Typed Pydantic       │
  │                      │ from Ollama HTTP     │ outputs, automatic   │
  │                      │ responses            │ retry, validation.   │
  │                      │                      │ Stage 07 currently   │
  │                      │                      │ regex-parses raw LLM │
  │                      │                      │ text                 │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ litellm              │ Separate             │ One interface,       │
  │                      │ ollama_runner.py +   │ auto-fallback.       │
  │                      │ gemini_client.py     │ Ollama down →        │
  │                      │                      │ Gemini. Rate limits  │
  │                      │                      │ → next key           │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ ChromaDB             │ Keyword-based resume │ Embedding-based      │
  │                      │ filtering (stage 09) │ semantic matching.   │
  │                      │                      │ Your keyword .py     │
  │                      │                      │ files cap at         │
  │                      │                      │ hardcoded limits     │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ pydantic (more use)  │ Dicts everywhere     │ You already import   │
  │                      │ across pipeline      │ it. Adding typed     │
  │                      │ stages               │ models between       │
  │                      │                      │ stages catches data  │
  │                      │                      │ corruption early     │
  └──────────────────────┴──────────────────────┴──────────────────────┘

  Medium impact, depends on PDF source quality

  ┌──────────────────────┬──────────────────────┬──────────────────────┐
  │ Tool                 │ Replaces             │ Why                  │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ docling (IBM) or     │ stages 01-04 for     │ PDF → structured     │
  │ marker               │ clean PDFs           │ markdown preserving  │
  │                      │                      │ reading order. Won't │
  │                      │                      │ help with scanned    │
  │                      │                      │ newspaper images     │
  │                      │                      │ though               │
  └──────────────────────┴──────────────────────┴──────────────────────┘

  Would NOT help

  - Kafka/RabbitMQ — sequential batch, not streaming. Wrong pattern.
  - Airflow/Prefect/Dagster — 14-stage sequential script with .done files
   works. These add a scheduler server for no gain at single-machine
  scale.
  - Spark/Dask/Ray — single-machine, <20 GB/day. Your ProcessPoolExecutor
   is fine.
  - dbt/Airbyte/Fivetran — wrong domain (document processing, not tabular
   data)
