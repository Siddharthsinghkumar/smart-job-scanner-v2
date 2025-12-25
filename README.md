# 🚀 Smart Job Scanner v2  
### A High-Throughput Job Intelligence & Data Refinery

**Smart Job Scanner v2** is an autonomous **ETL (Extract, Transform, Load) + Intelligence pipeline** that bridges the gap between traditional print media and modern job search.

It continuously ingests unstructured newspaper PDFs, processes **~15–20 GB of raw visual data per run**, and distills it into **high-signal, relevance-scored job alerts** delivered via Telegram.

The system is designed for **hardware-constrained environments**, predictable execution, and long-running scheduled operation.

---

## ✨ What This System Does

- Automatically downloads daily newspaper PDFs (web + Telegram sources)
- Converts PDFs into images and isolates job advertisements using computer vision
- Performs multilingual OCR (English & Hindi)
- Extracts structured job data using local and cloud LLMs
- **Performs semantic similarity matching with a normalized scoring system**
- Sends concise, relevance-weighted Telegram alerts
- Maintains persistent state to prevent duplicate notifications

---

## 🏗️ Engineering Architecture  
### A Stateful 12-Stage Pipeline

The system is implemented as a **deterministic, stateful, 12-stage pipeline**, moving from raw pixels to semantic job intelligence.

---

### 1️⃣ Ingestion & Vision (src/01 – src/03)

- **Resilient Ingestion**  
  Multi-channel downloader (web scraping + Telegram API fallbacks) with retry logic and backoff handling.

- **Dynamic DPI Estimation**  
  Adaptive scaling engine computes optimal DPI per page to balance OCR accuracy against GPU VRAM limits.

- **Layout Intelligence**  
  Parallelized block detection using OpenCV to isolate job advertisements from general newspaper noise.

---

### 2️⃣ OCR & Extraction Engine (src/04 – src/07)

- **Hardware-Aware OCR**  
  Custom EasyOCR pipeline with:
  - GPU VRAM governor (capped at ~3800 MB for NVIDIA 3050 Ti stability)
  - Multi-core CPU fallback paths for reliability

- **Stateful Local LLM Inference (Ollama)**  
  SQLite-backed checkpointing ensures crash-safe execution.  
  If processing halts at page *N*, it resumes from *N+1* with no data loss.

- **Adaptive Token Windowing**  
  Dynamically adjusts generation limits based on OCR text density to optimize inference speed.

---

### 3️⃣ Semantic Filtering & Delivery (src/08 – src/12)

- **Multi-Stage Verification Gate**  
  OCR and local LLM outputs are treated as **unstructured, noisy data** and passed through a cloud-based LLM gate (Gemini/GPT) for schema validation and filtering.

- **Semantic Scoring**  
  Sentence-Transformers compute cosine similarity between extracted jobs and dynamically updated resume profiles.

- **Delivery**  
  High-signal, relevance-ranked job alerts are delivered via Telegram.

- **Data Lifecycle Management**  
  Automated cleanup enforces a **24-hour retention policy** on transient image data to prevent disk exhaustion.

---

## 🔁 End-to-End Pipeline Flow

```mermaid
graph TD
    A[PDF Ingestion: Web / Telegram] --> B[Vision: Dynamic DPI Conversion]
    B --> C[Layout Detection: Smart Blocks]
    C --> D[OCR: Multilingual EasyOCR (GPU)]
    D --> E[NLP: Argos Translation]
    E --> F[Batching: Ollama Input Prep]
    F --> G[LLM: Local Extraction (Ollama)]
    G --> H[Verification: Cloud LLM Gate]
    H --> I[Matching: Semantic Embedding Score]
    I --> J[State: SQLite Deduplication]
    J --> K[Alert: Telegram Notification]
    K --> L[Cleanup: 24h Data Retention]
```

---

## 📐 Scale, Performance & Data Reduction

- **Processing Volume**  
  12–16 newspapers per run  
  ~15–20 GB transient image + OCR data per execution

- **Processing Speed**  
  End-to-end execution time: **40–60 minutes** per full batch  
  (hardware and OCR-engine dependent)

- **Resource Balancing**  
  Concurrent CPU/GPU utilization:
  - GPU: OCR + local LLM inference  
  - CPU: preprocessing, layout detection, batching

- **Data-to-Insight Compression**  
  ~20 GB of raw scan data → **~5 KB of structured job alerts**  
  **Compression ratio: ~4,000,000 : 1**

---

## ⚙️ Orchestration & Automation

The system uses a **two-layer orchestration model**:

- **`scheduler.sh`**  
  Time-based automation layer responsible for triggering scheduled runs.

- **`run_pipeline.sh`**  
  Pipeline control plane that:
  - Executes stages in deterministic order  
  - Skips completed steps via persisted state  
  - Supports safe retries and crash recovery  

This separation cleanly decouples *when* the system runs from *how* the pipeline executes.

---

## 🗄️ Data Persistence & State Tracking

SQLite databases are used for:

- Processed job tracking & deduplication  
- Resume matching history  
- LLM interaction logging  
- Crash-safe incremental progress  

This enables predictable behavior across repeated scheduled executions.

---

## 📊 Operational Notes

- Field-tested with **30+ consecutive days of zero-intervention scheduled runs**
- Comprehensive logging enables post-run auditing of:
  - OCR accuracy
  - Matching decisions
  - Hardware utilization

---

## 🔬 R&D & Benchmarking

This project includes extensive benchmarking of alternative engines:

- **PaddleOCR** – Deprecated due to CUDA compatibility issues
- **Tesseract** – Rejected for poor accuracy on low-contrast regional fonts
- **Argos Translate** – Selected for fast CPU-based batch translation, freeing GPU for OCR/LLM workloads

---

## 🛠️ Tech Stack

- **Language:** Python 3.x  
- **Vision & OCR:** OpenCV, EasyOCR, PyMuPDF (fitz)  
- **LLMs:** Ollama (local), Google Gemini / GPT (cloud validation)  
- **Embeddings:** Sentence-Transformers (BERT/RoBERTa)  
- **Automation & Scraping:** Selenium, Telethon (Telegram Client)  
- **Persistence:** SQLite3 (state, history, deduplication)  
- **Orchestration:** Bash (`scheduler.sh`, `run_pipeline.sh`)  
- **Notifications:** Telegram Bot API  

---

## 👤 Author

**Siddharth Singh**  
B.Tech Electrical Engineering (2024)  

Focus: Designing **resource-aware, hardware-constrained AI systems** that operate continuously under real-world constraints.
