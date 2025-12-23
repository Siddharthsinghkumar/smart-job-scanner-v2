# 🚀 Smart Job Scanner v2

Smart Job Scanner v2 is an autonomous job discovery and matching system that monitors daily newspapers, extracts job advertisements using OCR and LLMs, and delivers relevant opportunities directly via Telegram.

The system targets job notifications that are often published only in print (such as government and institutional postings) and is engineered with an emphasis on **predictable execution, hardware awareness, and operational stability**, allowing it to run continuously once scheduled.

---

## ✨ What This System Does

- Automatically downloads daily newspaper PDFs  
- Converts PDFs into images and detects advertisement layouts  
- Performs multilingual OCR (English & Hindi)  
- Extracts structured job data using local and cloud LLMs  
- **Performs semantic similarity matching with a normalized scoring system, generating relevance-weighted alerts**  
- Sends concise, personalized Telegram notifications  
- Maintains persistent state to prevent duplicate alerts  

---

## 🏗️ High-Level Workflow

1. **Ingestion** – Scheduled download of target newspapers  
2. **Vision Processing** – PDF-to-image conversion and layout detection  
3. **OCR** – Multilingual text extraction from detected blocks  
4. **Parsing** – Local LLM converts noisy OCR output into structured JSON  
5. **Multi-Stage Verification Gate** – Cloud LLM performs final schema validation and filtering on noisy OCR/LLM outputs  
6. **Matching** – Semantic comparison with resume profiles  
7. **Notification** – Telegram alerts with job metadata and relevance score  

---

## 🧠 Key Design Decisions

- **Hardware-Aware Scheduling**  
  Compute-heavy OCR and LLM workloads are scheduled during off-peak hours to minimize contention on shared systems and reduce sustained thermal load on consumer hardware.

- **Local-First, Cloud-Validated Processing**  
  Primary extraction and parsing are handled locally, with cloud LLMs used selectively as a final validation layer where accuracy matters most.

- **Structured Validation Gate**  
  OCR and local LLM outputs are treated as unstructured, noisy data and passed through a cloud-based LLM gate for schema validation, filtering, and consistency checks.

- **Stateful, Idempotent Execution**  
  SQLite-backed state tracking enables safe re-runs, incremental progress, and recovery from partial failures without reprocessing completed steps.

---

## 📐 Scale & Data Reduction

- **Processing Volume:**  
  Designed to process **12–16 daily newspapers per run**, generating approximately **15–20 GB of transient image and OCR data** during peak stages.

- **Processing Speed:**  
  Average end-to-end execution time of **40–60 minutes** for a full 16-newspaper batch (~20 GB raw throughput), depending on OCR engine and hardware profile.

- **Resource Balancing:**  
  Optimized for concurrent CPU/GPU utilization, balancing GPU-based OCR (EasyOCR) and LLM inference (Ollama) with multi-core CPU preprocessing stages.

- **Data-to-Insight Compression:**  
  Large volumes of raw, unstructured scan data are distilled into **a few kilobytes of structured, relevance-ranked job alerts**, achieving a high signal-to-noise ratio for human consumption.

---

## 🛠️ Tech Stack

- **Language:** Python 3.x  
- **OCR Engines:** EasyOCR, Tesseract (GPU/CPU variants evaluated)  
- **LLMs:** Ollama (local), Google Gemini / GPT (cloud)  
- **Embeddings:** Sentence-Transformers  
- **Automation & Scraping:** Selenium, Telethon (Telegram Client)  
- **Databases:** SQLite (state, history, and interaction tracking)  
- **Orchestration:** Bash scripts (`scheduler.sh`, `run_pipeline.sh`)  
- **Notifications:** Telegram Bot API  

---

## 📂 Project Structure

```
src/
  - Core numbered scripts implement the stable 12-stage processing pipeline
  - Experimental components form an OCR & LLM benchmarking suite,
    enabling A/B evaluation of engines and performance trade-offs

data/
  - Persistent storage for PDFs, images, OCR outputs, structured job data,
    resume profiles, and SQLite databases used for state tracking
  - Includes an automated 24-hour data lifecycle policy to maintain storage health

logs/
  - Execution logs for monitoring, performance analysis, and post-run inspection
```

---

## ⚙️ Orchestration & Automation

The system uses a two-layer orchestration model:

- **scheduler.sh**  
  Acts as the time-based automation layer, triggering scheduled executions and invoking the pipeline controller.

- **run_pipeline.sh**  
  Serves as the pipeline control plane, executing processing stages in deterministic order, skipping completed steps via persisted state, and supporting safe retries.

This separation cleanly decouples *when* the system runs from *how* the pipeline executes.

---

## 🗄️ Data Persistence & State Tracking

SQLite databases are used to:

- Track processed job postings and shortlist history  
- Prevent duplicate alerts across runs  
- Enable incremental processing and crash-safe recovery  
- Support auditing of matching logic and pipeline behavior  

---

## 📊 Operational Notes

- Field-tested with **30+ consecutive days of zero-intervention scheduled runs**  
- Comprehensive logging enables post-run auditing of extraction quality, matching decisions, and hardware performance  

---

## 🔐 License

This project is licensed under the **Apache License 2.0**.

---

## 👤 Author

**Siddharth Singh**  
Electrical Engineering background  
AI / ML • Systems Automation  

Focused on building predictable, resource-aware systems that operate continuously under real-world constraints.
