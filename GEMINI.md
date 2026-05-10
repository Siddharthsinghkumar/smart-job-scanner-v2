# Smart Job Scanner v2 - Project Context

Smart Job Scanner v2 is a sophisticated **OCR + LLM ETL pipeline** designed to automate the process of finding jobs from newspaper PDFs. It extracts job advertisements using computer vision and OCR, filters them using Large Language Models (LLMs), and sends notifications to Telegram.

## 🚀 Architecture Overview

The project follows a linear stagewise pipeline architecture. Each stage is a discrete Python script that processes data and writes intermediate results to `run_state/` or `data/`.

### Core Technologies
- **Python 3.10+**: Primary programming language.
- **EasyOCR**: Text extraction from detected image crops.
- **YOLO (Ultralytics)**: Object detection for identifying job advertisement blocks.
- **PyTorch (CUDA 12.1)**: Deep learning framework for vision models.
- **Google Gemini API**: High-level LLM filtering and data extraction.
- **Telethon / Telegram Bot API**: Notification and alert system.
- **Argos Translate**: Local machine translation for multi-language support.

---

## 🛠 Building and Running

### Setup
1. **Virtual Environment**: The project uses a local venv in `4_env/`.
   ```bash
   python -m venv 4_env
   source 4_env/bin/activate
   pip install -r requirements.txt
   ```
2. **Environment Variables**: Configure `.env` from `.env.example`.
   ```bash
   cp .env.example .env
   # Edit .env with API keys (Google Gemini, Telegram)
   ```

### Key Commands (via Makefile)
- `make run`: Executes the full end-to-end pipeline via `scripts/run_pipeline.py`.
- `make dry-run`: Validates configuration and paths without executing heavy stages.
- `make test`: Runs the `pytest` suite.
- `make health`: Runs `scripts/health_check.py` to verify environment and dependencies.

---

## 📋 Pipeline Stages

The pipeline consists of 11+ stages (ordered by `scripts/run_pipeline.sh` and `scripts/run_pipeline.py`):

1.  **Stage 01**: PDF to Page Images (`src/pipeline/stage01_pdf_to_images.py`)
2.  **Stage 02**: Block Detection (`src/pipeline/stage02_block_detection.py`)
3.  **Stage 03**: OCR (`src/pipeline/stage03_ocr.py`) - *Isolated subprocess version with VRAM management.*
4.  **Stage 05**: Translation (`src/pipeline/stage05_translation.py`)
5.  **Stage 07**: LLM Extraction (`src/pipeline/stage07_llm_extraction.py`)
6.  **Stage 10**: Notification (`src/pipeline/stage10_notification.py`)
7.  **Stage 11**: Cleanup (`src/pipeline/stage11_cleanup.py`)

---

## 📂 Directory Structure

- `/src/pipeline`: Core stage implementation scripts.
- `/src/vision`: Detector logic and YOLO integrations.
- `/src/ocr`: EasyOCR wrappers and preprocessing.
- `/configs`: JSON/YAML configurations for detectors and paths.
- `/data`:
    - `raw_pdfs`: Input newspaper PDFs.
    - `pdf2img`: Rendered page images.
    - `crops`: Cropped job advertisement images.
- `/run_state`: Intermediate manifests (`page_manifest.jsonl`, `ocr_manifest.jsonl`) and progress markers.
- `/logs`: Detailed logs per stage and per run.
- `/artifacts`: Model weights (`.pt`) and evaluation charts.
- `/tools`: Utility scripts for benchmarking, training, and dataset preparation.

---

## 💡 Development Conventions

- **Manifest-First**: Stages communicate via JSONL manifests in `run_state/`.
- **Validation**: Always run `make health` before a full run to ensure CUDA and API keys are ready.
- **Error Recovery**: Stage 03 (OCR) includes subprocess isolation and OOM recovery (downscaling).
- **VRAM Sensitivity**: The system is tuned for 4GB VRAM. Avoid running multiple GPU-intensive stages in parallel.
- **GPU Hygiene**: Before running benchmarks or GPU-intensive scripts, manually verify via `nvidia-smi` or `src/utils/gpu_utils.py` that no other processes (like `ollama`) are occupying the GPU to ensure accurate performance readings and stability.
- **Command Monitoring**: While running any pipeline or shell command, actively monitor the elapsed time. Compare it against expected durations from previous successful runs. Do not allow a command to run blindly for hours. If a command exceeds its expected time by more than 2x, investigate the logs immediately and terminate the process if it appears stuck or in a loop.
- **Testing**: New features should be validated with `pytest` in the `tests/` directory.

---

## 🛡️ Autonomous Watchdog & Stall Recovery

To prevent long-term idleness and ensure efficient resource usage, the following mandates are in effect for all autonomous operations:

### 1. Proactive Monitoring
- **Tailing Logs**: Every turn when a background process is active, the agent **MUST** tail the relevant log file (e.g., `session_log.txt`, `challenge_output.log`) to verify progress.
- **Progress Validation**: Compare current log state against previous turns. If the log has not grown and CPU/GPU usage is low for >10 minutes, the agent **MUST** investigate.
- **Heartbeat Check**: For unified scripts with heartbeat files (e.g., `heartbeat_bench_log.txt`), check the timestamp of the last entry.

### 2. Stall Mandates (2x Rule)
- **Time Threshold**: If a command exceeds its expected duration (based on history or ~100s/PDF) by more than **2x**, or runs for more than **30 minutes** without significant log updates, it is considered "Stalled."
- **Recovery Action**:
    1.  Immediately capture a snapshot of `top`, `nvidia-smi`, and the last 50 lines of the log.
    2.  Terminate the stalled process and its children (`kill -9`).
    3.  Analyze the snapshot to identify the bottleneck (e.g., VRAM OOM, deadlock, thermal throttling).
    4.  Propose a strategy change (e.g., reducing concurrency, clearing caches) before restarting.

### 3. Cleanup Hygiene
- **Session Start**: At the beginning of any multi-step task, check for and kill orphaned background processes from previous sessions to avoid resource contention.
- **Process Tracking**: Maintain an internal list of active PIDs and ensure they are reaped or accounted for before declaring a task complete.
- **VRAM Clearance**: Between major stages (e.g., Stage 3 to Stage 5), explicitly check that no GPU processes are lingering.

### 4. Active Status Reporting
- **Visibility Mandate**: The agent **MUST NOT** remain silent for more than 15 minutes while a background process is running.
- **Progress Snapshots**: If a task is expected to take >30 minutes, the agent **MUST** perform a "Status Turn" every ~10% of progress or ~15 minutes. A Status Turn consists of:
    1.  Reading the last 10 lines of the active log.
    2.  Checking CPU/GPU utilization.
    3.  Reporting the current percentage/ETA to the user.
- **No Blind Waiting**: Never issue a command that waits indefinitely for a multi-hour process without backgrounding it and providing a heartbeat.

---

## 🚨 Fatal Hardware Error Protocol (CUDA)

If a **"CUDA unknown error"** or any persistent CUDA initialization failure occurs:

1.  **IMMEDIATE STOP**: The agent **MUST** cease all execution attempts immediately. Do not attempt "fixes," environment changes, or CPU fallbacks for GPU-native tasks.
2.  **REPORT & SUGGEST**: Report the exact error string and the state of `nvidia-smi` (especially Power/VRAM metrics). Formally suggest a **System Restart** to the user.
3.  **NO CPU FALLBACK**: Never pivot GPU-intensive tasks (OCR, YOLO) to CPU unless specifically authorized by the user, as this violates performance targets and masks hardware instability.

---

## 🏗️ Architectural Integrity Mandates

### 1. Lazy Library Loading (Non-Negotiable)
- **Rule**: Heavy libraries (`torch`, `ultralytics`, `easyocr`, `stanza`) **MUST** be imported locally within worker functions or main execution blocks.
- **Why**: Global imports trigger CUDA context deadlocks in `spawn/fork` multiprocessing environments.
- **Violation Consequence**: Any script found with top-level heavy imports is considered broken and must be fixed before execution.
