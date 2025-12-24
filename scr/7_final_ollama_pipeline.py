#!/usr/bin/env python3
"""
File: 7_final_ollama_pipeline.py
Purpose:
    High-reliability batch extraction pipeline for newspaper job postings using Ollama (local LLM).
    This script processes OCR-converted text files, identifies job listings, and saves structured results.
    It includes automatic checkpointing, SQLite-based resume system, adaptive token windowing for speed,
    GPU inference via Ollama, crash-resilience, and detailed logging.

What it does:
    - Reads OCR-processed .txt pages from data/batch_inputs/
    - Sends text to local model (OpenHermes via Ollama) with job-extraction prompt
    - Saves full response & extracted jobs into organized folders
    - Stores processed pages in SQLite DB
    - Auto-resumes from last processed page on crash or restart
    - Supports optional `--fresh`, `--newspaper`, and `--debug` flags
    - Auto warm-up of model to avoid slow first-request

Key Features:
    ✅ Token-adaptive generation window based on text size (fast + stable)
    ✅ Checkpoint + SQLite ensures no lost progress
    ✅ Logs GPU stats & runtime summary
    ✅ Handles OCR noise & empty pages gracefully
    ✅ Fully offline, privacy-safe, runs on your GPU

Usage:
    python 7_final_ollama_pipeline.py                # resume & process everything (hybrid enabled by default)
    python 7_final_ollama_pipeline.py --fresh        # ignore checkpoint, start from page 1
    python 7_final_ollama_pipeline.py --debug        # show token window decisions & debugging info
    python 7_final_ollama_pipeline.py --newspaper TOI # only process Times of India pages
    python 7_final_ollama_pipeline.py --force        # ignore DB/checkpoint, process everything
    python 7_final_ollama_pipeline.py --no-hybrid    # disable hybrid CPU+GPU processing
    python 7_final_ollama_pipeline.py --gpu-endpoint http://gpu-server:11434 --cpu-endpoint http://cpu-server:11434

Dependencies:
    - Ollama running model `openhermes`
    - Python 3
    - SQLite3
    - psutil

"""
import os
import re
import logging
import requests
import time
import sys
import signal
import socket
import subprocess
import psutil
import sqlite3
import argparse
import json
import hashlib
import urllib3
import statistics
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

# ===== PERSISTENT REQUESTS SESSION WITH RETRIES =====
# global session to reuse TCP connection and keep-alive
SESSION = requests.Session()
# Retry constructor compatibility across urllib3 versions
retry_kwargs = dict(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
# prefer allowed_methods but fallback to method_whitelist if constructor expects it
try:
    retry_kwargs["allowed_methods"] = frozenset(["POST", "GET"])
    retry_strategy = Retry(**retry_kwargs)
except TypeError:
    retry_kwargs["method_whitelist"] = frozenset(["POST", "GET"])
    retry_strategy = Retry(**retry_kwargs)

adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=retry_strategy)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)

# 🆕 THREAD-LOCAL SESSIONS FOR CONCURRENT WORKERS
thread_local = threading.local()

def get_session():
    """🆕 Return a thread-local requests.Session with adapter mounted."""
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        # reuse same adapter config
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        # copy same defaults
        s.headers.update({"User-Agent": "ollama-batch/1.0"})
        thread_local.session = s
    return thread_local.session

# optional: a default short timeout to protect stuck requests (seconds)
# 🆕 INCREASED REQUEST TIMEOUT TO AVOID TRUNCATION
DEFAULT_REQUEST_TIMEOUT = 240  # seconds; increased from 120 for CPU endpoints

# 🆕 MODEL LATENCY PROFILING
MODEL_LATENCIES = []

def time_model_call(func, *f_args, **f_kwargs):
    """Time model calls and collect latency statistics. Avoid shadowing global 'args'."""
    t0 = time.time()
    try:
        res = func(*f_args, **f_kwargs)
    except Exception as e:
        # Ensure exceptions from the wrapped call bubble up but are logged
        logging.error(f"Exception inside timed call: {e}")
        raise

    dt = time.time() - t0
    try:
        MODEL_LATENCIES.append(dt)
        # log summary every 20 calls (only access CLI args via the global 'args' safely)
        if len(MODEL_LATENCIES) % 20 == 0:
            if len(MODEL_LATENCIES) >= 20:
                p95 = statistics.quantiles(MODEL_LATENCIES, n=20)[-1]
            else:
                p95 = max(MODEL_LATENCIES) if MODEL_LATENCIES else 0
            avg_time = statistics.mean(MODEL_LATENCIES) if MODEL_LATENCIES else 0
            logging.info(f"Model latencies (n={len(MODEL_LATENCIES)}): avg {avg_time:.2f}s, p95 {p95:.2f}s")
            # only print debug console when args exists and debug is enabled
            if 'args' in globals() and getattr(args, 'debug', False):
                console(f"⏱️  Model latencies (n={len(MODEL_LATENCIES)}): avg {avg_time:.2f}s, p95 {p95:.2f}s")
    except Exception as e:
        # defensive: never let reporting break the main pipeline
        logging.debug(f"time_model_call reporting failed: {e}")

    return res

# logging helper
total_files = 0
total_jobs = 0
errors = []

# 🆕 DATABASE WRITE LOCK FOR THREAD SAFETY
db_write_lock = threading.Lock()

# ───── Config ─────
OLLAMA_MODEL = "openhermes"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
PORT = 11434

# 🆕 JOB KEYWORDS FOR PRE-FILTERING
JOB_KEYWORDS = [
    "vacancy", "vacancies", "apply", "applications", "walk-in", "walk in", 
    "interview", "recruitment", "job", "posts", "post", "salary", "walk-in-interview",
    "appointment", "employment", "hiring", "position", "opening", "opportunity",
    "requirement", "wanted", "needed", "candidate", "eligible", "qualification"
]

batch_input_dir = Path("data/test_data")
output_dir = Path("data/batch_output")
output_dir.mkdir(parents=True, exist_ok=True)

jobs_root = Path("data/Jobs_found_final")
jobs_root.mkdir(parents=True, exist_ok=True)

# ───── Database & Checkpoint System ─────
db_file = Path("data/processing_state.db")
checkpoint_file = Path("data/processing_checkpoint.txt")

serve_proc = None
start_time = datetime.now()

# ───── Rich Console Setup ─────
RICH_CONSOLE = Console()

# ───── Prompt Template ─────
PROMPT_TEMPLATE = """You are an expert assistant that extracts job postings from raw OCR text.

### Task
From the input text, identify *all job postings* (ignore ads, tenders, unrelated notices).

Return results in **two sections**:

---

📌 **Concise Listings**
- Format each job in **one line**:
  Employer | Position | Salary | Deadline | Contact

📝 **Full Job Descriptions**
- For each job, include a **detailed expanded version**:
  Employer:
  Position:
  Eligibility / Requirements:
  Remuneration:
  Deadline:
  Contact:
  Full Posting Text: (retain all relevant details, do not omit)

---

### Example Output
📌 Concise Listings
- Indraprastha Power Corp | Multi Tasking Staff | Rs. 22,500/month | July 23, 2025 | URC 35 Inf Bde Shankar Vihar

📝 Full Job Descriptions
Employer: Indraprastha Power Corporation Limited
Position: Multi Tasking Staff
Eligibility: Matriculation, computer knowledge, dependants preferred
Remuneration: Rs. 22,500 per month
Deadline: July 23, 2025
Contact: URC 35 Inf Bde Shankar Vihar
Full Posting Text: "Application are invited for hiring of 02 x Multi Tasking Staff (MTS)..."

---

Now process the following input text and output in the above format:
[NEWSPAPER_CONTENT]
{content}
[/NEWSPAPER_CONTENT]
""".strip()

# 🆕 BATCH PROCESSING PROMPT TEMPLATE
BATCH_PROMPT_TEMPLATE = """You are an expert assistant that extracts job postings from raw OCR text.

### Task
Process multiple newspaper pages in one batch. For EACH file below, identify *all job postings* (ignore ads, tenders, unrelated notices).

Return results for EACH file in this JSON format:
{
  "files": [
    {
      "filename": "filename1.txt",
      "concise_listings": ["Employer | Position | Salary | Deadline | Contact", ...],
      "full_descriptions": ["Employer: ...\nPosition: ...", ...]
    },
    {
      "filename": "filename2.txt", 
      "concise_listings": [...],
      "full_descriptions": [...]
    }
  ]
}

IMPORTANT: Return ONLY valid JSON, no other text.

Now process these files:

===BATCH_START===
{content}
===BATCH_END===
""".strip()

# ───── Logging Setup ─────
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_name = f"final_ollama_run_{timestamp}.log"

logging.basicConfig(
    filename=log_dir / log_name,
    filemode='a',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# 🆕 QUIET TERMINAL OUTPUT SETUP
def setup_quiet_console():
    """Set up console handler for quiet operation (only warnings/errors by default)."""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only show warnings/errors in terminal
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

def console(msg, level="info"):
    """🆕 Controlled console output - only shows in debug mode or for warnings/errors."""
    if args.debug:
        print(msg)
    elif level == "warning":
        logging.warning(msg)
    elif level == "error":
        logging.error(msg)
    # Otherwise, the message only goes to log file via logging.info()

# ───── Performance Optimization Functions ─────
def file_md5_text(content: str) -> str:
    """🆕 Generate MD5 hash of text content for deduplication."""
    return hashlib.md5(content.encode('utf-8', errors='ignore')).hexdigest()

def read_and_prefilter_files(file_paths, max_workers=None):
    """🆕 Read files in parallel, fast pre-filter, return list of (Path, content) that passed filter.
       Mark skipped files in DB to keep consistent state.
    """
    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 4))
    results = []
    skipped_count = 0

    def _read(path):
        try:
            text = path.read_text(encoding='utf-8').replace("\x00", "").strip()
            return (path, text)
        except Exception as e:
            logging.error(f"Error reading {path}: {e}")
            return (path, "")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_read, p): p for p in file_paths}
        for fut in as_completed(futures):
            p = futures[fut]
            path, content = fut.result()
            # cheap prefilter
            if not content:
                mark_file_processed(path, has_jobs=False, job_count=0, skipped_fast=True)
                skipped_count += 1
                continue
            # use your likely_contains_job (keeps debug behavior)
            if not likely_contains_job(content):
                mark_file_processed(path, has_jobs=False, job_count=0, skipped_fast=True)
                skipped_count += 1
                continue
            # passed -> keep
            results.append((path, content))
    if args.debug:
        console(f"🔎 Prefilter result: {len(results)} files passed, {skipped_count} skipped")
    return results

def likely_contains_job(text):
    """🆕 Fast pre-filter to skip pages that likely have no job postings."""
    if not text:
        return False
    
    # 🆕 In testing (--force), don't prefilter at all
    if 'args' in globals() and getattr(args, 'force', False):
        return True

    lower_text = text.lower()
    # Count keyword matches
    matches = sum(1 for keyword in JOB_KEYWORDS if keyword in lower_text)
    
    # Tune sensitivity: require at least 2 keyword matches to reduce false negatives
    contains_job = matches >= 2
    
    if args.debug and not contains_job:
        console(f"🔍 Pre-filter: Skipping - only {matches} job keyword(s) found")
    
    return contains_job

def create_file_batches(prefetched_files, max_chars=5000, max_files=2):
    """🆕 Batch files together to reduce API calls - using smaller batches for better reliability."""
    batches = []
    current_batch = []
    current_chars = 0
    
    for file_path, content in prefetched_files:
        content_length = len(content)
        
        # Skip if content too large for batching
        if content_length > max_chars * 0.7:  # If single file is >70% of batch limit
            if current_batch:  # Finish current batch first
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            # Process large file individually
            batches.append([(file_path, content)])
            continue
            
        # Start new batch if limits exceeded
        if (current_batch and 
            (current_chars + content_length > max_chars or len(current_batch) >= max_files)):
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        
        current_batch.append((file_path, content))
        current_chars += content_length
    
    if current_batch:
        batches.append(current_batch)
    
    if args.debug:
        console(f"📦 Created {len(batches)} batches from {len(prefetched_files)} files")
        for i, batch in enumerate(batches):
            total_chars = sum(len(content) for _, content in batch)
            console(f"  Batch {i+1}: {len(batch)} files, {total_chars} chars")
    
    return batches

def format_batch_content(batch_files):
    """🆕 Format multiple files into a single batch prompt."""
    batch_content = []
    for file_path, content in batch_files:
        batch_content.append(f"FILE: {file_path.name}")
        batch_content.append("---CONTENT_START---")
        batch_content.append(content)
        batch_content.append("---CONTENT_END---")
        batch_content.append("")  # Empty line between files
    
    return "\n".join(batch_content)

def safe_format_prompt(template, **kwargs):
    """Safely substitute only the {content} placeholder without interpreting other braces."""
    content = kwargs.get("content", "")
    if "{content}" not in template:
        # nothing to replace; return template unchanged
        return template
    try:
        # only replace the single {content} token — avoids KeyError from other braces
        return template.replace("{content}", content)
    except Exception as e:
        logging.error("safe_format_prompt replacement error: %s", e)
        Path("logs/template_error.txt").write_text(
            f"Template: {template}\n\n---CONTENT---\n{content}", encoding="utf-8"
        )
        console(f"❌ Template replacement error: {e}", "error")
        raise

def parse_batch_response(response_text, batch_files):
    """Robust JSON extractor: try various safe strategies to recover a JSON object
       containing { "files": [...] } from noisy model outputs.
    """
    # 🆕 Save raw response for debugging before any parsing attempts
    raw_response_dir = Path("logs/raw_responses")
    raw_response_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    first_filename = batch_files[0][0].name if batch_files else "unknown"
    raw_response_file = raw_response_dir / f"raw_response_{first_filename}_{timestamp}.txt"
    raw_response_file.write_text(response_text, encoding="utf-8")
    
    try:
        raw = response_text.strip()

        # 1) Drop Markdown fences like ```json ... ``` or ``` ... ```
        if raw.startswith("```"):
            # remove the first fence and trailing fence if present
            fence_end = raw.find("```", 3)
            if fence_end != -1:
                raw = raw.split("```", 2)[1].strip()

        # 2) Try to find the largest {...} block (first { to last })
        if "{" in raw and "}" in raw:
            first = raw.find("{")
            last = raw.rfind("}")
            candidate = raw[first:last+1]
            try:
                data = json.loads(candidate)
            except Exception:
                # try a safer approach below
                data = None
        else:
            data = None

        # 3) If that failed, try to locate the `"files"` key and parse a surrounding object
        if not data:
            m = re.search(r'("files"\s*:\s*\[)', raw)
            if m:
                idx = m.start()
                # find previous '{' before idx
                start = raw.rfind("{", 0, idx)
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = raw[start:end+1]
                    try:
                        data = json.loads(candidate)
                    except Exception:
                        data = None

        # 4) Last attempt: try json.loads on the whole text (some endpoints return plain JSON)
        if not data:
            try:
                data = json.loads(raw)
            except Exception:
                data = None

        # 5) 🆕 PLAIN TEXT FALLBACK: If JSON parsing fails, try to extract from plain text
        if not data:
            # Try to salvage from plain text by looking for FILE: markers and listings
            file_results = {}
            lines = raw.split('\n')
            current_file = None
            current_listings = []
            
            for i, line in enumerate(lines):
                # Look for FILE: markers
                if line.strip().startswith("FILE:"):
                    # Save previous file's results
                    if current_file and current_listings:
                        file_results[current_file] = {
                            "concise_listings": current_listings,
                            "full_descriptions": []
                        }
                    
                    # Start new file
                    current_file = line.strip().replace("FILE:", "").strip()
                    current_listings = []
                
                # Look for concise listings (lines starting with "- ")
                elif line.strip().startswith("- "):
                    current_listings.append(line.strip())
            
            # Save the last file's results
            if current_file and current_listings:
                file_results[current_file] = {
                    "concise_listings": current_listings,
                    "full_descriptions": []
                }
            
            if file_results:
                console(f"🔄 Using plain text fallback for {len(file_results)} files", "warning")
                return file_results

        if not data:
            # Debug: show a helpful slice of the raw response
            console(f"❌ Failed to parse batch response: unable to locate valid JSON", "error")
            console(f"💾 Raw response saved to: {raw_response_file}")
            if args.debug:
                console(f"📄 Raw response (first 2000 chars):\n{response_text[:2000]}")
            return {}

        # Map results to filenames
        file_results = {}
        for file_data in data.get("files", []):
            filename = file_data.get("filename", "")
            file_results[filename] = {
                "concise_listings": file_data.get("concise_listings", []),
                "full_descriptions": file_data.get("full_descriptions", [])
            }

        return file_results

    except Exception as e:
        console(f"❌ Exception while parsing batch response: {e}", "error")
        console(f"💾 Raw response saved to: {raw_response_file}")
        if args.debug:
            console(f"📄 Raw response (first 2000 chars):\n{response_text[:2000]}")
        return {}

# ───── Hybrid Processing Functions ─────
def split_batches_for_hybrid(batches, char_threshold=3000):
    """🆕 Split batches into gpu_batches and cpu_batches using a heuristic:
       If total chars in a batch > char_threshold → GPU, else CPU.
    """
    gpu_batches = []
    cpu_batches = []
    for b in batches:
        total_chars = sum(len(content) for _, content in b)
        if total_chars > char_threshold:
            gpu_batches.append(b)
        else:
            cpu_batches.append(b)
    return gpu_batches, cpu_batches

def run_batches_on_endpoint(endpoint, batches, endpoint_name="endpoint"):
    """🆕 Send batches sequentially to a specific endpoint (used by ThreadPoolExecutor)."""
    if args.debug:
        console(f"🔀 Starting {endpoint_name} worker with {len(batches)} batches")
    
    for batch_index, batch in enumerate(batches):
        if args.debug:
            console(f"  {endpoint_name}: Processing batch {batch_index + 1}/{len(batches)}")
            
        if len(batch) > 1:
            batch_content = format_batch_content(batch)
            # 🆕 USE SAFE FORMATTING WITH ERROR HANDLING
            batch_prompt = safe_format_prompt(BATCH_PROMPT_TEMPLATE, content=batch_content)
            # 🆕 USE KEYWORD ARGUMENTS FOR CLARITY
            response_text = time_model_call(
                query_model_safe, batch_prompt, len(batch_content),
                is_batch=True, endpoint=endpoint
            )
            if response_text:
                file_results = parse_batch_response(response_text, batch)
                for file_path, content in batch:
                    process_file_result(file_path, content, file_results.get(file_path.name, {}))
            else:
                # fallback to single-file processing using this endpoint
                if args.debug:
                    console(f"  {endpoint_name}: Batch failed, falling back to single-file processing")
                for file_path, content in batch:
                    # force per-file call to the specific endpoint
                    # 🆕 USE SAFE FORMATTING WITH ERROR HANDLING
                    prompt = safe_format_prompt(PROMPT_TEMPLATE, content=content)
                    # 🆕 USE KEYWORD ARGUMENTS FOR CLARITY
                    output_text = time_model_call(
                        query_model_safe, prompt, len(content),
                        is_batch=False, endpoint=endpoint
                    )
                    if output_text:
                        newspaper_name = file_path.stem.split("_p")[0]
                        subdir = output_dir / newspaper_name
                        subdir.mkdir(parents=True, exist_ok=True)
                        jobs_subdir = jobs_root / newspaper_name
                        jobs_subdir.mkdir(parents=True, exist_ok=True)
                        process_model_response(file_path, newspaper_name, subdir, jobs_subdir, output_text)
        else:
            # single-file batch
            for file_path, content in batch:
                # 🆕 USE SAFE FORMATTING WITH ERROR HANDLING
                prompt = safe_format_prompt(PROMPT_TEMPLATE, content=content)
                # 🆕 USE KEYWORD ARGUMENTS FOR CLARITY
                output_text = time_model_call(
                    query_model_safe, prompt, len(content),
                    is_batch=False, endpoint=endpoint
                )
                if output_text:
                    newspaper_name = file_path.stem.split("_p")[0]
                    subdir = output_dir / newspaper_name
                    subdir.mkdir(parents=True, exist_ok=True)
                    jobs_subdir = jobs_root / newspaper_name
                    jobs_subdir.mkdir(parents=True, exist_ok=True)
                    process_model_response(file_path, newspaper_name, subdir, jobs_subdir, output_text)

def process_batches_concurrent(gpu_batches, cpu_batches, gpu_endpoint, cpu_endpoint, max_workers=2):
    """🆕 Run gpu_batches on gpu_endpoint and cpu_batches on cpu_endpoint in parallel."""
    jobs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        if gpu_batches:
            jobs.append(ex.submit(run_batches_on_endpoint, gpu_endpoint, gpu_batches, "GPU"))
        if cpu_batches:
            jobs.append(ex.submit(run_batches_on_endpoint, cpu_endpoint, cpu_batches, "CPU"))
        # wait for jobs to finish
        for fut in jobs:
            try:
                fut.result()
            except Exception as e:
                console(f"❌ Error in concurrent batch processing: {e}", "error")
                logging.error(f"Concurrent batch processing error: {e}")

# ───── Database Helpers ─────
def init_database():
    """🆕 Initialize SQLite database for tracking processing state with improved concurrency."""
    # open with a slightly larger timeout to avoid transient "database is locked"
    conn = sqlite3.connect(db_file, timeout=30)
    cursor = conn.cursor()
    # Use WAL journal mode to improve concurrent read/writes from multiple threads
    try:
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
    except Exception as e:
        logging.warning(f"⚠️ Could not set PRAGMAs: {e}")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_files (
            file_path TEXT PRIMARY KEY,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            has_jobs BOOLEAN DEFAULT FALSE,
            job_count INTEGER DEFAULT 0,
            skipped_fast BOOLEAN DEFAULT FALSE
        )
    ''')
    conn.commit()
    conn.close()

def mark_file_processed(file_path, has_jobs=False, job_count=0, skipped_fast=False):
    """🆕 Mark a file as processed in the database with thread safety.

    NOTE: If the script is run with --force, DB writes are skipped so you can run
    baseline/test runs without changing the database.
    """
    # If args exists and force is set, skip writing to DB (baseline/test mode).
    try:
        if 'args' in globals() and getattr(args, 'force', False):
            if getattr(args, 'debug', False):
                console(f"⚡ Skipping DB write for {file_path} because --force is set")
            return
    except Exception:
        # If anything goes wrong checking args, proceed to write the DB as before.
        pass

    # ensure only one thread writes at a time
    with db_write_lock:
        conn = sqlite3.connect(db_file, timeout=30)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO processed_files 
            (file_path, processed_at, has_jobs, job_count, skipped_fast)
            VALUES (?, ?, ?, ?, ?)
        ''', (str(file_path), datetime.now(), has_jobs, job_count, skipped_fast))
        conn.commit()
        conn.close()

def get_processed_files():
    """🆕 Get all processed files from database with thread safety."""
    # small read lock to avoid racing with writes
    with db_write_lock:
        if not db_file.exists():
            return set()
        conn = sqlite3.connect(db_file, timeout=30)
        cursor = conn.cursor()
        cursor.execute('SELECT file_path FROM processed_files')
        processed = {Path(row[0]) for row in cursor.fetchall()}
        conn.close()
    return processed

# ───── Checkpoint Helpers ─────
def save_checkpoint_for_next(current_file):
    """Save the NEXT file as checkpoint (not current)."""
    batch_files = sorted(batch_input_dir.rglob("*.txt"))
    try:
        idx = batch_files.index(current_file)
        next_file = batch_files[idx + 1] if idx + 1 < len(batch_files) else ""
    except ValueError:
        next_file = ""
    
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            f.write(f"# Checkpoint {datetime.now()}\n{next_file}")
        logging.info(f"💾 Checkpoint saved for next: {next_file}")
    except Exception as e:
        logging.error(f"❌ Error saving checkpoint: {e}")

def load_checkpoint():
    """Load the next file to process from checkpoint."""
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                for line in reversed(lines):
                    if not line.startswith('#'):
                        return line
        except Exception as e:
            logging.error(f"❌ Error loading checkpoint: {e}")
    return None

def clear_checkpoint():
    """Clear checkpoint file when processing is complete."""
    try:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logging.info("🧹 Checkpoint cleared")
    except Exception as e:
        logging.error(f"❌ Error clearing checkpoint: {e}")

# ───── Improved Model Helpers ─────
def wait_for_model(max_wait=60, endpoint=None):
    """Wait until the model is ready with exponential backoff."""
    endpoint = endpoint or OLLAMA_ENDPOINT
    console(f"⌛ Waiting for model at {endpoint} to become ready...")
    for attempt in range(max_wait):
        try:
            # 🆕 USE THREAD-LOCAL SESSION
            res = get_session().post(
                endpoint,
                json={"model": OLLAMA_MODEL, "prompt": "Say hello", "stream": False},
                timeout=3
            )
            if res.status_code == 200:
                logging.info(f"✅ Model at {endpoint} is ready.")
                console(f"✅ Model at {endpoint} is ready.")
                return True
        except:
            pass
        
        # Exponential backoff: 1, 2, 4, 8 seconds...
        sleep_time = min(2 ** attempt, 10)
        time.sleep(sleep_time)
    
    return False

def query_model_safe(prompt_text, content_length, retries=3, is_batch=False, endpoint=None):
    """Send prompt to model with retry logic and auto-restart.
       endpoint: custom endpoint (e.g. CPU server). If None, uses OLLAMA_ENDPOINT.
    """
    endpoint = endpoint or OLLAMA_ENDPOINT
    
    # 🆕 IMPROVED ADAPTIVE TOKEN WINDOW with batch support
    if is_batch:
        # Larger token window for batches
        if content_length < 8000:
            max_tokens = 3000
            token_setting = "BATCH_SMALL"
        elif content_length < 20000:
            max_tokens = 6000  
            token_setting = "BATCH_MEDIUM"
        else:
            max_tokens = 8000
            token_setting = "BATCH_LARGE"
    else:
        # Single file token window
        if content_length < 2000:
            max_tokens = 1200
            token_setting = "SMALL"
        elif content_length < 6000:
            max_tokens = 2500
            token_setting = "MEDIUM"  
        else:
            max_tokens = 4000
            token_setting = "LARGE"
    
    # 🆕 Only log token info in debug mode to reduce spam
    if args.debug:
        mode = "BATCH" if is_batch else "SINGLE"
        endpoint_name = "default" if endpoint == OLLAMA_ENDPOINT else endpoint
        console(f"📊 {mode} Content: {content_length} chars → {token_setting} window ({max_tokens} tokens) → {endpoint_name}")
    logging.info(f"📊 Content length: {content_length} chars → Using {token_setting} token window ({max_tokens} tokens) → {endpoint}")
    
    for attempt in range(retries):
        try:
            # 🆕 USE THREAD-LOCAL SESSION
            session = get_session()
            
            # 🆕 IMPROVED OPTIONS: Don't use stop token for batch/JSON outputs
            options = {
                "num_predict": max_tokens,
                "temperature": 0.0
            }
            # Avoid a restrictive stop token when expecting structured / JSON outputs,
            # which can cause premature truncation. If you really want a stop token for
            # short single-file outputs, you could set it only for non-batch calls.
            if not is_batch:
                # optional — keep the simple newline-double-stop for small single-file outputs
                options["stop"] = ["\n\n"]
            # else: for batch calls expecting JSON, do not set stop (let model finish).
            
            res = session.post(
                endpoint,
                json={
                    "model": OLLAMA_MODEL, 
                    "prompt": prompt_text, 
                    "stream": False,
                    "options": options
                },
                timeout=DEFAULT_REQUEST_TIMEOUT
            )
            res.raise_for_status()
            
            # 🆕 IMPROVED RESPONSE PARSING
            try:
                # prefer JSON->response if available
                j = res.json()
                text = j.get("response") or j.get("text") or j.get("output") or ""
                # fallback: sometimes the body is plain text
                if not text:
                    text = res.text
            except Exception:
                text = res.text

            response_text = text.strip()
            
            # 🆕 LOG RESPONSE LENGTH FOR DIAGNOSTICS
            logging.info(f"[MODEL_RESP] endpoint={endpoint} len={len(response_text)}")
            if args.debug:
                console(f"[MODEL_RESP] endpoint={endpoint} len={len(response_text)}")
            
            if response_text:
                return response_text
            else:
                if args.debug:
                    console(f"⚠️ Empty response from {endpoint}, retry {attempt + 1}/{retries}")
                logging.warning(f"⚠️ Empty response from {endpoint}, retry {attempt + 1}/{retries}")
                
        except requests.exceptions.RequestException as e:
            logging.warning(f"Model request error at {endpoint} (attempt {attempt+1}): {e}")
            if attempt == 0:
                # quick hint: try restarting the server once (only for default endpoint)
                if endpoint == OLLAMA_ENDPOINT:
                    if is_port_open(PORT):
                        logging.info("Port seems open; retrying request.")
                    else:
                        logging.warning("Port closed: attempting to restart Ollama.")
                        shutdown_ollama(force=True)
                        start_ollama_serve()
                        if not wait_for_model(20):
                            logging.error("Model failed to come up after restart.")
                else:
                    # For custom endpoints, just wait and retry
                    logging.info(f"Custom endpoint {endpoint} may be temporarily unavailable")
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            logging.error(f"❌ Model query failed at {endpoint} (attempt {attempt + 1}/{retries}): {e}")
            console(f"❌ Model query failed at {endpoint} (attempt {attempt + 1}/{retries}): {e}")
        
        # Exponential backoff between retries
        time.sleep(2 * (attempt + 1))
    
    console(f"❌ Failed after all retries at {endpoint}")
    return ""

# ───── Other Helpers ─────
def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_ollama_serve():
    """Start Ollama serve if not already running."""
    global serve_proc
    logging.info("🧠 Starting Ollama model server...")
    if is_port_open(PORT):
        logging.warning("⚠️ Port 11434 already in use. Assuming Ollama is running.")
        console("⚠️ Ollama already running.")
        return
    serve_proc = subprocess.Popen(["ollama", "serve"], preexec_fn=os.setpgrp)
    logging.info("✅ Ollama server launched.")
    console("✅ Ollama server launched.")

def shutdown_ollama(force=False):
    """Shut down Ollama server cleanly."""
    global serve_proc
    if serve_proc and serve_proc.poll() is None:
        logging.info("🛑 Shutting down Ollama server...")
        console("🛑 Shutting down Ollama server...")
        os.killpg(os.getpgid(serve_proc.pid), signal.SIGTERM)
        serve_proc.wait(timeout=10)
        logging.info("✅ Ollama terminated.")
        console("✅ Ollama terminated.")
    elif force:
        # Kill any lingering Ollama processes
        logging.info("🛑 Force shutdown of Ollama processes...")
        console("🛑 Force shutdown of Ollama processes...")
        subprocess.run(["pkill", "-f", "ollama"])
        logging.info("✅ Ollama terminated (force).")
        console("✅ Ollama terminated (force).")

def handle_exit(signum, frame):
    console("🛑 Caught exit signal, cleaning up...")
    logging.info("🛑 Caught exit signal, cleaning up...")

    # Final check of running models
    check_ollama_ps()

    shutdown_ollama()
    sys.exit(0)

def check_ollama_ps():
    try:
        out = subprocess.check_output(["ollama", "ps"]).decode()
        console("📊 Ollama models running:\n" + out)
        logging.info(f"Ollama ps:\n{out}")
    except Exception as e:
        console(f"⚠️ Could not check ollama ps: {e}")

def log_gpu_status():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
        )
        util, mem_used, mem_total = out.decode().strip().split(", ")
        logging.info(f"🖥️ Ollama GPU Status → Utilization: {util}%, VRAM: {mem_used}/{mem_total} MiB")
        if args.debug:
            console(f"🖥️ GPU: {util}% util, {mem_used}/{mem_total} MiB VRAM")
    except Exception as e:
        logging.warning(f"⚠️ Could not fetch GPU usage: {e}")
        if args.debug:
            console("⚠️ Could not fetch GPU usage")

def print_progress(processed_count, total_count, start_time):
    """Print progress with ETA - only in debug mode or as tqdm."""
    if total_count == 0:
        return
    
    if args.debug:  # Only show detailed progress in debug mode
        elapsed = datetime.now() - start_time
        percent = (processed_count / total_count) * 100
        elapsed_seconds = elapsed.total_seconds()
        
        if processed_count > 0:
            time_per_file = elapsed_seconds / processed_count
            remaining_files = total_count - processed_count
            eta_seconds = time_per_file * remaining_files
            eta_str = f"{eta_seconds // 60:.0f}m {eta_seconds % 60:.0f}s"
        else:
            eta_str = "calculating..."
        
        console(f"📊 Progress: {processed_count}/{total_count} ({percent:.1f}%) - ETA: {eta_str}")

# ───── Main Processing ─────
def main():
    global total_files, total_jobs, errors

    console("🚀 Starting Ollama batch processing...")
    logging.info("🚀 Starting Ollama batch processing...")

    # Fix: Argument precedence - default to resume, override with fresh
    resume_mode = not args.fresh

    # Initialize database
    init_database()

    # Get all batch files
    batch_files = sorted(batch_input_dir.rglob("*.txt"))
    if not batch_files:
        console("⚠️ No batch files found in data/batch_inputs/")
        return

    # Filter by newspaper FIRST (before checkpoint logic)
    if args.newspaper:
        batch_files = [f for f in batch_files if args.newspaper in str(f)]
        if not batch_files:
            console(f"⚠️ No files found for newspaper: {args.newspaper}")
            return

    # Get already processed files from database
    already_processed = get_processed_files()
    
    # Load checkpoint to find where we left off
    checkpoint = load_checkpoint()
    
    files_to_process = []
    start_index = 0
    
    # Fix: Use resume_mode instead of checking args.fresh directly
    if resume_mode and checkpoint:
        # Find the position of the checkpoint file in the sorted list
        checkpoint_path = Path(checkpoint)
        try:
            start_index = batch_files.index(checkpoint_path)
            console(f"🔁 Resuming from checkpoint: {checkpoint_path.name}")
            logging.info(f"🔁 Resuming from checkpoint: {checkpoint_path.name}")
        except ValueError:
            # Checkpoint file not found in current batch, start from beginning
            console("⚠️ Checkpoint file not found in filtered list, starting from beginning")
            logging.warning("⚠️ Checkpoint file not found in filtered list, starting from beginning")
            start_index = 0
    elif args.fresh:
        console("🆕 Fresh start requested, ignoring checkpoint")
        logging.info("🆕 Fresh start requested")
        start_index = 0
    else:
        console("ℹ️  Resume by default, but no checkpoint found - starting from beginning")
        logging.info("ℹ️  Resume by default but no checkpoint found")
    
    # 🆕 FORCE MODE: Ignore DB and checkpoint, process all files
    if args.force:
        # Force-run: ignore DB and checkpoint; process all filtered files
        if args.debug:
            console("⚡ Force flag set: ignoring DB/checkpoint — processing all found files")
        files_to_process = list(batch_files[start_index:])  # start_index still honors --newspaper / fresh logic
    else:
        for i in range(start_index, len(batch_files)):
            file_path = batch_files[i]
            if file_path not in already_processed:
                files_to_process.append(file_path)

    total_processed_count = len(already_processed)
    console(f"📊 Total: {len(batch_files)} files, {total_processed_count} processed, {len(files_to_process)} remaining")
    logging.info(f"📊 Files - Total: {len(batch_files)}, Processed: {total_processed_count}, Remaining: {len(files_to_process)}")

    if not files_to_process:
        console("✅ All files have already been processed!")
        clear_checkpoint()
        return

    # 🆕 PERFORMANCE OPTIMIZATION: Parallel file reading and pre-filtering
    console("🔍 Prefetching and pre-filtering files...")
    passed_files = read_and_prefilter_files(files_to_process, max_workers=args.workers)
    if not passed_files:
        console("✅ Nothing to send to model after prefilter. Exiting.")
        return

    # 🆕 PERFORMANCE OPTIMIZATION: Create batches for processing - USING SMALLER BATCHES
    batches = create_file_batches(passed_files, max_chars=5000, max_files=2)
    
    # 🆕 IMPROVEMENT: Model warm-up prompt to avoid cold start penalty
    console("🔥 Warming up model with initial prompt...")
    # 🆕 USE KEYWORD ARGUMENTS FOR CLARITY
    warm_up_result = time_model_call(
        query_model_safe,
        "just say ready",
        len("just say ready"),
        retries=1,
        is_batch=False,
    )
    if warm_up_result:
        console("✅ Model warmed up and ready")
    else:
        console("⚠️ Model warm-up had issues, but continuing...")

    # 🆕 HYBRID PROCESSING: Split batches between CPU and GPU endpoints (enabled by default)
    if args.hybrid:
        gpu_endpoint = args.gpu_endpoint or OLLAMA_ENDPOINT
        cpu_endpoint = args.cpu_endpoint or OLLAMA_ENDPOINT
        
        # Ensure endpoints are ready
        if gpu_endpoint != OLLAMA_ENDPOINT:
            console(f"🔌 Checking GPU endpoint: {gpu_endpoint}")
            if not wait_for_model(endpoint=gpu_endpoint):
                console(f"❌ GPU endpoint {gpu_endpoint} not ready, falling back to default")
                gpu_endpoint = OLLAMA_ENDPOINT
        
        if cpu_endpoint != OLLAMA_ENDPOINT:
            console(f"🔌 Checking CPU endpoint: {cpu_endpoint}")
            if not wait_for_model(endpoint=cpu_endpoint):
                console(f"❌ CPU endpoint {cpu_endpoint} not ready, falling back to default")
                cpu_endpoint = OLLAMA_ENDPOINT
        
        # split batches heuristically - USING SMALLER THRESHOLD
        gpu_batches, cpu_batches = split_batches_for_hybrid(batches, char_threshold=3000)
        if args.debug:
            console(f"🔀 Hybrid mode: {len(gpu_batches)} GPU batches, {len(cpu_batches)} CPU batches")
            if gpu_batches:
                gpu_chars = sum(sum(len(c) for _, c in b) for b in gpu_batches)
                console(f"  GPU batches: {gpu_chars} total chars")
            if cpu_batches:
                cpu_chars = sum(sum(len(c) for _, c in b) for b in cpu_batches)
                console(f"  CPU batches: {cpu_chars} total chars")
        
        # run concurrently
        process_batches_concurrent(gpu_batches, cpu_batches, gpu_endpoint, cpu_endpoint, max_workers=min(2, args.workers))
    else:
        # 🆕 RICH PROGRESS BAR FOR BATCH PROCESSING (non-hybrid mode)
        if not args.debug:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total} batches)"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=RICH_CONSOLE,
                transient=False,
            )
            
            with progress:
                task = progress.add_task("Processing batches...", total=len(batches))
                
                for batch_index, batch in enumerate(batches):
                    progress.update(task, description=f"Batch {batch_index+1}/{len(batches)}")
                    process_batch(batch_index, batch, batches)
                    progress.advance(task)
        else:
            # Debug mode: show detailed output
            for batch_index, batch in enumerate(batches):
                console(f"📦 Processing batch {batch_index + 1}/{len(batches)} with {len(batch)} files")
                process_batch(batch_index, batch, batches)

    # Clear checkpoint when all files are processed
    clear_checkpoint()
    console("✅ Finished all batch files.")
    logging.info("✅ Finished all batch files.")
    
    # 🆕 FINAL LATENCY SUMMARY
    if MODEL_LATENCIES:
        avg_time = statistics.mean(MODEL_LATENCIES)
        max_time = max(MODEL_LATENCIES)
        min_time = min(MODEL_LATENCIES)
        if len(MODEL_LATENCIES) >= 4:
            p95 = statistics.quantiles(MODEL_LATENCIES, n=20)[-1] if len(MODEL_LATENCIES) >= 20 else max(MODEL_LATENCIES)
        else:
            p95 = max_time
        
        latency_summary = f"""
⏱️  Model Latency Summary:
- Calls: {len(MODEL_LATENCIES)}
- Average: {avg_time:.2f}s
- Min/Max: {min_time:.2f}s / {max_time:.2f}s
- 95th percentile: {p95:.2f}s
"""
        console(latency_summary)
        logging.info(latency_summary)

def process_batch(batch_index, batch, batches):
    """🆕 Process a single batch of files."""
    # 🆕 BATCH PROCESSING LOGIC
    if len(batch) > 1:
        # Process as batch
        batch_content = format_batch_content(batch)
        # 🆕 USE SAFE FORMATTING WITH ERROR HANDLING
        batch_prompt = safe_format_prompt(BATCH_PROMPT_TEMPLATE, content=batch_content)
        
        # 🆕 USE TIMED MODEL CALL FOR PROFILING
        # 🆕 USE KEYWORD ARGUMENTS FOR CLARITY
        response_text = time_model_call(
            query_model_safe, batch_prompt, len(batch_content),
            is_batch=True
        )
        
        if response_text:
            file_results = parse_batch_response(response_text, batch)
            
            # Process each file in the batch
            for file_path, content in batch:
                process_file_result(file_path, content, file_results.get(file_path.name, {}))
        else:
            console(f"❌ Batch {batch_index + 1} failed, processing files individually")
            # Fallback: process files individually
            for file_path, content in batch:
                process_single_file(file_path, content)
    else:
        # Single file processing (for batches with 1 file or fallback)
        for file_path, content in batch:
            # 🆕 USE TIMED MODEL CALL FOR PROFILING
            process_single_file(file_path, content)

def process_single_file(file_path, content):
    """🆕 Process a single file with the model."""
    global total_files, total_jobs
    
    newspaper_name = file_path.stem.split("_p")[0]
    subdir = output_dir / newspaper_name
    subdir.mkdir(parents=True, exist_ok=True)

    jobs_subdir = jobs_root / newspaper_name
    jobs_subdir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        console(f"📄 Processing: {file_path.name} → Newspaper: {newspaper_name}")

    try:
        # Note: Content already pre-filtered, but we double-check for safety
        if not content or not likely_contains_job(content):
            if args.debug:
                console(f"⚡ Fast-skip: No job keywords in {file_path.name}")
            mark_file_processed(file_path, has_jobs=False, job_count=0, skipped_fast=True)
            return

        # 🆕 USE SAFE FORMATTING WITH ERROR HANDLING
        prompt = safe_format_prompt(PROMPT_TEMPLATE, content=content)
        # 🆕 USE TIMED MODEL CALL FOR PROFILING
        # 🆕 USE KEYWORD ARGUMENTS FOR CLARITY
        output_text = time_model_call(
            query_model_safe, prompt, len(content),
            is_batch=False
        ).strip()

        if not output_text:
            console(f"⚠️ Empty response for {file_path.name} after retries", "warning")
            return

        process_model_response(file_path, newspaper_name, subdir, jobs_subdir, output_text)
        
    except Exception as e:
        errors.append(f"{file_path.name}: {e}")
        logging.error(f"❌ Error processing {file_path.name}: {e}")
        console(f"❌ Error processing {file_path.name}: {e}")

def process_file_result(file_path, content, file_result):
    """🆕 Process a file from batch results."""
    global total_files, total_jobs
    
    newspaper_name = file_path.stem.split("_p")[0]
    subdir = output_dir / newspaper_name
    subdir.mkdir(parents=True, exist_ok=True)

    jobs_subdir = jobs_root / newspaper_name
    jobs_subdir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        console(f"📄 Processing batch result: {file_path.name}")

    try:
        # Reconstruct response text from batch result
        if file_result:
            concise_listings = file_result.get("concise_listings", [])
            full_descriptions = file_result.get("full_descriptions", [])
            
            # Reconstruct the response format expected by process_model_response
            output_lines = []
            if concise_listings:
                output_lines.append("📌 Concise Listings")
                for listing in concise_listings:
                    output_lines.append(f"- {listing}")
                output_lines.append("")
            
            if full_descriptions:
                output_lines.append("📝 Full Job Descriptions")
                for desc in full_descriptions:
                    output_lines.append(desc)
                    output_lines.append("")
            
            output_text = "\n".join(output_lines).strip()
        else:
            # No result from batch, fall back to single processing
            if args.debug:
                console(f"🔄 Batch failed for {file_path.name}, falling back to single processing")
            process_single_file(file_path, content)
            return

        if output_text:
            process_model_response(file_path, newspaper_name, subdir, jobs_subdir, output_text)
        else:
            console(f"⚠️ Empty batch result for {file_path.name}", "warning")
            mark_file_processed(file_path, has_jobs=False, job_count=0)
            
    except Exception as e:
        errors.append(f"{file_path.name}: {e}")
        logging.error(f"❌ Error processing batch result for {file_path.name}: {e}")
        console(f"❌ Error processing batch result for {file_path.name}: {e}")

def process_model_response(file_path, newspaper_name, subdir, jobs_subdir, output_text):
    """🆕 Common processing for model responses."""
    global total_files, total_jobs
    
    # Save full response
    response_path = subdir / (file_path.stem + "_response.txt")
    response_path.write_text(output_text, encoding="utf-8")
    logging.info(f"[💾] Saved response: {response_path}")
    if args.debug:
        console(f"💾 Saved response → {response_path}")

    # Extract job listings
    lines = output_text.splitlines()
    relevant = [line for line in lines if line.strip().startswith("- ")]

    total_files += 1
    job_count = len(relevant)
    has_jobs = job_count > 0
    
    if has_jobs:
        total_jobs += job_count
        jobs_path = jobs_subdir / (file_path.stem + "_jobs.txt")
        jobs_path.write_text("\n".join(relevant), encoding="utf-8")
        logging.info(f"[🎯] Saved jobs: {jobs_path}")
        if args.debug:
            console(f"🎯 Jobs saved → {jobs_path} ({job_count} jobs)")
    else:
        logging.warning(f"[🕵️] No jobs in {file_path.name}")
        if args.debug:
            console(f"🕵️ No jobs found in {file_path.name}")

    # Mark as processed in database
    mark_file_processed(file_path, has_jobs, job_count)
    
    # Save checkpoint for NEXT file (not current)
    save_checkpoint_for_next(file_path)

# ───── CLI Argument Parsing ─────
def parse_arguments():
    """
    Simplified CLI:
      - --force     : keep (used for testing to ignore DB/checkpoint)
      - hybrid      : enabled by default (no need to pass --hybrid)
      - --no-hybrid : optional to explicitly disable hybrid mode
    Other flags are kept with sensible defaults so existing code paths continue to work.
    """
    parser = argparse.ArgumentParser(description='Ollama Batch Processing Pipeline (minimal CLI defaults)')

    # Minimal / Primary flags the user asked for
    parser.add_argument('--force', action='store_true',
                        help='Force processing: ignore DB/checkpoint and process everything found in input dir')

    # Keep --fresh and --debug as optional useful helpers (harmless defaults)
    parser.add_argument('--fresh', action='store_true', help='Ignore checkpoint and start fresh')
    parser.add_argument('--debug', action='store_true', help='Enable debug output including token window info')

    # Hybrid: enabled by default, provide --no-hybrid to disable
    parser.add_argument('--hybrid', dest='hybrid', action='store_true',
                        help='(kept for compatibility) enable hybrid CPU+GPU processing (enabled by default)')
    parser.add_argument('--no-hybrid', dest='hybrid', action='store_false',
                        help='Disable hybrid processing (useful to force single-endpoint mode)')
    parser.set_defaults(hybrid=True)

    # Other flags kept so the script runs unchanged (default values)
    parser.add_argument('--gpu-endpoint', type=str, default=None, help='GPU Ollama endpoint (overrides OLLAMA_ENDPOINT for GPU tasks)')
    parser.add_argument('--cpu-endpoint', type=str, default=None, help='CPU Ollama endpoint for small tasks')
    parser.add_argument('--workers', type=int, default=2, help='Number of worker threads for prefetch or parallel dispatch')
    parser.add_argument('--newspaper', type=str, default=None, help='Process only specific newspaper (optional)')

    return parser.parse_args()

# ───── Entrypoint ─────
if __name__ == "__main__":
    args = parse_arguments()
    
    # 🆕 SETUP QUIET CONSOLE OUTPUT
    setup_quiet_console()
    
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        start_ollama_serve()
        if wait_for_model():
            log_gpu_status()
            check_ollama_ps()   # ✅ startup check
            main()
        else:
            logging.error("❌ Model did not become ready in time.")
            console("❌ Model did not become ready in time.")
    finally:
        check_ollama_ps()       # ✅ final check before shutdown
        shutdown_ollama()

        try:
            nvidia_info = subprocess.check_output(["nvidia-smi"]).decode()
            driver_version = re.search(r"Driver Version:\s+(\S+)", nvidia_info).group(1)
            cuda_version = re.search(r"CUDA Version:\s+(\S+)", nvidia_info).group(1)
            vram_match = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", nvidia_info)
            if vram_match:
                vram_used, vram_total = vram_match.groups()
            else:
                vram_used, vram_total = "N/A", "N/A"
        except Exception as e:
            driver_version, cuda_version, vram_used, vram_total = "N/A", "N/A", "N/A", "N/A"
            logging.error(f"⚠️ Could not fetch GPU info: {e}")

        cpu_usage = psutil.cpu_percent(interval=1)
        runtime = datetime.now() - start_time

        # 🆕 FIXED SUMMARY FOR --FORCE MODE
        if getattr(args, 'force', False):
            # In testing, show progress relative to the files we attempted in this run
            total_processed = total_files
            total_available = len(list(batch_input_dir.rglob('*.txt')))
        else:
            total_processed = len(get_processed_files()) if db_file.exists() else total_files
            total_available = len(list(batch_input_dir.rglob('*.txt')))

        summary = f"""
✅ Summary:
- Total files processed in this run: {total_files}
- Total jobs found in this run: {total_jobs}
- Cumulative files processed: {total_processed}/{total_available}
- Errors encountered: {len(errors)}
  {errors if errors else 'None'}
- Final GPU VRAM used: {vram_used} MiB
- GPU driver version: {driver_version}
- CUDA version: {cuda_version}
- CPU usage during run: {cpu_usage}%
- Total runtime: {runtime}
"""
        print(summary)  # Always print final summary
        logging.info(summary)

        try:
            subprocess.run(["ollama", "stop", OLLAMA_MODEL], check=False)
            logging.info(f"🧹 Ollama model '{OLLAMA_MODEL}' stopped, VRAM freed.")
        except Exception as e:
            logging.warning(f"Failed to stop Ollama model: {e}")