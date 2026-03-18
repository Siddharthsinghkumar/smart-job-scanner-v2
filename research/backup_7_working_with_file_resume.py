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
from pathlib import Path
from datetime import datetime

# logging helper
total_files = 0
total_jobs = 0
errors = []

# ───── Config ─────
OLLAMA_MODEL = "openhermes"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
PORT = 11434

batch_input_dir = Path("data/batch_inputs")
output_dir = Path("data/batch_output")
output_dir.mkdir(parents=True, exist_ok=True)

jobs_root = Path("data/Jobs_found_final")
jobs_root.mkdir(parents=True, exist_ok=True)

# ───── Database & Checkpoint System ─────
db_file = Path("data/processing_state.db")
checkpoint_file = Path("data/processing_checkpoint.txt")

serve_proc = None
start_time = datetime.now()

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

# ───── Database Helpers ─────
def init_database():
    """Initialize SQLite database for tracking processing state."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_files (
            file_path TEXT PRIMARY KEY,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            has_jobs BOOLEAN DEFAULT FALSE,
            job_count INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def mark_file_processed(file_path, has_jobs=False, job_count=0):
    """Mark a file as processed in the database."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO processed_files 
        (file_path, processed_at, has_jobs, job_count)
        VALUES (?, ?, ?, ?)
    ''', (str(file_path), datetime.now(), has_jobs, job_count))
    conn.commit()
    conn.close()

def get_processed_files():
    """Get all processed files from database."""
    conn = sqlite3.connect(db_file)
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
def wait_for_model(max_wait=60):
    """Wait until the model is ready with exponential backoff."""
    print("⌛ Waiting for model to become ready...")
    for attempt in range(max_wait):
        try:
            res = requests.post(
                OLLAMA_ENDPOINT,
                json={"model": OLLAMA_MODEL, "prompt": "Say hello", "stream": False},
                timeout=3
            )
            if res.status_code == 200:
                logging.info("✅ Model is ready.")
                print("✅ Model is ready.")
                return True
        except:
            pass
        
        # Exponential backoff: 1, 2, 4, 8 seconds...
        sleep_time = min(2 ** attempt, 10)
        time.sleep(sleep_time)
    
    return False

def query_model_safe(prompt_text, retries=3):
    """Send prompt to model with retry logic and auto-restart."""
    for attempt in range(retries):
        try:
            res = requests.post(
                OLLAMA_ENDPOINT,
                json={
                    "model": OLLAMA_MODEL, 
                    "prompt": prompt_text, 
                    "stream": False,
                    "options": {
                        "num_predict": 4000,  # Limit output tokens for stability
                        "temperature": 0.1    # More deterministic output
                    }
                },
                timeout=600
            )
            res.raise_for_status()
            response_text = res.json().get("response", "").strip()
            
            if response_text:
                return response_text
            else:
                print(f"⚠️ Empty response, retry {attempt + 1}/{retries}")
                logging.warning(f"⚠️ Empty response, retry {attempt + 1}/{retries}")
                
        except requests.exceptions.ConnectionError as e:
            print(f"🔁 Connection error, restarting Ollama (attempt {attempt + 1}/{retries})")
            logging.warning(f"🔁 Connection error, restarting Ollama: {e}")
            shutdown_ollama(force=True)
            start_ollama_serve()
            if not wait_for_model(30):
                continue
        except Exception as e:
            logging.error(f"❌ Model query failed (attempt {attempt + 1}/{retries}): {e}")
            print(f"❌ Model query failed (attempt {attempt + 1}/{retries}): {e}")
        
        # Exponential backoff between retries
        time.sleep(2 * (attempt + 1))
    
    print("❌ Failed after all retries")
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
        print("⚠️ Ollama already running.")
        return
    serve_proc = subprocess.Popen(["ollama", "serve"], preexec_fn=os.setpgrp)
    logging.info("✅ Ollama server launched.")
    print("✅ Ollama server launched.")

def shutdown_ollama(force=False):
    """Shut down Ollama server cleanly."""
    global serve_proc
    if serve_proc and serve_proc.poll() is None:
        logging.info("🛑 Shutting down Ollama server...")
        print("🛑 Shutting down Ollama server...")
        os.killpg(os.getpgid(serve_proc.pid), signal.SIGTERM)
        serve_proc.wait(timeout=10)
        logging.info("✅ Ollama terminated.")
        print("✅ Ollama terminated.")
    elif force:
        # Kill any lingering Ollama processes
        logging.info("🛑 Force shutdown of Ollama processes...")
        print("🛑 Force shutdown of Ollama processes...")
        subprocess.run(["pkill", "-f", "ollama"])
        logging.info("✅ Ollama terminated (force).")
        print("✅ Ollama terminated (force).")

def handle_exit(signum, frame):
    print("🛑 Caught exit signal, cleaning up...")
    logging.info("🛑 Caught exit signal, cleaning up...")

    # Final check of running models
    check_ollama_ps()

    shutdown_ollama()
    sys.exit(0)

def check_ollama_ps():
    try:
        out = subprocess.check_output(["ollama", "ps"]).decode()
        print("📊 Ollama models running:\n", out)
        logging.info(f"Ollama ps:\n{out}")
    except Exception as e:
        print(f"⚠️ Could not check ollama ps: {e}")

def log_gpu_status():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
        )
        util, mem_used, mem_total = out.decode().strip().split(", ")
        logging.info(f"🖥️ Ollama GPU Status → Utilization: {util}%, VRAM: {mem_used}/{mem_total} MiB")
        print(f"🖥️ Ollama GPU Status → Utilization: {util}%, VRAM: {mem_used}/{mem_total} MiB")
    except Exception as e:
        logging.warning(f"⚠️ Could not fetch GPU usage: {e}")
        print("⚠️ Could not fetch GPU usage")

def print_progress(processed_count, total_count, start_time):
    """Print progress with ETA."""
    if total_count == 0:
        return
    
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
    
    print(f"📊 Progress: {processed_count}/{total_count} ({percent:.1f}%) - ETA: {eta_str}")

# ───── Main Processing ─────
def main():
    global total_files, total_jobs, errors

    print("🚀 Starting Ollama batch processing...")
    logging.info("🚀 Starting Ollama batch processing...")

    # Fix: Argument precedence - default to resume, override with fresh
    resume_mode = not args.fresh

    # Initialize database
    init_database()

    # Get all batch files
    batch_files = sorted(batch_input_dir.rglob("*.txt"))
    if not batch_files:
        print("⚠️ No batch files found in data/batch_inputs/")
        return

    # 🆕 IMPROVEMENT 1: Filter by newspaper FIRST (before checkpoint logic)
    if args.newspaper:
        batch_files = [f for f in batch_files if args.newspaper in str(f)]
        if not batch_files:
            print(f"⚠️ No files found for newspaper: {args.newspaper}")
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
            print(f"🔁 Resuming from checkpoint: {checkpoint_path.name}")
            logging.info(f"🔁 Resuming from checkpoint: {checkpoint_path.name}")
        except ValueError:
            # Checkpoint file not found in current batch, start from beginning
            print("⚠️ Checkpoint file not found in filtered list, starting from beginning")
            logging.warning("⚠️ Checkpoint file not found in filtered list, starting from beginning")
            start_index = 0
    elif args.fresh:
        print("🆕 Fresh start requested, ignoring checkpoint")
        logging.info("🆕 Fresh start requested")
        start_index = 0
    else:
        # 🆕 IMPROVEMENT 4: Better UX message for resume with no checkpoint
        print("ℹ️  Resume by default, but no checkpoint found - starting from beginning")
        logging.info("ℹ️  Resume by default but no checkpoint found")
    
    # Get files to process (from checkpoint position, excluding already processed)
    for i in range(start_index, len(batch_files)):
        file_path = batch_files[i]
        if file_path not in already_processed:
            files_to_process.append(file_path)

    total_processed_count = len(already_processed)
    print(f"📊 Total files: {len(batch_files)}, Already processed: {total_processed_count}, Remaining: {len(files_to_process)}")
    logging.info(f"📊 Files - Total: {len(batch_files)}, Processed: {total_processed_count}, Remaining: {len(files_to_process)}")

    if not files_to_process:
        print("✅ All files have already been processed!")
        clear_checkpoint()
        return

    # 🆕 IMPROVEMENT 5: Model warm-up prompt to avoid cold start penalty
    print("🔥 Warming up model with initial prompt...")
    warm_up_result = query_model_safe("just say ready", retries=1)
    if warm_up_result:
        print("✅ Model warmed up and ready")
    else:
        print("⚠️ Model warm-up had issues, but continuing...")

    for i, batch_file in enumerate(files_to_process, 1):
        newspaper_name = batch_file.stem.split("_p")[0]
        subdir = output_dir / newspaper_name
        subdir.mkdir(parents=True, exist_ok=True)

        jobs_subdir = jobs_root / newspaper_name
        jobs_subdir.mkdir(parents=True, exist_ok=True)

        # Print progress with ETA
        current_processed = total_processed_count + i
        print_progress(current_processed, len(batch_files), start_time)
        print(f"📄 [{i}/{len(files_to_process)}] Processing: {batch_file.name}  →  Newspaper: {newspaper_name}")
        logging.info(f"📄 [{i}/{len(files_to_process)}] Processing: {batch_file.name}")

        try:
            raw_content = batch_file.read_text(encoding="utf-8")
            
            # 🆕 IMPROVEMENT 3: OCR text cleanup - remove null bytes and strip
            raw_content = raw_content.replace("\x00", "").strip()
            
            if not raw_content:
                print(f"⚠️ Empty content after cleanup for {batch_file.name}")
                logging.warning(f"⚠️ Empty content after cleanup for {batch_file.name}")
                continue

            prompt = PROMPT_TEMPLATE.format(content=raw_content)

            output_text = query_model_safe(prompt).strip()

            if not output_text:
                print(f"⚠️ Empty response for {batch_file.name} after retries")
                logging.warning(f"⚠️ Empty response for {batch_file.name} after retries")
                # Don't mark as processed if we never got a valid response
                continue

            response_path = subdir / (batch_file.stem + "_response.txt")
            response_path.write_text(output_text, encoding="utf-8")
            logging.info(f"[💾] Saved response: {response_path}")
            print(f"💾 Saved response → {response_path}")

            lines = output_text.splitlines()
            relevant = [line for line in lines if line.strip().startswith("- ")]

            total_files += 1
            job_count = len(relevant)
            has_jobs = job_count > 0
            
            if has_jobs:
                total_jobs += job_count
                jobs_path = jobs_subdir / (batch_file.stem + "_jobs.txt")
                jobs_path.write_text("\n".join(relevant), encoding="utf-8")
                logging.info(f"[🎯] Saved jobs: {jobs_path}")
                print(f"🎯 Jobs saved → {jobs_path} ({job_count} jobs)")
            else:
                logging.warning(f"[🕵️] No jobs in {batch_file.name}")
                print(f"🕵️ No jobs found in {batch_file.name}")

            # Mark as processed in database (even if no jobs found)
            mark_file_processed(batch_file, has_jobs, job_count)
            
            # Save checkpoint for NEXT file (not current)
            save_checkpoint_for_next(batch_file)

        except Exception as e:
            errors.append(f"{batch_file.name}: {e}")
            logging.error(f"❌ Error processing {batch_file.name}: {e}")
            print(f"❌ Error processing {batch_file.name}: {e}")

    # Clear checkpoint when all files are processed
    clear_checkpoint()
    print("✅ Finished all batch files.")
    logging.info("✅ Finished all batch files.")

# ───── CLI Argument Parsing ─────
def parse_arguments():
    parser = argparse.ArgumentParser(description='Ollama Batch Processing Pipeline')
    parser.add_argument('--newspaper', type=str, help='Process only specific newspaper')
    parser.add_argument('--fresh', action='store_true', help='Ignore checkpoint and start fresh')
    return parser.parse_args()

# ───── Entrypoint ─────
if __name__ == "__main__":
    args = parse_arguments()
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
            print("❌ Model did not become ready in time.")
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

        # Calculate total progress including previously processed files
        total_processed = len(get_processed_files()) if db_file.exists() else total_files
        
        summary = f"""
✅ Summary:
- Total files processed in this run: {total_files}
- Total jobs found in this run: {total_jobs}
- Cumulative files processed: {total_processed}/{len(list(batch_input_dir.rglob('*.txt')))}
- Errors encountered: {len(errors)}
  {errors if errors else 'None'}
- Final GPU VRAM used: {vram_used} MiB
- GPU driver version: {driver_version}
- CUDA version: {cuda_version}
- CPU usage during run: {cpu_usage}%
- Total runtime: {runtime}
"""
        print(summary)
        logging.info(summary)

        try:
            subprocess.run(["ollama", "stop", OLLAMA_MODEL], check=False)
            logging.info(f"🧹 Ollama model '{OLLAMA_MODEL}' stopped, VRAM freed.")
        except Exception as e:
            logging.warning(f"Failed to stop Ollama model: {e}")