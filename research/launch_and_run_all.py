import subprocess
import requests
import time
import sys
import signal
import logging
import socket
from pathlib import Path
from datetime import datetime

# ───── Config ─────
OLLAMA_MODEL = "mistral"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
PAGE_TEXTS_DIR = Path("data/page_texts")
BATCH_OUTPUT_DIR = Path("data/batch_output")
KEYWORDS = ["job", "recruitment", "intern", "hiring", "walk-in", "vacancy", "apply", "eligibility"]
PORT = 11434

serve_proc = None
start_time = datetime.now()

# ───── Logging Setup ─────
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%H-%M__%d-%m-%Y")
log_path = log_dir / f"ollama_run_{timestamp}.log"

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ───── Port Checker ─────
def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# ───── Start Ollama ─────
def start_ollama_serve():
    global serve_proc
    retry = 0
    logging.info("🧠 Starting Ollama model server...")
    while is_port_open(PORT) and retry < 10:
        logging.warning("⚠️ Port 11434 in use, retrying in 2s...")
        time.sleep(2)
        retry += 1
    if is_port_open(PORT):
        logging.error("❌ Port still busy. Exiting.")
        sys.exit(1)
    serve_proc = subprocess.Popen(["ollama", "serve"])
    logging.info("✅ Ollama server launched.")

# ───── Wait for Readiness ─────
def wait_for_model(max_wait=30):
    logging.info("⌛ Waiting for model to become ready...")
    for _ in range(max_wait):
        try:
            res = requests.post(
                OLLAMA_ENDPOINT,
                json={"model": OLLAMA_MODEL, "prompt": "Say hello", "stream": False},
                timeout=3
            )
            if res.status_code == 200:
                logging.info("✅ Model is ready.")
                return True
        except:
            pass
        time.sleep(1)
    return False

# ───── Send Prompt ─────
def query_model(prompt_text):
    try:
        res = requests.post(
            OLLAMA_ENDPOINT,
            json={"model": OLLAMA_MODEL, "prompt": prompt_text, "stream": False},
            timeout=60
        )
        return res.json()["response"]
    except Exception as e:
        logging.error(f"❌ Model query failed: {e}")
        return ""

# ───── Extract Job Lines ─────
def filter_job_lines(text):
    lines = text.splitlines()
    return [line for line in lines if any(k in line.lower() for k in KEYWORDS)]

# ───── Process Pages ─────
def process_pages():
    all_txt_files = list(PAGE_TEXTS_DIR.rglob("*.txt"))
    if not all_txt_files:
        msg = f"⚠️ No .txt files found under {PAGE_TEXTS_DIR.resolve()}"
        logging.warning(msg)
        print(msg)
        return

    for file in all_txt_files:
        logging.info(f"📄 Processing: {file.relative_to(PAGE_TEXTS_DIR)}")
        try:
            with open(file, "r", encoding="utf-8") as f:
                prompt = f.read()

            response = query_model(prompt)

            rel_path = file.relative_to(PAGE_TEXTS_DIR)
            subdir = BATCH_OUTPUT_DIR / rel_path.parent
            subdir.mkdir(parents=True, exist_ok=True)

            # Save full response
            full_out = subdir / f"{file.stem}_response.txt"
            with open(full_out, "w", encoding="utf-8") as f:
                f.write(response)
            logging.info(f"[💾] Saved full response: {full_out}")

            # Save filtered job lines
            filtered = filter_job_lines(response)
            if filtered:
                job_out = subdir / f"{file.stem}_jobs.txt"
                with open(job_out, "w", encoding="utf-8") as f:
                    f.write("\n".join(filtered))
                logging.info(f"[🎯] Saved job lines: {job_out}")
            else:
                logging.info(f"[🕵️] No job lines found in {file.name}")

        except Exception as e:
            logging.error(f"❌ Failed on {file.name}: {e}")
            raise e

# ───── Shutdown Ollama ─────
def shutdown_ollama():
    global serve_proc
    if serve_proc:
        logging.info("🛑 Shutting down Ollama...")
        serve_proc.terminate()
        serve_proc.wait()
        logging.info("✅ Ollama terminated.")

# ───── Clean Exit ─────
def handle_exit(sig, frame):
    logging.warning("⚠️ Exit signal received.")
    shutdown_ollama()
    sys.exit(0)

# ───── Main ─────
if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        start_ollama_serve()
        if wait_for_model():
            process_pages()
        else:
            logging.error("❌ Model did not become ready in time.")
    finally:
        shutdown_ollama()
        runtime = datetime.now() - start_time
        logging.info(f"⏱️ Total runtime: {runtime}")
