#!/usr/bin/env python3
import re
import shutil
import subprocess
from pathlib import Path
import logging
from datetime import datetime

# ---------------- Logging Setup ---------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
REFINED_DIR = DATA_DIR / "job_blocks_refined"
ERROR_DIR = DATA_DIR / "gpu_ocr_errors"
ERROR_DIR.mkdir(parents=True, exist_ok=True)

# log file path
ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_file = LOGS_DIR / f"gpu_ocr_helper_{ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gpu_ocr_helper")


# ---------------- Helpers ---------------- #
def find_latest_log():
    logs = sorted(LOGS_DIR.glob("easyocr_extract_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def parse_failed_images(log_file):
    failed = []
    patterns = [
        re.compile(r".*\[⚠ missing\]\s+(.+?\.png)"),
        re.compile(r".*\[🚫\]\s+(.+?\.png)"),
        re.compile(r".*OCR failed on\s+(.+?\.png)"),
        re.compile(r".*\[✖\].*?\s+(.+?\.png)"),   # catches EasyOCR crashed
        re.compile(r".*\[❌\]\s+No text from:\s+(.+?\.png)"),  # catches No text
    ]
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for pattern in patterns:
                m = pattern.search(line)
                if m:
                    failed.append(m.group(1).strip())
    return failed


def move_failed_images(failed_files):
    moved, missing = [], []
    for fname in failed_files:
        found = False
        for subfolder in REFINED_DIR.iterdir():
            candidate = subfolder / fname
            if candidate.exists():
                target_dir = ERROR_DIR / subfolder.name
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, target_dir / fname)
                logger.warning(f"[➡ moved] {fname} -> {target_dir}")
                moved.append(fname)
                found = True
                break
        if not found:
            logger.warning(f"[⚠ not found] {fname}")
            missing.append(fname)
    return moved, missing


def run_cpu_script():
    cpu_script = BASE_DIR / "src" / "4-59_gpu_ocr_healper_cpu_easyocr.py"
    if cpu_script.exists():
        logger.info("🚀 Running CPU Easy OCR script on error images...")
        subprocess.run(["python", str(cpu_script)], check=False)
        logger.info("✅ CPU OCR finished.")
        return True
    else:
        logger.error(f"❌ 4-5_gpu_ocr_healper_cpu_easyocr not found : {cpu_script}")
        return False


# ---------------- Main ---------------- #
def main():
    latest_log = find_latest_log()
    if not latest_log:
        logger.error("❌ No GPU OCR log files found.")
        return

    logger.info(f"📄 Using latest GPU OCR log: {latest_log.name}")

    failed_files = parse_failed_images(latest_log)
    if not failed_files:
        logger.info("✨ No failed images found in log.")
        return

    logger.info(f"⚠️ Found {len(failed_files)} failed images. Moving to {ERROR_DIR}...")
    moved, missing = move_failed_images(failed_files)

    cpu_ran = False
    if moved:
        cpu_ran = run_cpu_script()
    else:
        logger.info("🟡 Nothing to process with CPU OCR.")

    # -------- Summary --------
    logger.info("\n========== SUMMARY ==========")
    logger.info(f"GPU log checked: {latest_log.name}")
    logger.info(f"Total failed images found: {len(failed_files)}")
    logger.info(f"Moved to error folder: {len(moved)}")
    logger.info(f"Missing from disk: {len(missing)}")
    logger.info(f"CPU OCR script run: {'YES' if cpu_ran else 'NO'}")
    logger.info("=============================")


if __name__ == "__main__":
    main()
