#!/usr/bin/env python3
"""
2_run_smart_detector_batch_summary.py – Optimized Smart Block Detection Runner
Features:
- Parallel processing with clean per-PDF summaries
- Tracks processed, skipped, failed pages
- Shows total runtime summary
- Less terminal spam, detailed logs remain in /logs/
"""

import sys
import os
import logging
import time
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from collections import Counter
# ───────────── Calling Helper Script ─────────────
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.vision.block_detector import detect_connected_blocks

# ───────────── Setup Logging ─────────────
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_file = log_dir / f"smart_block_detector_parallel_{timestamp}.log"

logging.basicConfig(
    filename=log_file,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# ───────────── Worker: Detect or Skip ─────────────
def process_image(img_path_str):
    img_path = Path(img_path_str)
    pdf_folder = img_path.parent.name
    base_name = img_path.stem

    debug_name = f"debug_p{base_name.split('_p')[-1]}.png" if "_p" in base_name else f"{base_name}_debug.png"
    debug_path = Path("data/job_blocks_smart") / pdf_folder / debug_name

    if debug_path.exists():
        logging.info(f"[⏩] Skipped {img_path.name}: already processed")
        return ("skipped", pdf_folder)

    try:
        blocks, _ = detect_connected_blocks(str(img_path), debug=True)
        logging.info(f"[✓] {img_path.name}: {len(blocks)} block(s) detected")
        return ("processed", pdf_folder)
    except Exception as e:
        logging.error(f"[✖] {img_path.name} failed: {e}")
        return ("failed", pdf_folder)

# ───────────── Batch Runner ─────────────
def run_parallel_detector():
    start_time = time.time()
    logging.info("🔍 Parallel Smart Block Detection Batch Started")
    print("🔁 Collecting images...")

    base_dir = Path("data/pdf2img")
    all_images = []
    folder_map = {}

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        logging.info(f"📂 Processing PDF Folder: {folder.name}")
        imgs = sorted(folder.glob("*.png"))
        if imgs:
            folder_map[folder.name] = len(imgs)
            all_images.extend(str(img) for img in imgs)

    if not all_images:
        print("[!] No images found under data/pdf2img/*/")
        return

    num_workers = min(cpu_count() // 2 or 1, 6)
    print(f"🚀 Using {num_workers} parallel workers on {len(all_images)} pages...")

    results = []
    with Pool(processes=num_workers) as pool:
        for r in pool.imap_unordered(process_image, all_images):
            results.append(r)

    # ─── Aggregate Results ───────────────────────────
    counts = Counter(r[0] for r in results)
    total_time = time.time() - start_time
    logging.info("✅ Parallel Smart Block Detection Batch Completed")

    print("\n========== SUMMARY ==========")
    print(f"🏁 Total time: {total_time:.2f}s")
    print(f"📄 Total PDFs processed: {len(folder_map)}")
    print(f"🧾 Total pages scanned: {len(all_images)}")
    print(f"✅ Processed: {counts['processed']}")
    print(f"⏩ Skipped: {counts['skipped']}")
    print(f"❌ Failed: {counts['failed']}")
    print("==============================")
    print(f"[✓] Log saved to {log_file}")

    logging.info(
        f"SUMMARY - PDFs: {len(folder_map)}, Pages: {len(all_images)}, "
        f"Processed: {counts['processed']}, Skipped: {counts['skipped']}, "
        f"Failed: {counts['failed']}, TotalTime: {total_time:.2f}s"
    )

# ───────────── Entrypoint ─────────────
if __name__ == "__main__":
    run_parallel_detector()
