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
import argparse
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from collections import Counter
# ───────────── Calling Helper Script ─────────────
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.vision.block_detector import detect_connected_blocks
from src.utils.pipeline_config import load_config, get_path

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

def _detector_score_from_bbox(img_w, img_h, x, y, w, h):
    # Metadata-only confidence proxy derived from relative area.
    page_area = max(1, img_w * img_h)
    box_area = max(1, w * h)
    return round(min(1.0, max(0.0, box_area / page_area)), 4)


def _write_detector_metadata(page_name, detections, detections_output_dir):
    out_path = Path(detections_output_dir) / f"{page_name}.json"
    existing = {"page": page_name, "detections": []}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {"page": page_name, "detections": []}
        except Exception:
            existing = {"page": page_name, "detections": []}

    kept = []
    for entry in existing.get("detections", []):
        if isinstance(entry, dict) and entry.get("stage") != "detector":
            kept.append(entry)

    payload = {
        "page": page_name,
        "detections": detections + kept,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

# ───────────── Worker: Detect or Skip ─────────────
def process_image(task):
    img_path_str, blocks_output_dir = task
    img_path = Path(img_path_str)
    pdf_folder = img_path.parent.name
    page_name = img_path.name
    base_name = img_path.stem

    debug_name = f"debug_p{base_name.split('_p')[-1]}.png" if "_p" in base_name else f"{base_name}_debug.png"
    debug_path = Path(blocks_output_dir) / pdf_folder / debug_name

    if debug_path.exists():
        logging.info(f"[⏩] Skipped {img_path.name}: already processed")
        return ("skipped", pdf_folder, page_name, [])

    try:
        blocks, _ = detect_connected_blocks(str(img_path), save_base_dir=str(blocks_output_dir), debug=True)
        img_w, img_h = 1, 1
        try:
            import cv2  # local import keeps worker startup light

            page_img = cv2.imread(str(img_path))
            if page_img is not None:
                img_h, img_w = page_img.shape[:2]
        except Exception:
            pass

        detections = []
        for idx, (x, y, w, h) in enumerate(blocks):
            score = _detector_score_from_bbox(img_w, img_h, x, y, w, h)
            detections.append(
                {
                    "id": f"detector_{base_name}_{idx}",
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "score": score,
                    "stage": "detector",
                    "page": page_name,
                    "block_index": idx,
                }
            )
        logging.info(f"[✓] {img_path.name}: {len(blocks)} block(s) detected")
        return ("processed", pdf_folder, page_name, detections)
    except Exception as e:
        logging.error(f"[✖] {img_path.name} failed: {e}")
        return ("failed", pdf_folder, page_name, [])

# ───────────── Batch Runner ─────────────
def run_parallel_detector(config_path="configs/pipeline_paths.json"):
    config = load_config(config_path)
    images_output_dir = get_path("images_output", config)
    blocks_output_dir = get_path("blocks_output", config)
    detections_output_dir = get_path("detections_output", config)

    print("[CONFIG]")
    print(f"images_output = {images_output_dir}")
    print(f"blocks_output = {blocks_output_dir}")
    print(f"detections_output = {detections_output_dir}")

    detections_output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    logging.info("🔍 Parallel Smart Block Detection Batch Started")
    print("🔁 Collecting images...")

    base_dir = images_output_dir
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
        print(f"[!] No images found under {base_dir}/*/")
        return

    num_workers = min(cpu_count() // 2 or 1, 6)
    print(f"🚀 Using {num_workers} parallel workers on {len(all_images)} pages...")
    task_args = [(img, str(blocks_output_dir)) for img in all_images]

    results = []
    with Pool(processes=num_workers) as pool:
        for r in pool.imap_unordered(process_image, task_args):
            results.append(r)
            if len(r) >= 4 and r[0] == "processed":
                _, _, page_name, detections = r
                _write_detector_metadata(page_name, detections, detections_output_dir)

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
    parser = argparse.ArgumentParser(description="Stage02 block detection runner")
    parser.add_argument("--config", default="configs/pipeline_paths.json", help="Path to pipeline paths config")
    cli_args = parser.parse_args()
    run_parallel_detector(config_path=cli_args.config)
