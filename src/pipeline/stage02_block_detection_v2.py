#!/usr/bin/env python3
"""
Stage02 v2 block detector runner.

This keeps the same Stage2 contracts (block files + detector metadata schema)
while using the experimental v2 detector implementation.
"""

from __future__ import annotations

import argparse
import json
import logging
from src.utils.logging_utils import configure_logging
logger = configure_logging("stage02_block_detection_v2")
import os
import sys
import time
from collections import Counter
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.pipeline_config import get_path, load_config


log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_file = log_dir / f"smart_block_detector_v2_{timestamp}.log"

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

DETECTION_PARAMS_V2_PATH = Path("configs/detection_params_v2.json")
DETECTION_PARAM_V2_DEFAULTS = {
    "min_area": 1200,
    "max_area": 250000,
    "min_width": 55,
    "min_height": 28,
    "max_box_area_ratio": 0.22,
    "max_box_height_ratio": 0.42,
    "column_gap_density_threshold": 0.04,
    "column_min_width_ratio": 0.14,
    "column_gap_min_width": 14,
    "row_blank_density_ratio": 0.012,
    "row_gap_min_height": 12,
    "row_min_height": 26,
    "segment_min_width": 70,
    "segment_expand_px": 4,
    "merge_iou_threshold": 0.62,
    "merge_gap_px": 6,
    "min_fill_density": 0.012,
    "max_fill_density": 0.52,
    "header_exclude_ratio": 0.04,
    "footer_exclude_ratio": 0.03,
    "debug_draw_scores": 1,
}

_DETECTOR_FN = None


def _get_detector():
    global _DETECTOR_FN
    if _DETECTOR_FN is None:
        from src.vision.block_detector_v2 import detect_connected_blocks_v2

        _DETECTOR_FN = detect_connected_blocks_v2
    return _DETECTOR_FN


def _load_detection_params_v2() -> dict:
    params = dict(DETECTION_PARAM_V2_DEFAULTS)
    if DETECTION_PARAMS_V2_PATH.exists():
        try:
            loaded = json.loads(DETECTION_PARAMS_V2_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key in params:
                    if key in loaded:
                        params[key] = loaded[key]
        except Exception as exc:
            logging.warning(f"[!] Failed to load {DETECTION_PARAMS_V2_PATH}: {exc}")
    return params


def _export_detector_v2_env_once(params: dict) -> None:
    for key, value in params.items():
        os.environ[f"DETECTOR_V2_{key.upper()}"] = str(value)
    logging.info(f"[config] Loaded detector params from {DETECTION_PARAMS_V2_PATH}")


def _detector_score_from_bbox(img_w, img_h, x, y, w, h):
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


def detect_page_blocks_v2(img_path_str, blocks_output_dir, debug=True):
    img_path = Path(img_path_str)
    pdf_folder = img_path.parent.name
    page_name = img_path.name
    base_name = img_path.stem

    debug_name = f"debug_p{base_name.split('_p')[-1]}.png" if "_p" in base_name else f"{base_name}_debug.png"
    debug_path = Path(blocks_output_dir) / pdf_folder / debug_name

    if debug and debug_path.exists():
        return {
            "status": "skipped",
            "pdf_folder": pdf_folder,
            "page_name": page_name,
            "detections": [],
            "block_paths": [],
            "error": None,
        }

    try:
        detector = _get_detector()
        blocks, block_paths, block_scores = detector(str(img_path), save_base_dir=str(blocks_output_dir), debug=debug)
        img_w, img_h = 1, 1
        try:
            import cv2

            page_img = cv2.imread(str(img_path))
            if page_img is not None:
                img_h, img_w = page_img.shape[:2]
        except Exception:
            pass

        detections = []
        for idx, (x, y, w, h) in enumerate(blocks):
            score = (
                float(block_scores[idx])
                if isinstance(block_scores, list) and idx < len(block_scores)
                else _detector_score_from_bbox(img_w, img_h, x, y, w, h)
            )
            detections.append(
                {
                    "id": f"detector_v2_{base_name}_{idx}",
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "score": round(float(score), 4),
                    "stage": "detector",
                    "page": page_name,
                    "block_index": idx,
                }
            )

        return {
            "status": "processed",
            "pdf_folder": pdf_folder,
            "page_name": page_name,
            "detections": detections,
            "block_paths": [str(p) for p in block_paths],
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "pdf_folder": pdf_folder,
            "page_name": page_name,
            "detections": [],
            "block_paths": [],
            "error": str(exc),
        }


def process_image(task):
    img_path_str, blocks_output_dir = task
    result = detect_page_blocks_v2(img_path_str, blocks_output_dir, debug=True)
    status = result["status"]
    page_name = result["page_name"]
    pdf_folder = result["pdf_folder"]

    if status == "processed":
        logging.info(f"[v2] [✓] {page_name}: {len(result['detections'])} block(s) detected")
        return ("processed", pdf_folder, page_name, result["detections"])
    if status == "skipped":
        logging.info(f"[v2] [⏩] Skipped {page_name}: already processed")
        return ("skipped", pdf_folder, page_name, [])

    logging.error(f"[v2] [✖] {page_name} failed: {result.get('error')}")
    return ("failed", pdf_folder, page_name, [])


def run_parallel_detector_v2(config_path="configs/pipeline_paths.json"):
    detector_params = _load_detection_params_v2()
    _export_detector_v2_env_once(detector_params)

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
    logging.info("🔍 v2 Smart Block Detection Batch Started")
    print("🔁 Collecting images for v2 detector...")

    all_images = []
    folder_map = {}
    for folder in sorted(images_output_dir.iterdir()):
        if not folder.is_dir():
            continue
        imgs = sorted(folder.glob("*.png"))
        if imgs:
            folder_map[folder.name] = len(imgs)
            all_images.extend(str(img) for img in imgs)

    if not all_images:
        print(f"[!] No images found under {images_output_dir}/*/")
        return

    num_workers = min(cpu_count() // 2 or 1, 6)
    print(f"🚀 [v2] Using {num_workers} parallel workers on {len(all_images)} pages...")
    task_args = [(img, str(blocks_output_dir)) for img in all_images]

    results = []
    with Pool(processes=num_workers) as pool:
        for r in pool.imap_unordered(process_image, task_args):
            results.append(r)
            if len(r) >= 4 and r[0] == "processed":
                _, _, page_name, detections = r
                _write_detector_metadata(page_name, detections, detections_output_dir)

    counts = Counter(r[0] for r in results)
    total_time = time.time() - start_time
    logging.info("✅ v2 Smart Block Detection Batch Completed")

    print("\n========== V2 SUMMARY ==========")
    print(f"🏁 Total time: {total_time:.2f}s")
    print(f"📄 Total PDFs processed: {len(folder_map)}")
    print(f"🧾 Total pages scanned: {len(all_images)}")
    print(f"✅ Processed: {counts['processed']}")
    print(f"⏩ Skipped: {counts['skipped']}")
    print(f"❌ Failed: {counts['failed']}")
    print("================================")
    print(f"[✓] Log saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage02 block detection v2 runner")
    parser.add_argument("--config", default="configs/pipeline_paths.json", help="Path to pipeline paths config")
    cli_args = parser.parse_args()
    run_parallel_detector_v2(config_path=cli_args.config)
