#!/usr/bin/env python3
"""Stage02 v4 recall-first block detector runner.

Keeps Stage2 contracts stable while maximizing proposal recall.
"""

from __future__ import annotations

import argparse
import json
import logging
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
log_file = log_dir / f"smart_block_detector_v4_recall_{timestamp}.log"

logging.basicConfig(
    filename=log_file,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

DETECTION_PARAMS_V4_PATH = Path("configs/detection_params_v4.json")
DETECTION_PARAM_V4_DEFAULTS = {
    "min_area": 2600,
    "min_width": 72,
    "min_height": 44,
    "max_area_ratio": 0.52,
    "column_gap_density_threshold": 0.02,
    "column_gap_min_width": 8,
    "column_min_width_ratio": 0.07,
    "row_blank_density_ratio": 0.004,
    "row_gap_min_height": 8,
    "row_min_height": 40,
    "segment_min_width": 70,
    "segment_expand_px": 8,
    "combo_span_max": 3,
    "dense_cc_min_area": 420,
    "dense_cc_max_area_ratio": 0.24,
    "min_fill_density": 0.002,
    "max_fill_density": 0.985,
    "dedup_iou": 0.985,
    "max_boxes": 180,
    "header_exclude_ratio": 0.0,
    "footer_exclude_ratio": 0.0,
}

_DETECTOR_FN = None


def _get_detector():
    global _DETECTOR_FN
    if _DETECTOR_FN is None:
        from src.vision.block_detector_v4_recall import detect_connected_blocks_v4_recall

        _DETECTOR_FN = detect_connected_blocks_v4_recall
    return _DETECTOR_FN


def _load_detection_params_v4() -> dict:
    params = dict(DETECTION_PARAM_V4_DEFAULTS)
    if DETECTION_PARAMS_V4_PATH.exists():
        try:
            loaded = json.loads(DETECTION_PARAMS_V4_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key in params:
                    if key in loaded:
                        params[key] = loaded[key]
        except Exception as exc:
            logging.warning(f"[!] Failed to load {DETECTION_PARAMS_V4_PATH}: {exc}")
    return params


def _export_detector_v4_env_once(params: dict) -> None:
    for key, value in params.items():
        os.environ[f"DETECTOR_V4_{key.upper()}"] = str(value)
    logging.info(f"[config] Loaded detector params from {DETECTION_PARAMS_V4_PATH}")


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


def detect_page_blocks_v4(img_path_str, blocks_output_dir, debug=True):
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
                    "id": f"detector_v4_{base_name}_{idx}",
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
    result = detect_page_blocks_v4(img_path_str, blocks_output_dir, debug=True)
    status = result["status"]
    page_name = result["page_name"]
    pdf_folder = result["pdf_folder"]

    if status == "processed":
        logging.info(f"[v4] [✓] {page_name}: {len(result['detections'])} proposal block(s) detected")
        return ("processed", pdf_folder, page_name, result["detections"])
    if status == "skipped":
        logging.info(f"[v4] [⏩] Skipped {page_name}: already processed")
        return ("skipped", pdf_folder, page_name, [])

    logging.error(f"[v4] [✖] {page_name} failed: {result.get('error')}")
    return ("failed", pdf_folder, page_name, [])


def run_parallel_detector_v4(config_path="configs/pipeline_paths.json"):
    detector_params = _load_detection_params_v4()
    _export_detector_v4_env_once(detector_params)

    config = load_config(config_path)
    images_output_dir = get_path("images_output", config)
    blocks_output_dir = get_path("blocks_output", config)
    detections_output_dir = get_path("detections_output", config)

    print("[CONFIG]")
    print(f"images_output = {images_output_dir}")
    print(f"blocks_output = {blocks_output_dir}")
    print(f"detections_output = {detections_output_dir}")
    print(f"v4_max_boxes = {detector_params.get('max_boxes')}")

    detections_output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    logging.info("🔍 v4 recall-first block detection batch started")
    print("🔁 Collecting images for v4 detector...")

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
    print(f"🚀 [v4] Using {num_workers} parallel workers on {len(all_images)} pages...")
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
    logging.info("✅ v4 recall-first block detection batch completed")

    print("\n========== V4 SUMMARY ==========")
    print(f"🏁 Total time: {total_time:.2f}s")
    print(f"📄 Total PDFs processed: {len(folder_map)}")
    print(f"🧾 Total pages scanned: {len(all_images)}")
    print(f"✅ Processed: {counts['processed']}")
    print(f"⏩ Skipped: {counts['skipped']}")
    print(f"❌ Failed: {counts['failed']}")
    print("================================")
    print(f"[✓] Log saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage02 block detection v4 recall-first runner")
    parser.add_argument("--config", default="configs/pipeline_paths.json", help="Path to pipeline paths config")
    cli_args = parser.parse_args()
    run_parallel_detector_v4(config_path=cli_args.config)
