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
from src.utils.logging_utils import configure_logging
logger = configure_logging("stage02_block_detection")
import time
import json
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from collections import Counter
import cv2
# ───────────── Calling Helper Script ─────────────
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.pipeline_config import load_config, get_path
from src.pipeline.pipeline_metadata import (
    read_page_manifest_jsonl,
    write_crop_manifest_jsonl,
    generate_crop_id,
    compute_normalized_bbox,
)

# ───────────── Setup Logging ─────────────
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_file = log_dir / f"smart_block_detector_parallel_{timestamp}.log"

s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

DETECTION_PARAMS_PATH = Path("configs/detection_params.json")
DETECTION_PARAM_DEFAULTS = {
    "contour_area_min": 1500,
    "contour_area_max": 500000,
    "aspect_ratio_min": 0.5,
    "aspect_ratio_max": 5.0,
    "block_merge_distance": 40,
    "morphology_kernel_size": 5,
}

# Metadata manifest paths
DEFAULT_PAGE_MANIFEST_JSONL = Path("run_state/page_manifest.jsonl")
DEFAULT_CROP_MANIFEST_JSONL = Path("run_state/crop_manifest.jsonl")
DEFAULT_CROPS_OUTPUT_DIR = Path("data/crops")

_DETECTOR_FN = None


def _get_detector():
    global _DETECTOR_FN
    if _DETECTOR_FN is None:
        from src.vision.block_detector import detect_connected_blocks

        _DETECTOR_FN = detect_connected_blocks
    return _DETECTOR_FN


def _load_detection_params() -> dict:
    params = dict(DETECTION_PARAM_DEFAULTS)
    if DETECTION_PARAMS_PATH.exists():
        try:
            loaded = json.loads(DETECTION_PARAMS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key in params:
                    if key in loaded:
                        params[key] = loaded[key]
        except Exception as exc:
            logging.warning(f"[!] Failed to load {DETECTION_PARAMS_PATH}: {exc}")
    return params


def _export_detector_env_once(params: dict) -> None:
    env_map = {
        "DETECTOR_CONTOUR_AREA_MIN": params["contour_area_min"],
        "DETECTOR_CONTOUR_AREA_MAX": params["contour_area_max"],
        "DETECTOR_ASPECT_RATIO_MIN": params["aspect_ratio_min"],
        "DETECTOR_ASPECT_RATIO_MAX": params["aspect_ratio_max"],
        "DETECTOR_BLOCK_MERGE_DISTANCE": params["block_merge_distance"],
        "DETECTOR_MORPHOLOGY_KERNEL_SIZE": params["morphology_kernel_size"],
    }
    for name, value in env_map.items():
        os.environ[name] = str(value)
    logging.info(f"[config] Loaded detector params from {DETECTION_PARAMS_PATH}")

def _compute_overlap_group_ids(blocks_xyxy: list) -> list:
    """
    Assign overlap_group_id to each bbox. Bboxes with IoU > 0.05 share a group.
    Isolated bboxes get their own single-box group hash.
    Returns list of hex hashes, same length as blocks_xyxy.
    """
    import hashlib
    n = len(blocks_xyxy)
    if n == 0:
        return []

    # Union-find
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        union_area = area_a + area_b - inter
        return inter / max(1, union_area)

    for i in range(n):
        for j in range(i + 1, n):
            if iou(blocks_xyxy[i], blocks_xyxy[j]) > 0.05:
                union(i, j)

    # Build group hash per root
    group_members: dict = {}
    for i in range(n):
        root = find(i)
        group_members.setdefault(root, []).append(blocks_xyxy[i])

    group_hashes: dict = {}
    for root, members in group_members.items():
        key = "_".join(f"{x1},{y1},{x2},{y2}" for x1, y1, x2, y2 in sorted(members))
        group_hashes[root] = hashlib.md5(key.encode()).hexdigest()[:8]

    return [group_hashes[find(i)] for i in range(n)]


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


def detect_page_blocks(img_path_str, blocks_output_dir, debug=True, page_metadata=None, crops_output_dir=None):
    """
    Reusable per-page detector helper for streaming and batch runners.
    Returns a dict with status, detections and block artifact paths.

    Args:
        img_path_str: Path to input image
        blocks_output_dir: Where to save block debug outputs
        debug: Whether to use debug mode and skip existing
        page_metadata: Optional dict with page metadata (page_id, image_width, image_height, etc.)
        crops_output_dir: Optional directory to save cropped detection images

    Returns:
        dict with status, detections, block_paths, and crop_metadata (if crops_output_dir provided)
    """
    img_path = Path(img_path_str)
    pdf_folder = img_path.parent.name
    page_name = img_path.name
    base_name = img_path.stem

    debug_name = f"debug_p{base_name.split('_p')[-1]}.png" if "_p" in base_name else f"{base_name}_debug.png"
    debug_path = Path(blocks_output_dir) / pdf_folder / debug_name

    try:
        detector = _get_detector()
        t_detect_start = time.time()
        blocks, block_paths = detector(str(img_path), save_base_dir=str(blocks_output_dir), debug=debug)
        page_detect_time = time.time() - t_detect_start

        img_w, img_h = 1, 1
        try:
            page_img = cv2.imread(str(img_path))
            if page_img is not None:
                img_h, img_w = page_img.shape[:2]
        except Exception:
            pass

        # Compute overlap groups for all bboxes on this page
        blocks_xyxy = [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in blocks]
        overlap_group_ids = _compute_overlap_group_ids(blocks_xyxy)
        per_crop_detect_time = round(page_detect_time / max(1, len(blocks)), 4)

        detections = []
        crop_metadata = []

        for idx, (x, y, w, h) in enumerate(blocks):
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            score = _detector_score_from_bbox(img_w, img_h, x1, y1, w, h)
            detections.append(
                {
                    "id": f"detector_{base_name}_{idx}",
                    "bbox": [x1, y1, x2, y2],
                    "score": score,
                    "stage": "detector",
                    "page": page_name,
                    "block_index": idx,
                }
            )

            # Save crop image and generate metadata if metadata provided
            if page_metadata and crops_output_dir:
                try:
                    crop_path = _save_crop_image(
                        page_img if 'page_img' in locals() else cv2.imread(str(img_path)),
                        x1, y1, x2, y2,
                        base_name, idx,
                        crops_output_dir
                    )
                    if crop_path:
                        # Compute normalized bbox
                        norm_bbox = compute_normalized_bbox(x1, y1, x2, y2, img_w, img_h)

                        page_id = page_metadata.get('page_id')
                        crop_id = generate_crop_id(page_id, idx) if page_id else None

                        crop_info = {
                            "crop_id": crop_id,
                            "page_id": page_id,
                            "doc_id": page_metadata.get('doc_id'),
                            "pdf_path": page_metadata.get('pdf_path'),
                            "page_image_path": page_metadata.get('image_path'),
                            "crop_image_path": str(crop_path),
                            "newspaper_name": page_metadata.get('newspaper_name'),
                            "issue_date": page_metadata.get('issue_date'),
                            "page_index0": page_metadata.get('page_index0'),
                            "page_number1": page_metadata.get('page_number1'),
                            "detector_model": "block_detector_v2",  # Could be parameterized
                            "detector_checkpoint": str(blocks_output_dir),
                            "detector_conf": 0.022,  # Default threshold
                            "bbox_xyxy_abs": [x1, y1, x2, y2],
                            "bbox_xyxy_norm": [
                                norm_bbox['x1_norm'],
                                norm_bbox['y1_norm'],
                                norm_bbox['x2_norm'],
                                norm_bbox['y2_norm'],
                            ],
                            "crop_width": max(0, x2 - x1),
                            "crop_height": max(0, y2 - y1),
                            "page_width": img_w,
                            "page_height": img_h,
                            "padding_px": 0,
                            "x_center_norm": norm_bbox['center_x_norm'],
                            "y_center_norm": norm_bbox['center_y_norm'],
                            "area_norm": norm_bbox['area_norm'],
                            "crop_aspect_ratio": round(max(0, x2 - x1) / max(1, max(0, y2 - y1)), 4),
                            "detect_time_sec": per_crop_detect_time,
                            "overlap_group_id": overlap_group_ids[idx] if idx < len(overlap_group_ids) else None,
                        }
                        crop_metadata.append(crop_info)
                except Exception as e:
                    logging.warning(f"[!] Failed to save crop for {page_name} detection {idx}: {e}")

        return {
            "status": "processed",
            "pdf_folder": pdf_folder,
            "page_name": page_name,
            "detections": detections,
            "block_paths": [str(p) for p in block_paths],
            "crop_metadata": crop_metadata,
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "pdf_folder": pdf_folder,
            "page_name": page_name,
            "detections": [],
            "block_paths": [],
            "crop_metadata": [],
            "error": str(exc),
        }


def _save_crop_image(
    page_img,
    x1: int, y1: int, x2: int, y2: int,
    page_base_name: str,
    crop_idx: int,
    crops_output_dir
) -> Path | None:
    """
    Extract and save a cropped image.

    Returns:
        Path to saved crop file or None if failed
    """
    if page_img is None:
        return None

    crops_dir = Path(crops_output_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Clip coordinates to image bounds
        h, w = page_img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        crop = page_img[y1:y2, x1:x2]
        crop_filename = f"{page_base_name}_crop{crop_idx:04d}.png"
        crop_path = crops_dir / crop_filename

        cv2.imwrite(str(crop_path), crop)
        return crop_path
    except Exception as e:
        logging.warning(f"[!] Failed to save crop image: {e}")
        return None

def _should_skip_processed_pages() -> bool:
    return os.environ.get("DETECTOR_FORCE_REPROCESS", "false").lower() != "true"

# ───────────── Worker: Detect or Skip ─────────────
def process_image(task):
    img_path_str, blocks_output_dir, page_metadata, crops_output_dir = task
    
    img_path = Path(img_path_str)
    pdf_folder = img_path.parent.name
    base_name = img_path.stem
    debug_name = f"debug_p{base_name.split('_p')[-1]}.png" if "_p" in base_name else f"{base_name}_debug.png"
    debug_path = Path(blocks_output_dir) / pdf_folder / debug_name
    
    if debug_path.exists() and _should_skip_processed_pages():
        logging.info(f"[⏩] Skipped {img_path.name}: already processed")
        return ("skipped", pdf_folder, img_path.name, [], [])

    result = detect_page_blocks(
        img_path_str,
        blocks_output_dir,
        debug=True,
        page_metadata=page_metadata,
        crops_output_dir=crops_output_dir
    )
    status = result["status"]
    page_name = result["page_name"]
    pdf_folder = result["pdf_folder"]

    if status == "processed":
        logging.info(f"[✓] {page_name}: {len(result['detections'])} block(s) detected")
        return ("processed", pdf_folder, page_name, result["detections"], result.get("crop_metadata", []))
    if status == "skipped":
        logging.info(f"[⏩] Skipped {page_name}: already processed")
        return ("skipped", pdf_folder, page_name, [], [])

    logging.error(f"[✖] {page_name} failed: {result.get('error')}")
    return ("failed", pdf_folder, page_name, [], [])


def _validate_crop_manifest(all_crop_metadata: list, page_manifest: dict) -> None:
    """Validate crop manifest: crop files exist, bboxes valid and normalized, page links valid."""
    missing_crops = []
    invalid_bboxes = []
    out_of_range_bboxes = []
    broken_page_links = []
    page_ids_in_manifest = set(page_manifest.keys())

    for row in all_crop_metadata:
        crop_path = row.get("crop_image_path")
        if crop_path and not Path(crop_path).exists():
            missing_crops.append(crop_path)

        bbox_norm = row.get("bbox_xyxy_norm", [])
        if len(bbox_norm) == 4:
            x1n, y1n, x2n, y2n = bbox_norm
            if x2n <= x1n or y2n <= y1n:
                invalid_bboxes.append(row.get("crop_id"))
            if not (0.0 <= x1n <= 1.0 and 0.0 <= y1n <= 1.0 and 0.0 <= x2n <= 1.0 and 0.0 <= y2n <= 1.0):
                out_of_range_bboxes.append(row.get("crop_id"))
        else:
            invalid_bboxes.append(row.get("crop_id"))

        page_id = row.get("page_id")
        if page_id and page_id not in page_ids_in_manifest:
            broken_page_links.append(page_id)

    warn_count = len(missing_crops) + len(invalid_bboxes) + len(out_of_range_bboxes) + len(broken_page_links)
    if warn_count == 0:
        print(f"[✓] Crop manifest validation passed ({len(all_crop_metadata)} crops)")
        logging.info(f"[✓] Crop manifest validation passed ({len(all_crop_metadata)} crops)")
    else:
        if missing_crops:
            logging.warning(f"[!] {len(missing_crops)} crop files missing from disk")
            print(f"[!] {len(missing_crops)} crop files missing from disk")
        if invalid_bboxes:
            logging.warning(f"[!] {len(invalid_bboxes)} crops with degenerate bboxes")
            print(f"[!] {len(invalid_bboxes)} crops with degenerate bboxes")
        if out_of_range_bboxes:
            logging.warning(f"[!] {len(out_of_range_bboxes)} crops with bbox coords outside [0,1]")
            print(f"[!] {len(out_of_range_bboxes)} crops with bbox coords outside [0,1]")
        if broken_page_links:
            logging.warning(f"[!] {len(broken_page_links)} crops with page_id not in page manifest")
            print(f"[!] {len(broken_page_links)} crops with page_id not in page manifest")


# ───────────── Batch Runner ─────────────
def run_parallel_detector(config_path="configs/pipeline_paths.json", page_manifest_path=None, crops_output_dir=None):
    detector_params = _load_detection_params()
    _export_detector_env_once(detector_params)

    config = load_config(config_path)
    images_output_dir = get_path("images_output", config)
    blocks_output_dir = get_path("blocks_output", config)
    detections_output_dir = get_path("detections_output", config)

    # Use defaults if not provided
    if not page_manifest_path:
        page_manifest_path = DEFAULT_PAGE_MANIFEST_JSONL
    if not crops_output_dir:
        crops_output_dir = DEFAULT_CROPS_OUTPUT_DIR

    print("[CONFIG]")
    print(f"images_output = {images_output_dir}")
    print(f"blocks_output = {blocks_output_dir}")
    print(f"detections_output = {detections_output_dir}")
    print(f"page_manifest = {page_manifest_path}")
    print(f"crops_output = {crops_output_dir}")

    detections_output_dir.mkdir(parents=True, exist_ok=True)
    Path(crops_output_dir).mkdir(parents=True, exist_ok=True)

    # Load page metadata manifest from Stage 1
    page_manifest = read_page_manifest_jsonl(Path(page_manifest_path))
    print(f"📋 Loaded {len(page_manifest)} pages from manifest")
    logging.info(f"📋 Loaded {len(page_manifest)} pages from manifest")

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

    # Build task args with metadata lookup
    task_args = []
    for img_path_str in all_images:
        img_path = Path(img_path_str)
        page_name = img_path.name

        # Find metadata for this page by matching image path
        page_meta = None
        for page_id, meta in page_manifest.items():
            if Path(meta.get('image_path', '')).name == page_name:
                page_meta = meta
                break

        task_args.append((img_path_str, str(blocks_output_dir), page_meta, str(crops_output_dir)))

    results = []
    all_crop_metadata = []

    with Pool(processes=num_workers) as pool:
        for r in pool.imap_unordered(process_image, task_args):
            results.append(r)
            if len(r) >= 5 and r[0] == "processed":
                _, _, page_name, detections, crop_metadata = r
                _write_detector_metadata(page_name, detections, detections_output_dir)
                if crop_metadata:
                    all_crop_metadata.extend(crop_metadata)
            elif len(r) >= 5 and r[0] == "skipped":
                _, _, page_name, detections, crop_metadata = r
                if crop_metadata:
                    all_crop_metadata.extend(crop_metadata)

    # ─── Write Crop Manifest ───────────────────────────
    try:
        crop_manifest_path = Path(DEFAULT_CROP_MANIFEST_JSONL)
        write_crop_manifest_jsonl(all_crop_metadata, crop_manifest_path)
        print(f"📋 Crop manifest written: {crop_manifest_path} ({len(all_crop_metadata)} crops)")
        logging.info(f"📋 Crop manifest written: {crop_manifest_path} ({len(all_crop_metadata)} crops)")
    except Exception as e:
        print(f"[!] Failed to write crop manifest: {e}")
        logging.error(f"[!] Failed to write crop manifest: {e}")

    # ─── Post-Detection Validation ────────────────────
    _validate_crop_manifest(all_crop_metadata, page_manifest)

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
    print(f"🖼️  Total crops saved: {len(all_crop_metadata)}")
    print("==============================")
    print(f"[✓] Log saved to {log_file}")

    logging.info(
        f"SUMMARY - PDFs: {len(folder_map)}, Pages: {len(all_images)}, "
        f"Processed: {counts['processed']}, Skipped: {counts['skipped']}, "
        f"Failed: {counts['failed']}, Crops: {len(all_crop_metadata)}, TotalTime: {total_time:.2f}s"
    )

# ───────────── Entrypoint ─────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage02 block detection runner")
    parser.add_argument("--config", default="configs/pipeline_paths.json", help="Path to pipeline paths config")
    parser.add_argument(
        "--page-manifest",
        default=str(DEFAULT_PAGE_MANIFEST_JSONL),
        help="Path to page manifest JSONL from Stage 1"
    )
    parser.add_argument(
        "--crops-output",
        default=str(DEFAULT_CROPS_OUTPUT_DIR),
        help="Output directory for cropped detection images"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if debug images exist"
    )
    cli_args = parser.parse_args()
    
    if cli_args.force:
        os.environ["DETECTOR_FORCE_REPROCESS"] = "true"

    run_parallel_detector(
        config_path=cli_args.config,
        page_manifest_path=cli_args.page_manifest,
        crops_output_dir=cli_args.crops_output
    )
