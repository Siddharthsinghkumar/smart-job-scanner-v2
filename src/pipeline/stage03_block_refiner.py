import cv2
import numpy as np
import os
import time
import logging
import json
import argparse
import pytesseract
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import random
import re
import sys
from collections import Counter
from tqdm import tqdm
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.pipeline_config import load_config, get_path

# ───── Config ─────
DEBUG_SAMPLE_RATE = 0.0  # Save only 0% of skipped blocks for debugging
TESSERACT_CONFIG = "--oem 1 --psm 6"  # Cached config for faster OCR
BATCH_SAVE_SIZE = 80  # Increased batch size for better I/O performance
REFINER_VERBOSE_BLOCK_LOGS = os.getenv("REFINER_VERBOSE_BLOCK_LOGS", "0").strip() == "1"
REFINER_MAP_CHUNKSIZE = 16
_DETECTOR_PAGE_CACHE = {}
_DETECTOR_PAGE_CACHE_MAX = 4096


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


REFINER_MAX_WORDS = _env_int("REFINER_MAX_WORDS", 12)
REFINER_MAX_CHARS = _env_int("REFINER_MAX_CHARS", 50)
REFINER_MIN_CONFIDENCE = _env_int("REFINER_MIN_CONFIDENCE", 40)
REFINER_GRAPHIC_THRESHOLD = _env_float("REFINER_GRAPHIC_THRESHOLD", 0.12)
REFINER_MIN_GAP_WIDTH = _env_int("REFINER_MIN_GAP_WIDTH", 20)
REFINER_MIN_COL_WIDTH = _env_int("REFINER_MIN_COL_WIDTH", 200)
REFINER_WIDTH_THRESH = _env_int("REFINER_WIDTH_THRESH", 800)
STAGE3_MODE_CHOICES = ("normal", "passthrough", "recall_friendly")
GEOMETRY_PARENT_TOLERANCE = 6

# ───── Warm up Tesseract ─────
def warm_up_tesseract():
    """Preload Tesseract model to avoid slow first calls"""
    try:
        dummy_image = np.ones((32, 128), np.uint8) * 255
        _ = pytesseract.image_to_string(dummy_image, config=TESSERACT_CONFIG)
    except Exception:
        pass  # Warm-up failed but process will continue

# Call warm-up at module load
warm_up_tesseract()

# ───── Optimized Image Loading ─────
def fast_read_image(path):
    """
    Use OpenCV as default loader for performance with thousands of small files.
    Only use Pillow for very large files (>10MB) as fallback.
    """
    try:
        file_size = os.path.getsize(path)
        # Use Pillow only for very large files (>10MB), OpenCV for everything else
        if file_size > 10_000_000:  # >10 MB → very large file
            with Image.open(path) as im:
                return np.array(im.convert("RGB"))
        else:
            # Default: OpenCV (faster for thousands of small PNGs)
            return cv2.imread(str(path))
    except Exception:
        # Fallback to OpenCV if anything fails
        return cv2.imread(str(path))


# ───── Filters ─────
def analyze_text_block(
    image,
    max_words=REFINER_MAX_WORDS,
    max_chars=REFINER_MAX_CHARS,
    min_confidence=REFINER_MIN_CONFIDENCE,
    return_details=False,
):
    """
    Runs OCR once and decides:
    - If text is too short
    - If text is visible (based on confidence)
    """
    try:
        # Convert to grayscale (handles both BGR and RGB)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Use faster Otsu thresholding instead of adaptive threshold
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        data = pytesseract.image_to_data(
            thresh, 
            config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT
        )
        text = " ".join([t for t in data['text'] if t.strip() != ""]).strip()
        words = text.split()

        # Check confidence
        has_text = any(int(conf) > min_confidence and t.strip() != ""
                       for t, conf in zip(data['text'], data['conf']))

        # Short text check
        is_short = len(words) <= max_words or len(text) <= max_chars

        details = {
            "word_count": len(words),
            "text_length": len(text),
            "nonempty_tokens": len([t for t in data["text"] if str(t).strip()]),
        }
        if return_details:
            return is_short, has_text, details
        return is_short, has_text
    except Exception:
        if return_details:
            return True, False, {
                "word_count": 0,
                "text_length": 0,
                "nonempty_tokens": 0,
            }
        return True, False  # Treat failure as short + unreadable


def is_graphic_like(image, threshold=REFINER_GRAPHIC_THRESHOLD):
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, bin_img = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    non_zero = np.count_nonzero(bin_img)
    area = image.shape[0] * image.shape[1]
    density = non_zero / area
    return density < threshold


# ───── Logging Setup ─────
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_name = f"block_refiner_{timestamp}.log"

logging.basicConfig(
    filename=log_dir / log_name,
    filemode='a',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Ensure terminal output is not buffered
sys.stdout.reconfigure(line_buffering=True)


# ───── Column Split Logic ─────
def vertical_split(
    image,
    min_gap_width=REFINER_MIN_GAP_WIDTH,
    min_col_width=REFINER_MIN_COL_WIDTH,
):
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 15
    )

    vertical_proj = np.sum(thresh, axis=0)
    norm = (vertical_proj - np.min(vertical_proj)) / \
           (np.max(vertical_proj) - np.min(vertical_proj) + 1e-5)

    is_gap = norm < 0.08
    splits = []
    start = None
    for x, val in enumerate(is_gap):
        if val and start is None:
            start = x
        elif not val and start is not None:
            if x - start >= min_gap_width:
                splits.append((start, x))
            start = None

    col_bounds = []
    last_x = 0
    for (gap_start, gap_end) in splits:
        if gap_start - last_x >= min_col_width:
            col_bounds.append((last_x, gap_start))
        last_x = gap_end

    if image.shape[1] - last_x >= min_col_width:
        col_bounds.append((last_x, image.shape[1]))

    return col_bounds


def maybe_save_debug(image, path):
    """Save skipped images only with certain probability to reduce I/O."""
    if DEBUG_SAMPLE_RATE <= 0.0:
        return
    if random.random() < DEBUG_SAMPLE_RATE:
        cv2.imwrite(str(path), image)


def batch_save_images(save_buffer):
    """Save multiple images in batch to reduce I/O overhead"""
    for path, img in save_buffer:
        cv2.imwrite(str(path), img)


def _parse_block_reference(stem):
    match = re.match(r"^(?P<page_stem>.+)_block(?P<idx>\d+)$", stem)
    if not match:
        return None, None
    return match.group("page_stem"), int(match.group("idx"))


def _load_detector_entry(page_name, block_index, detections_dir):
    if page_name is None or block_index is None:
        return None

    cache_key = (str(detections_dir), str(page_name))
    page_index = _DETECTOR_PAGE_CACHE.get(cache_key)
    if page_index is None:
        detections_path = Path(detections_dir) / f"{page_name}.json"
        page_index = {}
        if detections_path.exists():
            try:
                payload = json.loads(detections_path.read_text(encoding="utf-8"))
                for entry in payload.get("detections", []):
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("stage") != "detector":
                        continue
                    try:
                        idx = int(entry.get("block_index", -1))
                    except Exception:
                        continue
                    page_index[idx] = entry
            except Exception:
                page_index = {}
        _DETECTOR_PAGE_CACHE[cache_key] = page_index
        if len(_DETECTOR_PAGE_CACHE) > _DETECTOR_PAGE_CACHE_MAX:
            _DETECTOR_PAGE_CACHE.pop(next(iter(_DETECTOR_PAGE_CACHE)))

    return page_index.get(int(block_index))


def _safe_int_bbox(raw_bbox):
    if not (isinstance(raw_bbox, list) and len(raw_bbox) == 4):
        return None
    try:
        return [int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3])]
    except Exception:
        return None


def _inside_or_near(parent_bbox, child_bbox, tolerance=GEOMETRY_PARENT_TOLERANCE):
    if parent_bbox is None or child_bbox is None:
        return None
    px1, py1, px2, py2 = parent_bbox
    cx1, cy1, cx2, cy2 = child_bbox
    return (
        cx1 >= px1 - tolerance
        and cy1 >= py1 - tolerance
        and cx2 <= px2 + tolerance
        and cy2 <= py2 + tolerance
    )


def _inside_page_bounds(page_dims, bbox):
    if page_dims is None or bbox is None:
        return None
    try:
        pw, ph = int(page_dims[0]), int(page_dims[1])
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except Exception:
        return None
    return x1 >= 0 and y1 >= 0 and x2 <= pw and y2 <= ph and x2 > x1 and y2 > y1


def _summarize_geometry_audit(rows):
    page_issue_counter = Counter()
    dim_mismatch = 0
    out_of_bounds = 0
    outside_parent = 0
    missing_detector = 0
    block_mismatch = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        flags = set(row.get("status_flags", []))
        page = str(row.get("page_name") or row.get("page") or "")
        issue_count = 0
        if "crop_dim_mismatch" in flags:
            dim_mismatch += 1
            issue_count += 1
        if "projected_out_of_bounds" in flags:
            out_of_bounds += 1
            issue_count += 1
        if "projected_outside_parent" in flags:
            outside_parent += 1
            issue_count += 1
        if "missing_detector_entry" in flags:
            missing_detector += 1
            issue_count += 1
        if "block_index_mismatch" in flags:
            block_mismatch += 1
            issue_count += 1
        if issue_count and page:
            page_issue_counter[page] += issue_count

    return {
        "total_blocks_audited": len(rows),
        "blocks_with_crop_dimension_mismatch": dim_mismatch,
        "blocks_with_projected_out_of_bounds_boxes": out_of_bounds,
        "blocks_with_projected_outside_parent_boxes": outside_parent,
        "blocks_missing_detector_entries": missing_detector,
        "blocks_with_block_index_mismatch": block_mismatch,
        "pages_most_affected": [
            {"page_name": page, "issue_count": count}
            for page, count in page_issue_counter.most_common(20)
        ],
    }


def _summarize_rejections(rows):
    reason_counter = Counter()
    page_counter = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        page = str(row.get("page_name") or "")
        if page:
            page_counter[page] += 1
        for reason in row.get("rejection_reasons", []):
            reason_counter[str(reason)] += 1

    return {
        "total_rejected_candidates": len(rows),
        "reason_counts": dict(reason_counter),
        "pages_most_affected": [
            {"page_name": page, "rejection_count": count}
            for page, count in page_counter.most_common(20)
        ],
    }


def _upsert_refined_metadata(page_name, refined_entries, detections_dir):
    detections_path = Path(detections_dir) / f"{page_name}.json"
    payload = {"page": page_name, "detections": []}
    if detections_path.exists():
        try:
            loaded = json.loads(detections_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {"page": page_name, "detections": []}

    keep = [
        d for d in payload.get("detections", [])
        if isinstance(d, dict) and d.get("stage") != "refined"
    ]
    payload["page"] = page_name
    payload["detections"] = keep + refined_entries
    detections_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def process_image_file(task):
    file_path = Path(task["file_path"])
    width_thresh = int(task["width_thresh"])
    debug_dirs = task["debug_dirs"]
    refined_dir = Path(task["refined_dir"])
    detections_dir = Path(task["detections_dir"])
    stage3_mode = str(task.get("stage3_mode", "normal"))
    geometry_audit_enabled = bool(task.get("geometry_audit", False))
    rejection_log_enabled = bool(task.get("rejection_log", False))
    page_dims_map = task.get("page_dims_map") or {}
    parent_tolerance = int(task.get("parent_tolerance", GEOMETRY_PARENT_TOLERANCE))

    result = {
        "saved": 0,
        "skipped_tiny": 0,
        "skipped_graphic": 0,
        "skipped_text": 0,
        "skipped_unreadable": 0,
        "skipped_error": 0,
        "logs": [],
        "refined_metadata": [],
        "geometry_audit_rows": [],
        "rejection_rows": [],
    }

    save_buffer = []

    try:
        image = fast_read_image(str(file_path))
        if image is None:
            result["skipped_unreadable"] += 1
            if rejection_log_enabled and stage3_mode == "normal":
                result["rejection_rows"].append(
                    {
                        "page_name": None,
                        "block_index": None,
                        "source_crop_file_path": str(file_path),
                        "rejection_reasons": ["unreadable"],
                        "bbox_dimensions": None,
                        "crop_dimensions": None,
                        "text_count": 0,
                        "ocr_text_length": 0,
                        "too_large": False,
                        "too_small": False,
                        "no_text": False,
                        "under_letter_threshold": False,
                        "graphic": False,
                        "unreadable": True,
                        "invalid_projection": False,
                        "other": False,
                    }
                )
            return result

        h, w = image.shape[:2]
        base_name = file_path.stem
        page_stem, block_index = _parse_block_reference(base_name)
        page_name = f"{page_stem}.png" if page_stem else None
        page_dims = page_dims_map.get(page_name)

        detector_entry = _load_detector_entry(page_name, block_index, detections_dir) if (page_name and block_index is not None) else None
        detector_bbox = _safe_int_bbox(detector_entry.get("bbox") if isinstance(detector_entry, dict) else None)
        detector_score = float(detector_entry.get("score", 1.0)) if isinstance(detector_entry, dict) else 1.0
        detector_block_index = None
        try:
            detector_block_index = int(detector_entry.get("block_index")) if isinstance(detector_entry, dict) else None
        except Exception:
            detector_block_index = None
        block_index_mismatch = (
            block_index is not None and detector_block_index is not None and detector_block_index != int(block_index)
        )

        expected_crop_w = None
        expected_crop_h = None
        crop_dim_mismatch = False
        if detector_bbox is not None:
            expected_crop_w = max(0, int(detector_bbox[2]) - int(detector_bbox[0]))
            expected_crop_h = max(0, int(detector_bbox[3]) - int(detector_bbox[1]))
            crop_dim_mismatch = abs(expected_crop_w - int(w)) > 1 or abs(expected_crop_h - int(h)) > 1

        # Stage 3A (geometry phase): build candidates via split or passthrough.
        if stage3_mode == "passthrough":
            candidates = [(image, base_name, 0, w)]
        elif w <= width_thresh:
            candidates = [(image, base_name, 0, w)]
        else:
            candidates = [
                (image[:, x1:x2], f"{base_name}_split{i}", x1, x2)
                for i, (x1, x2) in enumerate(vertical_split(image))
            ]
            if not candidates:
                if rejection_log_enabled:
                    result["rejection_rows"].append(
                        {
                            "page_name": page_name,
                            "block_index": block_index,
                            "source_crop_file_path": str(file_path),
                            "rejection_reasons": ["too_large"],
                            "bbox_dimensions": {
                                "width": expected_crop_w,
                                "height": expected_crop_h,
                            },
                            "crop_dimensions": {"width": int(w), "height": int(h)},
                            "text_count": 0,
                            "ocr_text_length": 0,
                            "too_large": True,
                            "too_small": False,
                            "no_text": False,
                            "under_letter_threshold": False,
                            "graphic": False,
                            "unreadable": False,
                            "invalid_projection": False,
                            "other": False,
                        }
                    )

        for crop, name, local_x1, local_x2 in candidates:
            reasons = []
            text_meta = {"word_count": None, "text_length": None}
            accepted = True

            # Stage 3B (semantic rejection phase). recall_friendly keeps only tiny-box guard.
            if stage3_mode == "normal":
                if crop.shape[1] < 50 or crop.shape[0] < 30:
                    reasons.append("too_small")
                    result["skipped_tiny"] += 1
                    maybe_save_debug(crop, debug_dirs["tiny"] / f"{name}.png")
                elif is_graphic_like(crop):
                    reasons.append("graphic")
                    result["skipped_graphic"] += 1
                    maybe_save_debug(crop, debug_dirs["graphic"] / f"{name}.png")
                else:
                    is_short, has_text, text_meta = analyze_text_block(crop, return_details=True)
                    if not has_text:
                        reasons.append("no_text")
                    if is_short:
                        reasons.append("under_letter_threshold")
                    if reasons:
                        result["skipped_text"] += 1
                        maybe_save_debug(crop, debug_dirs["text"] / f"{name}.png")
            elif stage3_mode == "recall_friendly":
                if crop.shape[1] < 40 or crop.shape[0] < 24:
                    reasons.append("too_small")
                    result["skipped_tiny"] += 1
                    maybe_save_debug(crop, debug_dirs["tiny"] / f"{name}.png")

            if reasons:
                accepted = False
                if REFINER_VERBOSE_BLOCK_LOGS:
                    result["logs"].append(f"[{name}] Status: {'+'.join(reasons)} - skipped")
                if rejection_log_enabled and stage3_mode == "normal":
                    invalid_projection = detector_bbox is None
                    result["rejection_rows"].append(
                        {
                            "page_name": page_name,
                            "block_index": block_index,
                            "source_crop_file_path": str(file_path),
                            "candidate_name": name,
                            "rejection_reasons": reasons + (["invalid_projection"] if invalid_projection else []),
                            "bbox_dimensions": {
                                "width": expected_crop_w,
                                "height": expected_crop_h,
                            },
                            "crop_dimensions": {"width": int(crop.shape[1]), "height": int(crop.shape[0])},
                            "text_count": text_meta.get("word_count"),
                            "ocr_text_length": text_meta.get("text_length"),
                            "too_large": "too_large" in reasons,
                            "too_small": "too_small" in reasons,
                            "no_text": "no_text" in reasons,
                            "under_letter_threshold": "under_letter_threshold" in reasons,
                            "graphic": "graphic" in reasons,
                            "unreadable": False,
                            "invalid_projection": invalid_projection,
                            "other": False,
                        }
                    )

            projected_bbox = None
            if detector_bbox is not None:
                det_x1, det_y1, det_x2, det_y2 = detector_bbox
                if stage3_mode == "passthrough":
                    projected_bbox = [det_x1, det_y1, det_x2, det_y2]
                else:
                    projected_bbox = [
                        int(det_x1 + local_x1),
                        int(det_y1),
                        int(det_x1 + local_x2),
                        int(det_y2),
                    ]

            inside_bounds = _inside_page_bounds(page_dims, projected_bbox)
            inside_parent = _inside_or_near(detector_bbox, projected_bbox, tolerance=parent_tolerance)
            status_flags = []
            if detector_entry is None:
                status_flags.append("missing_detector_entry")
            if block_index_mismatch:
                status_flags.append("block_index_mismatch")
            if crop_dim_mismatch:
                status_flags.append("crop_dim_mismatch")
            if projected_bbox is not None and inside_bounds is False:
                status_flags.append("projected_out_of_bounds")
            if projected_bbox is not None and detector_bbox is not None and inside_parent is False:
                status_flags.append("projected_outside_parent")

            if geometry_audit_enabled:
                result["geometry_audit_rows"].append(
                    {
                        "page_name": page_name,
                        "block_index": block_index,
                        "source_crop_file_path": str(file_path),
                        "candidate_name": name,
                        "stage3_mode": stage3_mode,
                        "accepted": accepted,
                        "source_stage2_bbox_page_xyxy": detector_bbox,
                        "expected_crop_size_from_stage2_bbox": {
                            "width": expected_crop_w,
                            "height": expected_crop_h,
                        },
                        "actual_crop_image_size": {"width": int(w), "height": int(h)},
                        "candidate_crop_size": {"width": int(crop.shape[1]), "height": int(crop.shape[0])},
                        "stage3_local_bbox_before_projection": [
                            int(local_x1),
                            0,
                            int(local_x2),
                            int(crop.shape[0]),
                        ],
                        "stage3_projected_page_bbox": projected_bbox,
                        "page_dimensions": (
                            {"width": int(page_dims[0]), "height": int(page_dims[1])}
                            if page_dims is not None
                            else None
                        ),
                        "projected_inside_page_bounds": inside_bounds,
                        "projected_inside_or_near_parent_stage2_bbox": inside_parent,
                        "scale_or_offset_applied": {
                            "x_offset": int(local_x1),
                            "y_offset": 0,
                            "x_scale": 1.0,
                            "y_scale": 1.0,
                        },
                        "status_flags": status_flags,
                    }
                )

            if not accepted:
                continue

            save_buffer.append((refined_dir / f"{name}.png", crop))
            result["saved"] += 1
            if REFINER_VERBOSE_BLOCK_LOGS:
                result["logs"].append(f"[{name}] Status: ok - queued")

            if page_name and projected_bbox is not None:
                result["refined_metadata"].append(
                    {
                        "id": f"refined_{name}",
                        "bbox": projected_bbox,
                        "score": round(detector_score, 4),
                        "stage": "refined",
                        "page": page_name,
                        "block_index": block_index,
                        "source_detector_id": detector_entry.get("id") if isinstance(detector_entry, dict) else None,
                        "source_detector_bbox": detector_bbox,
                        "stage3_mode": stage3_mode,
                    }
                )

            if len(save_buffer) >= BATCH_SAVE_SIZE:
                batch_save_images(save_buffer)
                save_buffer.clear()

        if save_buffer:
            batch_save_images(save_buffer)

    except Exception as exc:
        result["skipped_error"] += 1
        result["logs"].append(f"[{file_path.name}] error: {exc}")
        if rejection_log_enabled and stage3_mode == "normal":
            result["rejection_rows"].append(
                {
                    "page_name": None,
                    "block_index": None,
                    "source_crop_file_path": str(file_path),
                    "rejection_reasons": ["other"],
                    "bbox_dimensions": None,
                    "crop_dimensions": None,
                    "text_count": 0,
                    "ocr_text_length": 0,
                    "too_large": False,
                    "too_small": False,
                    "no_text": False,
                    "under_letter_threshold": False,
                    "graphic": False,
                    "unreadable": False,
                    "invalid_projection": False,
                    "other": True,
                }
            )
        if save_buffer:
            try:
                batch_save_images(save_buffer)
            except Exception:
                pass

    return result


# ───── Process Pool Initializer ─────
def process_initializer():
    """Initialize each worker process with Tesseract warm-up"""
    warm_up_tesseract()


def refine_page_blocks(
    pdf_folder: str,
    page_name: str,
    input_base: str = "data/job_blocks_smart",
    output_base: str = "data/job_blocks_refined",
    width_thresh: int = REFINER_WIDTH_THRESH,
    detections_output: str = "run_state/detections",
    write_metadata: bool = False,
    enable_debug_sampling: bool = False,
    stage3_mode: str = "normal",
    geometry_audit: bool = False,
    rejection_log: bool = False,
    images_base: str = "data/pdf2img",
    parent_tolerance: int = GEOMETRY_PARENT_TOLERANCE,
):
    """
    Reusable per-page refiner helper for streaming and batch runners.
    Returns aggregate per-page refinement stats and refined metadata entries.
    """
    input_base_path = Path(input_base)
    output_base_path = Path(output_base)
    detections_output_path = Path(detections_output)
    detections_output_path.mkdir(parents=True, exist_ok=True)

    subfolder = input_base_path / str(pdf_folder)
    refined_dir = output_base_path / str(pdf_folder)
    refined_dir.mkdir(parents=True, exist_ok=True)

    page_stem = Path(page_name).stem
    block_files = sorted(
        [f for f in subfolder.glob(f"{page_stem}_block*.png") if not f.name.startswith("debug_")]
    )

    debug_base = Path("data/refiner_skipped")
    debug_enabled = enable_debug_sampling or DEBUG_SAMPLE_RATE > 0.0
    if debug_enabled:
        debug_dirs = {
            "tiny": debug_base / "tiny",
            "graphic": debug_base / "graphic",
            "text": debug_base / "text",
        }
        for d in debug_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
    else:
        # Sampling is disabled by default and DEBUG_SAMPLE_RATE is typically 0.0,
        # but process_image_file expects these paths to exist.
        debug_dirs = {
            "tiny": debug_base / "tiny",
            "graphic": debug_base / "graphic",
            "text": debug_base / "text",
        }
        for d in debug_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "page": page_name,
        "pdf_folder": str(pdf_folder),
        "input_blocks": len(block_files),
        "saved": 0,
        "skipped_tiny": 0,
        "skipped_graphic": 0,
        "skipped_text": 0,
        "skipped_unreadable": 0,
        "skipped_error": 0,
        "refined_metadata": [],
        "logs": [],
        "geometry_audit_rows": [],
        "rejection_rows": [],
    }

    page_dims_map = {}
    if geometry_audit:
        images_base_path = Path(images_base)
        for img_path in images_base_path.rglob("*.png"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            page_dims_map[img_path.name] = (int(iw), int(ih))

    for block_file in block_files:
        res = process_image_file(
            {
                "file_path": str(block_file),
                "width_thresh": width_thresh,
                "debug_dirs": debug_dirs,
                "refined_dir": str(refined_dir),
                "detections_dir": str(detections_output_path),
                "stage3_mode": stage3_mode,
                "geometry_audit": geometry_audit,
                "rejection_log": rejection_log,
                "page_dims_map": page_dims_map,
                "parent_tolerance": parent_tolerance,
            }
        )
        aggregate["saved"] += res.get("saved", 0)
        aggregate["skipped_tiny"] += res.get("skipped_tiny", 0)
        aggregate["skipped_graphic"] += res.get("skipped_graphic", 0)
        aggregate["skipped_text"] += res.get("skipped_text", 0)
        aggregate["skipped_unreadable"] += res.get("skipped_unreadable", 0)
        aggregate["skipped_error"] += res.get("skipped_error", 0)
        aggregate["logs"].extend(res.get("logs", []))
        aggregate["refined_metadata"].extend(res.get("refined_metadata", []))
        aggregate["geometry_audit_rows"].extend(res.get("geometry_audit_rows", []))
        aggregate["rejection_rows"].extend(res.get("rejection_rows", []))

    if write_metadata:
        _upsert_refined_metadata(page_name, aggregate["refined_metadata"], detections_output_path)

    return aggregate


# ───── Main Refinement ─────
def refine_blocks_all(
    input_base="data/job_blocks_smart",
    output_base="data/job_blocks_refined",
    width_thresh=REFINER_WIDTH_THRESH,
    detections_output="run_state/detections",
    stage3_mode="normal",
    geometry_audit=False,
    geometry_audit_output="run_state/stage3_geometry_audit.json",
    geometry_audit_summary_output="run_state/stage3_geometry_audit_summary.json",
    rejection_log=False,
    rejection_log_output="run_state/stage3_rejection_log.json",
    rejection_summary_output="run_state/stage3_rejection_summary.json",
    images_base="data/pdf2img",
    parent_tolerance=GEOMETRY_PARENT_TOLERANCE,
):
    start = time.time()
    if stage3_mode not in STAGE3_MODE_CHOICES:
        raise ValueError(f"Unsupported stage3_mode={stage3_mode}. Expected one of {STAGE3_MODE_CHOICES}")

    input_base = Path(input_base)
    output_base = Path(output_base)
    detections_output = Path(detections_output)
    output_base.mkdir(parents=True, exist_ok=True)
    detections_output.mkdir(parents=True, exist_ok=True)

    TARGET_ONLY = None  # for testing
    total, saved = 0, 0
    skipped_tiny = 0
    skipped_graphic = 0
    skipped_text = 0
    skipped_unreadable = 0
    skipped_error = 0
    refined_by_page = {}
    geometry_rows = []
    rejection_rows = []

    debug_base = Path("data/refiner_skipped")
    debug_dirs = {
        "tiny": debug_base / "tiny",
        "graphic": debug_base / "graphic",
        "text": debug_base / "text"
    }
    if DEBUG_SAMPLE_RATE > 0.0:
        for d in debug_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    page_dims_map = {}
    if geometry_audit:
        images_base_path = Path(images_base)
        for img_path in images_base_path.rglob("*.png"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            page_dims_map[img_path.name] = (int(iw), int(ih))

    force_sequential = bool(geometry_audit or rejection_log or stage3_mode == "passthrough")

    logging.info(f"📂 Starting multi-folder refinement: {input_base}")
    print(f"🚀 Starting refinement of: {input_base}", flush=True)
    print(f"[stage3] mode={stage3_mode} geometry_audit={int(bool(geometry_audit))} rejection_log={int(bool(rejection_log))}", flush=True)

    for subfolder in input_base.iterdir():
        if not subfolder.is_dir():
            continue
        if TARGET_ONLY and subfolder.name != TARGET_ONLY:
            continue

        logging.info(f"📁 Processing subfolder: {subfolder.name}")
        print(f"📁 Processing: {subfolder.name}", flush=True)
        
        refined_dir = output_base / subfolder.name
        refined_dir.mkdir(parents=True, exist_ok=True)

        block_files = sorted([f for f in subfolder.glob("*.png")
                              if not f.name.startswith("debug_")])

        args_list = [
            {
                "file_path": str(file_path),
                "width_thresh": width_thresh,
                "debug_dirs": debug_dirs,
                "refined_dir": str(refined_dir),
                "detections_dir": str(detections_output),
                "stage3_mode": stage3_mode,
                "geometry_audit": geometry_audit,
                "rejection_log": rejection_log,
                "page_dims_map": page_dims_map,
                "parent_tolerance": parent_tolerance,
            }
            for file_path in block_files
        ]

        if force_sequential:
            iterator = (
                process_image_file(arg)
                for arg in tqdm(args_list, total=len(args_list), desc=f"🧩 Refining {subfolder.name}")
            )
        else:
            max_workers = min(10, multiprocessing.cpu_count())
            executor = ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=process_initializer,
            )
            iterator = tqdm(
                executor.map(process_image_file, args_list, chunksize=REFINER_MAP_CHUNKSIZE),
                total=len(args_list),
                desc=f"🧩 Refining {subfolder.name}",
            )

        try:
            for res in iterator:
                total += 1
                saved += res["saved"]
                skipped_tiny += res["skipped_tiny"]
                skipped_graphic += res["skipped_graphic"]
                skipped_text += res["skipped_text"]
                skipped_unreadable += res["skipped_unreadable"]
                skipped_error += res["skipped_error"]
                geometry_rows.extend(res.get("geometry_audit_rows", []))
                rejection_rows.extend(res.get("rejection_rows", []))
                for md in res.get("refined_metadata", []):
                    page_name = md.get("page")
                    if not page_name:
                        continue
                    refined_by_page.setdefault(page_name, []).append(md)

                for log_entry in res.get("logs", []):
                    logging.info(log_entry)
        finally:
            if not force_sequential:
                executor.shutdown(wait=True)

        # Periodic terminal update
        print(f"[{subfolder.name}] {saved}/{total} processed...", flush=True)

    # Final terminal summary
    elapsed = time.time() - start
    total_skipped = skipped_tiny + skipped_graphic + skipped_text + skipped_unreadable + skipped_error
    
    print(f"\n✅ Completed refinement in {elapsed:.1f}s", flush=True)
    print(f"📦 Saved: {saved} | ⏩ Skipped: {total_skipped}", flush=True)
    print(f"  └── Tiny: {skipped_tiny}", flush=True)
    print(f"  └── Graphic: {skipped_graphic}", flush=True)
    print(f"  └── Text: {skipped_text}", flush=True)
    print(f"  └── Unreadable: {skipped_unreadable}", flush=True)
    print(f"  └── Errors: {skipped_error}", flush=True)

    logging.info(f"✅ Completed refinement.")
    logging.info(f"📦 Saved refined blocks: {saved}")
    logging.info(f"🧮 Total input: {total}")
    logging.info(f"🚫 Skipped (tiny): {skipped_tiny}")
    logging.info(f"🚫 Skipped (graphic-like): {skipped_graphic}")
    logging.info(f"🚫 Skipped (short text): {skipped_text}")
    logging.info(f"🚫 Skipped (unreadable): {skipped_unreadable}")
    logging.info(f"🚫 Skipped (errors): {skipped_error}")
    for page_name, entries in refined_by_page.items():
        _upsert_refined_metadata(page_name, entries, detections_output)

    if geometry_audit:
        geometry_audit_output = Path(geometry_audit_output)
        geometry_audit_summary_output = Path(geometry_audit_summary_output)
        geometry_payload = {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "stage3_mode": stage3_mode,
            "parent_tolerance_px": int(parent_tolerance),
            "rows": geometry_rows,
        }
        geometry_summary = _summarize_geometry_audit(geometry_rows)
        geometry_summary["generated_at_utc"] = geometry_payload["generated_at_utc"]
        geometry_summary["stage3_mode"] = stage3_mode
        geometry_audit_output.parent.mkdir(parents=True, exist_ok=True)
        geometry_audit_summary_output.parent.mkdir(parents=True, exist_ok=True)
        geometry_audit_output.write_text(json.dumps(geometry_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        geometry_audit_summary_output.write_text(json.dumps(geometry_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[stage3] geometry audit: {geometry_audit_output}", flush=True)
        print(f"[stage3] geometry audit summary: {geometry_audit_summary_output}", flush=True)

    if rejection_log and stage3_mode == "normal":
        rejection_log_output = Path(rejection_log_output)
        rejection_summary_output = Path(rejection_summary_output)
        rejection_payload = {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "stage3_mode": stage3_mode,
            "rows": rejection_rows,
        }
        rejection_summary = _summarize_rejections(rejection_rows)
        rejection_summary["generated_at_utc"] = rejection_payload["generated_at_utc"]
        rejection_summary["stage3_mode"] = stage3_mode
        rejection_log_output.parent.mkdir(parents=True, exist_ok=True)
        rejection_summary_output.parent.mkdir(parents=True, exist_ok=True)
        rejection_log_output.write_text(json.dumps(rejection_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        rejection_summary_output.write_text(json.dumps(rejection_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[stage3] rejection log: {rejection_log_output}", flush=True)
        print(f"[stage3] rejection summary: {rejection_summary_output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage03 block refiner")
    parser.add_argument("--config", default="configs/pipeline_paths.json", help="Path to pipeline paths config")
    parser.add_argument(
        "--stage3-mode",
        choices=list(STAGE3_MODE_CHOICES),
        default="normal",
        help="Stage3 mode: normal refinement/filtering or passthrough metadata mirroring detector boxes",
    )
    parser.add_argument("--geometry-audit", action="store_true", help="Write per-block geometry audit JSON + summary")
    parser.add_argument("--rejection-log", action="store_true", help="Write explicit Stage3 normal-mode rejection reasons")
    parser.add_argument(
        "--geometry-audit-output",
        default="run_state/stage3_geometry_audit.json",
        help="Path for detailed geometry audit JSON",
    )
    parser.add_argument(
        "--geometry-audit-summary-output",
        default="run_state/stage3_geometry_audit_summary.json",
        help="Path for geometry audit summary JSON",
    )
    parser.add_argument(
        "--rejection-log-output",
        default="run_state/stage3_rejection_log.json",
        help="Path for detailed rejection log JSON",
    )
    parser.add_argument(
        "--rejection-summary-output",
        default="run_state/stage3_rejection_summary.json",
        help="Path for rejection summary JSON",
    )
    parser.add_argument(
        "--geometry-parent-tolerance",
        type=int,
        default=GEOMETRY_PARENT_TOLERANCE,
        help="Tolerance in pixels when checking projected bbox inside parent detector bbox",
    )
    cli_args = parser.parse_args()

    config = load_config(cli_args.config)
    images_output = get_path("images_output", config)
    blocks_output = get_path("blocks_output", config)
    refined_output = get_path("refined_output", config)
    detections_output = get_path("detections_output", config)

    print("[CONFIG]", flush=True)
    print(f"images_output = {images_output}", flush=True)
    print(f"blocks_output = {blocks_output}", flush=True)
    print(f"refined_output = {refined_output}", flush=True)
    print(f"detections_output = {detections_output}", flush=True)
    print(f"stage3_mode = {cli_args.stage3_mode}", flush=True)
    print(f"geometry_audit = {cli_args.geometry_audit}", flush=True)
    print(f"rejection_log = {cli_args.rejection_log}", flush=True)

    start = time.time()
    refine_blocks_all(
        input_base=str(blocks_output),
        output_base=str(refined_output),
        detections_output=str(detections_output),
        stage3_mode=cli_args.stage3_mode,
        geometry_audit=cli_args.geometry_audit,
        geometry_audit_output=cli_args.geometry_audit_output,
        geometry_audit_summary_output=cli_args.geometry_audit_summary_output,
        rejection_log=cli_args.rejection_log,
        rejection_log_output=cli_args.rejection_log_output,
        rejection_summary_output=cli_args.rejection_summary_output,
        images_base=str(images_output),
        parent_tolerance=cli_args.geometry_parent_tolerance,
    )
    end = time.time()
    logging.info(f"⏱️ Elapsed time: {end - start:.2f} seconds")
