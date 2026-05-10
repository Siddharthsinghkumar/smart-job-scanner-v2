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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import random
import re
import sys
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

        return is_short, has_text
    except Exception:
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


def process_image_file(args):
    file_path_str, width_thresh, debug_dirs, refined_dir, detections_dir = args
    file_path = Path(file_path_str)
    result = {
        "saved": 0,
        "skipped_tiny": 0,
        "skipped_graphic": 0,
        "skipped_text": 0,
        "skipped_unreadable": 0,
        "skipped_error": 0,
        "logs": [],
        "refined_metadata": []
    }

    save_buffer = []  # Buffer for batch saving

    try:
        # Use optimized image loading (OpenCV default, Pillow for very large files)
        image = fast_read_image(str(file_path))
        if image is None:
            result["skipped_unreadable"] += 1
            return result

        h, w = image.shape[:2]
        base_name = file_path.stem
        page_stem, block_index = _parse_block_reference(base_name)
        page_name = f"{page_stem}.png" if page_stem else None
        detector_entry = None
        detector_bbox = None
        detector_score = 1.0

        blocks = [(image, base_name, 0, w)] if w <= width_thresh else [
            (image[:, x1:x2], f"{base_name}_split{i}", x1, x2)
            for i, (x1, x2) in enumerate(vertical_split(image))
        ]

        for crop, name, local_x1, local_x2 in blocks:
            if crop.shape[1] < 50 or crop.shape[0] < 30:
                result["skipped_tiny"] += 1
                maybe_save_debug(crop, debug_dirs["tiny"] / f"{name}.png")
                if REFINER_VERBOSE_BLOCK_LOGS:
                    result["logs"].append(f"[{name}] Status: tiny - skipped")
                continue

            if is_graphic_like(crop):
                result["skipped_graphic"] += 1
                maybe_save_debug(crop, debug_dirs["graphic"] / f"{name}.png")
                if REFINER_VERBOSE_BLOCK_LOGS:
                    result["logs"].append(f"[{name}] Status: graphic - skipped")
                continue

            is_short, has_text = analyze_text_block(crop)
            if is_short or not has_text:
                result["skipped_text"] += 1
                maybe_save_debug(crop, debug_dirs["text"] / f"{name}.png")
                if REFINER_VERBOSE_BLOCK_LOGS:
                    result["logs"].append(f"[{name}] Status: text - skipped")
                continue

            # Add to save buffer instead of immediate write
            save_buffer.append((refined_dir / f"{name}.png", crop))
            if REFINER_VERBOSE_BLOCK_LOGS:
                result["logs"].append(f"[{name}] Status: ok - queued")
            result["saved"] += 1
            if page_name and block_index is not None:
                if detector_entry is None:
                    detector_entry = _load_detector_entry(page_name, block_index, detections_dir)
                    detector_bbox = detector_entry.get("bbox") if detector_entry else None
                    detector_score = float(detector_entry.get("score", 1.0)) if detector_entry else 1.0

            if page_name and detector_bbox and len(detector_bbox) == 4:
                det_x1, det_y1, det_x2, det_y2 = [int(v) for v in detector_bbox]
                refined_bbox = [
                    int(det_x1 + local_x1),
                    int(det_y1),
                    int(det_x1 + local_x2),
                    int(det_y2),
                ]
                result["refined_metadata"].append(
                    {
                        "id": f"refined_{name}",
                        "bbox": refined_bbox,
                        "score": round(detector_score, 4),
                        "stage": "refined",
                        "page": page_name,
                        "source_detector_id": detector_entry.get("id"),
                        "source_detector_bbox": [det_x1, det_y1, det_x2, det_y2],
                    }
                )

            # Batch save when buffer reaches threshold (larger batch size)
            if len(save_buffer) >= BATCH_SAVE_SIZE:
                batch_save_images(save_buffer)
                save_buffer.clear()

        # Save any remaining images in buffer
        if save_buffer:
            batch_save_images(save_buffer)

    except Exception:
        result["skipped_error"] += 1
        # Try to save any buffered images even if error occurred
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
    }

    for block_file in block_files:
        res = process_image_file(
            (str(block_file), width_thresh, debug_dirs, refined_dir, detections_output_path)
        )
        aggregate["saved"] += res.get("saved", 0)
        aggregate["skipped_tiny"] += res.get("skipped_tiny", 0)
        aggregate["skipped_graphic"] += res.get("skipped_graphic", 0)
        aggregate["skipped_text"] += res.get("skipped_text", 0)
        aggregate["skipped_unreadable"] += res.get("skipped_unreadable", 0)
        aggregate["skipped_error"] += res.get("skipped_error", 0)
        aggregate["logs"].extend(res.get("logs", []))
        aggregate["refined_metadata"].extend(res.get("refined_metadata", []))

    if write_metadata:
        _upsert_refined_metadata(page_name, aggregate["refined_metadata"], detections_output_path)

    return aggregate


# ───── Main Refinement ─────
def refine_blocks_all(input_base="data/job_blocks_smart",
                      output_base="data/job_blocks_refined",
                      width_thresh=REFINER_WIDTH_THRESH,
                      detections_output="run_state/detections"):
    start = time.time()
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

    debug_base = Path("data/refiner_skipped")
    debug_dirs = {
        "tiny": debug_base / "tiny",
        "graphic": debug_base / "graphic",
        "text": debug_base / "text"
    }
    if DEBUG_SAMPLE_RATE > 0.0:
        for d in debug_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    logging.info(f"📂 Starting multi-folder refinement: {input_base}")
    print(f"🚀 Starting refinement of: {input_base}", flush=True)

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

        max_workers = min(14, multiprocessing.cpu_count())
        args_list = [
            (str(file_path), width_thresh, debug_dirs, refined_dir, detections_output)
            for file_path in block_files
        ]

        # Use tqdm for real-time progress tracking with process initializer
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=process_initializer
        ) as executor:
            for res in tqdm(
                executor.map(process_image_file, args_list, chunksize=REFINER_MAP_CHUNKSIZE),
                total=len(args_list),
                desc=f"🧩 Refining {subfolder.name}"
            ):
                total += 1
                saved += res["saved"]
                skipped_tiny += res["skipped_tiny"]
                skipped_graphic += res["skipped_graphic"]
                skipped_text += res["skipped_text"]
                skipped_unreadable += res["skipped_unreadable"]
                skipped_error += res["skipped_error"]
                for md in res.get("refined_metadata", []):
                    page_name = md.get("page")
                    if not page_name:
                        continue
                    refined_by_page.setdefault(page_name, []).append(md)

                for log_entry in res.get("logs", []):
                    logging.info(log_entry)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage03 block refiner")
    parser.add_argument("--config", default="configs/pipeline_paths.json", help="Path to pipeline paths config")
    cli_args = parser.parse_args()

    config = load_config(cli_args.config)
    blocks_output = get_path("blocks_output", config)
    refined_output = get_path("refined_output", config)
    detections_output = get_path("detections_output", config)

    print("[CONFIG]", flush=True)
    print(f"blocks_output = {blocks_output}", flush=True)
    print(f"refined_output = {refined_output}", flush=True)
    print(f"detections_output = {detections_output}", flush=True)

    start = time.time()
    refine_blocks_all(
        input_base=str(blocks_output),
        output_base=str(refined_output),
        detections_output=str(detections_output),
    )
    end = time.time()
    logging.info(f"⏱️ Elapsed time: {end - start:.2f} seconds")
