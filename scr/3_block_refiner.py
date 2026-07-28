import cv2
import numpy as np
import os
import time
import logging
import pytesseract
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import random
import sys
from tqdm import tqdm
from PIL import Image

# ───── Config ─────
DEBUG_SAMPLE_RATE = 0.0  # Save only 0% of skipped blocks for debugging
TESSERACT_CONFIG = "--oem 1 --psm 6"  # Cached config for faster OCR
BATCH_SAVE_SIZE = 80  # Increased batch size for better I/O performance

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
def analyze_text_block(image, max_words=12, max_chars=50, min_confidence=40):
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


def is_graphic_like(image, threshold=0.12):
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
def vertical_split(image, min_gap_width=20, min_col_width=200):
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
    if random.random() < DEBUG_SAMPLE_RATE:
        cv2.imwrite(str(path), image)


def batch_save_images(save_buffer):
    """Save multiple images in batch to reduce I/O overhead"""
    for path, img in save_buffer:
        cv2.imwrite(str(path), img)


def process_image_file(args):
    file_path_str, width_thresh, debug_dirs, refined_dir = args
    file_path = Path(file_path_str)
    result = {
        "saved": 0,
        "skipped_tiny": 0,
        "skipped_graphic": 0,
        "skipped_text": 0,
        "skipped_unreadable": 0,
        "skipped_error": 0,
        "logs": []
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

        blocks = [(image, base_name)] if w <= width_thresh else [
            (image[:, x1:x2], f"{base_name}_split{i}")
            for i, (x1, x2) in enumerate(vertical_split(image))
        ]

        for crop, name in blocks:
            if crop.shape[1] < 50 or crop.shape[0] < 30:
                result["skipped_tiny"] += 1
                maybe_save_debug(crop, debug_dirs["tiny"] / f"{name}.png")
                result["logs"].append(f"[{name}] Status: tiny - skipped")
                continue

            if is_graphic_like(crop):
                result["skipped_graphic"] += 1
                maybe_save_debug(crop, debug_dirs["graphic"] / f"{name}.png")
                result["logs"].append(f"[{name}] Status: graphic - skipped")
                continue

            is_short, has_text = analyze_text_block(crop)
            if is_short or not has_text:
                result["skipped_text"] += 1
                maybe_save_debug(crop, debug_dirs["text"] / f"{name}.png")
                result["logs"].append(f"[{name}] Status: text - skipped")
                continue

            # Add to save buffer instead of immediate write
            save_buffer.append((refined_dir / f"{name}.png", crop))
            result["logs"].append(f"[{name}] Status: ok - queued")
            result["saved"] += 1

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


# ───── Main Refinement ─────
def refine_blocks_all(input_base="data/job_blocks_smart",
                      output_base="data/job_blocks_refined",
                      width_thresh=800):
    input_base = Path(input_base)
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    TARGET_ONLY = None  # for testing
    total, saved = 0, 0
    skipped_tiny = 0
    skipped_graphic = 0
    skipped_text = 0
    skipped_unreadable = 0
    skipped_error = 0

    debug_base = Path("data/refiner_skipped")
    debug_dirs = {
        "tiny": debug_base / "tiny",
        "graphic": debug_base / "graphic",
        "text": debug_base / "text"
    }
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
            (str(file_path), width_thresh, debug_dirs, refined_dir)
            for file_path in block_files
        ]

        # Use tqdm for real-time progress tracking with process initializer
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=process_initializer
        ) as executor:
            for res in tqdm(
                executor.map(process_image_file, args_list),
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


if __name__ == "__main__":
    start = time.time()
    refine_blocks_all()
    end = time.time()
    logging.info(f"⏱️ Elapsed time: {end - start:.2f} seconds")