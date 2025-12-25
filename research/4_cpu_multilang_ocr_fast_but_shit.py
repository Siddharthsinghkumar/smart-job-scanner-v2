#!/usr/bin/env python3
"""
cpu_multilang_ocr.py

CPU-only Tesseract-based OCR pipeline (Hindi + English friendly).
Now optimized with multiprocessing (per-page parallel OCR).
"""

import cv2
import pytesseract
import logging
import numpy as np
from pathlib import Path
import re
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count  # <-- added

# ───── Config ─────
DEFAULT_LANGS = "hin+eng"  # multilingual default
PSM_FALLBACKS = [6, 4, 7, 3]  # fallback PSM modes
GIBBERISH_RATIO_THRESHOLD = 0.30

# Image resize thresholds
WIDTH_UPSCALE_SMALL = 800
WIDTH_UPSCALE_MEDIUM = 1200
UPSCALE_SMALL = 2.0
UPSCALE_MEDIUM = 1.5
UPSCALE_LARGE = 1.0

# ───── Logging Setup ─────
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_name = f"extract_text_{timestamp}.log"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(log_dir / log_name, mode='a', encoding='utf-8')
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

# ───── Regex for gibberish detection ─────
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
LATIN_ALNUM_RE = re.compile(r'[A-Za-z0-9]')


# ───── Helpers ─────
def get_lang_for_newspaper(newspaper_name: str) -> str:
    n = newspaper_name.lower()
    hindi_keys = ("db", "dainik", "navbharat", "amar", "hindi", "amarujala", "bhaskar", "jantak")
    english_keys = ("the", "indianexpress", "express", "times", "hindu", "ie", "th", "english")
    if any(k in n for k in hindi_keys):
        return "hin+eng"
    if any(k in n for k in english_keys):
        return "eng"
    return DEFAULT_LANGS


def is_gibberish(text: str) -> bool:
    if not text:
        return True
    stripped = re.sub(r'\s+', '', text)
    if not stripped:
        return True
    valid_chars = DEVANAGARI_RE.findall(stripped)
    valid_count = len(valid_chars) + len(LATIN_ALNUM_RE.findall(stripped))
    ratio = valid_count / len(stripped)
    return ratio < GIBBERISH_RATIO_THRESHOLD


def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\x0c', '')
    text = re.sub(r'�+', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    if w < WIDTH_UPSCALE_SMALL:
        fx = UPSCALE_SMALL
    elif w < WIDTH_UPSCALE_MEDIUM:
        fx = UPSCALE_MEDIUM
    else:
        fx = UPSCALE_LARGE

    if fx != 1.0:
        gray = cv2.resize(gray, None, fx=fx, fy=fx, interpolation=cv2.INTER_LINEAR)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    if np.mean(blur) < 127:
        clean = cv2.bitwise_not(clean)

    return clean


def tesseract_ocr_image(image: np.ndarray, langs: str, image_name: str = "?") -> str:
    best_text = ""
    for psm in PSM_FALLBACKS:
        try:
            config = f"--oem 3 --psm {psm}"
            text = pytesseract.image_to_string(image, lang=langs, config=config)
            text = clean_ocr_text(text)
            if text and not is_gibberish(text):
                return text
            if len(text) > len(best_text):
                best_text = text
        except Exception as e:
            logging.error(f"[✖] OCR crashed on {image_name}, psm={psm}: {e}")
            continue
    return best_text


def extract_text_from_image(image_path: Path, langs: str = DEFAULT_LANGS):
    """Return (success_count, fail_count, text) for one image."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return (0, 1, "")

        processed = preprocess_for_ocr(image)
        text = tesseract_ocr_image(processed, langs=langs, image_name=image_path.name)
        if text:
            return (1, 0, text.strip())
        else:
            return (0, 1, "")
    except Exception:
        return (0, 1, "")


# ───── OCR Folder Processing ─────
def extract_text_by_page_folder(refined_folder: Path):
    newspaper_name = refined_folder.name
    logger.info(f"📰 Processing newspaper folder: {newspaper_name}")

    output_dir = Path("data/page_texts") / newspaper_name
    output_dir.mkdir(parents=True, exist_ok=True)

    langs = get_lang_for_newspaper(newspaper_name)
    logger.info(f"[lang] Using languages '{langs}' for: {newspaper_name}")

    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    all_imgs = []
    for e in exts:
        all_imgs.extend(list(refined_folder.glob(e)))
    all_imgs = sorted(all_imgs, key=lambda p: p.name)

    page_groups = defaultdict(list)
    for img_path in all_imgs:
        match = re.search(r"_p(\d+)", img_path.name, re.IGNORECASE)
        if match:
            page_number = match.group(1)
            page_groups[page_number].append(img_path)
        else:
            page_groups["0"].append(img_path)

    if not page_groups:
        logger.warning(f"[🚫] No valid page-labeled images found in {newspaper_name}")
        return (0, 0, 0, 0)

    logger.info(f"📄 Found {len(page_groups)} page(s) in: {newspaper_name}")

    total_pages = 0
    total_blocks = 0
    success_blocks = 0
    failed_blocks = 0
    max_used_workers = 0

    for page, img_paths in sorted(page_groups.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        total_pages += 1
        logger.info(f"📃 Page {page}: {len(img_paths)} blocks")

        results = []
        max_workers = min(8, len(img_paths))  # limit workers    for Tesseract 8 give best core to time efficiency for 24mb cache keep cache/core to =< 3
        used_workers=max_workers
        max_used_workers = max(max_used_workers, used_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_text_from_image, img_path, langs): img_path for img_path in img_paths}
            for future in as_completed(futures):
                img_name = futures[future].name
                s, f, text = future.result()
                total_blocks += 1
                success_blocks += s
                failed_blocks += f

                if text:
                    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                    if lines:
                        results.append("\n".join(lines))
                    else:
                        logger.warning(f"[🗑️] Blank-lines-only from: {img_name}")
                else:
                    logger.warning(f"[❌] No text from: {img_name}")

        if results:
            page_text = "\n\n".join(results)
            out_file = output_dir / f"{newspaper_name}_p{page}_text.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(page_text)
            logger.info(f"[💾] Saved page {page} to {out_file.name}")
        else:
            logger.warning(f"[❌] No text extracted for page {page}")

    return (total_pages, total_blocks, success_blocks, failed_blocks, max_used_workers)


# ───── Entry Point ─────
def extract_all_folders(base_dir="data/gpu_ocr_errors"):
    start_time = datetime.now()

    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"[🚫] Base path does not exist: {base_dir}")
        return

    folders = [f for f in base_path.iterdir() if f.is_dir()]
    if not folders:
        logger.warning("[🚫] No folders found in gpu_ocr_errors")
        return

    total_pages = total_blocks = success_blocks = failed_blocks = 0
    max_used_workers = 0

    for folder in sorted(folders, key=lambda p: p.name):
        p, b, s, f, w = extract_text_by_page_folder(folder)
        total_pages += p
        total_blocks += b
        success_blocks += s
        failed_blocks += f
        max_used_workers = max(max_used_workers, w)

    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info("✅ All folders processed.")
    logger.info("========== SUMMARY ==========")
    logger.info(f"Total newspapers: {len(folders)}")
    logger.info(f"Total pages processed: {total_pages}")
    logger.info(f"Total blocks processed: {total_blocks}")
    logger.info(f"Blocks with text: {success_blocks}")
    logger.info(f"Blocks failed: {failed_blocks}")
    logger.info(f"🖥️ CPU cores available: {cpu_count()}")
    logger.info(f"⚡ Max worker processes used: {max_used_workers}")
    logger.info(f"⏱️ Total time taken: {elapsed:.2f} seconds")
    logger.info("=============================")


if __name__ == "__main__":
    extract_all_folders()
