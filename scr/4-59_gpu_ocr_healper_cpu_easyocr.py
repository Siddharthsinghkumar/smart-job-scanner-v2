#!/usr/bin/env python3
"""
cpu_easyocr_only.py

CPU-only EasyOCR pipeline (Hindi + English friendly).
"""

import easyocr
import logging
from pathlib import Path
import re
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

# ───── Config ─────
AVAILABLE_LANGS = ["hi", "en"]  # fixed, no dynamic detection
DEFAULT_LANGS = ["hi", "en"]

# ───── Logging Setup ─────
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_name = f"easyocr_cpu_extract_{timestamp}.log"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(log_dir / log_name, mode="a", encoding="utf-8")
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

# ───── Hindi detection ─────
if "hi" in AVAILABLE_LANGS:
    HINDI_LANG = "hi"
else:
    HINDI_LANG = None

logger.info(f"🔍 EasyOCR available languages: {AVAILABLE_LANGS}")
if HINDI_LANG:
    logger.info(f"✅ Hindi support detected: {HINDI_LANG}")
else:
    logger.warning("⚠️ No Hindi support found, will use English only!")

# ───── Cached Reader ─────
_reader_cache = {}

def get_reader(langs):
    key = "+".join(langs)
    if key not in _reader_cache:
        _reader_cache[key] = easyocr.Reader(langs, gpu=False)
    return _reader_cache[key]

# ───── Helpers ─────
def get_lang_for_newspaper(newspaper_name: str):
    n = newspaper_name.lower()
    hindi_keys = (
        "db", "dainik", "navbharat", "amar", "hindi", "amarujala", "bhaskar", "jantak",
    )
    english_keys = (
        "the", "indianexpress", "express", "times", "hindu", "ie", "th", "english",
    )

    if any(k in n for k in hindi_keys) and HINDI_LANG:
        return [HINDI_LANG, "en"]
    if any(k in n for k in english_keys):
        return ["en"]
    return [HINDI_LANG, "en"] if HINDI_LANG else ["en"]

def easyocr_image(image_path: Path, langs):
    try:
        reader = get_reader(langs)
        results = reader.readtext(str(image_path), detail=0, paragraph=True)
        if results:
            return (1, 0, "\n".join([r.strip() for r in results if r.strip()]))
        else:
            return (0, 1, "")
    except Exception as e:
        logger.error(f"[✖] EasyOCR crashed on {image_path.name}: {e}")
        return (0, 1, "")

# ───── OCR Folder Processing ─────
def extract_text_by_page_folder(refined_folder: Path):
    newspaper_name = refined_folder.name
    logger.info(f"📰 Processing newspaper folder: {newspaper_name}")

    output_dir = Path("data/page_texts") / newspaper_name
    output_dir.mkdir(parents=True, exist_ok=True)

    langs = get_lang_for_newspaper(newspaper_name)
    logger.info(f"[lang] Using languages '{'+'.join(langs)}' for: {newspaper_name}")

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
        return (0, 0, 0, 0, 0)

    logger.info(f"📄 Found {len(page_groups)} page(s) in: {newspaper_name}")

    total_pages = total_blocks = success_blocks = failed_blocks = 0
    max_used_workers = 0

    for page, img_paths in sorted(
        page_groups.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]
    ):
        total_pages += 1
        logger.info(f"📃 Page {page}: {len(img_paths)} blocks")

        results = []
        max_workers = min(4, len(img_paths))
        used_workers = max_workers
        max_used_workers = max(max_used_workers, used_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(easyocr_image, img_path, langs): img_path
                for img_path in img_paths
            }
            for future in as_completed(futures):
                img_name = futures[future].name
                s, f, text = future.result()
                total_blocks += 1
                success_blocks += s
                failed_blocks += f

                if text:
                    results.append(text)
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
    logger.info(f"⚡ Max worker threads used: {max_used_workers}")
    logger.info(f"⏱️ Total time taken: {elapsed:.2f} seconds")
    logger.info("=============================")


if __name__ == "__main__":
    extract_all_folders()
