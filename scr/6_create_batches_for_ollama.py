import os
import re
import logging
from pathlib import Path
from datetime import datetime

# ───── Config ─────
INPUT_BASE_DIR = Path("data/all_eng_text")
OUTPUT_DIR = Path("data/batch_inputs")
MAX_WORDS_PER_BATCH = 1000
OVERLAP_WORDS = 50

# ───── Logging Setup ─────
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_name = f"batch_create_{timestamp}.log"

logging.basicConfig(
    filename=log_dir / log_name,
    filemode='a',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ───── Helpers ─────
def extract_page_number(filename):
    """Extract page number from filename like ..._p12_text.txt"""
    match = re.search(r'_p(\d+)_text\.txt$', filename)
    return int(match.group(1)) if match else -1

def extract_prefix(filename):
    """Extract prefix like TH_Delhi-30-06-2025_p1_text.txt → TH_Delhi-30-06-2025"""
    return filename.split("_p")[0].replace(" ", "_")

def load_page_texts(folder):
    files = sorted(
        [f for f in folder.glob("*_p*_text.txt")],
        key=lambda f: extract_page_number(f.name)
    )

    if not files:
        logging.warning(f"⚠️ No text files in: {folder.name}")
        return []

    pages = []
    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                text = f.read().strip()
                words = text.split()
                if words:
                    pages.append({"file": file.name, "words": words})
                else:
                    logging.info(f"⚠️ Skipped empty file: {file.name}")
        except Exception as e:
            logging.error(f"✖️ Error reading {file.name}: {e}")

    return pages

def split_page_into_batches(words, max_words=MAX_WORDS_PER_BATCH, overlap=OVERLAP_WORDS):
    """Split one page into multiple batches with overlap."""
    if len(words) <= max_words:
        return [words]  # whole page fits into one batch

    batches = []
    start = 0
    while start < len(words):
        end = start + max_words
        batch_words = words[start:end]
        batches.append(batch_words)

        # Move forward with overlap
        start = end - overlap

        if start >= len(words):
            break

    return batches

def write_batches(pages, output_dir, prefix):
    subdir = output_dir / prefix
    subdir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for page in pages:
        page_num = extract_page_number(page["file"])
        page_batches = split_page_into_batches(page["words"])

        for i, word_list in enumerate(page_batches, start=1):
            if not word_list:
                logging.warning(f"⚠️ Skipped empty batch for page {page_num} #{i}")
                continue

            text_block = " ".join(word_list)
            out_name = f"{prefix}_p{page_num}_batch_{i}.txt"   # ✅ include page number
            batch_file = subdir / out_name

            try:
                with open(batch_file, "w", encoding="utf-8") as f:
                    f.write(text_block)
                logging.info(f"✅ Saved {out_name} ({len(word_list)} words)")
                saved_files.append(batch_file)
            except Exception as e:
                logging.error(f"✖️ Failed to write {out_name}: {e}")

    return saved_files


# ───── Main ─────
def main():
    logging.info("🚀 Starting per-page batch generation...")

    subfolders = [f for f in INPUT_BASE_DIR.iterdir() if f.is_dir()]
    if not subfolders:
        logging.error("🛑 No subfolders found in input base.")
        return

    for folder in subfolders:
        logging.info(f"📰 Processing: {folder.name}")
        pages = load_page_texts(folder)

        if not pages:
            logging.warning(f"⚠️ No valid pages in: {folder.name}")
            continue

        prefix = extract_prefix(pages[0]["file"])
        logging.info(f"📄 Loaded {len(pages)} pages from: {prefix}")

        files = write_batches(pages, OUTPUT_DIR, prefix)
        logging.info(f"📁 Saved {len(files)} batch files for {prefix}")

    logging.info("✅ All folders processed.")

if __name__ == "__main__":
    main()
