import cv2
import pytesseract
import logging
from pathlib import Path
import re
from collections import defaultdict
from datetime import datetime

# ───── Logging Setup ─────
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_name = f"extract_text_{timestamp}.log"

logging.basicConfig(
    filename=log_dir / log_name,
    filemode='a',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ───── OCR Helper ─────
def extract_text_from_image(image_path):
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning(f"[⚠️] Unreadable image: {image_path.name}")
            return ""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11
        )
        text = pytesseract.image_to_string(thresh, config="--psm 6")
        return text.strip()
    except Exception as e:
        logging.error(f"[✖] OCR failed on {image_path.name}: {e}")
        return ""

# ───── Folder Processing ─────
def extract_text_by_page_folder(refined_folder: Path):
    newspaper_name = refined_folder.name
    logging.info(f"📰 Processing newspaper folder: {newspaper_name}")

    output_dir = Path("data/page_texts") / newspaper_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group image blocks by page number
    page_groups = defaultdict(list)
    for img_path in sorted(refined_folder.glob("*.png")):
        match = re.search(r"_p(\d+)", img_path.name)
        if match:
            page_number = match.group(1)
            page_groups[page_number].append(img_path)

    if not page_groups:
        logging.warning(f"[🚫] No valid page-labeled images found in {newspaper_name}")
        return

    logging.info(f"📄 Found {len(page_groups)} page(s) in: {newspaper_name}")

    for page, img_paths in page_groups.items():
        logging.info(f"📃 Page {page}: {len(img_paths)} blocks")
        text_blocks = []

        for img_path in img_paths:
            text = extract_text_from_image(img_path)
            if text:
                text_blocks.append(text)
            else:
                logging.warning(f"[🗑️] No text from: {img_path.name}")

        if text_blocks:
            page_text = "\n\n".join(text_blocks)
            out_file = output_dir / f"{newspaper_name}_p{page}_text.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(page_text)
            logging.info(f"[💾] Saved page {page} to {out_file.name}")
        else:
            logging.warning(f"[❌] No text extracted for page {page}")

# ───── Entry Point ─────
def extract_all_folders(base_dir="data/job_blocks_refined"):
    base_path = Path(base_dir)
    folders = [f for f in base_path.iterdir() if f.is_dir()]
    
    if not folders:
        logging.warning("[🚫] No folders found in job_blocks_refined")
        return

    for folder in folders:
        extract_text_by_page_folder(folder)

    logging.info("✅ All folders processed.")

if __name__ == "__main__":
    extract_all_folders()
