import os
import logging
from datetime import datetime
from paddleocr import PaddleOCR
from PIL import Image
import pynvml

# =======================
# Logging Setup
# =======================
log_filename = "ocr_gpu_processing.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# =======================
# GPU Info Helper
# =======================
def get_gpu_info():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # first GPU
        name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        total_mb = mem_info.total // (1024 * 1024)
        used_mb = mem_info.used // (1024 * 1024)
        return {
            "name": name,
            "memory_total": total_mb,
            "memory_used": used_mb,
            "utilization": util.gpu
        }
    except Exception as e:
        logger.warning(f"Could not fetch GPU stats: {e}")
        return None

# =======================
# OCR Setup
# =======================
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ch',             # Use Chinese as base as flag to enable multi language
    ocr_version='PP-OCRv5',
    device="gpu"
)

# =======================
# Counters
# =======================
counters = {
    "newspapers": 0,
    "pages": 0,
    "blocks": 0,
    "blocks_with_text": 0,
    "failed": 0
}

# =======================
# OCR per image
# =======================
def process_image(img_path):
    try:
        result = ocr.ocr(img_path, cls=True)
        blocks = len(result[0]) if result and result[0] else 0
        with_text = sum(1 for line in result[0] if line[1][0].strip()) if blocks else 0
        return {
            "pages": 1,
            "blocks": blocks,
            "blocks_with_text": with_text,
            "failed": 0
        }
    except Exception as e:
        logger.error(f"❌ Failed OCR on {img_path}: {e}")
        return {"pages": 1, "blocks": 0, "blocks_with_text": 0, "failed": 1}

# =======================
# Process folder
# =======================
def run_ocr_on_newspapers(base_dir):
    start_time = datetime.now()
    folders = [os.path.join(base_dir, f) for f in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, f))]

    for folder in folders:
        counters["newspapers"] += 1
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
                    img_path = os.path.join(root, file)
                    stats = process_image(img_path)
                    counters["pages"] += stats["pages"]
                    counters["blocks"] += stats["blocks"]
                    counters["blocks_with_text"] += stats["blocks_with_text"]
                    counters["failed"] += stats["failed"]

    elapsed = (datetime.now() - start_time).total_seconds()

    # GPU stats
    gpu_info = get_gpu_info()

    # =======================
    # Summary
    # =======================
    logger.info("✅ All folders processed.")
    logger.info("========== SUMMARY ==========")
    logger.info(f"Total newspapers: {counters['newspapers']}")
    logger.info(f"Total pages processed: {counters['pages']}")
    logger.info(f"Total blocks processed: {counters['blocks']}")
    logger.info(f"Blocks with text: {counters['blocks_with_text']}")
    logger.info(f"Blocks failed: {counters['failed']}")

    if gpu_info:
        logger.info(f"🖥️ GPU: {gpu_info['name']}")
        logger.info(f"📦 GPU Memory: {gpu_info['memory_total']} MB total, {gpu_info['memory_used']} MB used")
        logger.info(f"📊 GPU Utilization: {gpu_info['utilization']}%")
    else:
        logger.info("⚠️ GPU stats not available.")

    logger.info(f"⏱️ Total time taken: {elapsed:.2f} seconds")
    logger.info("=============================")


# =======================
# Main Entry
# =======================
if __name__ == "__main__":
    base_directory = "newspapers"  # your dataset folder
    run_ocr_on_newspapers(base_directory)
