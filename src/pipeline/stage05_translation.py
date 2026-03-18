import os
import warnings
import logging
from pathlib import Path
from datetime import datetime

# VERY IMPORTANT: set env + filter warnings BEFORE importing noisy libs

# Quiet environment-based noise before torch/stanza loads
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"      # quieter torch C++ logs
os.environ["CUDA_MODULE_LOADING"] = "LAZY"       # avoid noisy CUDA init prints
os.environ["PYTHONWARNINGS"] = "ignore"          # optional: suppress warnings on import

# Filter Python warnings early (module-level / future warnings)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

# Optional: silence specific loggers that are noisy
logging.getLogger("stanza").setLevel(logging.ERROR)
logging.getLogger("argostranslate").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Now import the other libraries after the above block
import time
import torch
import argostranslate.translate
import multiprocessing
import platform

# ---------------- LOGGING SETUP ----------------
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_name = f"translation_{timestamp}.log"

# Configure root logger: DEBUG for file, WARNING for console
logger = logging.getLogger()
# Remove existing handlers (prevents duplication on re-run)
if logger.hasHandlers():
    logger.handlers.clear()

logger.setLevel(logging.DEBUG)   # allow DEBUG to propagate to handlers

# File handler — store everything (DEBUG+)
fh = logging.FileHandler(log_dir / log_name, mode="a", encoding="utf-8")
fh.setLevel(logging.DEBUG)

# Console handler — only show warnings/errors to terminal
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

# Don't bubble logs up to ancestor loggers (avoid duplicate printing)
logger.propagate = False
# ------------------------------------------------

# ---------------- CONFIG ----------------
INPUT_ROOT_DIR = Path("data/page_texts")
OUTPUT_ROOT_DIR = Path("data/all_eng_text")
FROM_LANG = "hi"
TO_LANG = "en"
# ----------------------------------------

# --- Devanagari digit normalization ---
DEVANAGARI_DIGITS = "०१२३४५६७८९"
WESTERN_DIGITS = "0123456789"
DIGIT_TRANS = str.maketrans(DEVANAGARI_DIGITS, WESTERN_DIGITS)


def is_hindi(text: str, threshold: float = 0.2) -> bool:
    """Check if text is primarily Hindi (Devanagari script)."""
    total_chars = len(text)
    if total_chars == 0:
        return False
    devanagari_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    ratio = devanagari_chars / total_chars
    return ratio >= threshold


def smart_translate_block(text: str, from_lang="hi", to_lang="en") -> str:
    """Translate block if Hindi, else keep original."""
    if not text.strip():
        return text

    # Normalize numerals
    text = text.translate(DIGIT_TRANS)

    if is_hindi(text):
        try:
            return argostranslate.translate.translate(text, from_lang, to_lang)
        except Exception as e:
            logger.error(f"❌ Translation failed: {e}")
            return text
    else:
        return text


def main():
    start_time_total = time.time()
    print("🚀 Starting smart translation script (line-by-line)...")

    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Argos packages
    installed_packages = argostranslate.translate.get_installed_languages()
    from_lang_obj = next((l for l in installed_packages if l.code == FROM_LANG), None)
    to_lang_obj = next((l for l in installed_packages if l.code == TO_LANG), None)

    if not from_lang_obj or not to_lang_obj:
        logger.error(f"❌ No translation package found for {FROM_LANG}->{TO_LANG}")
        return

    # Process every folder
    for folder in INPUT_ROOT_DIR.iterdir():
        if not folder.is_dir():
            continue

        print(f"📂 Processing folder: {folder.name}")
        folder_start_time = time.time()
        output_folder = OUTPUT_ROOT_DIR / folder.name
        output_folder.mkdir(parents=True, exist_ok=True)

        text_files = list(folder.glob("*.txt"))
        if not text_files:
            logger.warning(f"⚠️ No .txt files in {folder.name}")
            continue

        files_processed = 0

        for file_path in text_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                output_lines = []
                start_file_time = time.time()
                for i, line in enumerate(lines, 1):
                    translated_line = smart_translate_block(line)
                    if translated_line != line:
                        logger.debug(f"🌐 Line {i}: translated")
                    else:
                        logger.debug(f"📄 Line {i}: kept original")
                    output_lines.append(translated_line)

                duration = time.time() - start_file_time
                output_text = "".join(output_lines)

                # Save to output folder
                out_path = output_folder / file_path.name
                with open(out_path, "w", encoding="utf-8") as out:
                    out.write(output_text)

                logger.info(f"✅ {file_path.name} → {out_path.name} ({duration:.2f}s)")
                files_processed += 1

            except Exception as e:
                logger.error(f"❌ Failed to process {file_path}: {e}")

        folder_elapsed = time.time() - folder_start_time
        print(f"   ✅ Completed {files_processed} files in {folder_elapsed:.2f}s\n")

    # Summary
    elapsed_total = time.time() - start_time_total
    print("================ SUMMARY ================")
    print(f"Total time: {elapsed_total:.2f}s")
    print(f"Output folder: {OUTPUT_ROOT_DIR}")
    print("==========================================")

    # Log the detailed summary
    logger.info("\n📊 DETAILED SUMMARY")
    logger.info(f"Input root folder : {INPUT_ROOT_DIR}")
    logger.info(f"Output root folder: {OUTPUT_ROOT_DIR}")
    logger.info(f"Total script duration: {elapsed_total:.2f}s")

    # System info (CPU not GPU, since Argos is CPU-based)
    logger.info("\n🔎 SYSTEM INFO")
    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA available (not used by Argos): %s", torch.cuda.is_available())
    logger.info("Device used: CPU")
    logger.info("CPU: %s", platform.processor())
    logger.info("Total cores: %d", multiprocessing.cpu_count())
    logger.info("Translation summary completed.")


if __name__ == "__main__":
    main()