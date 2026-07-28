#!/usr/bin/env python3
"""
from src.utils.logging_utils import configure_logging
logger = configure_logging("stage05_translation_manifest")
Stage 05: Translation (Manifest-First)
Optimized for 4GB VRAM & Multi-core CPU.
Reads run_state/ocr_manifest.jsonl -> writes run_state/translated_manifest.jsonl
"""

import os, sys, json, time, multiprocessing as mp
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# Configuration
IN_MANIFEST = PROJ_ROOT / "run_state" / "ocr_manifest.jsonl"
OUT_MANIFEST = PROJ_ROOT / "run_state" / "translated_manifest.jsonl"
FROM_LANG = "hi"
TO_LANG = "en"
BATCH_SIZE = 100

DEVANAGARI_DIGITS = "०१२३४५६७८९"
WESTERN_DIGITS = "0123456789"
DIGIT_TRANS = str.maketrans(DEVANAGARI_DIGITS, WESTERN_DIGITS)

def is_hindi(text, threshold=0.2):
    if not text or not isinstance(text, str): return False
    text = text.translate(DIGIT_TRANS)
    devanagari_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    return (devanagari_chars / len(text)) >= threshold if len(text) > 0 else False

def translate_worker(items):
    """Worker function for Argos Translate."""
    import argostranslate.translate
    
    # Pre-load packages in worker
    installed_packages = argostranslate.translate.get_installed_languages()
    # Note: Loading might be slow, we should do it once per process.
    
    results = []
    for item in items:
        text = item.get("ocr_text_raw", "")
        if is_hindi(text):
            try:
                translated = argostranslate.translate.translate(text, FROM_LANG, TO_LANG)
                item["ocr_text_eng"] = translated
                item["translation_status"] = "translated"
            except Exception as e:
                item["ocr_text_eng"] = text
                item["translation_status"] = f"error: {str(e)}"
        else:
            item["ocr_text_eng"] = text
            item["translation_status"] = "skipped_english"
        results.append(item)
    return results

def main():
    print(f"🚀 Starting Stage 05 (Translation) on {IN_MANIFEST}")
    start_time = time.time()
    
    if not IN_MANIFEST.exists():
        print(f"❌ Input manifest not found: {IN_MANIFEST}")
        sys.exit(1)

    with open(IN_MANIFEST, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    total = len(lines)
    print(f"📦 Loaded {total} blocks.")

    # Check for Hindi presence to avoid heavy process spawning if not needed
    has_hindi = False
    all_data = []
    for line in lines:
        data = json.loads(line)
        all_data.append(data)
        if not has_hindi and is_hindi(data.get("ocr_text_raw", "")):
            has_hindi = True
    
    if not has_hindi:
        print("✅ No Hindi detected. Fast-tracking English manifest.")
        with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
            for data in all_data:
                data["ocr_text_eng"] = data["ocr_text_raw"]
                data["translation_status"] = "no_hindi_detected"
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(f"🏁 Finished in {time.time() - start_time:.2f}s")
        return

    print("🌐 Hindi detected. Spawning translation workers...")
    # Split into batches
    batches = [all_data[i:i + BATCH_SIZE] for i in range(0, len(all_data), BATCH_SIZE)]
    
    num_procs = max(1, mp.cpu_count() - 2)
    with mp.Pool(processes=num_procs) as pool:
        results = []
        for i, batch_res in enumerate(pool.imap(translate_worker, batches)):
            results.extend(batch_res)
            if i % 10 == 0:
                print(f"Progress: {min(total, (i+1)*BATCH_SIZE)}/{total} blocks processed...")
                sys.stdout.flush()

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"✅ Translation complete. Saved to {OUT_MANIFEST}")
    print(f"🏁 Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
