#!/usr/bin/env python3
"""
from src.utils.logging_utils import configure_logging
logger = configure_logging("stage04_ocr_fast")
Stage 04: Fast Single-Process GPU OCR
Optimized for 4GB VRAM. Avoids multiprocessing CUDA errors.
Processes all job blocks from data/job_blocks_refined.
"""

import os
import sys
import json
import time
import cv2
import easyocr
import torch
from pathlib import Path
from tqdm import tqdm

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# Config
IN_DIR = PROJ_ROOT / "data/job_blocks_refined"
OUT_MANIFEST = PROJ_ROOT / "run_state/ocr_manifest.jsonl"
BATCH_SIZE = 16  # Moderate batch size for 4GB VRAM

def main():
    print(f"🚀 Starting Fast GPU OCR...")
    start_time = time.time()
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA NOT AVAILABLE. Falling back to CPU...")
        reader = easyocr.Reader(['en'], gpu=False)
    else:
        print("🧠 Loading EasyOCR (GPU)...")
        try:
            reader = easyocr.Reader(['en'], gpu=True)
        except Exception as e:
            print(f"⚠️ GPU Load failed: {e}. Falling back to CPU...")
            reader = easyocr.Reader(['en'], gpu=False)
    
    all_blocks = []
    folders = sorted([f for f in IN_DIR.iterdir() if f.is_dir()])
    
    for folder in folders:
        newspaper = folder.name
        print(f"📂 Scanning {newspaper}...")
        files = sorted(list(folder.glob("*.jpg")))
        for f in files:
            # Filename format: <news>_p<p_idx>_c<c_idx>.jpg
            parts = f.stem.split("_")
            p_idx = int(parts[-2].replace("p", "")) - 1
            c_idx = int(parts[-1].replace("c", ""))
            all_blocks.append({
                "path": str(f),
                "newspaper": newspaper,
                "p_idx": p_idx,
                "c_idx": c_idx
            })

    print(f"📦 Total blocks to process: {len(all_blocks)}")
    
    # Open manifest in write mode
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f_out:
        for i in tqdm(range(0, len(all_blocks), BATCH_SIZE)):
            batch = all_blocks[i:i+BATCH_SIZE]
            
            for item in batch:
                try:
                    # Individual read for accuracy and simple logic
                    # EasyOCR's readtext on a list of images is sometimes buggy with paragraph=True
                    results = reader.readtext(item['path'], paragraph=True)
                    text = " ".join([res[1] for r in results for res in (r if isinstance(r, list) else [r])]) # Robustness
                    # Wait, EasyOCR return varies. Let's be surgical.
                    text = ""
                    for res in results:
                        if isinstance(res, (list, tuple)) and len(res) >= 2:
                            text += res[1] + " "
                    
                    entry = {
                        "p_idx": item['p_idx'],
                        "c_idx": item['c_idx'],
                        "ocr_text_raw": text.strip(),
                        "status": "ok",
                        "newspaper_name": item['newspaper'],
                        "pdf_path": f"data/raw_pdfs/{item['newspaper']}.pdf" # Heuristic
                    }
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f_out.flush()
                except Exception as e:
                    print(f"Error processing {item['path']}: {e}")

    print(f"✅ OCR Complete. Saved to {OUT_MANIFEST}")
    print(f"🏁 Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
