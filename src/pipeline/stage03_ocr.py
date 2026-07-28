#!/usr/bin/env python3
"""
from src.utils.logging_utils import configure_logging
logger = configure_logging("stage03_ocr")
Step 3: OCR pipeline (v4.4 - Infinite-Stream).
Optimized for high-volume cross-validation.
"""

import sys
import json
import time
import multiprocessing
import os
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

# Force Proj Root
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

from src.pipeline.pipeline_metadata import read_crop_manifest_jsonl
from src.pipeline.step3_ocr_utilities import normalize_text

def ocr_worker(crops, temp_output, pdf_path):
    from easyocr import Reader
    reader = Reader(['en'], gpu=True, detector="dbnet18", cudnn_benchmark=True)
    results = []
    
    # Pre-open doc for any missing metadata
    import fitz
    doc = fitz.open(str(pdf_path))
    
    for i in range(0, len(crops), 32): # Increased batch
        batch = crops[i:i + 32]
        batch_imgs = []
        valid_batch_crops = []
        
        for crop in batch:
            p = Path(crop['crop_image_path'])
            if not p.is_absolute(): p = PROJ_ROOT / p
            
            img = cv2.imread(str(p))
            if img is None: continue
            
            # Sonic Resize
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * (128/h)), 128))
            batch_imgs.append(img)
            valid_batch_crops.append(crop)
            
        if not batch_imgs: continue
        
        try:
            # Efficient loop-batching
            for j, img in enumerate(batch_imgs):
                res = reader.readtext(img, paragraph=True)
                text = " ".join([r[1] for r in res])
                entry = valid_batch_crops[j].copy()
                entry.update({"ocr_text_raw": text, "status": "ok"})
                results.append(entry)
        except: pass

    with open(temp_output, "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    doc.close()

def run_step3_ocr(manifest_path=Path("run_state/crop_manifest.jsonl"), output_path=Path("run_state/ocr_manifest.jsonl")):
    if not manifest_path.exists(): return
    crops_manifest = read_crop_manifest_jsonl(manifest_path)
    if not crops_manifest: return
    
    # We assume one PDF per run in this validator
    first_crop = list(crops_manifest.values())[0]
    pdf_path = first_crop['pdf_path']
    name = first_crop['newspaper_name']
    
    temp_dir = Path("run_state/tmp_ocr")
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    temp_out = temp_dir / "results.jsonl"

    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=ocr_worker, args=(list(crops_manifest.values()), temp_out, pdf_path))
    p.start()
    p.join()

    # Finalize
    if temp_out.exists():
        shutil.copy(temp_out, output_path)
        print(f"✅ Stage 3 Complete for {name}")

if __name__ == "__main__":
    run_step3_ocr()
