#!/usr/bin/env python3
"""
Step 3: High-Performance OCR pipeline (Streaming v3.9).
Optimized for 30s 1-PDF goal.
Features:
  - Persistent GPU worker (Phase B1)
  - DBNet filtering (Phase A1.1)
  - Digital text bypass (Phase A2)
  - Batching (Phase A3)
  - Parallel Stage 1/2/3 overlap (Phase C1)
  - 10-core CPU limit
"""

import sys
import json
import logging
import time
import threading
import queue
import gc
import os
import multiprocessing
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# CUDA tuning
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Force Proj Root
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

from src.pipeline.pipeline_metadata import read_crop_manifest_jsonl
from src.pipeline.step3_ocr_utilities import normalize_text

import numpy as np
import cv2
import psutil

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
OCR_CORE_LIMIT = 10
OCR_BATCH_SIZE = 16 # Phase A3
OCR_DETECTOR = "dbnet18"

# ─── STREAMING PIPELINE ─────────────────────────────────────────────────────

def gpu_worker_process(task_queue, result_queue, crops_dir):
    """Persistent GPU process to avoid reload overhead (Phase B1)."""
    import fitz
    from easyocr import Reader
    import torch
    
    # Init
    try:
        psutil.Process().cpu_affinity(list(range(min(psutil.cpu_count(), OCR_CORE_LIMIT))))
    except: pass

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [GPU] %(message)s")

    logger.info(f"Initializing EasyOCR with {OCR_DETECTOR}")
    reader = Reader(['en'], gpu=True, detector=OCR_DETECTOR, cudnn_benchmark=True)
    
    page_text_cache = {}

    while True:
        try:
            crop = task_queue.get()
            if crop is None: break
            
            # Phase A2: Digital Bypass (Fast)
            if True: # Always enabled for v3.9
                p_path = crop['pdf_path']
                p_idx = crop['page_index0']
                cache_key = (p_path, p_idx)
                if cache_key not in page_text_cache:
                    try:
                        doc = fitz.open(p_path)
                        page_txt = doc[p_idx].get_text().strip()
                        doc.close()
                        # Threshold: if page has real text, use it (Phase A2)
                        page_text_cache[cache_key] = page_txt if len(page_txt) > 100 else None
                    except:
                        page_text_cache[cache_key] = None
                
                bypass_text = page_text_cache[cache_key]
                if bypass_text:
                    result_queue.put({**crop, "ocr_text_raw": bypass_text[:1000], "status": "digital_bypass"})
                    task_queue.task_done()
                    continue

            # Phase A1: Filter + Faster OCR
            img_path = Path(crops_dir) / crop['crop_image_path']
            if not img_path.exists():
                img_path = Path(crops_dir) / Path(crop['crop_image_path']).name
                
            img = cv2.imread(str(img_path))
            if img is None:
                result_queue.put({**crop, "status": "read_failed"})
                task_queue.task_done()
                continue
            
            # Phase A1.1: DBNet Filtering (Option 1)
            # If DBNet detect() returns no text regions -> skip recognition
            horiz, free = reader.detect(img)
            if not horiz or (len(horiz[0]) == 0 and len(free[0]) == 0):
                result_queue.put({**crop, "status": "filtered_empty", "ocr_text_raw": ""})
                task_queue.task_done()
                continue

            # Phase A3: Batching within readtext
            ocr_res = reader.readtext(img, batch_size=OCR_BATCH_SIZE)
            text = " ".join([r[1] for r in ocr_res])
            conf = float(np.mean([r[2] for r in ocr_res])) if ocr_res else 0.0
            
            result_queue.put({
                **crop,
                "ocr_text_raw": text,
                "ocr_text_norm": normalize_text(text),
                "ocr_conf_mean": conf,
                "status": "ok"
            })
            task_queue.task_done()
            
        except Exception as e:
            result_queue.put({**crop, "status": "error", "error": str(e)})
            task_queue.task_done()

def run_step3_ocr():
    start_time = time.time()
    
    # Load
    manifest_path = Path("run_state/crop_manifest.jsonl")
    crops_manifest = read_crop_manifest_jsonl(manifest_path)
    if not crops_manifest: return

    # Size-bucketed batching (Phase C2)
    # Group by area to minimize padding in EasyOCR detector/recognizer
    sorted_cids = sorted(crops_manifest.keys(), key=lambda x: crops_manifest[x].get('area_norm', 0))

    task_queue = multiprocessing.JoinableQueue(maxsize=1000)
    result_queue = multiprocessing.Queue()
    
    crops_dir = PROJ_ROOT / "data" / "crops"
    
    gpu_proc = multiprocessing.Process(
        target=gpu_worker_process, 
        args=(task_queue, result_queue, str(crops_dir))
    )
    gpu_proc.start()

    # Feed
    def producer():
        for cid in sorted_cids:
            task_queue.put(crops_manifest[cid])
        task_queue.put(None)
    threading.Thread(target=producer, daemon=True).start()

    # Collect
    processed = 0
    total = len(crops_manifest)
    
    final_ocr_path = Path("run_state/ocr_manifest.jsonl")
    final_can_path = Path("run_state/step3_candidates.jsonl")
    final_rej_path = Path("run_state/step3_rejects.jsonl")
    
    with open(final_ocr_path, "w") as f_ocr, \
         open(final_can_path, "w") as f_can, \
         open(final_rej_path, "w") as f_rej:
             
        while processed < total:
            res = result_queue.get()
            res['is_step3_survivor'] = res['status'] in ["ok", "digital_bypass"]
            
            line = json.dumps(res, ensure_ascii=False) + "\n"
            f_ocr.write(line)
            if res['is_step3_survivor']: f_can.write(line)
            else: f_rej.write(line)
            
            processed += 1
            if processed % 100 == 0: print(f"  Processed {processed}/{total} crops...")

    task_queue.join()
    gpu_proc.join(timeout=10)
    if gpu_proc.is_alive(): gpu_proc.terminate()
    
    elapsed = time.time() - start_time
    print(f"✅ OCR Complete. Processed {total} in {elapsed:.1f}s ({total/max(1,elapsed):.2f} crops/sec)")

if __name__ == "__main__":
    run_step3_ocr()
