#!/usr/bin/env python3
"""
Streaming Pipeline Orchestrator (v3.20).
OpenCV Detector with proper output.
"""

import sys
import os
import time
import json
import multiprocessing as mp
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import cv2
import fitz # PyMuPDF

# Force Proj Root
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from src.pipeline.step3_ocr_utilities import normalize_text

# Configuration
RENDER_WORKERS = 4
OCR_BATCH_SIZE = 16
OCR_DETECTOR = "dbnet18"

# ─── WORKERS ────────────────────────────────────────────────────────────────

def render_worker(pdf_queue, page_queue):
    while True:
        pdf_task = pdf_queue.get()
        if pdf_task is None: break
        pdf_path = Path(pdf_task['pdf_path'])
        try:
            doc = fitz.open(str(pdf_path))
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text().strip()
                if len(text) > 100:
                    page_queue.put({"type": "digital", "doc_id": pdf_task['doc_id'], "page_idx": page_idx, "text": text[:1000], "pdf_path": str(pdf_path)})
                    continue
                pix = page.get_pixmap(dpi=300)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                page_queue.put({"type": "scan", "doc_id": pdf_task['doc_id'], "page_idx": page_idx, "img": img, "pdf_path": str(pdf_path), "w": pix.w, "h": pix.h})
            doc.close()
        except Exception as e: print(f"[Render] Error: {e}")

def detect_worker(page_queue, crop_queue):
    from src.vision.block_detector import detect_connected_blocks
    while True:
        page_task = page_queue.get()
        if page_task is None:
            crop_queue.put(None)
            break
        if page_task['type'] == "digital":
            crop_queue.put(page_task)
            continue
        tmp_img = f"tmp_p{page_task['page_idx']}.png"
        cv2.imwrite(tmp_img, page_task['img'])
        blocks, _ = detect_connected_blocks(tmp_img, debug=False)
        if os.path.exists(tmp_img): os.remove(tmp_img)
        
        # print(f"DEBUG: Page {page_task['page_idx']} found {len(blocks)} blocks")
        for i, (x, y, bw, bh) in enumerate(blocks):
            crop_img = page_task['img'][y:y+bh, x:x+bw]
            crop_queue.put({
                "type": "crop", "doc_id": page_task['doc_id'], "page_idx": page_task['page_idx'], "crop_idx": i,
                "img": crop_img, "bbox_norm": [x/page_task['w'], y/page_task['h'], (x+bw)/page_task['w'], (y+bh)/page_task['h']]
            })

def ocr_worker(crop_queue, result_queue):
    from easyocr import Reader
    reader = Reader(['en'], gpu=True, detector=OCR_DETECTOR, cudnn_benchmark=True)
    while True:
        crop_task = crop_queue.get()
        if crop_task is None:
            result_queue.put(None)
            break
        if crop_task['type'] == "digital":
            result_queue.put({"id": f"{crop_task['doc_id']}_p{crop_task['page_idx']}", "text": crop_task['text'], "status": "digital"})
            continue
        ocr_res = reader.readtext(crop_task['img'], batch_size=OCR_BATCH_SIZE)
        text = " ".join([r[1] for r in ocr_res])
        result_queue.put({"id": f"{crop_task['doc_id']}_p{crop_task['page_idx']}_c{crop_task['crop_idx']}", "text": text, "status": "ok"})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)
    pdf_p = Path(args.pdf)
    pdfs = [pdf_p] if pdf_p.is_file() else list(pdf_p.glob("*.pdf"))
    
    print(f"🌊 Streaming Pipeline (v3.20) Start: {len(pdfs)} PDFs")
    start_time = time.time()
    pdf_q = mp.Queue()
    page_q = mp.Queue(maxsize=1) 
    crop_q = mp.Queue(maxsize=100)
    res_q = mp.Queue()
    for p in pdfs: pdf_q.put({"pdf_path": str(p), "doc_id": p.stem})
    for _ in range(RENDER_WORKERS): pdf_q.put(None)
    d_p = mp.Process(target=detect_worker, args=(page_q, crop_q))
    o_p = mp.Process(target=ocr_worker, args=(crop_q, res_q))
    d_p.start()
    o_p.start()
    threads = [threading.Thread(target=render_worker, args=(pdf_q, page_q)) for _ in range(RENDER_WORKERS)]
    for t in threads: t.start()
    for t in threads: t.join()
    page_q.put(None)
    count = 0
    while True:
        res = res_q.get()
        if res is None: break
        count += 1
        if count % 50 == 0: print(f"  Processed {count} items...")
    d_p.join(); o_p.join()
    elapsed = time.time() - start_time
    print(f"✅ Complete: {count} items in {elapsed:.2f}s ({count/max(1,elapsed):.2f} items/sec)")
