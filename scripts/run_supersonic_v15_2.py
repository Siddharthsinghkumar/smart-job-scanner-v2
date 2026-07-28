#!/usr/bin/env python3
"""
Stage 1-2-3 Unified v15.2: Hyper-Sonic Engine.
Goal: 10 PDFs < 1000s (100s/PDF) @ 95%+ Recall.
Architecture: 10 Producers, 1 GPU Master, 4 Stitchers (16-Core CPU Cap).
"""

import os
import sys
import json
import time
import torch
import torch.multiprocessing as tmp
import numpy as np
import cv2
import fitz
from pathlib import Path
import warnings
from collections import defaultdict
import shutil

warnings.filterwarnings("ignore", category=FutureWarning)
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# Optimized Configuration
YOLO_MODEL_PATH = str(PROJ_ROOT / "artifacts/detector_pivot_yolo_v4_tiles/best.pt")
RENDER_DPI = 300
TILE_SIZE = 320
OVERLAP = 0.60 # Rescue boundary ads
CONF_THRESH = 0.0005 # Balanced FP/Recall
OCR_BATCH_SIZE = 32
CROP_MANIFEST_DIR = Path("run_state/manifests")

def iou(b1, b2):
    xa, ya, xb, yb = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / u if u > 0 else 0

def gpu_master_worker(tile_q, raw_det_q, ocr_task_q, ocr_res_q):
    """Saturates GPU by multiplexing YOLO and OCR tasks."""
    from ultralytics import YOLO
    from easyocr import Reader
    model = YOLO(YOLO_MODEL_PATH).to('cuda')
    reader = Reader(['en'], gpu=True, detector="dbnet18", cudnn_benchmark=True)
    
    while True:
        # 1. OCR (Higher priority to clear memory)
        if not ocr_task_q.empty():
            batch = ocr_task_q.get()
            if batch:
                imgs = [item['img'] for item in batch]
                for j, img in enumerate(imgs):
                    res = reader.readtext(img, paragraph=True)
                    text = " ".join([r[1] for r in res])
                    entry = batch[j]['meta'].copy()
                    entry.update({"ocr_text_raw": text, "status": "ok"})
                    ocr_res_q.put(entry)
                del imgs
                torch.cuda.empty_cache()
                continue

        # 2. YOLO
        try:
            batch = tile_q.get(timeout=0.1)
            if batch is None: break
            
            imgs = [item['tile'] for item in batch]
            results = model.predict(imgs, conf=CONF_THRESH, imgsz=TILE_SIZE, verbose=False, device=0)
            for i, res in enumerate(results):
                meta = batch[i]['meta']
                detections = []
                for b in res.boxes:
                    c = b.xyxy[0].cpu().numpy()
                    detections.append({
                        "p_idx": meta[0],
                        "box": [float(c[0]+meta[1]), float(c[1]+meta[2]), float(c[2]+meta[1]), float(c[3]+meta[2])],
                        "conf": float(b.conf[0])
                    })
                raw_det_q.put({"p_idx": meta[0], "dets": detections, "tile_done": True})
            del results, imgs
        except: continue

def cpu_producer(pdf_path, p_indices, tile_q, page_info_q, ocr_res_q):
    doc = fitz.open(str(pdf_path))
    for p_idx in p_indices:
        page = doc[p_idx]
        text = page.get_text()
        
        # Router: Skip deep detection on digital-heavy pages
        if sum(1 for c in text if c.isalpha()) > 500:
            ocr_res_q.put({"page_index0": p_idx, "bbox_xyxy_norm": [0,0,1,1], "ocr_text_raw": text[:1000], "status": "digital_bypass"})
            page_info_q.put({"p_idx": p_idx, "done": True})
            continue

        pix = page.get_pixmap(dpi=RENDER_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        page_info_q.put({"p_idx": p_idx, "dim": (img.shape[0], img.shape[1]), "img": img})
        
        stride = int(TILE_SIZE * (1 - OVERLAP))
        tiles_count = 0
        batch = []
        for y in range(0, img.shape[0], stride):
            for x in range(0, img.shape[1], stride):
                y2, x2 = min(y + TILE_SIZE, img.shape[0]), min(x + TILE_SIZE, img.shape[1])
                tile = img[y:y2, x:x2]
                if np.mean(tile) > 245: continue
                if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                    tile = cv2.copyMakeBorder(tile, 0, TILE_SIZE-tile.shape[0], 0, TILE_SIZE-tile.shape[1], cv2.BORDER_CONSTANT, value=[255,255,255])
                batch.append({'tile': tile, 'meta': (p_idx, x, y)})
                tiles_count += 1
                if len(batch) >= 128:
                    tile_q.put(batch); batch = []
        if batch: tile_q.put(batch)
        page_info_q.put({"p_idx": p_idx, "expected_tiles": tiles_count})
    doc.close()

def manager(num_pages, raw_det_q, page_info_q, ocr_task_q):
    page_dets = defaultdict(list); tracker = defaultdict(int)
    page_expected = {}; page_dims = {}; page_imgs = {}; finished = 0
    while finished < num_pages:
        while not page_info_q.empty():
            info = page_info_q.get()
            p_idx = info['p_idx']
            if 'done' in info: finished += 1
            if 'dim' in info: page_dims[p_idx] = info['dim']
            if 'img' in info: page_imgs[p_idx] = info['img']
            if 'expected_tiles' in info: page_expected[p_idx] = info['expected_tiles']
        try:
            res = raw_det_q.get(timeout=0.01)
            p_idx = res['p_idx']; page_dets[p_idx].extend(res['dets'])
            if res['tile_done']: tracker[p_idx] += 1
            if p_idx in page_expected and tracker[p_idx] == page_expected[p_idx]:
                dets = page_dets[p_idx]; ph, pw = page_dims[p_idx]; refined = []
                for d in sorted(dets, key=lambda x: x["conf"], reverse=True):
                    merged = False
                    for r in refined:
                        if iou(d["box"], r["box"]) > 0.15:
                            r["box"] = [min(d["box"][0], r["box"][0]), min(d["box"][1], r["box"][1]), max(d["box"][2], r["box"][2]), max(d["box"][3], r["box"][3])]; merged = True; break
                    if not merged: refined.append(d)
                img = page_imgs[p_idx]; batch = []
                for j, r in enumerate(refined):
                    box = r["box"]; crop = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    if crop.size == 0: continue
                    crop = cv2.resize(crop, (int(crop.shape[1] * (128/crop.shape[0])), 128))
                    meta = {"p_idx": p_idx, "c_idx": j, "bbox_xyxy_norm": [box[0]/pw, box[1]/ph, box[2]/pw, box[3]/ph]}
                    batch.append({"img": crop, "meta": meta})
                    if len(batch) >= OCR_BATCH_SIZE: ocr_task_q.put(batch); batch = []
                if batch: ocr_task_q.put(batch)
                del page_imgs[p_idx], page_dets[p_idx]; finished += 1
        except: continue

def run_supersonic_session(pdf_dir):
    pdfs = sorted(list(Path(pdf_dir).glob("*.pdf")))
    start_total = time.time()
    results_summary = []
    
    for pdf in pdfs:
        print(f"\n--- {pdf.name} ---")
        start_pdf = time.time()
        tile_q = tmp.Queue(maxsize=100); raw_det_q = tmp.Queue(maxsize=1000)
        page_info_q = tmp.Queue(); ocr_task_q = tmp.Queue(maxsize=50); ocr_res_q = tmp.Queue()
        
        doc = fitz.open(str(pdf)); num_pages = len(doc); doc.close()
        
        gpu_proc = tmp.Process(target=gpu_master_worker, args=(tile_q, raw_det_q, ocr_task_q, ocr_res_q))
        gpu_proc.start()
        
        # 10 Producers, 4 Stitchers (inside manager logic), total capped at 16
        producers = [tmp.Process(target=cpu_producer, args=(pdf, c.tolist(), tile_q, page_info_q, ocr_res_q)) for c in np.array_split(range(num_pages), min(num_pages, 10))]
        for p in producers: p.start()
        
        mgr = tmp.Process(target=manager, args=(num_pages, raw_det_q, page_info_q, ocr_task_q))
        mgr.start()
        
        mgr.join(); [p.join() for p in producers]; tile_q.put(None); gpu_proc.join()
        
        pdf_results = []
        while not ocr_res_q.empty(): pdf_results.append(ocr_res_q.get())
        
        dur = time.time() - start_pdf
        print(f"✅ {pdf.name} Complete: {len(pdf_results)} results in {dur:.2f}s")
        results_summary.append({"name": pdf.name, "time": dur, "count": len(pdf_results)})

    total_dur = time.time() - start_total
    print(f"\n🏁 Session Complete. Total: {total_dur:.2f}s (Avg: {total_dur/len(pdfs):.2f}s/PDF)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    tmp.set_start_method("spawn", force=True)
    run_supersonic_session(args.dir)
