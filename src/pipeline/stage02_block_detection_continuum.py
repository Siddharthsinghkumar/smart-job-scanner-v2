#!/usr/bin/env python3
"""
from src.utils.logging_utils import configure_logging
logger = configure_logging("stage02_block_detection_continuum")
Stage 02 v11.2: Continuum Engine.
Strategy: Warm Model Persistence across multiple PDFs. 
Eliminates the 10s "nothing" gap between files.
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
import shutil
from pathlib import Path
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# Configuration
YOLO_MODEL_PATH = str(PROJ_ROOT / "artifacts/detector_pivot_yolo_v4_tiles/best.pt")
RENDER_DPI = 300
TILE_SIZE = 320
OVERLAP = 0.50 
CONF_THRESH = 0.0001 
NUM_GPU_WORKERS = 4 
NUM_CPU_PRODUCERS = 8 
NUM_CPU_STITCHERS = 6 
CROP_MANIFEST = Path("run_state/crop_manifest.jsonl")

def iou(b1, b2):
    xa, ya, xb, yb = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / u if u > 0 else 0

def gpu_worker(in_q, raw_det_q, model_path, worker_id):
    from ultralytics import YOLO
    model = YOLO(model_path).to('cuda')
    # Warmup once per session, not per PDF
    _ = model.predict(np.zeros((320,320,3), dtype=np.uint8), verbose=False)
    
    while True:
        batch = in_q.get()
        if batch is None: break # End of entire corpus
        
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

def cpu_producer(pdf_path, p_indices, tile_q, page_info_q):
    doc = fitz.open(str(pdf_path))
    for p_idx in p_indices:
        page = doc[p_idx]
        pix = page.get_pixmap(dpi=RENDER_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        h, w = img.shape[:2]
        
        page_info_q.put({"p_idx": p_idx, "dim": (h, w)})
        stride = int(TILE_SIZE * (1 - OVERLAP))
        tiles_count = 0
        batch = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2, x2 = min(y + TILE_SIZE, h), min(x + TILE_SIZE, w)
                tile = img[y:y2, x:x2]
                if np.mean(tile) > 245: continue
                if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                    tile = cv2.copyMakeBorder(tile, 0, TILE_SIZE-tile.shape[0], 0, TILE_SIZE-tile.shape[1], cv2.BORDER_CONSTANT, value=[255,255,255])
                
                batch.append({'tile': tile, 'meta': (p_idx, x, y)})
                tiles_count += 1
                if len(batch) >= 64:
                    tile_q.put(batch)
                    batch = []
        if batch: tile_q.put(batch)
        page_info_q.put({"p_idx": p_idx, "expected_tiles": tiles_count})
    doc.close()

def feeder_manager(raw_det_q, page_info_q, stitch_q, num_pages):
    page_dets = defaultdict(list)
    page_tile_tracker = defaultdict(int)
    page_expected = {}
    page_dims = {}
    finished_pages = 0
    while finished_pages < num_pages:
        while not page_info_q.empty():
            info = page_info_q.get()
            if 'dim' in info: page_dims[info['p_idx']] = info['dim']
            if 'expected_tiles' in info: page_expected[info['p_idx']] = info['expected_tiles']
        try:
            res = raw_det_q.get(timeout=0.1)
            p_idx = res['p_idx']
            page_dets[p_idx].extend(res['dets'])
            if res['tile_done']: page_tile_tracker[p_idx] += 1
            if p_idx in page_expected and page_tile_tracker[p_idx] == page_expected[p_idx]:
                if p_idx in page_dims:
                    stitch_q.put((p_idx, page_dets[p_idx], page_dims[p_idx]))
                    del page_dets[p_idx]
                    finished_pages += 1
        except: continue
    for _ in range(NUM_CPU_STITCHERS): stitch_q.put(None)

def stitch_worker(stitch_q, final_q, pdf_path):
    doc = fitz.open(str(pdf_path))
    while True:
        task = stitch_q.get()
        if task is None: break
        p_idx, dets, (ph, pw) = task
        refined = []
        for d in sorted(dets, key=lambda x: x["conf"], reverse=True):
            merged = False
            for r in refined:
                if iou(d["box"], r["box"]) > 0.15:
                    r["box"] = [min(d["box"][0], r["box"][0]), min(d["box"][1], r["box"][1]),
                               max(d["box"][2], r["box"][2]), max(d["box"][3], r["box"][3])]
                    merged = True; break
            if not merged: refined.append(d)
        
        page_results = []
        for j, r in enumerate(refined):
            box = r["box"]
            page_results.append({
                "crop_id": f"p{p_idx+1}_c{j}",
                "page_index0": p_idx,
                "bbox_xyxy_norm": [box[0]/pw, box[1]/ph, box[2]/pw, box[3]/ph],
                "status": "ok"
            })
        final_q.put(page_results)
    doc.close()

def process_one_pdf(pdf_path, tile_q, raw_det_q, page_info_q, stitch_q, final_q):
    """Processes a single PDF through the existing warm pipeline."""
    doc = fitz.open(str(pdf_path))
    num_pages = len(doc); doc.close()
    
    # 1. Start Producers
    chunks = np.array_split(range(num_pages), min(num_pages, NUM_CPU_PRODUCERS))
    producers = [tmp.Process(target=cpu_producer, args=(pdf_path, c.tolist(), tile_q, page_info_q)) for c in chunks]
    for p in producers: p.start()
    
    # 2. Start Stitchers
    stitchers = [tmp.Process(target=stitch_worker, args=(stitch_q, final_q, pdf_path)) for i in range(NUM_CPU_STITCHERS)]
    for s in stitchers: s.start()
    
    # 3. Start Feeder
    feeder = tmp.Process(target=feeder_manager, args=(raw_det_q, page_info_q, stitch_q, num_pages))
    feeder.start()
    
    # 4. Wait for this PDF to finish
    results = []
    for _ in range(num_pages):
        results.extend(final_q.get())
    
    feeder.join()
    for p in producers: p.join()
    for s in stitchers: s.join()
    
    return results

def run_continuum(pdf_dir):
    print(f"🚀 v11.2 Continuum: Processing directory {pdf_dir}")
    pdfs = sorted(list(Path(pdf_dir).glob("*.pdf")))
    
    # Persistent Queues
    tile_q = tmp.Queue(maxsize=100)
    raw_det_q = tmp.Queue(maxsize=2000)
    page_info_q = tmp.Queue()
    stitch_q = tmp.Queue()
    final_q = tmp.Queue()
    
    # 1. Start Persistent GPU Workers (Warm models)
    gpu_procs = [tmp.Process(target=gpu_worker, args=(tile_q, raw_det_q, YOLO_MODEL_PATH, i)) for i in range(NUM_GPU_WORKERS)]
    for p in gpu_procs: p.start()
    
    for pdf in pdfs:
        start_pdf = time.time()
        print(f"\n--- Starting {pdf.name} ---")
        
        # Cleanup crops dir safely
        if os.path.exists("data/crops"): shutil.rmtree("data/crops")
        os.makedirs("data/crops", exist_ok=True)
        
        results = process_one_pdf(pdf, tile_q, raw_det_q, page_info_q, stitch_q, final_q)
        
        # Save manifest per PDF
        out_manifest = Path(f"run_state/crop_manifest_{pdf.stem}.jsonl")
        with open(out_manifest, "w") as f:
            for c in results:
                c.update({"pdf_path": str(pdf), "newspaper_name": pdf.stem})
                f.write(json.dumps(c) + "\n")
        
        dur = time.time() - start_pdf
        print(f"✅ {pdf.name} Finished in {dur:.2f}s")

    # Shutdown session
    for _ in range(NUM_GPU_WORKERS): tile_q.put(None)
    for p in gpu_procs: p.join()
    print("\n🏁 Continuum Session Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    tmp.set_start_method('spawn', force=True)
    run_continuum(args.dir)
