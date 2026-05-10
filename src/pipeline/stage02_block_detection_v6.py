#!/usr/bin/env python3
"""
Stage 02 v11.1: Sonic-Stream Core Logic.
Optimized for modular calling by Hybrid Router.
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

warnings.filterwarnings("ignore", category=FutureWarning)
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# Configuration
YOLO_MODEL_PATH = str(PROJ_ROOT / "artifacts/detector_pivot_yolo_v4_tiles/best.pt")
RENDER_DPI = 300
TILE_SIZE = 320
OVERLAP = 0.50 
CONF_THRESH = 0.0001 
CROP_MANIFEST = Path("run_state/crop_manifest.jsonl")

def iou(b1, b2):
    xa, ya, xb, yb = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / u if u > 0 else 0

def gpu_worker(in_q, raw_det_q, model_path, worker_id):
    from ultralytics import YOLO
    model = YOLO(model_path).to('cuda')
    _ = model.predict(np.zeros((320,320,3), dtype=np.uint8), verbose=False)
    while True:
        batch = in_q.get()
        if batch is None: break
        imgs = [item['tile'] for item in batch]
        results = model.predict(imgs, conf=CONF_THRESH, imgsz=TILE_SIZE, verbose=False, device="cpu")
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
        torch.cuda.empty_cache()

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
                if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                    tile = cv2.copyMakeBorder(tile, 0, TILE_SIZE-tile.shape[0], 0, TILE_SIZE-tile.shape[1], cv2.BORDER_CONSTANT, value=[255,255,255])
                batch.append({'tile': tile, 'meta': (p_idx, x, y)})
                tiles_count += 1
                if len(batch) >= 128:
                    tile_q.put(batch)
                    batch = []
        if batch: tile_q.put(batch)
        page_info_q.put({"p_idx": p_idx, "expected_tiles": tiles_count})
    doc.close()

def feeder_manager(raw_det_q, page_info_q, stitch_q, num_pages_to_wait):
    page_dets = defaultdict(list)
    page_tile_tracker = defaultdict(int)
    page_expected = {}
    page_dims = {}
    finished_pages = 0
    while finished_pages < num_pages_to_wait:
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
    for _ in range(6): stitch_q.put(None)

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
        
        # High-res crop extraction
        page = doc[p_idx]
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        page_results = []
        for j, r in enumerate(refined):
            box = r["box"]
            crop_filename = f"{Path(pdf_path).stem}_p{p_idx+1}_c{j:04d}.png"
            crop_path = Path("data/crops") / crop_filename
            crop_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            if crop_img.size > 0:
                cv2.imwrite(str(crop_path), crop_img)
            
            page_results.append({
                "crop_id": f"p{p_idx+1}_c{j}",
                "page_index0": p_idx,
                "bbox_xyxy_norm": [box[0]/pw, box[1]/ph, box[2]/pw, box[3]/ph],
                "crop_image_path": f"data/crops/{crop_filename}",
                "status": "ok"
            })
        final_q.put(page_results)
    doc.close()

def run_v9_5_internal(pdf_path, page_indices):
    num_pages = len(page_indices)
    if num_pages == 0: return []
    Path("data/crops").mkdir(parents=True, exist_ok=True)
    
    tile_q = tmp.Queue(maxsize=100)
    raw_det_q = tmp.Queue(maxsize=2000)
    page_info_q = tmp.Queue()
    stitch_q = tmp.Queue()
    final_q = tmp.Queue()
    
    gpu_procs = [tmp.Process(target=gpu_worker, args=(tile_q, raw_det_q, YOLO_MODEL_PATH, i)) for i in range(4)]
    for p in gpu_procs: p.start()
    
    chunks = np.array_split(page_indices, min(num_pages, 8))
    producers = [tmp.Process(target=cpu_producer, args=(pdf_path, c.tolist(), tile_q, page_info_q)) for c in chunks]
    for p in producers: p.start()
    
    stitchers = [tmp.Process(target=stitch_worker, args=(stitch_q, final_q, pdf_path)) for i in range(6)]
    for s in stitchers: s.start()
    
    feeder = tmp.Process(target=feeder_manager, args=(raw_det_q, page_info_q, stitch_q, num_pages))
    feeder.start()
    
    results = []
    for _ in range(num_pages):
        results.extend(final_q.get())
    
    feeder.join(); [p.join() for p in producers]
    for _ in range(4): tile_q.put(None)
    [p.join() for p in gpu_procs]; [s.join() for s in stitchers]
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    args = parser.parse_args()
    tmp.set_start_method('spawn', force=True)
    results = run_v9_5_internal(Path(args.pdf), list(range(fitz.open(args.pdf).page_count)))
    with open(CROP_MANIFEST, "w") as f:
        for c in results:
            c.update({"pdf_path": args.pdf, "newspaper_name": Path(args.pdf).stem})
            f.write(json.dumps(c) + "\n")
    print(f"✅ v11.1 Complete: {len(results)} ads.")
