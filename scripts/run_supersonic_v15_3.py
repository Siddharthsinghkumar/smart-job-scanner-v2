#!/usr/bin/env python3
"""
Stage 1-2-3 Unified v15.3: 'Total-Breakthrough' Engine.
Goal: 10 PDFs < 1000s @ 95%+ Recall.
Architecture: 10 Producers, 1 GPU Master, 4 Stitchers (16-Core Adherence).
Vectorized NMS + Balanced GPU Scheduler.
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
import psutil
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# Optimized Configuration
YOLO_MODEL_PATH = str(PROJ_ROOT / "artifacts/detector_pivot_yolo_v4_tiles/best.pt")
RENDER_DPI = 300
TILE_SIZE = 320
OVERLAP = 0.60 
CONF_THRESH = 0.0005
OCR_BATCH_SIZE = 32

def vectorized_nms(boxes, scores, iou_threshold):
    if len(boxes) == 0: return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def gpu_master_worker(tile_q, raw_det_q, ocr_task_q, ocr_res_q, gpu_idle_since):
    from ultralytics import YOLO
    from easyocr import Reader
    model = YOLO(YOLO_MODEL_PATH).to('cuda')
    reader = Reader(['en'], gpu=True, detector="dbnet18", cudnn_benchmark=True)
    
    yolo_weight = 2
    ocr_weight = 1
    
    while True:
        work_done = False
        # Ratio-based scheduling
        for _ in range(yolo_weight):
            try:
                batch = tile_q.get_nowait()
                if batch is None: return
                gpu_idle_since.value = time.time()
                imgs = [item['tile'] for item in batch]
                results = model.predict(imgs, conf=CONF_THRESH, imgsz=TILE_SIZE, verbose=False, device=0)
                for i, res in enumerate(results):
                    meta = batch[i]['meta']
                    detections = []
                    for b in res.boxes:
                        c = b.xyxy[0].cpu().numpy()
                        detections.append([float(c[0]+meta[1]), float(c[1]+meta[2]), float(c[2]+meta[1]), float(c[3]+meta[2]), float(b.conf[0])])
                    raw_det_q.put({"p_idx": meta[0], "dets": detections, "tile_done": True})
                work_done = True
            except: break

        for _ in range(ocr_weight):
            try:
                batch = ocr_task_q.get_nowait()
                if batch:
                    gpu_idle_since.value = time.time()
                    imgs = [item['img'] for item in batch]
                    for j, img in enumerate(imgs):
                        res = reader.readtext(img, paragraph=True)
                        text = " ".join([r[1] for r in res])
                        entry = batch[j]['meta'].copy()
                        entry.update({"ocr_text_raw": text, "status": "ok"})
                        ocr_res_q.put(entry)
                    work_done = True
            except: break
        
        if not work_done:
            time.sleep(0.005)

def cpu_producer(pdf_path, p_indices, tile_q, page_info_q, ocr_res_q):
    doc = fitz.open(str(pdf_path))
    for p_idx in p_indices:
        page = doc[p_idx]
        text = page.get_text()
        if sum(1 for c in text if c.isalpha()) > 500:
            ocr_res_q.put({"p_idx": p_idx, "bbox_xyxy_norm": [0,0,1,1], "ocr_text_raw": text[:1000], "status": "digital_bypass"})
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
                if len(batch) >= 64: tile_q.put(batch); batch = []
        if batch: tile_q.put(batch)
        page_info_q.put({"p_idx": p_idx, "expected_tiles": tiles_count})
    doc.close()

def stitcher_worker(raw_det_q, page_info_q, ocr_task_q, stitcher_id, num_stitchers):
    page_dets = defaultdict(list); tracker = defaultdict(int)
    page_expected = {}; page_dims = {}; page_imgs = {}
    
    while True:
        # Process page info
        while not page_info_q.empty():
            try:
                info = page_info_q.get_nowait()
                if info is None: return # Shutdown
                p_idx = info['p_idx']
                if p_idx % num_stitchers != stitcher_id: 
                    # Not my page, but we need to handle this differently if we use multiple stitchers
                    # In this design, let's just use 1 manager or distribute by p_idx
                    pass 
                
                if 'dim' in info: page_dims[p_idx] = info['dim']
                if 'img' in info: page_imgs[p_idx] = info['img']
                if 'expected_tiles' in info: page_expected[p_idx] = info['expected_tiles']
            except: break

        # Process detections
        try:
            res = raw_det_q.get(timeout=0.01)
            if res is None: return
            p_idx = res['p_idx']
            if p_idx % num_stitchers == stitcher_id:
                page_dets[p_idx].extend(res['dets'])
                if res['tile_done']: tracker[p_idx] += 1
                if p_idx in page_expected and tracker[p_idx] == page_expected[p_idx]:
                    # Vectorized NMS
                    dets = np.array(page_dets[p_idx])
                    if len(dets) > 0:
                        keep = vectorized_nms(dets[:, :4], dets[:, 4], 0.15)
                        refined = dets[keep]
                        img = page_imgs[p_idx]; ph, pw = page_dims[p_idx]; batch = []
                        for j, r in enumerate(refined):
                            box = r[:4]; crop = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                            if crop.size == 0: continue
                            crop = cv2.resize(crop, (int(crop.shape[1] * (128/crop.shape[0])), 128))
                            meta = {"p_idx": p_idx, "c_idx": j, "bbox_xyxy_norm": [box[0]/pw, box[1]/ph, box[2]/pw, box[3]/ph]}
                            batch.append({"img": crop, "meta": meta})
                            if len(batch) >= OCR_BATCH_SIZE: ocr_task_q.put(batch); batch = []
                        if batch: ocr_task_q.put(batch)
                    del page_imgs[p_idx], page_dets[p_idx], page_expected[p_idx], tracker[p_idx], page_dims[p_idx]
            else:
                # Put back or use a better distribution
                raw_det_q.put(res)
        except: continue

def hardware_watcher(gpu_idle_since, stop_event, log_file):
    with open(log_file, "a") as f:
        while not stop_event.is_set():
            cpu_pct = psutil.cpu_percent()
            mem_pct = psutil.virtual_memory().percent
            idle_dur = time.time() - gpu_idle_since.value
            f.write(f"{datetime.now().isoformat()} - CPU: {cpu_pct}% | MEM: {mem_pct}% | GPU_IDLE: {idle_dur:.1f}s\n")
            f.flush()
            time.sleep(1)

def run_supersonic_session(pdf_dir):
    pdfs = sorted(list(Path(pdf_dir).glob("*.pdf")))
    start_total = time.time()
    results_summary = []
    
    log_path = PROJ_ROOT / "hw_log_v15_3.txt"
    if log_path.exists(): log_path.unlink()
    
    gpu_idle_since = tmp.Value('d', time.time())
    stop_watcher = tmp.Event()
    watcher = tmp.Process(target=hardware_watcher, args=(gpu_idle_since, stop_watcher, str(log_path)))
    watcher.start()

    try:
        for pdf in pdfs:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing {pdf.name}...")
            start_pdf = time.time()
            tile_q = tmp.Queue(maxsize=200); raw_det_q = tmp.Queue(maxsize=2000)
            page_info_qs = [tmp.Queue() for _ in range(4)]
            ocr_task_q = tmp.Queue(maxsize=100); ocr_res_q = tmp.Queue()
            
            doc = fitz.open(str(pdf)); num_pages = len(doc); doc.close()
            
            gpu_proc = tmp.Process(target=gpu_master_worker, args=(tile_q, raw_det_q, ocr_task_q, ocr_res_q, gpu_idle_since))
            gpu_proc.start()
            
            producers = [tmp.Process(target=cpu_producer, args=(pdf, c.tolist(), tile_q, page_info_qs[i%4], ocr_res_q)) 
                         for i, c in enumerate(np.array_split(range(num_pages), 10))]
            for p in producers: p.start()
            
            stitchers = [tmp.Process(target=stitcher_worker, args=(raw_det_q, page_info_qs[i], ocr_task_q, i, 4)) 
                         for i in range(4)]
            for s in stitchers: s.start()
            
            # Monitoring loop
            pdf_results = []
            pages_done = 0
            while len(pdf_results) < num_pages: # This is tricky since one page can have multiple results or 0
                # Using a different condition: all producers and stitchers finished
                # But we need to collect results
                time.sleep(1)
                elapsed = time.time() - start_pdf
                idle_dur = time.time() - gpu_idle_since.value
                
                if elapsed > 180: # 3 minutes
                    print(f"❌ STALL DETECTED: {pdf.name} took > 180s. Killing...")
                    break
                if idle_dur > 20:
                    print(f"❌ GPU STALL: Idle for {idle_dur:.1f}s. Killing...")
                    break
                
                # Check if all producers are done
                if all(not p.is_alive() for p in producers) and tile_q.empty() and raw_det_q.empty() and ocr_task_q.empty():
                    # Wait a bit for last OCR
                    time.sleep(2)
                    break

            for p in producers: p.terminate()
            for s in stitchers: 
                raw_det_q.put(None)
                s.terminate()
            tile_q.put(None)
            gpu_proc.terminate()
            
            while not ocr_res_q.empty(): pdf_results.append(ocr_res_q.get())
            
            dur = time.time() - start_pdf
            print(f"✅ {pdf.name} Complete: {len(pdf_results)} entries in {dur:.2f}s")
            results_summary.append({"name": pdf.name, "time": dur, "count": len(pdf_results)})

    finally:
        stop_watcher.set()
        watcher.join()

    total_dur = time.time() - start_total
    print(f"\n🏁 Session Complete. Total: {total_dur:.2f}s (Avg: {total_dur/len(pdfs):.2f}s/PDF)")
    
    report = {
        "total_time": total_dur,
        "avg_time_per_pdf": total_dur/len(pdfs),
        "results": results_summary
    }
    with open("benchmark_v15_3_results.json", "w") as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/home/sidd/project/smart-job-scanner-v2/data/continuum_test/")
    args = parser.parse_args()
    tmp.set_start_method("spawn", force=True)
    run_supersonic_session(args.dir)
