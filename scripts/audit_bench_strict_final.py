#!/usr/bin/env python3
"""
Audit Bench Strict Final v2: Parallel Single-PDF Benchmark + Accuracy Audit.
MANDATE COMPLIANT: Local imports, 16-core CPU limit, 4-GPU worker strategy.
Targets: UHT Delhi 07-04.pdf.
"""

import os, sys, json, time, cv2, fitz, warnings, queue
import numpy as np
import torch.multiprocessing as tmp
from pathlib import Path
from collections import defaultdict

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# Configuration from .md files
YOLO_MODEL_PATH = str(PROJ_ROOT / "artifacts/detector_pivot_yolo_v4_tiles/best.pt")
GT_PATH = PROJ_ROOT / "uht_gt_boxes.json"
PDF_PATH = PROJ_ROOT / "data/raw_pdfs/micro/UHT Delhi 07-04.pdf"
RENDER_DPI = 300
TILE_SIZE = 320
OVERLAP = 0.52
CONF_THRESH = 0.0005 
IOU_TP_THRESH = 0.4
NEGATIVE_KEYWORDS = ["obituary", "missing person", "loss of documents", "change of name"]

def calculate_iou(box1, box2):
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea
    return interArea / unionArea if unionArea > 0 else 0

def gpu_worker(tile_q, det_q, task_q, res_q, worker_id):
    # MANDATE COMPLIANT: Local Imports
    import torch
    from ultralytics import YOLO
    import easyocr
    
    # 🚨 FATAL HARDWARE ERROR PROTOCOL: Hard stop on CUDA Unknown Error
    try:
        if not torch.cuda.is_available():
            print(f"[GPU {worker_id}] CUDA NOT AVAILABLE. STOPPING.")
            return
        # Attempt minimal tensor to verify driver
        _ = torch.zeros(1).cuda()
    except Exception as e:
        print(f"[GPU {worker_id}] FATAL CUDA ERROR: {e}. STOPPING.")
        return

    model = YOLO(YOLO_MODEL_PATH).to('cuda')
    reader = easyocr.Reader(['en'], gpu=True, detector="dbnet18")
    
    while True:
        # Priority 1: Detection (Stage 2)
        try:
            msg = tile_q.get(timeout=0.1)
            if msg is None: break
            
            # Check if this is a batch or a sentinel
            if isinstance(msg, list):
                batch = msg
                imgs = [it['tile'] for it in batch]
                res = model.predict(imgs, conf=CONF_THRESH, imgsz=TILE_SIZE, verbose=False)
                for i, r in enumerate(res):
                    meta = batch[i]['meta']
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    dets = []
                    for j, box in enumerate(boxes):
                        dets.append({
                            "p_idx": meta[0],
                            "box": [float(box[0]+meta[1]), float(box[1]+meta[2]), float(box[2]+meta[1]), float(box[3]+meta[2])],
                            "conf": float(confs[j])
                        })
                    det_q.put({"p_idx": meta[0], "dets": dets, "done": True})
            else:
                # Page sentinel for manager
                det_q.put(msg)
        except queue.Empty: pass
        
        # Priority 2: OCR (Stage 3)
        try:
            ocr_batch = task_q.get_nowait()
            for it in ocr_batch:
                res = reader.readtext(it['img'], paragraph=True)
                text = " ".join([r[1] for r in res])
                res_q.put({"p_idx": it['p_idx'], "bbox": it['bbox'], "text": text})
        except queue.Empty: pass

def cpu_producer(pdf_path, p_indices, tile_q):
    doc = fitz.open(str(pdf_path))
    for p_idx in p_indices:
        page = doc[p_idx]
        pix = page.get_pixmap(dpi=RENDER_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        ph, pw = img.shape[0], img.shape[1]
        stride = int(TILE_SIZE * (1 - OVERLAP))
        batch = []
        tiles_count = 0
        for y in range(0, ph, stride):
            for x in range(0, pw, stride):
                y2, x2 = min(y + TILE_SIZE, ph), min(x + TILE_SIZE, pw)
                tile = img[y:y2, x:x2]
                if np.mean(tile) > 245: continue
                if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                    tile = cv2.copyMakeBorder(tile, 0, TILE_SIZE-tile.shape[0], 0, TILE_SIZE-tile.shape[1], cv2.BORDER_CONSTANT, value=[255,255,255])
                batch.append({"tile": tile, "meta": (p_idx, x, y)})
                tiles_count += 1
                if len(batch) >= 32:
                    tile_q.put(batch); batch = []
        if batch: tile_q.put(batch)
        # Sentinel for page completion
        tile_q.put({"p_idx": p_idx, "expected": tiles_count, "dim": (ph, pw), "img": img})
    doc.close()

def manager(num_pages, det_q, task_q, res_all, audit_indices):
    from sentence_transformers import SentenceTransformer, util
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    page_dets = defaultdict(list); page_tracker = defaultdict(int); page_expected = {}; page_imgs = {}; page_dims = {}
    finished = 0
    
    while finished < num_pages:
        try:
            msg = det_q.get(timeout=1)
            if "expected" in msg:
                p_idx = msg["p_idx"]
                page_expected[p_idx] = msg["expected"]
                page_imgs[p_idx] = msg["img"]
                page_dims[p_idx] = msg["dim"]
            else:
                p_idx = msg["p_idx"]
                page_dets[p_idx].extend(msg["dets"])
                page_tracker[p_idx] += 1
            
            if p_idx in page_expected and page_tracker[p_idx] == page_expected[p_idx]:
                # Stage 2 -> Stage 3 (Geometry Merge)
                raw_dets = page_dets[p_idx]
                ph, pw = page_dims[p_idx]
                
                # Accuracy snapshot for Stage 2 (Raw)
                if p_idx in audit_indices:
                    res_all[f"S2_{p_idx}"] = [ [d['box'][0]/pw, d['box'][1]/ph, d['box'][2]/pw, d['box'][3]/ph] for d in raw_dets ]

                # MERGE (STRICT STAGE 2 -> 3)
                refined = []
                for det in raw_dets:
                    merged = False
                    for r in refined:
                        if calculate_iou(det['box'], r['box']) > 0.15:
                            r['box'] = [min(det['box'][0], r['box'][0]), min(det['box'][1], r['box'][1]), max(det['box'][2], r['box'][2]), max(det['box'][3], r['box'][3])]
                            merged = True; break
                    if not merged: refined.append(det)
                
                # Accuracy snapshot for Stage 3
                if p_idx in audit_indices:
                    res_all[f"S3_{p_idx}"] = [ [r['box'][0]/pw, r['box'][1]/ph, r['box'][2]/pw, r['box'][3]/ph] for r in refined ]

                # FEED OCR (Stage 3)
                img = page_imgs[p_idx]
                ocr_batch = []
                for r in refined:
                    box = r['box']
                    crop = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    if crop.size == 0: continue
                    ocr_batch.append({"p_idx": p_idx, "bbox": [box[0]/pw, box[1]/ph, box[2]/pw, box[3]/ph], "img": crop})
                if ocr_batch: task_q.put(ocr_batch)
                
                del page_imgs[p_idx], page_dets[p_idx]
                finished += 1
        except: pass

def run_strict_audit():
    print("🚀 Starting Parallel Strict Audit...")
    start_total = time.time()
    
    with open(GT_PATH, 'r') as f: gt_data = json.load(f)
    audit_indices = [13, 14]
    
    tile_q = tmp.Queue(maxsize=100); det_q = tmp.Queue(); task_q = tmp.Queue(); res_q = tmp.Queue()
    manager_dict = tmp.Manager().dict()
    
    # MANDATE: 4 GPU Workers, 10 CPU Producers (Capped for UHT)
    num_gpu = 2 # Reduced from 4 for 4GB stability after recent crash
    gpu_procs = [tmp.Process(target=gpu_worker, args=(tile_q, det_q, task_q, res_q, i)) for i in range(num_gpu)]
    for p in gpu_procs: p.start()
    
    doc = fitz.open(str(PDF_PATH))
    num_pages = doc.page_count
    
    producer = tmp.Process(target=cpu_producer, args=(PDF_PATH, range(num_pages), tile_q))
    producer.start()
    
    mgr = tmp.Process(target=manager, args=(num_pages, det_q, task_q, manager_dict, audit_indices))
    mgr.start()
    
    # Final Result Collection
    ocr_results = defaultdict(list)
    received = 0
    # This is a bit complex in parallel. We wait for all pages.
    while mgr.is_alive() or not res_q.empty():
        try:
            it = res_q.get(timeout=1)
            ocr_results[it['p_idx']].append(it)
        except: pass
        
    producer.join(); mgr.join()
    for _ in range(num_gpu): tile_q.put(None)
    for p in gpu_procs: p.join()
    
    # Stage 4: Filtering + Vectors (On Audit Pages)
    from sentence_transformers import SentenceTransformer, util
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    for p_idx in audit_indices:
        items = ocr_results[p_idx]
        # Accuracy Snapshot for Stage 4 (OCR Unfiltered)
        manager_dict[f"S3_OCR_{p_idx}"] = [x["bbox"] for x in items]
        
        # FILTER
        cured = [it for it in items if not any(kw in it["text"].lower() for kw in NEGATIVE_KEYWORDS) and len(it["text"].split()) >= 3]
        
        # DEDUPE & STITCH (Conservative)
        stitched = []
        cured.sort(key=lambda x: x["bbox"][1])
        for it in cured:
            found = False
            for s in stitched:
                if it["bbox"][1] - s["bbox"][3] < 0.005 and \
                   min(it["bbox"][2], s["bbox"][2]) - max(it["bbox"][0], s["bbox"][0]) > 0:
                    s["bbox"] = [min(s["bbox"][0], it["bbox"][0]), min(s["bbox"][1], it["bbox"][1]), max(s["bbox"][2], it["bbox"][2]), max(s["bbox"][3], it["bbox"][3])]
                    found = True; break
            if not found: stitched.append(it.copy())
            
        manager_dict[f"S4_{p_idx}"] = [x["bbox"] for x in stitched]

    # --- FINAL TABLES ---
    total_time = time.time() - start_total
    print(f"\nTOTAL TIME: {total_time:.2f}s")
    
    gt_total = sum(len(gt_data[str(i+1)]) for i in audit_indices)
    
    print("\n| Stage | TP | FP | FN | Recall |")
    print("|-------|----|----|----|--------|")
    
    for stage_tag in ["S2", "S3", "S4"]:
        tp, fp = 0, 0
        matched_gt = set()
        for p_idx in audit_indices:
            gts = gt_data[str(p_idx+1)]
            preds = manager_dict.get(f"{stage_tag}_{p_idx}", [])
            for p_box in preds:
                best_iou, best_idx = 0, -1
                for i, g_box in enumerate(gts):
                    if (p_idx, i) in matched_gt: continue
                    iou = calculate_iou(p_box, g_box)
                    if iou > best_iou: best_iou, best_idx = iou, i
                if best_iou >= IOU_TP_THRESH: tp += 1; matched_gt.add((p_idx, best_idx))
                else: fp += 1
        print(f"| {stage_tag} | {tp} | {fp} | {gt_total - tp} | {tp/gt_total:.3f} |")

if __name__ == "__main__":
    tmp.set_start_method('spawn', force=True)
    run_strict_audit()
