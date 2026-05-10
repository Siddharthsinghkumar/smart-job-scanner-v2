#!/usr/bin/env python3
"""
Audit Bench v1: Single-PDF Benchmark + Accuracy Audit (Stages 1-4 ONLY).
MANDATE COMPLIANT: Local imports inside functions.
Targets: UHT Delhi 07-04.pdf (Pages 14-15 labeled).
"""

import os, sys, json, time, cv2, fitz, warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# Config
YOLO_MODEL_PATH = str(PROJ_ROOT / "artifacts/detector_pivot_yolo_v4_tiles/best.pt")
GT_PATH = PROJ_ROOT / "uht_gt_boxes.json"
PDF_PATH = PROJ_ROOT / "data/raw_pdfs/micro/UHT Delhi 07-04.pdf"
RENDER_DPI = 300
TILE_SIZE = 320
OVERLAP = 0.52
CONF_THRESH = 0.001
IOU_TP_THRESH = 0.4

def calculate_iou(box1, box2):
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea
    return interArea / unionArea if unionArea > 0 else 0

def run_benchmark_audit():
    # MANDATE COMPLIANT: Local Imports
    import torch
    from ultralytics import YOLO
    import easyocr

    print(f"🚀 Initializing Audit Bench for: {PDF_PATH.name}")
    sys.stdout.flush()
    start_total = time.time()
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 0
        print(f"🧠 GPU Ready: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA NOT AVAILABLE.")
    sys.stdout.flush()
    
    # Load Models
    model = YOLO(YOLO_MODEL_PATH)
    if device == 0: model.to('cuda')
    reader = easyocr.Reader(['en'], gpu=(device == 0))
    
    # Load GT
    with open(GT_PATH, 'r') as f:
        gt_data = json.load(f)
    # Mapping: GT "14" -> idx 13, GT "15" -> idx 14
    target_indices = [13, 14]
    gt_total = sum(len(gt_data[str(i+1)]) for i in target_indices)

    # Timing buckets
    timers = {"Stage 1": 0, "Stage 2": 0, "Stage 3": 0, "Stage 4": 0}
    
    # Results buckets for Accuracy
    boxes_s2 = {idx: [] for idx in target_indices}
    boxes_s3 = {idx: [] for idx in target_indices}
    boxes_s4 = {idx: [] for idx in target_indices}

    doc = fitz.open(str(PDF_PATH))
    total_pages = doc.page_count
    
    for p_idx in range(total_pages):
        # Stage 1: Render
        s1_start = time.time()
        page = doc[p_idx]
        pix = page.get_pixmap(dpi=RENDER_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        ph, pw = img.shape[0], img.shape[1]
        timers["Stage 1"] += (time.time() - s1_start)
        
        stride = int(TILE_SIZE * (1 - OVERLAP))
        tiles = []; metas = []
        for y in range(0, ph, stride):
            for x in range(0, pw, stride):
                y2, x2 = min(y + TILE_SIZE, ph), min(x + TILE_SIZE, pw)
                tile = img[y:y2, x:x2]
                if np.mean(tile) > 245: continue
                if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                    tile = cv2.copyMakeBorder(tile, 0, TILE_SIZE-tile.shape[0], 0, TILE_SIZE-tile.shape[1], cv2.BORDER_CONSTANT, value=[255,255,255])
                tiles.append(tile)
                metas.append((x, y))
        
        # Stage 2: Detection
        s2_start = time.time()
        all_dets = []
        if tiles:
            for i in range(0, len(tiles), 32):
                res = model.predict(tiles[i:i+32], conf=CONF_THRESH, imgsz=TILE_SIZE, verbose=False)
                for j, r in enumerate(res):
                    boxes = r.boxes.xyxy.cpu().numpy()
                    tx, ty = metas[i+j]
                    for box in boxes:
                        det = [float(box[0]+tx), float(box[1]+ty), float(box[2]+tx), float(box[3]+ty)]
                        all_dets.append(det)
                        if p_idx in target_indices:
                            boxes_s2[p_idx].append([det[0]/pw, det[1]/ph, det[2]/pw, det[3]/ph])
        timers["Stage 2"] += (time.time() - s2_start)
        
        # Stage 3: Merge
        s3_start = time.time()
        refined = []
        for det in all_dets:
            merged = False
            for r in refined:
                if calculate_iou(det, r) > 0.15:
                    r[0], r[1] = min(det[0], r[0]), min(det[1], r[1])
                    r[2], r[3] = max(det[2], r[2]), max(det[3], r[3])
                    merged = True; break
            if not merged: refined.append(det)
        if p_idx in target_indices:
            for r in refined:
                boxes_s3[p_idx].append([r[0]/pw, r[1]/ph, r[2]/pw, r[3]/ph])
        timers["Stage 3"] += (time.time() - s3_start)
        
        # Stage 4: OCR
        s4_start = time.time()
        for r in refined:
            crop = img[int(r[1]):int(r[3]), int(r[0]):int(r[2])]
            if crop.size == 0: continue
            res = reader.readtext(crop, paragraph=True)
            if p_idx in target_indices:
                if res:
                    boxes_s4[p_idx].append([r[0]/pw, r[1]/ph, r[2]/pw, r[3]/ph])
        timers["Stage 4"] += (time.time() - s4_start)
            
        if (p_idx + 1) % 5 == 0:
            print(f"  Processed {p_idx+1}/{total_pages} pages...")
            sys.stdout.flush()

    doc.close()
    total_time = time.time() - start_total

    # --- Metrics Computation ---
    accuracy_results = []
    for stage_name, stage_boxes in [("Stage 2", boxes_s2), ("Stage 3", boxes_s3), ("Stage 4", boxes_s4)]:
        tp, fp = 0, 0
        missed_gt = []
        matches = []
        for p_idx in target_indices:
            gts = gt_data[str(p_idx+1)]
            preds = stage_boxes[p_idx]
            matched_gt_indices = set()
            for p_box in preds:
                best_iou, best_gt_idx = 0, -1
                for i, g_box in enumerate(gts):
                    if i in matched_gt_indices: continue
                    iou = calculate_iou(p_box, g_box)
                    if iou > best_iou: best_iou, best_gt_idx = iou, i
                if best_iou >= IOU_TP_THRESH:
                    tp += 1
                    matched_gt_indices.add(best_gt_idx)
                    matches.append((p_box, gts[best_gt_idx], best_iou))
                else:
                    fp += 1
            for i, g_box in enumerate(gts):
                if i not in matched_gt_indices:
                    missed_gt.append((p_idx+1, g_box))
        recall = tp / gt_total if gt_total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy_results.append({
            "Stage": stage_name, "TP": tp, "FP": fp, "FN": gt_total - tp, 
            "Recall": recall, "Precision": precision, 
            "Missed": missed_gt, "SampleMatches": matches[:3]
        })

    # --- Report Output ---
    print("\n### 1. Timing Table")
    print("| Stage | Time (s) |")
    print("|-------|----------|")
    for s, t in timers.items():
        print(f"| {s} | {t:.2f} |")
    print(f"| **TOTAL** | **{total_time:.2f}** |")

    print("\n### 2. Accuracy Table")
    print("| Stage | TP | FP | FN | Recall | Precision |")
    print("|-------|----|----|----|--------|-----------|")
    for res in accuracy_results:
        print(f"| {res['Stage']} | {res['TP']} | {res['FP']} | {res['FN']} | {res['Recall']:.3f} | {res['Precision']:.3f} |")

    print(f"\nGT total: {gt_total}")
    s2_tp, s3_tp, s4_tp = accuracy_results[0]['TP'], accuracy_results[1]['TP'], accuracy_results[2]['TP']
    print(f"TP Delta (S2->S3): {s3_tp - s2_tp}")
    print(f"TP Delta (S3->S4): {s4_tp - s3_tp}")
    s2_fp, s4_fp = accuracy_results[0]['FP'], accuracy_results[2]['FP']
    fp_red = (s2_fp - s4_fp) / s2_fp * 100 if s2_fp > 0 else 0
    print(f"FP reduction (S2->S4): {fp_red:.1f}%")

    print("\n### 3. Validation Checks")
    tp_decreased = "YES" if s3_tp < s2_tp or s4_tp < s3_tp else "NO"
    print(f"- Did TP decrease at any stage? {tp_decreased}")
    print(f"- Did Stage 3 improve recall over Stage 2? {'YES' if s3_tp > s2_tp else 'NO'}")
    print(f"- Did Stage 4 preserve or improve recall? {'YES' if s4_tp >= s3_tp else 'NO'}")
    print(f"- Is TOTAL time < 100s? {'YES' if total_time < 100 else 'NO'}")

    print("\n### 4. Debug Output")
    print("#### Missed GT boxes (FN) in Stage 4:")
    for m in accuracy_results[2]['Missed']: print(f"- Page {m[0]}: {m[1]}")
    print("\n#### Example matches (IoU pairs) in Stage 4:")
    for m in accuracy_results[2]['SampleMatches']: print(f"- Pred: {m[0]}, GT: {m[1]}, IoU: {m[2]:.4f}")

    if tp_decreased == "YES":
        print("\n🚨 HARD CONSTRAINT VIOLATED: TP decreased after Stage 2. Stopping.")
        sys.exit(1)

if __name__ == "__main__":
    run_benchmark_audit()
