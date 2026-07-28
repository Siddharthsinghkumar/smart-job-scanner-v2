#!/usr/bin/env python3
import os, sys, json, time, cv2, fitz
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import easyocr
import torch
from collections import defaultdict

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from src.pipeline.stage04_5_refiner import stitch_blocks, filter_by_keywords, deduplicate_vectors

# Config
YOLO_MODEL_PATH = str(PROJ_ROOT / "artifacts/detector_pivot_yolo_v4_tiles/best.pt")
GT_PATH = PROJ_ROOT / "uht_gt_boxes.json"
PDF_PATH = PROJ_ROOT / "data/raw_pdfs/micro/UHT Delhi 07-04.pdf"
RENDER_DPI = 300
TILE_SIZE = 320
OVERLAP = 0.52
CONF_THRESH = 0.001
IOU_THRESHOLD = 0.4 

def calculate_iou(box1, box2):
    xA, yA, xB, yB = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea
    return interArea / unionArea if unionArea > 0 else 0

def run_audit():
    print("🎯 Starting Final Stage-by-Stage Audit...")
    with open(GT_PATH, 'r') as f: gt_data = json.load(f)
    page_map = {13: "14", 14: "15"}; target_indices = [13, 14]
    model = YOLO(YOLO_MODEL_PATH)
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    doc = fitz.open(str(PDF_PATH))
    results_by_stage = { "Stage 2": defaultdict(list), "Stage 3": defaultdict(list), "Stage 4": defaultdict(list), "Stage 4.5": defaultdict(list) }
    
    for p_idx in target_indices:
        page = doc[p_idx]; pix = page.get_pixmap(dpi=RENDER_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        ph, pw = img.shape[0], img.shape[1]
        stride = int(TILE_SIZE * (1 - OVERLAP))
        raw_boxes = []
        for y in range(0, ph, stride):
            for x in range(0, pw, stride):
                y2, x2 = min(y + TILE_SIZE, ph), min(x + TILE_SIZE, pw)
                tile = img[y:y2, x:x2]
                if np.mean(tile) > 245: continue
                if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                    tile = cv2.copyMakeBorder(tile, 0, TILE_SIZE-tile.shape[0], 0, TILE_SIZE-tile.shape[1], cv2.BORDER_CONSTANT, value=[255,255,255])
                res = model.predict(tile, conf=CONF_THRESH, imgsz=TILE_SIZE, verbose=False)[0]
                for box in res.boxes.xyxy.cpu().numpy():
                    raw_boxes.append([(box[0]+x)/pw, (box[1]+y)/ph, (box[2]+x)/pw, (box[3]+y)/ph])
        results_by_stage["Stage 2"][p_idx] = raw_boxes
        
        refined_boxes = []
        for rb in raw_boxes:
            merged = False
            for r in refined_boxes:
                if calculate_iou(rb, r) > 0.15:
                    r[0], r[1] = min(rb[0], r[0]), min(rb[1], r[1])
                    r[2], r[3] = max(rb[2], r[2]), max(rb[3], r[3])
                    merged = True; break
            if not merged: refined_boxes.append(rb[:])
        results_by_stage["Stage 3"][p_idx] = refined_boxes
            
        ocr_blocks = []
        for j, box_norm in enumerate(refined_boxes):
            bx1, by1, bx2, by2 = int(box_norm[0]*pw), int(box_norm[1]*ph), int(box_norm[2]*pw), int(box_norm[3]*ph)
            crop = img[by1:by2, bx1:bx2]
            if crop.size == 0: continue
            res_ocr = reader.readtext(crop, paragraph=True)
            text = " ".join([it[1] for it in res_ocr])
            ocr_blocks.append({"page_index0": p_idx, "bbox_xyxy_norm": box_norm, "ocr_text_raw": text, "pdf_path": str(PDF_PATH)})
        results_by_stage["Stage 4"][p_idx] = ocr_blocks
        
        stitched = stitch_blocks(ocr_blocks)
        cured = filter_by_keywords(stitched)
        final = deduplicate_vectors(cured)
        results_by_stage["Stage 4.5"][p_idx] = final

    print("\n📊 Computing Final Metrics...")
    table = []
    total_gt = sum(len(gt_data[v]) for v in page_map.values())
    
    for stage in ["Stage 2", "Stage 3", "Stage 4", "Stage 4.5"]:
        tp, fp = 0, 0
        for p_idx in target_indices:
            gts = gt_data[page_map[p_idx]]
            preds = results_by_stage[stage][p_idx]
            if stage in ["Stage 4", "Stage 4.5"]: preds = [b['bbox_xyxy_norm'] for b in preds]
            
            matched_gt = set()
            for p_box in preds:
                best_iou, best_gt_idx = 0, -1
                for idx, g_box in enumerate(gts):
                    if idx in matched_gt: continue
                    iou = calculate_iou(p_box, g_box)
                    if iou > best_iou: best_iou = iou; best_gt_idx = idx
                if best_iou >= IOU_THRESHOLD: tp += 1; matched_gt.add(best_gt_idx)
                else: fp += 1
        fn = total_gt - tp
        recall = tp / total_gt if total_gt > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        table.append({"Stage": stage, "TP": tp, "FP": fp, "FN": fn, "Recall": recall, "Precision": precision})

    print("\n| Stage | TP | FP | FN | Recall | Precision |")
    print("|-------|----|----|----|--------|-----------|")
    for row in table:
        print(f"| {row['Stage']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['Recall']:.3f} | {row['Precision']:.3f} |")
    
    print(f"\nGT Total: {total_gt}")
    s2_fp = table[0]['FP']
    s45_fp = table[3]['FP']
    print(f"FP Reduction %: {(s2_fp - s45_fp) / s2_fp * 100:.1f}%")
    print(f"Recall Delta (S2 -> S4.5): {table[3]['Recall'] - table[0]['Recall']:.3f}")

if __name__ == "__main__":
    run_audit()
