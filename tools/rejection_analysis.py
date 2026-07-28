#!/usr/bin/env python3
import os, sys, json, time, cv2, fitz
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# Matching logic
IOU_THRESHOLD = 0.4

def calculate_iou(box1, box2):
    xA, yA, xB, yB = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea
    return interArea / unionArea if unionArea > 0 else 0

def run_rejection_audit():
    print("🔍 RESTORING RECALL: Identifying Killed Jobs...")
    gt = json.load(open("uht_gt_boxes.json"))
    page_map = {13: "14", 14: "15"}
    
    with open("run_state/ocr_manifest.jsonl") as f:
        ocr = [json.loads(line) for line in f if "UHT Delhi 07-04" in line]
        
    with open("run_state/refined_manifest.jsonl") as f:
        refined = [json.loads(line) for line in f if "UHT Delhi 07-04" in line]

    for p_idx in [13, 14]:
        gts = gt[page_map[p_idx]]
        page_ocr = [item for item in ocr if item.get("p_idx", item.get("page_index0")) == p_idx]
        page_refined = [item for item in refined if item.get("p_idx", item.get("page_index0")) == p_idx]
        
        # Find which OCR items matched GT but are NOT in refined
        for item in page_ocr:
            matched_gt = False
            for g_box in gts:
                if calculate_iou(item["bbox_xyxy_norm"], g_box) >= IOU_THRESHOLD:
                    matched_gt = True; break
            
            if matched_gt:
                # Is it in refined?
                survived = False
                for r_item in page_refined:
                    if calculate_iou(item["bbox_xyxy_norm"], r_item["bbox_xyxy_norm"]) > 0.9:
                        survived = True; break
                
                if not survived:
                    # Diagnose why
                    # 1. Stitching
                    from src.pipeline.stage04_5_refiner import stitch_blocks, filter_by_keywords, deduplicate_vectors
                    stitched = stitch_blocks([item])
                    s_survived = any(calculate_iou(item['bbox_xyxy_norm'], b['bbox_xyxy_norm']) > 0.9 for b in stitched)
                    
                    # 2. Keywords
                    cured = filter_by_keywords(stitched)
                    c_survived = any(calculate_iou(item['bbox_xyxy_norm'], b['bbox_xyxy_norm']) > 0.9 for b in cured)
                    
                    # 3. Vectors
                    final = deduplicate_vectors(cured)
                    f_survived = any(calculate_iou(item['bbox_xyxy_norm'], b['bbox_xyxy_norm']) > 0.9 for b in final)
                    
                    reason = "UNKNOWN"
                    if not s_survived: reason = "STITCHING (Merge/Box Shift)"
                    elif not c_survived: reason = "KEYWORDS/MIN_LENGTH"
                    elif not f_survived: reason = "VECTOR_DEDUPE"
                    
                    print(f"❌ KILL DETECTED | Reason: {reason} | Page {p_idx+1} | Text: '{item['ocr_text_raw'][:50]}...' ")

if __name__ == "__main__":
    run_rejection_audit()
