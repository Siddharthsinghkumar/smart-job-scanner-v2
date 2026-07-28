#!/usr/bin/env python3
import os, sys, json, time, cv2, fitz
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

def calculate_iou(box1, box2):
    xA, yA, xB, yB = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea
    return interArea / unionArea if unionArea > 0 else 0

def calculate_coverage(pred_box, gt_box):
    """How much of the GT box is covered by the Pred box."""
    xA, yA, xB, yB = max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1]), min(pred_box[2], gt_box[2]), min(pred_box[3], gt_box[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    gtArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    return interArea / gtArea if gtArea > 0 else 0

def run_final_audit():
    print("🎯 FINAL AUDIT: Coverage Analysis...")
    gt = json.load(open("uht_gt_boxes.json"))
    page_map = {13: "14", 14: "15"}
    
    with open("run_state/ocr_manifest.jsonl") as f:
        ocr = [json.loads(line) for line in f if "UHT Delhi 07-04" in line]
        
    with open("run_state/refined_manifest.jsonl") as f:
        refined = [json.loads(line) for line in f if "UHT Delhi 07-04" in line]

    print("\n| Stage | GT Matched | Recall |")
    print("|-------|------------|--------|")
    
    for name, data in [("Stage 4 (OCR)", ocr), ("Stage 4.5 (Refined)", refined)]:
        total_gt = 0
        matched_gt = set()
        
        for p_idx in [13, 14]:
            gts = gt[page_map[p_idx]]
            total_gt += len(gts)
            preds = [item for item in data if item.get("p_idx", item.get("page_index0")) == p_idx]
            
            for i, g_box in enumerate(gts):
                for p_item in preds:
                    p_box = p_item["bbox_xyxy_norm"]
                    # If prediction COVERS > 80% of GT, we count it as a find (even if merged)
                    if calculate_coverage(p_box, g_box) > 0.8:
                        matched_gt.add((p_idx, i))
                        break
        
        recall = len(matched_gt) / total_gt
        print(f"| {name} | {len(matched_gt)}/{total_gt} | {recall:.3f} |")

if __name__ == "__main__":
    run_final_audit()
