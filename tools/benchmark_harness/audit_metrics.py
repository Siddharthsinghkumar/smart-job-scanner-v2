#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def calculate_iou(box1, box2):
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea
    return interArea / unionArea if unionArea > 0 else 0

def evaluate_stage(preds, gt_data):
    tp, fp = 0, 0
    matched_gt = set()
    gt_total = 0
    
    preds_by_page = {}
    for p in preds:
        p_num = str(p["page_number"])
        if p_num not in preds_by_page:
            preds_by_page[p_num] = []
        preds_by_page[p_num].append(p["bbox"])
        
    missed_gt = []
    
    for page_num, gts in gt_data.items():
        gt_total += len(gts)
        page_preds = preds_by_page.get(page_num, [])
        page_matched = set()
        
        for p_box in page_preds:
            best_iou, best_idx = 0, -1
            for i, g_box in enumerate(gts):
                if i in page_matched: continue
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= 0.4:
                tp += 1
                page_matched.add(best_idx)
            else:
                fp += 1
                
        for i, g_box in enumerate(gts):
            if i not in page_matched:
                missed_gt.append({"page": page_num, "box": g_box})
                
    fn = gt_total - tp
    recall = tp / gt_total if gt_total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return {
        "tp": tp, "fp": fp, "fn": fn, "gt_total": gt_total,
        "recall": recall, "precision": precision,
        "missed_gt": missed_gt
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gt-file", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(args.gt_file, "r") as f:
        gt_data = json.load(f)

    for stage in ["stage2", "stage3", "stage4"]:
        pred_file = run_dir / f"stage_outputs/{stage}.json"
        if not pred_file.exists(): continue
        
        with open(pred_file, "r") as f:
            preds = json.load(f)
            
        metrics = evaluate_stage(preds, gt_data)
        
        with open(run_dir / f"metrics_{stage}.json", "w") as f:
            json.dump(metrics, f)

if __name__ == "__main__":
    main()
