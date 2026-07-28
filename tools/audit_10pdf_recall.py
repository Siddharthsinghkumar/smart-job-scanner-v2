
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import re

def calculate_iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def normalize_page_name(fname):
    base = Path(fname).name.lower()
    base = re.sub(r'^[a-f0-9]{8}-', '', base)
    base = base.replace('.png', '').replace('.jpg', '')
    base = re.sub(r'[^a-z0-9]', '', base)
    return base

def load_gt(labels_path):
    with open(labels_path) as f:
        labels_raw = json.load(f)
    gt_map = defaultdict(list)
    for entry in labels_raw:
        norm_name = normalize_page_name(entry['data']['image'])
        for ann in entry['annotations']:
            ads = [res for res in ann['result'] if res['type'] == 'rectanglelabels']
            for res in ads:
                v = res['value']
                gt_map[norm_name].append([
                    v['x']/100.0, v['y']/100.0, 
                    (v['x']+v['width'])/100.0, (v['y']+v['height'])/100.0
                ])
    return gt_map

def audit_pdf_manifest(manifest_path, gt_map):
    detections = []
    with open(manifest_path) as f:
        for line in f:
            item = json.loads(line)
            norm_name = normalize_page_name(item.get('page_image_path', ''))
            detections.append((norm_name, item['bbox_xyxy_norm']))
    
    # Get total GT ads across all pages mentioned in manifest
    manifest_pages = set(d[0] for d in detections)
    total_gt = sum(len(gt_map[pg]) for pg in manifest_pages)
    
    if total_gt == 0: return None
    
    tps = 0
    matched_gt = set()
    for pimg, det_box in detections:
        best_iou = 0
        best_idx = -1
        if pimg not in gt_map: continue
        for idx, gt_box in enumerate(gt_map[pimg]):
            if (pimg, idx) in matched_gt: continue
            iou = calculate_iou(det_box, gt_box)
            if iou >= 0.3 and iou > best_iou: # Loose matching for tiny ads
                best_iou = iou
                best_idx = idx
        if best_idx != -1:
            tps += 1
            matched_gt.add((pimg, best_idx))
            
    return {"tp": tps, "gt": total_gt, "recall": tps/total_gt}

if __name__ == "__main__":
    gt = load_gt("merged_fresh_labels.json")
    manifests = sorted(list(Path("run_state").glob("crop_manifest_*.jsonl")))
    
    print("\n=== 10-PDF HIGH-RECALL QUALITY AUDIT ===")
    print(f"{'PDF Corpus':<25} | {'TP':<5} | {'GT':<5} | {'Recall'}")
    print("-" * 55)
    
    all_recalls = []
    for m in manifests:
        res = audit_pdf_manifest(m, gt)
        if res:
            name = m.stem.replace("crop_manifest_", "")
            print(f"{name[:25]:<25} | {res['tp']:<5} | {res['gt']:<5} | {res['recall']:.2%}")
            all_recalls.append(res['recall'])
            
    if all_recalls:
        print("-" * 55)
        print(f"AVERAGE CORPUS RECALL: {sum(all_recalls)/len(all_recalls):.2%}")
