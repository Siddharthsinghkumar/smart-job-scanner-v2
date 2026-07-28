
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import re

def calculate_iou(box1, box2):
    xa, ya, xb, yb = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    u = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return inter / u if u > 0 else 0

def load_gt():
    with open("merged_fresh_labels.json") as f: labels = json.load(f)
    gt = defaultdict(list)
    for e in labels:
        img = e["data"]["image"]
        if "UHT Delhi 07-04" in img:
            p_num = int(re.search(r'_p(\d+)', img).group(1))
            for a in e["annotations"]:
                for r in a["result"]:
                    if r["type"] == "rectanglelabels":
                        v = r["value"]
                        gt[p_num].append([v["x"]/100, v["y"]/100, (v["x"]+v["width"])/100, (v["y"]+v["height"])/100])
    return gt

def audit():
    gt = load_gt()
    total_gt = sum(len(v) for v in gt.values())
    
    manifest = Path("run_state/crop_manifest.jsonl")
    if not manifest.exists(): return "FAIL: No manifest"
    
    dets = defaultdict(list)
    with open(manifest) as f:
        for line in f:
            item = json.loads(line)
            dets[item["page_index0"]+1].append(item["bbox_xyxy_norm"])
            
    tps = 0
    matched = defaultdict(set)
    for p_num, p_dets in dets.items():
        if p_num not in gt: continue
        for db in p_dets:
            for i, gb in enumerate(gt[p_num]):
                if i not in matched[p_num] and calculate_iou(db, gb) > 0.05:
                    tps += 1
                    matched[p_num].add(i)
                    break
    
    precision = tps / sum(len(v) for v in dets.values()) if dets else 0
    recall = tps / total_gt if total_gt > 0 else 0
    
    print(f"--- UHT DELHI QUALITY GATE ---")
    print(f"GT Total   : {total_gt}")
    print(f"TP Found   : {tps}")
    print(f"FP Count   : {sum(len(v) for v in dets.values()) - tps}")
    print(f"Recall     : {recall:.2%}")
    print(f"Precision  : {precision:.2%}")
    return tps >= 46

if __name__ == "__main__":
    audit()
