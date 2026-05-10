
import json
import re
from pathlib import Path

def get_iou(boxA, boxB):
    xa = max(boxA[0], boxB[0])
    ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2])
    yb = min(boxA[3], boxB[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)

def audit():
    labels_file = Path("merged_fresh_labels.json")
    candidates_file = Path("run_state/step3_candidates.jsonl")
    
    if not labels_file.exists() or not candidates_file.exists():
        print("Missing audit files.")
        return

    labels = json.loads(labels_file.read_text())
    gt_by_page = {}
    for task in labels:
        page = task["data"]["image"]
        gt_boxes = []
        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                if res["type"] == "rectanglelabels":
                    v = res["value"]
                    x1, y1 = v["x"]/100, v["y"]/100
                    x2, y2 = (v["x"]+v["width"])/100, (v["y"]+v["height"])/100
                    gt_boxes.append((x1, y1, x2, y2))
        gt_by_page[page] = gt_boxes

    # Load candidates
    found_tps = set()
    total_candidates = 0
    with open(candidates_file) as f:
        for line in f:
            c = json.loads(line)
            total_candidates += 1
            page = c["page_id"]
            if page in gt_by_page:
                bbox = c["bbox_xyxy_norm"]
                for i, gt in enumerate(gt_by_page[page]):
                    if get_iou(bbox, gt) > 0.4:
                        found_tps.add(f"{page}_{i}")

    total_gt = sum(len(v) for v in gt_by_page.values())
    recall = len(found_tps) / total_gt if total_gt > 0 else 0
    
    print(f"--- RECALL AUDIT ---")
    print(f"Total Ground Truth Ads: {total_gt}")
    print(f"True Positives Found:   {len(found_tps)}")
    print(f"Recall:                 {recall:.2%}")
    print(f"Total Candidates:       {total_candidates}")
    print(f"Precision:              {len(found_tps)/total_candidates:.2%}")

if __name__ == "__main__":
    audit()
