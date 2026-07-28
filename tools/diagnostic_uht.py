import json, re, sys
from collections import defaultdict
from pathlib import Path

def iou(b1, b2):
    xa, ya, xb, yb = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / u if u > 0 else 0

def run_diagnostic():
    labels_path = "merged_fresh_labels.json"
    if not Path(labels_path).exists(): return "No labels"
    
    with open(labels_path) as f: labels = json.load(f)
    gt = defaultdict(list)
    for e in labels:
        img = e["data"]["image"]
        if "UHT Delhi 07-04" in img:
            p_match = re.search(r"_p(\d+)", img)
            if not p_match: continue
            p_num = int(p_match.group(1))
            for a in e["annotations"]:
                for r in a["result"]:
                    if r["type"] == "rectanglelabels":
                        v = r["value"]
                        gt[p_num].append([v["x"]/100, v["y"]/100, (v["x"]+v["width"])/100, (v["y"]+v["height"])/100])

    manifest_path = "run_state/crop_manifest.jsonl"
    if not Path(manifest_path).exists(): return "No manifest"
    
    dets = defaultdict(list)
    with open(manifest_path) as f:
        for line in f:
            item = json.loads(line)
            dets[item["page_index0"]+1].append(item["bbox_xyxy_norm"])

    print("--- MISSING AD DIAGNOSTIC ---")
    total_misses = 0
    for p_num in sorted(gt.keys()):
        matched = set()
        for db in dets[p_num]:
            for i, gb in enumerate(gt[p_num]):
                if i not in matched and iou(db, gb) > 0.05:
                    matched.add(i)
                    break
        misses = [i for i in range(len(gt[p_num])) if i not in matched]
        if misses:
            print(f"Page {p_num}: Missed {len(misses)} ads. Indices: {misses}")
            total_misses += len(misses)
    print(f"Total Misses: {total_misses}")

if __name__ == "__main__":
    run_diagnostic()
