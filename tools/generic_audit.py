import json, re, sys
from pathlib import Path
from collections import defaultdict

def iou(b1, b2):
    xa, ya, xb, yb = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / u if u > 0 else 0

def audit_manifest(manifest_path, pdf_stem):
    if not Path(manifest_path).exists():
        return "MISSING", 0, 0, 0
        
    with open("merged_fresh_labels.json") as f: labels = json.load(f)
    gt = defaultdict(list)
    for e in labels:
        img = e["data"]["image"]
        if pdf_stem in img:
            p_match = re.search(r"_p(\d+)", img)
            if not p_match: continue
            p_num = int(p_match.group(1))
            for a in e["annotations"]:
                for r in a["result"]:
                    if r["type"] == "rectanglelabels":
                        v = r["value"]
                        gt[p_num].append([v["x"]/100, v["y"]/100, (v["x"]+v["width"])/100, (v["y"]+v["height"])/100])
    
    total_gt = sum(len(v) for v in gt.values())
    
    dets = defaultdict(list)
    with open(manifest_path) as f:
        for line in f:
            item = json.loads(line)
            # Both manifests use page_index0
            dets[item["page_index0"]+1].append(item["bbox_xyxy_norm"])
            
    tps = 0
    matched = defaultdict(set)
    for p_num, p_dets in dets.items():
        if p_num not in gt: continue
        for db in p_dets:
            for i, gb in enumerate(gt[p_num]):
                if i not in matched[p_num] and iou(db, gb) > 0.05:
                    tps += 1
                    matched[p_num].add(i)
                    break
    return "OK", total_gt, tps, sum(len(v) for v in dets.values())

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: generic_audit.py <manifest_path> <pdf_stem>")
        sys.exit(1)
    status, gt_cnt, tp_cnt, det_cnt = audit_manifest(sys.argv[1], sys.argv[2])
    print(f"{status}|{gt_cnt}|{tp_cnt}|{det_cnt}")
