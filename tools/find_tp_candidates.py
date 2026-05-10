import json, re, sys
from pathlib import Path
from collections import defaultdict

def iou(b1, b2):
    xa, ya, xb, yb = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / u if u > 0 else 0

with open("merged_fresh_labels.json") as f: labels = json.load(f)
gt = defaultdict(list)
for e in labels:
    if "UHT Delhi 07-04" in e["data"]["image"]:
        p_num = int(re.search(r"_p(\d+)", e["data"]["image"]).group(1))
        for a in e["annotations"]:
            for r in a["result"]:
                if r["type"] == "rectanglelabels":
                    v = r["value"]
                    gt[p_num].append([v["x"]/100, v["y"]/100, (v["x"]+v["width"])/100, (v["y"]+v["height"])/100])

with open("run_state/stage4_final_ads.jsonl") as f:
    dets = [json.loads(l) for l in f]

print("--- TP DISCOVERY ---")
found_ids = []
for db in dets:
    p_num = db["page_index0"] + 1
    if p_num not in gt: continue
    for i, gb in enumerate(gt[p_num]):
        if iou(db["bbox_xyxy_norm"], gb) > 0.05:
            print(f"TP FOUND: {db['crop_id']} | Text: {db.get('ocr_text_raw', '')[:50]}...")
            found_ids.append(db["crop_id"])
            break
