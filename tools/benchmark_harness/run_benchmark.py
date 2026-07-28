#!/usr/bin/env python3
import os, sys, json, time, cv2, fitz
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
import uuid

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# Configuration
YOLO_MODEL_PATH = str(PROJ_ROOT / "artifacts/detector_pivot_yolo_v2_tiles/best.pt")
RENDER_DPI = 300
CONF_THRESH = 0.0013 
TILE_SIZE = 1024
OVERLAP = 0.52

NEGATIVE_KEYWORDS = ["obituary", "missing person", "loss of documents", "change of name"]

def get_now():
    return time.time()

def log_event(events_file, event_type, stage):
    with open(events_file, "a") as f:
        f.write(json.dumps({"ts": get_now(), "event": event_type, "stage": stage}) + "\n")
    print(f"[{get_now()}] {event_type} - {stage}")
    sys.stdout.flush()

def calculate_iou(box1, box2):
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea
    return interArea / unionArea if unionArea > 0 else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = run_dir / "stage_outputs"
    stage_dir.mkdir(exist_ok=True)
    events_file = run_dir / "events.jsonl"

    pdf_path = Path(args.pdf)
    doc = fitz.open(str(pdf_path))

    run_info = {
        "run_id": str(uuid.uuid4())[:8],
        "pdf": args.pdf,
        "timestamp": get_now(),
        "page_count": doc.page_count
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f)

    # Clean existing
    if events_file.exists(): events_file.unlink()

    # Lazy loads
    import torch
    from ultralytics import YOLO
    import easyocr
    from sentence_transformers import SentenceTransformer, util

    # Preload models
    print("Loading models...")
    model = YOLO(YOLO_MODEL_PATH)
    if torch.cuda.is_available(): model.to('cuda')
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # --- STAGE 1: Rendering ---
    log_event(events_file, "stage_start", "stage1")
    images = []
    for p_idx in range(doc.page_count):
        pix = doc[p_idx].get_pixmap(dpi=RENDER_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        images.append(img)
    log_event(events_file, "stage_end", "stage1")

    # --- STAGE 2: Detection ---
    log_event(events_file, "stage_start", "stage2")
    stage2_preds = []
    stage2_boxes_by_page = defaultdict(list)
    tile_count = 0
    raw_detections = 0
    
    for p_idx in range(doc.page_count):
        img = images[p_idx]
        ph, pw = img.shape[0], img.shape[1]
        stride = int(TILE_SIZE * (1 - OVERLAP))
        
        tiles = []
        metas = []
        for y in range(0, ph, stride):
            for x in range(0, pw, stride):
                y2, x2 = min(y + TILE_SIZE, ph), min(x + TILE_SIZE, pw)
                tile = img[y:y2, x:x2]
                if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                    tile = cv2.copyMakeBorder(tile, 0, TILE_SIZE-tile.shape[0], 0, TILE_SIZE-tile.shape[1], cv2.BORDER_CONSTANT, value=[255,255,255])
                tiles.append(tile)
                metas.append((x, y))
                tile_count += 1
                
        if tiles:
            for i in range(0, len(tiles), 4):
                batch_tiles = tiles[i:i+4]
                res = model.predict(batch_tiles, conf=CONF_THRESH, imgsz=TILE_SIZE, verbose=False)
                for j, r in enumerate(res):
                    boxes = r.boxes.xyxy.cpu().numpy()
                    raw_detections += len(boxes)
                    tx, ty = metas[i+j]
                    for box in boxes:
                        norm_box = [float(box[0]+tx)/pw, float(box[1]+ty)/ph, float(box[2]+tx)/pw, float(box[3]+ty)/ph]
                        stage2_boxes_by_page[p_idx].append(norm_box)

    # Global NMS per page for Stage 2
    post_nms_detections = 0
    stage2_refined_by_page = defaultdict(list)
    for p_idx in range(doc.page_count):
        raw_dets = stage2_boxes_by_page[p_idx]
        refined = []
        for det in raw_dets:
            merged = False
            for r in refined:
                if calculate_iou(det, r) > 0.5: # Standard NMS threshold
                    r[0], r[1] = min(det[0], r[0]), min(det[1], r[1])
                    r[2], r[3] = max(det[2], r[2]), max(det[3], r[3])
                    merged = True; break
            if not merged: refined.append(det)
        
        for r in refined:
            stage2_preds.append({"page_number": p_idx + 1, "bbox": r})
            stage2_refined_by_page[p_idx].append(r)
            post_nms_detections += 1
    
    with open(stage_dir / "stage2.json", "w") as f:
        json.dump(stage2_preds, f)
    
    log_event(events_file, "metadata", {
        "stage": "stage2", 
        "tile_count": tile_count, 
        "raw_detections": raw_detections,
        "post_nms_detections": post_nms_detections
    })
    log_event(events_file, "stage_end", "stage2")

    # --- STAGE 3: OCR ---
    log_event(events_file, "stage_start", "stage3")
    stage3_preds = []
    stage3_boxes_by_page = defaultdict(list)
    boxes_into_ocr = 0
    
    for p_idx in range(doc.page_count):
        raw_dets = stage2_refined_by_page[p_idx]
        
        # Geometry Merge for OCR blocks (tighter/specific)
        refined = []
        for det in raw_dets:
            merged = False
            for r in refined:
                if calculate_iou(det, r) > 0.15:
                    r[0], r[1] = min(det[0], r[0]), min(det[1], r[1])
                    r[2], r[3] = max(det[2], r[2]), max(det[3], r[3])
                    merged = True; break
            if not merged: refined.append(det)
            
        img = images[p_idx]
        ph, pw = img.shape[0], img.shape[1]
        boxes_into_ocr += len(refined)
        
        for r in refined:
            bx1, by1, bx2, by2 = int(r[0]*pw), int(r[1]*ph), int(r[2]*pw), int(r[3]*ph)
            crop = img[by1:by2, bx1:bx2]
            text = ""
            if crop.size > 0:
                res_ocr = reader.readtext(crop, paragraph=True)
                text = " ".join([it[1] for it in res_ocr])
            
            if text.strip():
                stage3_preds.append({"page_number": p_idx + 1, "bbox": r, "text": text})
                stage3_boxes_by_page[p_idx].append({"bbox": r, "text": text})
                
    with open(stage_dir / "stage3.json", "w") as f:
        json.dump(stage3_preds, f)
    log_event(events_file, "metadata", {"stage": "stage3", "boxes_into_ocr": boxes_into_ocr})
    log_event(events_file, "stage_end", "stage3")

    # --- STAGE 4: Filtering + Vector semantics + stitching ---
    log_event(events_file, "stage_start", "stage4")
    stage4_preds = []
    
    for p_idx in range(doc.page_count):
        items = stage3_boxes_by_page[p_idx]
        
        # Filter
        cured = []
        for it in items:
            txt = it["text"].lower()
            if any(kw in txt for kw in NEGATIVE_KEYWORDS): continue
            if len(txt.split()) < 3: continue
            cured.append(it)
            
        # Vector Semantics & Deduplication
        if len(cured) > 1:
            texts = [x["text"] for x in cured]
            embs = embedder.encode(texts, convert_to_tensor=True)
            sims = util.cos_sim(embs, embs)
            to_skip = set()
            unique_cured = []
            for i in range(len(cured)):
                if i in to_skip: continue
                unique_cured.append(cured[i])
                for j in range(i+1, len(cured)):
                    if sims[i][j] > 0.98: to_skip.add(j)
        else:
            unique_cured = cured

        # Stitching
        stitched = []
        unique_cured.sort(key=lambda x: x["bbox"][1])
        for it in unique_cured:
            found = False
            for s in stitched:
                if it["bbox"][1] - s["bbox"][3] < 0.01 and \
                   min(it["bbox"][2], s["bbox"][2]) - max(it["bbox"][0], s["bbox"][0]) > 0:
                    s["text"] += " " + it["text"]
                    s["bbox"] = [
                        min(s["bbox"][0], it["bbox"][0]),
                        min(s["bbox"][1], it["bbox"][1]),
                        max(s["bbox"][2], it["bbox"][2]),
                        max(s["bbox"][3], it["bbox"][3])
                    ]
                    found = True; break
            if not found: stitched.append(it.copy())
            
        for f in stitched:
            stage4_preds.append({"page_number": p_idx + 1, "bbox": f["bbox"], "text": f["text"]})
            
    with open(stage_dir / "stage4.json", "w") as f:
        json.dump(stage4_preds, f)
    log_event(events_file, "stage_end", "stage4")

if __name__ == "__main__":
    main()
