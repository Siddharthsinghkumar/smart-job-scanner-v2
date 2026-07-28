#!/usr/bin/env python3
"""
Stable Single-Process Baseline (v17.4).
Used to guarantee baseline data when multiprocessing fails.
"""
import os, sys, json, time, torch, numpy as np, cv2, fitz, shutil, warnings
from pathlib import Path
from ultralytics import YOLO
from easyocr import Reader

warnings.filterwarnings("ignore")
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# Config
YOLO_MODEL_PATH = str(PROJ_ROOT / "artifacts/detector_pivot_yolo_v4_tiles/best.pt")
RENDER_DPI = 300
TILE_SIZE = 320
OVERLAP = 0.52
CONF_THRESH = 0.0005 

def iou(b1, b2):
    xa, ya, xb, yb = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / u if u > 0 else 0

def run_stable_baseline(pdf_path):
    print(f"🚀 Stable Baseline: {pdf_path.name}")
    start_wall = time.time()
    
    # Load Models
    model = YOLO(YOLO_MODEL_PATH).to('cuda')
    reader = Reader(['en'], gpu=True, detector="dbnet18", cudnn_benchmark=False)
    
    doc = fitz.open(str(pdf_path))
    final_results = []
    
    for p_idx in range(len(doc)):
        page = doc[p_idx]
        text = page.get_text()
        alpha_count = sum(1 for c in text if c.isalpha())
        
        # Bypass disabled for baseline audit
        # if alpha_count > 400: ...
        
        print(f"  Page {p_idx+1}: ANALYZING (alpha={alpha_count})")
        pix = page.get_pixmap(dpi=RENDER_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        ph, pw = img.shape[:2]
        
        # Detection
        stride = int(TILE_SIZE * (1 - OVERLAP))
        page_dets = []
        for y in range(0, ph, stride):
            for x in range(0, pw, stride):
                y2, x2 = min(y + TILE_SIZE, ph), min(x + TILE_SIZE, pw)
                tile = img[y:y2, x:x2]
                if np.mean(tile) > 245: continue
                if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                    tile = cv2.copyMakeBorder(tile, 0, TILE_SIZE-tile.shape[0], 0, TILE_SIZE-tile.shape[1], cv2.BORDER_CONSTANT, value=[255,255,255])
                
                res = model.predict(tile, conf=CONF_THRESH, imgsz=TILE_SIZE, verbose=False, device=0)[0]
                for b in res.boxes:
                    c = b.xyxy[0].cpu().numpy()
                    page_dets.append({"box": [float(c[0]+x), float(c[1]+y), float(c[2]+x), float(c[3]+y)], "conf": float(b.conf[0])})
        
        # Stitch
        refined = []
        for d in sorted(page_dets, key=lambda x: x["conf"], reverse=True):
            merged = False
            for r in refined:
                if iou(d["box"], r["box"]) > 0.15:
                    r["box"] = [min(d["box"][0], r["box"][0]), min(d["box"][1], r["box"][1]), max(d["box"][2], r["box"][2]), max(d["box"][3], r["box"][3])]; merged = True; break
            if not merged: refined.append(d)
            
        # OCR
        for j, r in enumerate(refined):
            box = r["box"]
            crop = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            if crop.size == 0: continue
            crop = cv2.resize(crop, (int(crop.shape[1] * (128/crop.shape[0])), 128))
            res = reader.readtext(crop, paragraph=True)
            ocr_text = " ".join([item[1] for item in res])
            final_results.append({
                "page_index0": p_idx,
                "crop_id": f"p{p_idx+1}_c{j}",
                "bbox_xyxy_norm": [box[0]/pw, box[1]/ph, box[2]/pw, box[3]/ph],
                "ocr_text_raw": ocr_text,
                "status": "ok"
            })
        print(f"  Page {p_idx+1}/{len(doc)} processed.")

    # Save manifest
    manifest_path = Path("run_state/ocr_manifest.jsonl")
    with open(manifest_path, "w") as f:
        for r in final_results:
            r.update({"pdf_path": str(pdf_path), "newspaper_name": pdf_path.stem})
            f.write(json.dumps(r) + "\n")
            
    print(f"✅ Finished in {time.time()-start_wall:.2f}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--pdf", required=True)
    args = parser.parse_args(); run_stable_baseline(Path(args.pdf))
