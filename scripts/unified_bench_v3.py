#!/usr/bin/env python3
"""
Unified Bench v3: Single-Process Pipeline for Stages 1-4.
MANDATE COMPLIANT: Local imports inside functions to prevent CUDA deadlocks.
"""

import os, sys, json, time, cv2, fitz, warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# Config
YOLO_MODEL_PATH = str(PROJ_ROOT / "artifacts/detector_pivot_yolo_v4_tiles/best.pt")
RENDER_DPI = 300
TILE_SIZE = 320
OVERLAP = 0.52
CONF_THRESH = 0.001

def iou(b1, b2):
    xa, ya, xb, yb = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / u if u > 0 else 0

def run_bench(pdf_dir):
    # 🆕 MANDATE COMPLIANT: Local Imports
    import torch
    from ultralytics import YOLO
    import easyocr

    print("🚀 Initializing Unified Mandate-Compliant Engine...")
    start_all = time.time()
    
    # Check GPU but handle failure gracefully
    device = 'cpu'
    try:
        if torch.cuda.is_available():
            print(f"🧠 GPU Detected: {torch.cuda.get_device_name(0)}")
            # Test a small tensor to see if driver is actually alive
            _ = torch.zeros(1).cuda()
            device = 0
            print("✅ GPU Communication Verified.")
        else:
            print("⚠️ GPU not available to Torch.")
    except Exception as e:
        print(f"❌ GPU Driver Error: {e}. Defaulting to CPU.")
        device = 'cpu'

    print(f"📡 Loading YOLO Model on {device}...")
    model = YOLO(YOLO_MODEL_PATH)
    if device == 0: model.to('cuda')
    
    print(f"📡 Loading EasyOCR Reader (GPU={device==0})...")
    reader = easyocr.Reader(['en'], gpu=(device == 0))
    
    pdfs = sorted(list(Path(pdf_dir).glob("*.pdf")))
    print(f"📦 Found {len(pdfs)} PDFs.")
    
    final_out = PROJ_ROOT / "run_state/ocr_manifest.jsonl"
    with open(final_out, "w") as f_out: pass 

    for pdf_path in pdfs:
        print(f"\n📄 Processing: {pdf_path.name}")
        pdf_start = time.time()
        doc = fitz.open(str(pdf_path))
        
        for p_idx in range(doc.page_count):
            page = doc[p_idx]
            pix = page.get_pixmap(dpi=RENDER_DPI)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            ph, pw = img.shape[0], img.shape[1]
            stride = int(TILE_SIZE * (1 - OVERLAP))
            tiles = []; metas = []
            
            for y in range(0, ph, stride):
                for x in range(0, pw, stride):
                    y2, x2 = min(y + TILE_SIZE, ph), min(x + TILE_SIZE, pw)
                    tile = img[y:y2, x:x2]
                    if np.mean(tile) > 245: continue
                    if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                        tile = cv2.copyMakeBorder(tile, 0, TILE_SIZE-tile.shape[0], 0, TILE_SIZE-tile.shape[1], cv2.BORDER_CONSTANT, value=[255,255,255])
                    tiles.append(tile)
                    metas.append((x, y))
            
            # 1. Detection
            all_dets = []
            if tiles:
                for i in range(0, len(tiles), 32):
                    batch_tiles = tiles[i:i+32]
                    batch_metas = metas[i:i+32]
                    results = model.predict(batch_tiles, conf=CONF_THRESH, imgsz=TILE_SIZE, verbose=False)
                    for j, res in enumerate(results):
                        boxes = res.boxes.xyxy.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()
                        tx, ty = batch_metas[j]
                        for k in range(len(boxes)):
                            box = boxes[k]
                            all_dets.append({
                                "box": [float(box[0]+tx), float(box[1]+ty), float(box[2]+tx), float(box[3]+ty)],
                                "conf": float(confs[k])
                            })
            
            # 2. Refinement
            refined = []
            for d in sorted(all_dets, key=lambda x: x["conf"], reverse=True):
                merged = False
                for r in refined:
                    if iou(d["box"], r["box"]) > 0.15:
                        r["box"] = [min(d["box"][0], r["box"][0]), min(d["box"][1], r["box"][1]), max(d["box"][2], r["box"][2]), max(d["box"][3], r["box"][3])]
                        merged = True; break
                if not merged: refined.append(d)
            
            # 3. OCR
            with open(final_out, "a", encoding="utf-8") as f_out:
                for j, r in enumerate(refined):
                    box = r["box"]
                    crop = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    if crop.size == 0: continue
                    res = reader.readtext(crop, paragraph=True)
                    text = " ".join([item[1] for item in res])
                    entry = {
                        "p_idx": p_idx, "c_idx": j, "ocr_text_raw": text, "status": "ok",
                        "bbox_xyxy_norm": [box[0]/pw, box[1]/ph, box[2]/pw, box[3]/ph],
                        "newspaper_name": pdf_path.stem, "pdf_path": str(pdf_path)
                    }
                    f_out.write(json.dumps(entry) + "\n")
            
            if (p_idx + 1) % 5 == 0:
                print(f"  Page {p_idx+1}/{doc.page_count} Done.")

        doc.close()
        print(f"✅ {pdf_path.name} Finished in {time.time()-pdf_start:.2f}s")

    print(f"\n🏁 Total Corpus Time: {time.time()-start_all:.2f}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    run_bench(args.dir)
