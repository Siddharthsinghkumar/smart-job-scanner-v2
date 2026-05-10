#!/usr/bin/env python3
"""
Unified CPU Bench: High-Parallelism CPU Pipeline for Stages 1-4.
Uses Tesseract OCR for speed on CPU.
Goal: Sub-100s per PDF for 2 PDFs.
"""

import os, sys, json, time, cv2, fitz, warnings
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pytesseract
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# Config
YOLO_MODEL_PATH = str(PROJ_ROOT / "artifacts/detector_pivot_yolo_v4_tiles/best.pt")
RENDER_DPI = 150 # Lower DPI for CPU speed
TILE_SIZE = 320
OVERLAP = 0.25  # Lower overlap for CPU speed
CONF_THRESH = 0.05 # Higher threshold to reduce OCR load

def iou(b1, b2):
    xa, ya, xb, yb = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / u if u > 0 else 0

def process_page(args):
    pdf_path, p_idx, model_path = args
    # Load model locally to avoid sharing issues
    model = YOLO(model_path)
    
    doc = fitz.open(str(pdf_path))
    page = doc[p_idx]
    pix = page.get_pixmap(dpi=RENDER_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    ph, pw = img.shape[0], img.shape[1]
    stride = int(TILE_SIZE * (1 - OVERLAP))
    
    all_dets = []
    for y in range(0, ph, stride):
        for x in range(0, pw, stride):
            y2, x2 = min(y + TILE_SIZE, ph), min(x + TILE_SIZE, pw)
            tile = img[y:y2, x:x2]
            if np.mean(tile) > 245: continue
            if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                tile = cv2.copyMakeBorder(tile, 0, TILE_SIZE-tile.shape[0], 0, TILE_SIZE-tile.shape[1], cv2.BORDER_CONSTANT, value=[255,255,255])
            
            res = model.predict(tile, conf=CONF_THRESH, imgsz=TILE_SIZE, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for k in range(len(boxes)):
                box = boxes[k]
                all_dets.append({
                    "box": [float(box[0]+x), float(box[1]+y), float(box[2]+x), float(box[3]+y)],
                    "conf": float(confs[k])
                })
    
    # Refinement
    refined = []
    for d in sorted(all_dets, key=lambda x: x["conf"], reverse=True):
        merged = False
        for r in refined:
            if iou(d["box"], r["box"]) > 0.15:
                r["box"] = [min(d["box"][0], r["box"][0]), min(d["box"][1], r["box"][1]), max(d["box"][2], r["box"][2]), max(d["box"][3], r["box"][3])]
                merged = True; break
        if not merged: refined.append(d)
    
    # OCR (Tesseract)
    results = []
    for j, r in enumerate(refined):
        box = r["box"]
        crop = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        if crop.size == 0: continue
        
        text = pytesseract.image_to_string(crop, lang='eng', config='--psm 6')
        results.append({
            "p_idx": p_idx,
            "c_idx": j,
            "ocr_text_raw": text.strip(),
            "status": "ok",
            "bbox_xyxy_norm": [box[0]/pw, box[1]/ph, box[2]/pw, box[3]/ph],
            "newspaper_name": Path(pdf_path).stem,
            "pdf_path": str(pdf_path)
        })
    
    doc.close()
    return results

def run_bench(pdf_dir):
    print("🚀 Starting Unified CPU Parallel Engine...")
    start_all = time.time()
    
    pdfs = sorted(list(Path(pdf_dir).glob("*.pdf")))
    tasks = []
    for pdf_path in pdfs:
        doc = fitz.open(str(pdf_path))
        for p_idx in range(doc.page_count):
            tasks.append((pdf_path, p_idx, YOLO_MODEL_PATH))
        doc.close()
    
    print(f"📦 Total Pages to process: {len(tasks)}")
    
    final_out = PROJ_ROOT / "run_state/ocr_manifest.jsonl"
    with open(final_out, "w", encoding="utf-8") as f_out:
        with ProcessPoolExecutor(max_workers=16) as executor:
            for page_results in tqdm(executor.map(process_page, tasks), total=len(tasks)):
                for entry in page_results:
                    f_out.write(json.dumps(entry) + "\n")

    print(f"\n🏁 Total Corpus Time: {time.time()-start_all:.2f}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    run_bench(args.dir)
