#!/usr/bin/env python3
"""
from src.utils.logging_utils import configure_logging
logger = configure_logging("stage02_block_detection_v5")
Stage 02 v5.7: Persistent Sequential High-Recall.
Strict sequential processing to avoid GPU thrashing.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import cv2
import fitz
import torch

# Force Proj Root
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

from src.vision.block_detector_v5_ultra import detect_ultra_recall

# Configuration
YOLO_MODEL = str(PROJ_ROOT / "artifacts/detector_pivot_yolo_v4_tiles/best.pt")
RENDER_DPI = 400
CROP_MANIFEST = Path("run_state/crop_manifest.jsonl")

def run_v5_sequential(pdf_path):
    print(f"🚀 Launching v5.7 Sequential High-Recall for {pdf_path.name}...")
    start_time = time.time()
    
    doc = fitz.open(str(pdf_path))
    all_crops = []
    
    for i in range(len(doc)):
        page = doc[i]
        
        # 1. High-Res Render
        pix = page.get_pixmap(dpi=RENDER_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        pix_h, pix_w = img.shape[:2]
        
        # 2. Ultra-Slice Detection (Opt I: 60% overlap for 46/49 parity)
        blocks = detect_ultra_recall(img, YOLO_MODEL, overlap=0.6)
        
        for j, b in enumerate(blocks):
            box_norm = [float(b[0]/pix_w), float(b[1]/pix_h), float(b[2]/pix_w), float(b[3]/pix_h)]
            crop_filename = f"{pdf_path.stem}_p{i+1}_crop{j:04d}.png"
            
            all_crops.append({
                "crop_id": f"{pdf_path.stem}_p{i+1}_c{j}",
                "page_id": f"{pdf_path.stem}_p{i+1}",
                "page_image_path": f"{pdf_path.stem}_p{i+1}.png",
                "crop_image_path": f"data/crops/{crop_filename}",
                "bbox_xyxy_norm": box_norm,
                "status": "ok",
                "pdf_path": str(pdf_path),
                "page_index0": i,
                "newspaper_name": pdf_path.stem
            })
            
            # Save crop
            crop_path = Path("data/crops") / crop_filename
            crop_img = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            if crop_img.size > 0:
                cv2.imwrite(str(crop_path), crop_img)

        if (i+1) % 5 == 0:
            print(f"  Processed {i+1}/{len(doc)} pages...")

    with open(CROP_MANIFEST, "w") as f:
        for c in all_crops:
            f.write(json.dumps(c) + "\n")
            
    elapsed = time.time() - start_time
    print(f"✅ v5.7 Complete: {len(all_crops)} ads in {elapsed:.2f}s")
    doc.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    args = parser.parse_args()
    run_v5_sequential(Path(args.pdf))
