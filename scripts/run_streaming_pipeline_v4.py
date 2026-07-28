#!/usr/bin/env python3
"""
Stable Vector Pipeline (v4.12).
Goal: Quality Parity with Stage 1/2/3 manifests.
"""

import sys
import os
import time
import json
import cv2
import fitz
import numpy as np
from pathlib import Path

# Force Proj Root
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from src.vision.block_detector import detect_connected_blocks

def run_stable_vector(pdf_path_obj):
    from easyocr import Reader
    print(f"🚀 Processing {pdf_path_obj.name}...")
    reader = Reader(['en'], gpu=True, detector="dbnet18", cudnn_benchmark=True)
    
    doc = fitz.open(str(pdf_path_obj))
    results = []
    
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=120)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Consistent filename for audit: "BS English-Delhi 07-04_p1.png"
        page_name = f"{pdf_path_obj.stem}_p{i+1}.png"
        
        tmp = f"/dev/shm/stable_{i}.png"
        cv2.imwrite(tmp, img)
        blocks, _ = detect_connected_blocks(tmp, debug=False)
        os.remove(tmp)
        
        for j, b in enumerate(blocks):
            x, y, w, h = b
            crop = img[y:y+h, x:x+w]
            
            # Vector Normalize
            ch, cw = crop.shape[:2]
            new_h = 64
            new_w = int(cw * (new_h / ch))
            crop_resized = cv2.resize(crop, (new_w, new_h))
            crop_grey = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2GRAY)
            
            # OCR
            ocr_res = reader.readtext(crop_grey, paragraph=True)
            text = " ".join([r[1] for r in ocr_res])
            
            results.append({
                "crop_id": f"{pdf_path_obj.stem}_p{i+1}_c{j}",
                "page_image_path": page_name,
                "bbox_xyxy_norm": [x/pix.w, y/pix.h, (x+w)/pix.w, (y+h)/pix.h],
                "ocr_text_raw": text,
                "is_step3_survivor": True,
                "status": "ok"
            })
    doc.close()
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    pdf_p = Path(args.pdf)
    pdfs = [pdf_p] if pdf_p.is_file() else sorted(list(pdf_p.glob("*.pdf")))
    
    all_res = []
    for p in pdfs:
        all_res.extend(run_stable_vector(p))
        
    with open(args.out, "w") as f:
        for r in all_res:
            f.write(json.dumps(r) + "\n")
    print(f"✅ Audit manifest written: {args.out}")
