#!/usr/bin/env python3
"""
Stage 02 v12.0: Hybrid-Router Engine.
Policy: 
- Digital/Sparse Pages -> v2 (OpenCV)
- Hard/Scan Pages -> v11.1 (Sonic-Stream)
"""

import os
import sys
import json
import time
import torch.multiprocessing as tmp
import numpy as np
import cv2
import fitz
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# Imports from existing engines
from src.pipeline.stage02_block_detection_v6 import run_v11_1_core # I will wrap v11.1 logic
from src.vision.block_detector import detect_connected_blocks # v2 Logic

def classify_page(page):
    """Simple router: Digital vs Scan."""
    text = page.get_text()
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count > 400:
        return "EASY"
    return "HARD"

def run_hybrid_stage2(pdf_path):
    print(f"🚀 v12.0 Hybrid Router: {pdf_path.name}")
    start_wall = time.time()
    
    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)
    
    # 1. Page Classification
    routing = []
    for i in range(num_pages):
        res = classify_page(doc[i])
        routing.append(res)
    
    easy_indices = [i for i, r in enumerate(routing) if r == "EASY"]
    hard_indices = [i for i, r in enumerate(routing) if r == "HARD"]
    
    print(f"  Routing: {len(easy_indices)} EASY, {len(hard_indices)} HARD pages.")
    
    final_crops = []
    
    # --- PHASE A: Fast Path (v2) ---
    if easy_indices:
        print(f"  [v2] Processing {len(easy_indices)} easy pages...")
        # Create temp images for OpenCV
        for p_idx in easy_indices:
            page = doc[p_idx]
            pix = page.get_pixmap(dpi=150)
            tmp_img = f"temp_p{p_idx}.png"
            pix.save(tmp_img)
            
            blocks, _ = detect_connected_blocks(tmp_img, debug=False)
            h, w = pix.height, pix.width
            for j, b in enumerate(blocks):
                final_crops.append({
                    "crop_id": f"p{p_idx+1}_c{j}",
                    "page_index0": p_idx,
                    "bbox_xyxy_norm": [b[0]/w, b[1]/h, (b[0]+b[2])/w, (b[1]+b[3])/h],
                    "status": "ok",
                    "source": "v2"
                })
            os.remove(tmp_img)

    # --- PHASE B: Rescue Path (v11.1) ---
    if hard_indices:
        print(f"  [v11.1] Rescuing {len(hard_indices)} hard pages...")
        # We reuse the logic from stage02_block_detection_v6
        # To avoid rewrite, I will call the core logic if possible 
        # or implement a mini-version here
        from src.pipeline.stage02_block_detection_v6 import run_v9_5_internal
        hard_crops = run_v9_5_internal(pdf_path, hard_indices)
        for c in hard_crops:
            c["source"] = "v11.1"
            final_crops.append(c)

    # Final Save
    CROP_MANIFEST = Path("run_state/crop_manifest.jsonl")
    with open(CROP_MANIFEST, "w") as f:
        for c in final_crops:
            c.update({"pdf_path": str(pdf_path), "newspaper_name": pdf_path.stem})
            f.write(json.dumps(c) + "\n")
            
    dur = time.time() - start_wall
    print(f"✅ Hybrid Complete: {len(final_crops)} ads in {dur:.2f}s")
    doc.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    args = parser.parse_args()
    run_hybrid_stage2(Path(args.pdf))
