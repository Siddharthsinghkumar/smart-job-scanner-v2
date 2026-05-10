#!/usr/bin/env python3
"""
Stage 02: Parallel Block Detection.
Overlaps detection across multiple pages using multiprocessing.
"""

import os
import sys
import json
import logging
import time
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any

# Force Proj Root
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

from src.utils.pipeline_config import get_path, load_config
from src.vision.block_detector import detect_connected_blocks

# Configuration
PARALLEL_WORKERS = 6
DEFAULT_PAGE_MANIFEST = Path("run_state/page_manifest.jsonl")
DEFAULT_CROP_MANIFEST = Path("run_state/crop_manifest.jsonl")

def process_page(page_item):
    """Worker function for one page."""
    img_path = page_item['image_path']
    base_name = Path(img_path).stem
    
    # 1. Detect
    blocks, _ = detect_connected_blocks(img_path, debug=False)
    
    # 2. Build metadata
    crops = []
    # Use real image size for normalization
    import cv2
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    for i, (bx, by, bw, bh) in enumerate(blocks):
        page_num_str = f"p{page_item['page_number1']}"
        crop_filename = f"{page_item['newspaper_name']}_{page_num_str}_crop{i:04d}.png"
        crop_path = Path("data/crops") / crop_filename
        
        # Save Crop
        crop_img = img[by:by+bh, bx:bx+bw]
        cv2.imwrite(str(crop_path), crop_img)
        
        crops.append({
            "crop_id": f"{page_item['doc_id']}_p{page_item['page_number1']}_c{i}",
            "page_id": page_item['page_id'],
            "doc_id": page_item['doc_id'],
            "pdf_path": page_item['pdf_path'],
            "page_image_path": img_path,
            "crop_image_path": f"data/crops/{crop_filename}",
            "newspaper_name": page_item['newspaper_name'],
            "page_index0": page_item['page_index0'],
            "page_number1": page_item['page_number1'],
            "bbox_xyxy_norm": [bx/w, by/h, (bx+bw)/w, (by+bh)/h],
            "status": "ok"
        })
    return crops

def run_stage2_parallel():
    start = time.time()
    if not DEFAULT_PAGE_MANIFEST.exists(): return
    
    pages = []
    with open(DEFAULT_PAGE_MANIFEST) as f:
        for line in f: pages.append(json.loads(line))
        
    print(f"🚀 Launching Parallel Detector on {len(pages)} pages...")
    
    with multiprocessing.Pool(PARALLEL_WORKERS) as pool:
        all_crop_lists = pool.map(process_page, pages)
        
    all_crops = [c for sublist in all_crop_lists for c in sublist]
    
    with open(DEFAULT_CROP_MANIFEST, "w") as f:
        for c in all_crops:
            f.write(json.dumps(c) + "\n")
            
    elapsed = time.time() - start
    print(f"✅ Parallel Detection Complete: {len(all_crops)} crops in {elapsed:.2f}s")

if __name__ == "__main__":
    run_stage2_parallel()
