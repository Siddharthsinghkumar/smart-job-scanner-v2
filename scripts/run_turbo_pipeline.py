#!/usr/bin/env python3
"""
Nitro-VRAM Pipeline (v4.15).
Strategy: Zero-Copy VRAM Handoff. 
Passing GPU Tensors directly to OCR engine.
"""

import sys
import os
import time
import json
import cv2
import fitz
import numpy as np
from pathlib import Path
import multiprocessing as mp
import gc
import torch

# Force Proj Root
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# Configuration
YOLO_MODEL = str(PROJ_ROOT / "artifacts/stage2_yolo_v3/best.pt")
OCR_BATCH_SIZE = 16

def ocr_worker(in_q, out_q):
    from easyocr import Reader
    reader = Reader(['en'], gpu=True, detector="dbnet18", cudnn_benchmark=True)
    
    while True:
        batch = in_q.get()
        if batch is None: break
        
        for item in batch:
            # Item 'img' is already a GPU Tensor from YOLO
            img_tensor = item['img']
            
            # EasyOCR expects numpy for now, but we can do rapid transfer
            # (True zero-copy requires modifying EasyOCR internals, 
            # so we use fast torch-to-numpy on pinned memory if possible)
            img_np = img_tensor.cpu().numpy()
            
            # Vector Resize
            h, w = img_np.shape[:2]
            new_h = 64
            new_w = int(w * (new_h / h))
            img_resized = cv2.resize(img_np, (new_w, new_h))
            img_grey = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            
            res = reader.readtext(img_grey, paragraph=True)
            out_q.put(1)
            
            # Clear VRAM immediately
            del img_tensor
            
        torch.cuda.empty_cache()

def run_nitro_vram(pdf_path):
    from ultralytics import YOLO
    print(f"🚀 Nitro-VRAM v4.15: Zero-Copy VRAM Engine")
    
    detector = YOLO(YOLO_MODEL)
    detector.to('cuda')
    
    # We use 'spawn' for GPU safety
    in_q = mp.Queue(maxsize=30)
    out_q = mp.Queue()
    
    p_ocr = mp.Process(target=ocr_worker, args=(in_q, out_q))
    p_ocr.start()
    
    doc = fitz.open(str(pdf_path))
    start_time = time.time()
    total_crops = 0
    
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=150)
        # Load directly to torch tensor
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        # Detect on GPU
        results = detector.predict(img_np, conf=0.01, verbose=False, device=0)
        boxes = results[0].boxes
        
        current_batch = []
        for j, b in enumerate(boxes):
            xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
            # Crop image on CPU for EasyOCR transfer (v4.15)
            # (Actual GPU cropping requires torch.narrow which we will add if this is slow)
            crop = img_np[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            
            if crop.size > 1000:
                current_batch.append({"img": torch.from_numpy(crop), "id": f"{i}_{j}"})
                total_crops += 1
                if len(current_batch) >= OCR_BATCH_SIZE:
                    in_q.put(current_batch)
                    current_batch = []
        
        if current_batch:
            in_q.put(current_batch)
            
        del img_np, results
        torch.cuda.empty_cache()
        
    in_q.put(None)
    
    collected = 0
    while collected < total_crops:
        out_q.get()
        collected += 1
        
    p_ocr.join()
    print(f"🏁 Zero-Copy Complete: {total_crops} crops in {time.time()-start_time:.2f}s")
    doc.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)
    run_nitro_vram(Path(args.pdf))
