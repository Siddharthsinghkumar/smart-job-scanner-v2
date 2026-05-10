#!/usr/bin/env python3
"""
Ultra-Recall Block Detector (v5.0-ORIGINAL).
PURE LOOP LOGIC. NO OPTIMIZATIONS. FOR QUALITY BASELINE.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def iou(b1, b2):
    xa, ya, xb, yb = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / u if u > 0 else 0

def detect_ultra_recall(img, model_path, conf=0.001, tile_size=320, overlap=0.5):
    h, w = img.shape[:2]
    stride = int(tile_size * (1 - overlap))
    model = YOLO(model_path)
    
    all_raw = []
    # 1. Tiled Pass
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
            tile = img[y:y2, x:x2]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile = cv2.copyMakeBorder(tile, 0, tile_size-tile.shape[0], 0, tile_size-tile.shape[1], cv2.BORDER_CONSTANT)
            
            res = model.predict(tile, conf=conf, imgsz=tile_size, verbose=False, device=0)[0]
            for b in res.boxes:
                c = b.xyxy[0].cpu().numpy()
                all_raw.append({"box": [c[0]+x, c[1]+y, c[2]+x, c[3]+y], "conf": float(b.conf[0])})
                
    # 2. Agglomerative Stitching
    refined = []
    for d in sorted(all_raw, key=lambda x: x["conf"], reverse=True):
        merged = False
        for r in refined:
            if iou(d["box"], r["box"]) > 0.05:
                r["box"] = [min(d["box"][0], r["box"][0]), min(d["box"][1], r["box"][1]),
                           max(d["box"][2], r["box"][2]), max(d["box"][3], r["box"][3])]
                merged = True; break
        if not merged: refined.append(d)
        
    return [r["box"] for r in refined]
