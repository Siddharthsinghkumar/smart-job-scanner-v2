#!/usr/bin/env python3
"""
Stage 05: Vector Surgeon (v5.1 - Hybrid Recovery).
Policy: Vector Similarity + Heuristic Safety-Net.
"""
import os, sys, json, time, torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# Force Proj Root
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# Configuration
INPUT_PATH = Path("run_state/stage4_final_ads.jsonl")
OUTPUT_PATH = Path("run_state/stage5_vector_ads.jsonl")
MODEL_NAME = 'all-MiniLM-L6-v2'
SIMILARITY_THRESHOLD = 0.25 # Lowered to rescue fuzzy matches
MERGE_THRESHOLD = 0.95

JOB_ANCHORS = [
    "Hiring teacher and faculty", "Requirement for engineer", "Walk-in interview", 
    "Vacancy for manager", "Apply with resume", "Wanted staff and principal"
]

def run_stage5_surgeon():
    print("🚀 Stage 05 Hybrid Surgeon (v5.1) started...")
    start_wall = time.time()
    
    if not INPUT_PATH.exists(): return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(MODEL_NAME, device=device)
    anchor_embeddings = model.encode(JOB_ANCHORS, convert_to_tensor=True)
    
    candidates = []
    with open(INPUT_PATH, "r") as f:
        for line in f: candidates.append(json.loads(line))
        
    if not candidates: return

    final_ads = []
    
    # Batch encoding for speed
    texts = [c.get("ocr_text_raw", "") for c in candidates]
    candidate_embeddings = model.encode(texts, convert_to_tensor=True)
    
    for i, emb in enumerate(candidate_embeddings):
        cand = candidates[i]
        text = texts[i].lower()
        
        # A. Semantic Score
        cos_scores = util.cos_sim(emb, anchor_embeddings)[0]
        max_score = float(torch.max(cos_scores))
        
        # B. HYBRID SAFETY NET
        # 1. High-Signal Heuristics (Keywords like Teacher/PGT/REQ)
        has_strong_marker = any(k in text for k in ["teacher", "tgt", "pgt", "req", "staff", "faculty"])
        # 2. Digit Signal (Phone numbers)
        has_phone_signal = sum(c.isdigit() for c in text) >= 8
        
        # C. DECISION
        if max_score > SIMILARITY_THRESHOLD or has_strong_marker or has_phone_signal:
            cand["vector_score"] = max_score
            final_ads.append(cand)

    with open(OUTPUT_PATH, "w") as f:
        for ad in final_ads:
            f.write(json.dumps(ad) + "\n")
            
    dur = time.time() - start_wall
    print(f"✅ Stage 5 Complete: {len(final_ads)} ads in {dur:.2f}s")

if __name__ == "__main__":
    run_stage5_surgeon()
