#!/usr/bin/env python3
"""
Stage 04.5: Semantic-Geospatial Refiner
Tuned for high recall preservation.
"""

import json
import time
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
# from sentence_transformers import SentenceTransformer, util (MOVED)

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# Configuration
IN_MANIFEST = PROJ_ROOT / "run_state" / "ocr_manifest.jsonl"
OUT_MANIFEST = PROJ_ROOT / "run_state" / "refined_manifest.jsonl"

# 1. Positional Stitching Config (Conservative)
MAX_VERT_GAP = 0.01  # 1% of page height
MAX_HORIZ_GAP = 0.005 # 0.5% of page width

# 2. Negative Keywords
NEGATIVE_KEYWORDS = [
    "obituary", "missing person", "loss of documents", "change of name", "property sale"
]

def load_manifest(path):
    data = []
    if not path.exists(): return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except: continue
    return data

def stitch_blocks(blocks):
    if not blocks: return []
    pages = defaultdict(list)
    for b in blocks:
        p_idx = b.get('page_index0') if 'page_index0' in b else b.get('p_idx', 0)
        pages[(b.get('pdf_path', 'unknown'), p_idx)].append(b)
    
    stitched_all = []
    for (pdf, p_idx), p_blocks in pages.items():
        p_blocks.sort(key=lambda x: (x['bbox_xyxy_norm'][1], x['bbox_xyxy_norm'][0]))
        merged = []
        for b in p_blocks:
            found = False
            for m in merged:
                m_box = m['bbox_xyxy_norm']; b_box = b['bbox_xyxy_norm']
                v_gap = b_box[1] - m_box[3]
                h_overlap = min(m_box[2], b_box[2]) - max(m_box[0], b_box[0])
                if 0 <= v_gap < MAX_VERT_GAP and h_overlap > 0:
                    m['ocr_text_raw'] += "\n" + b['ocr_text_raw']
                    m['bbox_xyxy_norm'] = [min(m_box[0], b_box[0]), min(m_box[1], b_box[1]), max(m_box[2], b_box[2]), max(m_box[3], b_box[3])]
                    found = True; break
            if not found: merged.append(b.copy())
        stitched_all.extend(merged)
    return stitched_all

def filter_by_keywords(blocks):
    filtered = []
    for b in blocks:
        text = b.get('ocr_text_raw', '').lower()
        if any(kw in text for kw in NEGATIVE_KEYWORDS): continue
        if len(text.split()) < 3: continue
        filtered.append(b)
    return filtered

def deduplicate_vectors(blocks):
    from sentence_transformers import SentenceTransformer, util
    if len(blocks) < 2: return blocks
    texts = [b.get('ocr_text_raw', '') for b in blocks]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    to_skip = set(); unique_blocks = []
    for i in range(len(blocks)):
        if i in to_skip: continue
        unique_blocks.append(blocks[i])
        for j in range(i + 1, len(blocks)):
            if cosine_scores[i][j] > 0.98: to_skip.add(j)
    return unique_blocks

def main():
    print(f"🚀 Starting Stage 04.5 (Refiner)...")
    start_time = time.time()
    raw_blocks = load_manifest(IN_MANIFEST)
    print(f"📦 Loaded {len(raw_blocks)} blocks.")
    stitched = stitch_blocks(raw_blocks)
    print(f"🧵 Stitched to {len(stitched)} blocks.")
    cured = filter_by_keywords(stitched)
    print(f"🧹 Filtered to {len(cured)} blocks.")
    final = deduplicate_vectors(cured)
    print(f"💎 Deduplicated to {len(final)} unique blocks.")
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        for b in final: f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"🏁 Refiner complete in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
