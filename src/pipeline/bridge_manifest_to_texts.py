#!/usr/bin/env python3
"""
Bridge Script: Manifest to Page Texts
Reads run_state/translated_manifest.jsonl -> writes data/all_eng_text/
Required to connect Unified Stage 1-2-3 with Stage 06/07.
"""

import os, sys, json
from pathlib import Path
from collections import defaultdict

PROJ_ROOT = Path(__file__).resolve().parents[2]
IN_MANIFEST = PROJ_ROOT / "run_state" / "translated_manifest.jsonl"
OUT_DIR = PROJ_ROOT / "data" / "all_eng_text"

def main():
    print(f"🌉 Bridging manifest to directory structure...")
    if not IN_MANIFEST.exists():
        print(f"❌ Input manifest not found: {IN_MANIFEST}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Group by newspaper and page
    grouped = defaultdict(lambda: defaultdict(list))
    
    with open(IN_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            news = data.get("newspaper_name", "unknown")
            p_idx = data.get("page_index0", 0)
            text = data.get("ocr_text_eng", "")
            if text.strip():
                grouped[news][p_idx].append(text)

    total_pages = 0
    for news, pages in grouped.items():
        news_dir = OUT_DIR / news
        news_dir.mkdir(parents=True, exist_ok=True)
        print(f"📰 News: {news} ({len(pages)} pages)")
        
        for p_idx, blocks in pages.items():
            # Combine blocks with double newline
            page_text = "\n\n".join(blocks)
            # Match naming convention: <prefix>_p<page_num>_text.txt
            # Note: Stage 6 uses extract_page_number which looks for _p(\d+)_text.txt
            out_file = news_dir / f"{news}_p{p_idx+1}_text.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(page_text)
            total_pages += 1

    print(f"🏁 Bridge complete. Created {total_pages} page files in {OUT_DIR}")

if __name__ == "__main__":
    main()
