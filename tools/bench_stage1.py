import fitz
import time
import os
import sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

# Adhere to CPU_MANDATE.md (Max 16 cores)
NUM_CORES = 16

def render_pdf_worker(pdf_path, dpi=120):
    """Raw rendering performance test (no disk write)."""
    start = time.time()
    try:
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        for page in doc:
            # Memory-Only Pixmap strategy
            pix = page.get_pixmap(dpi=dpi)
            # Simulate buffer access as numpy
            _ = np.frombuffer(pix.samples, dtype=np.uint8)
        doc.close()
        return time.time() - start, page_count
    except Exception as e:
        return 0, 0

def run_bench():
    pdf_dir = Path("data/benchmark_13_raw")
    all_pdfs = sorted(list(pdf_dir.glob("*.pdf")))
    test_pdfs = all_pdfs[:10]
    
    if len(test_pdfs) < 10:
        print(f"Only found {len(test_pdfs)} PDFs.")
        return

    print(f"🚀 Benchmarking Stage 1 (120 DPI, Memory-Only) on 10 PDFs...")
    print(f"   CPU Cores: {NUM_CORES}")
    print("-" * 50)
    
    # Run in parallel pool
    with mp.Pool(NUM_CORES) as pool:
        results = pool.map(render_pdf_worker, test_pdfs)
    
    total_time = 0
    total_pages = 0
    print(f"{'PDF Name':<30} | {'Pages':<5} | {'Time':<8}")
    print("-" * 50)
    
    for pdf, (dur, p_count) in zip(test_pdfs, results):
        if p_count > 0:
            print(f"{pdf.stem[:30]:<30} | {p_count:<5} | {dur:.3f}s")
            total_time += dur
            total_pages += p_count
            
    avg_pdf = total_time / 10
    avg_page = total_time / total_pages if total_pages > 0 else 0
    
    print("-" * 50)
    print(f"AVG TIME PER PDF  : {avg_pdf:.3f}s")
    print(f"AVG TIME PER PAGE : {avg_page*1000:.2f}ms")
    print(f"TOTAL TIME (RAW)  : {total_time:.2f}s")

if __name__ == "__main__":
    run_bench()
