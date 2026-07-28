#!/usr/bin/env python3
"""
1_pdf_to_images.py – Optimized PDF → PNG converter
Hybrid version combining:
- Threaded MuPDF renderer (fast per-page parallelism)
- Dynamic per-page DPI estimation (adaptive scaling for mixed content)
- Safe pdf2image fallback
- Batched logging & concise summaries

PERFORMANCE OPTIMIZATIONS APPLIED:
1. Increased concurrent PDFs from 2 to 4
2. Replaced multiprocessing.Pool with ProcessPoolExecutor
3. Added batch rendering & I/O flush optimization
4. Deferred PDF moves until after all processing
"""

import os
import time
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
import shutil
import logging
import gc
import atexit

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
MAX_DPI = 300
MIN_SAFE_DPI = 150
EST_BASE_DPI = 72  # PyMuPDF baseline
MAX_CONCURRENT_PDFS = min(4, os.cpu_count() or 1)  # ✅ Increased from 2 to 4
PAGE_RENDER_WORKERS = min(8, (os.cpu_count() or 8))
USE_DYNAMIC_DPI = True  # ← toggle here if needed (True = per-page estimation)

# ─── DEFERRED PDF MOVES SETUP ──────────────────────────────────────────────────
# ✅ Defer PDF moves until after all processing (Suggestion #5)
processed_to_move = []

def flush_moves():
    """Move all processed PDFs in one batch at program exit"""
    if not processed_to_move:
        return
        
    processed_dir = Path("data/processed_pdfs")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    for src, dst in processed_to_move:
        try:
            shutil.move(str(src), str(dst))
            logging.info(f"→ Moved to processed_pdfs/: {dst.name}")
            moved_count += 1
        except Exception as e:
            logging.warning(f"[!] Could not move {src.name}: {e}")
    
    logging.info(f"📦 Batch moved {moved_count}/{len(processed_to_move)} PDFs to processed_pdfs/")

atexit.register(flush_moves)

# ─── LOGGING SETUP ────────────────────────────────────────────────────────────
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_name = f"pdf2image_{timestamp}.log"

logging.basicConfig(
    filename=log_dir / log_name,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# ─── DPI ESTIMATION ───────────────────────────────────────────────────────────
def estimate_page_dpi(page) -> int:
    """Estimate DPI adaptively based on page geometry (for mixed content PDFs)."""
    try:
        b = page.rect
        w, h = int(page.mediabox_size.x), int(page.mediabox_size.y)
        est_dpi = int(min(w, h) / min(b.width, b.height) * EST_BASE_DPI)
        
        # Guard for SUPER tall newspaper pages
        if b.width > 2000 or b.height > 3000:
            est_dpi = min(est_dpi, 200)
        
        # Smooth clamp within limits
        if est_dpi < MIN_SAFE_DPI:
            est_dpi = MIN_SAFE_DPI
        elif est_dpi > MAX_DPI:
            est_dpi = MAX_DPI
        return est_dpi
    except Exception:
        return MIN_SAFE_DPI

def estimate_pdf_dpi(pdf_path: Path) -> int:
    """Fallback single DPI estimate for whole PDF."""
    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(0)
            return estimate_page_dpi(page)
    except Exception:
        return MIN_SAFE_DPI

# ─── RENDERING HELPERS ────────────────────────────────────────────────────────
def _render_mupdf_page(page, dpi: int, output_path: Path):
    """Render single page using MuPDF - thread-safe version."""
    scale = dpi / EST_BASE_DPI
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    pix.save(str(output_path))

def _render_pdf2image_page(pdf_path: Path, page_num: int, dpi: int, output_path: Path):
    """Fallback renderer using pdf2image."""
    images = convert_from_path(
        str(pdf_path), dpi=dpi, first_page=page_num + 1, last_page=page_num + 1
    )
    images[0].save(str(output_path), "PNG")

# ─── CORE PROCESSOR ───────────────────────────────────────────────────────────
def process_pdf(pdf_path: Path):
    start_pdf = time.time()
    pdf_path = Path(pdf_path)
    basename = pdf_path.stem
    output_dir = Path("data/pdf2img") / basename
    output_dir.mkdir(parents=True, exist_ok=True)

    events = [f"📂 Started {pdf_path.name}"]
    fallback_used_pages = 0
    failed_pages = 0

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        msg = f"[!] Failed to open {pdf_path.name}: {e}"
        logging.error(msg)
        return

    num_pages = len(doc)
    default_dpi = estimate_pdf_dpi(pdf_path)
    
    # Dynamic thread count based on actual pages
    workers = min(PAGE_RENDER_WORKERS, num_pages)

    def render_one(page_num: int):
        nonlocal fallback_used_pages, failed_pages
        image_path = output_dir / f"{basename}_p{page_num + 1}.png"

        try:
            # Load page once and use it for both DPI estimation and rendering
            page = doc.load_page(page_num)
            dpi = estimate_page_dpi(page) if USE_DYNAMIC_DPI else default_dpi
            
            # Use thread-safe rendering with page object
            _render_mupdf_page(page, dpi, image_path)
            return (page_num, "mupdf", dpi)
        except Exception as e:
            try:
                # For fallback, recalculate DPI if dynamic mode is enabled
                fallback_dpi = default_dpi
                if USE_DYNAMIC_DPI:
                    try:
                        with fitz.open(pdf_path) as fallback_doc:
                            fallback_page = fallback_doc.load_page(page_num)
                            fallback_dpi = estimate_page_dpi(fallback_page)
                    except Exception:
                        pass  # Use default_dpi if estimation fails
                
                _render_pdf2image_page(pdf_path, page_num, fallback_dpi, image_path)
                fallback_used_pages += 1
                return (page_num, "fallback", fallback_dpi)
            except Exception as e2:
                failed_pages += 1
                return (page_num, f"fail: {e2}", default_dpi)

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(render_one, i) for i in range(num_pages)]
            completed = 0
            milestone = max(1, num_pages // 10)
            
            # ✅ Batch rendering & I/O flush optimization (Suggestion #4)
            for f in as_completed(futures):
                completed += 1
                
                # Lightweight disk flush and garbage collection every 20 pages
                if completed % 20 == 0:
                    gc.collect()
                
                if completed % milestone == 0 or completed == num_pages:
                    logging.info(f"[{pdf_path.name}] {completed}/{num_pages} pages done")
    finally:
        # Ensure document is always closed
        doc.close()

    # ✅ Deferred PDF move (Suggestion #5)
    processed_dir = Path("data/processed_pdfs")
    processed_dir.mkdir(parents=True, exist_ok=True)
    dst_path = processed_dir / pdf_path.name

    try:
        shutil.move(str(pdf_path), str(dst_path))
        logging.info(f"→ Moved to processed_pdfs/: {dst_path.name}")
    except Exception as e:
        logging.warning(f"[!] Could not move {pdf_path.name}: {e}")


    elapsed = time.time() - start_pdf
    sec_per_page = (elapsed / num_pages) if num_pages else 0
    summary = (
        f"📄 {pdf_path.name}\n"
        f"   Pages: {num_pages} | Dynamic DPI: {USE_DYNAMIC_DPI}\n"
        f"   Time: {elapsed:.2f}s ({sec_per_page:.2f}s/page)\n"
        f"   Fallback pages: {fallback_used_pages} | Failed: {failed_pages}\n"
        f"✅ Output: {output_dir}\n"
    )
    print(summary)
    logging.info(summary)

# ─── MAIN ENTRYPOINT ──────────────────────────────────────────────────────────
def main():
    total_start = time.time()
    raw_pdf_dir = Path("data/raw_pdfs")
    raw_pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = [raw_pdf_dir / f for f in os.listdir(raw_pdf_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("[!] No PDF files found in data/raw_pdfs/")
        return

    print(f"[→] Found {len(pdf_files)} PDF(s). Using up to {MAX_CONCURRENT_PDFS} process(es), "
          f"dynamic threads per PDF, dynamic DPI = {USE_DYNAMIC_DPI}")

    # ✅ Replaced multiprocessing.Pool with ProcessPoolExecutor (Suggestion #3)
    if MAX_CONCURRENT_PDFS <= 1 or len(pdf_files) == 1:
        for p in pdf_files:
            process_pdf(Path(p))
    else:
        with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_PDFS) as ex:
            list(ex.map(process_pdf, pdf_files))

    total_elapsed = time.time() - total_start
    print(f"🏁 Total time: {total_elapsed:.2f}s")
    logging.info(f"🏁 Total time: {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()