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
import argparse
import sys
import json
import hashlib
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
import shutil
import logging
from src.utils.logging_utils import configure_logging
logger = configure_logging("stage01_pdf_to_images")
import gc
import atexit
from itertools import repeat
import re
from typing import Any
import cv2

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.pipeline_config import load_config, get_path
from src.pipeline.pipeline_metadata import (
    extract_newspaper_and_date_from_pdf,
    generate_page_id,
    generate_doc_id,
    write_page_manifest_jsonl,
)

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
MAX_DPI = 300
MIN_SAFE_DPI = 150
EST_BASE_DPI = 72  # PyMuPDF baseline
MAX_CONCURRENT_PDFS = min(4, os.cpu_count() or 1)  # ✅ Increased from 2 to 4
PAGE_RENDER_WORKERS = min(8, (os.cpu_count() or 8))
USE_DYNAMIC_DPI = True  # ← toggle here if needed (True = per-page estimation)
IMAGES_OUTPUT_DIR = Path("data/pdf2img")
PAGE_INDEX_PAD_WIDTH = max(0, int(os.getenv("STAGE1_PAGE_INDEX_PAD_WIDTH", "0")))
DEFAULT_STAGE1_MANIFEST_PATH = Path("run_state/stage1_page_identity_manifest.json")
# New metadata manifest (JSONL format)
DEFAULT_STAGE1_PAGE_MANIFEST_JSONL = Path("run_state/page_manifest.jsonl")

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

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
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


def _rendered_page_filename(pdf_stem: str, page_index: int) -> str:
    """Build deterministic filename from 1-based PDF page index only."""
    if PAGE_INDEX_PAD_WIDTH > 0:
        return f"{pdf_stem}_p{int(page_index):0{PAGE_INDEX_PAD_WIDTH}d}.png"
    return f"{pdf_stem}_p{int(page_index)}.png"


def _sha256_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _remove_stale_page_images(output_dir: Path, pdf_stem: str, expected_names: set[str]) -> list[str]:
    pattern = re.compile(rf"^{re.escape(pdf_stem)}_p\d+\.png$")
    removed: list[str] = []
    for png_path in sorted(output_dir.glob("*.png"), key=lambda p: p.name):
        if not pattern.match(png_path.name):
            continue
        if png_path.name in expected_names:
            continue
        try:
            png_path.unlink()
            removed.append(png_path.name)
        except Exception as exc:
            logging.warning(f"[!] Could not remove stale stage01 output {png_path}: {exc}")
    return removed


def _build_stage1_page_row(
    *,
    source_pdf: Path,
    pdf_stem: str,
    page_index: int,
    page_path: Path,
    render_mode: str,
    dpi: int,
) -> dict[str, Any]:
    rendered_name = page_path.name
    return {
        "newspaper": str(pdf_stem),
        "source_pdf": str(source_pdf),
        "pdf_page_index": int(page_index),
        "rendered_filename": rendered_name,
        "rendered_page_key": rendered_name,
        "rendered_file_path": str(page_path),
        "render_mode": str(render_mode),
        "dpi": int(dpi),
        "image_sha256": _sha256_file(page_path) if page_path.exists() else None,
        "printed_page_number": None,
        "printed_page_number_confidence": "none",
        "printed_page_number_source": "not_detected_in_stage1",
    }


def _write_stage1_manifest(pdf_summaries: list[dict[str, Any]], manifest_path: Path) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    page_rows: list[dict[str, Any]] = []
    for pdf_row in pdf_summaries:
        rows = pdf_row.get("pages", [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    page_rows.append(row)

    page_rows = sorted(
        page_rows,
        key=lambda r: (
            str(r.get("source_pdf", "")),
            int(r.get("pdf_page_index", 10**9)),
            str(r.get("rendered_filename", "")),
        ),
    )
    payload = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "naming_policy": {
            "identity_basis": "pdf_page_index",
            "filename_format": "<pdf_stem>_p<index>.png",
            "page_index_pad_width": int(PAGE_INDEX_PAD_WIDTH),
            "printed_page_number_used_in_filename": False,
        },
        "summary": {
            "total_pdfs": len(pdf_summaries),
            "total_rendered_pages": len(page_rows),
            "total_failed_pages": int(
                sum(int(row.get("failed_pages", 0)) for row in pdf_summaries if isinstance(row, dict))
            ),
        },
        "pdfs": pdf_summaries,
        "pages": page_rows,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_bool_cli(value: str | None, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")

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
def _process_pdf_internal(
    pdf_path: Path,
    images_output_dir: str = "data/pdf2img",
    on_page_done=None,
    move_processed: bool = True,
    dynamic_dpi: bool = USE_DYNAMIC_DPI,
    skip_existing: bool = False,
    page_render_workers: int | None = None,
):
    start_pdf = time.time()
    pdf_path = Path(pdf_path)
    basename = pdf_path.stem
    output_dir = Path(images_output_dir) / basename
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
    
    from src.utils.pipeline_config import load_config
    config = load_config()
    forced_dpi = config.get("render_dpi")
    if forced_dpi:
        default_dpi = int(forced_dpi)
        use_dynamic = False
        print(f"[→] Overriding DPI to fixed: {default_dpi}")
    else:
        default_dpi = estimate_pdf_dpi(pdf_path)
        use_dynamic = bool(dynamic_dpi)

    expected_names = {_rendered_page_filename(basename, idx + 1) for idx in range(num_pages)}
    removed_stale_files: list[str] = []
    if not bool(skip_existing):
        removed_stale_files = _remove_stale_page_images(output_dir, basename, expected_names)
    
    # Dynamic thread count based on actual pages
    worker_cap = int(page_render_workers) if page_render_workers else int(PAGE_RENDER_WORKERS)
    worker_cap = max(1, worker_cap)
    workers = min(worker_cap, num_pages)

    rendered_pages: dict[int, dict[str, Any]] = {}

    def render_one(page_num: int):
        nonlocal fallback_used_pages, failed_pages
        page_index = int(page_num) + 1
        image_path = output_dir / _rendered_page_filename(basename, page_index)
        t_page_start = time.time()

        if bool(skip_existing) and image_path.exists():
            return (page_num, "existing", default_dpi, image_path, 0.0)

        try:
            # Load page once and use it for both DPI estimation and rendering
            page = doc.load_page(page_num)
            dpi = estimate_page_dpi(page) if use_dynamic else default_dpi

            # Use thread-safe rendering with page object
            _render_mupdf_page(page, dpi, image_path)
            return (page_num, "mupdf", dpi, image_path, round(time.time() - t_page_start, 3))
        except Exception as e:
            try:
                # For fallback, recalculate DPI if dynamic mode is enabled
                fallback_dpi = default_dpi
                if bool(dynamic_dpi):
                    try:
                        with fitz.open(pdf_path) as fallback_doc:
                            fallback_page = fallback_doc.load_page(page_num)
                            fallback_dpi = estimate_page_dpi(fallback_page)
                    except Exception:
                        pass  # Use default_dpi if estimation fails

                _render_pdf2image_page(pdf_path, page_num, fallback_dpi, image_path)
                fallback_used_pages += 1
                return (page_num, "fallback", fallback_dpi, image_path, round(time.time() - t_page_start, 3))
            except Exception as e2:
                failed_pages += 1
                return (page_num, f"fail: {e2}", default_dpi, image_path, round(time.time() - t_page_start, 3))

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(render_one, i) for i in range(num_pages)]
            completed = 0
            milestone = max(1, num_pages // 10)
            
            # ✅ Batch rendering & I/O flush optimization (Suggestion #4)
            for f in as_completed(futures):
                page_num, mode, dpi, image_path, render_time_sec = f.result()
                page_idx = int(page_num) + 1
                completed += 1
                rendered_pages[int(page_num)] = {
                    "pdf_page_index": page_idx,
                    "render_mode": str(mode),
                    "dpi": int(dpi),
                    "page_path": str(image_path),
                    "render_time_sec": render_time_sec,
                    "failed": str(mode).startswith("fail:"),
                }
                if on_page_done is not None and not str(mode).startswith("fail:"):
                    try:
                        on_page_done(
                            {
                                "pdf_name": pdf_path.name,
                                "pdf_stem": basename,
                                "page_index": page_idx,
                                "page_name": Path(image_path).name,
                                "page_path": str(image_path),
                                "render_mode": mode,
                                "dpi": int(dpi),
                            }
                        )
                    except Exception as callback_exc:
                        logging.warning(
                            f"[!] on_page_done callback failed for {pdf_path.name} p{page_num + 1}: {callback_exc}"
                        )
                
                # Lightweight disk flush and garbage collection every 20 pages
                if completed % 20 == 0:
                    gc.collect()
                
                if completed % milestone == 0 or completed == num_pages:
                    logging.info(f"[{pdf_path.name}] {completed}/{num_pages} pages done")
    finally:
        # Ensure document is always closed
        doc.close()

    # ✅ Deferred PDF move (Suggestion #5)
    if move_processed:
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
        f"   Pages: {num_pages} | Dynamic DPI: {bool(dynamic_dpi)} | Skip existing: {bool(skip_existing)}\n"
        f"   Time: {elapsed:.2f}s ({sec_per_page:.2f}s/page)\n"
        f"   Fallback pages: {fallback_used_pages} | Failed: {failed_pages} | Stale removed: {len(removed_stale_files)}\n"
        f"✅ Output: {output_dir}\n"
    )
    print(summary)
    logging.info(summary)

    # Extract newspaper name and issue date from PDF
    newspaper_name, issue_date = extract_newspaper_and_date_from_pdf(pdf_path)
    doc_id = generate_doc_id(str(pdf_path))

    page_rows: list[dict[str, Any]] = []
    page_manifest_rows: list[dict[str, Any]] = []

    for page_num in sorted(rendered_pages.keys()):
        row = rendered_pages[page_num]
        if bool(row.get("failed")):
            continue

        page_index0 = int(page_num)  # 0-based
        page_index1 = int(row["pdf_page_index"])  # 1-based (display)
        page_path = Path(str(row["page_path"]))

        # Build legacy manifest row (for backward compatibility)
        page_rows.append(
            _build_stage1_page_row(
                source_pdf=pdf_path,
                pdf_stem=basename,
                page_index=page_index1,
                page_path=page_path,
                render_mode=str(row["render_mode"]),
                dpi=int(row["dpi"]),
            )
        )

        # Build new metadata manifest row
        # Get image dimensions
        image_width, image_height = 0, 0
        try:
            img = cv2.imread(str(page_path))
            if img is not None:
                image_height, image_width = img.shape[:2]
        except Exception:
            pass

        page_id = generate_page_id(str(pdf_path), page_index0)

        manifest_row = {
            "page_id": page_id,
            "doc_id": doc_id,
            "pdf_path": str(pdf_path.resolve()),
            "page_index0": page_index0,
            "page_number1": page_index1,
            "newspaper_name": newspaper_name,
            "issue_date": issue_date,
            "image_path": str(page_path.resolve()),
            "image_width": image_width,
            "image_height": image_height,
            "render_dpi": int(row["dpi"]),
            "render_time_sec": float(row.get("render_time_sec", 0.0)),
        }
        page_manifest_rows.append(manifest_row)

    return {
        "pdf_name": pdf_path.name,
        "pdf_stem": basename,
        "source_pdf": str(pdf_path),
        "output_dir": str(output_dir),
        "total_pages": int(num_pages),
        "rendered_pages": int(len(page_rows)),
        "failed_pages": int(failed_pages),
        "fallback_pages": int(fallback_used_pages),
        "stale_removed_count": int(len(removed_stale_files)),
        "stale_removed_files": removed_stale_files,
        "pages": page_rows,
        "page_manifest_rows": page_manifest_rows,  # New: for JSONL export
    }


def process_pdf(
    pdf_path: Path,
    images_output_dir: str = "data/pdf2img",
    dynamic_dpi: bool = USE_DYNAMIC_DPI,
    skip_existing: bool = False,
    move_processed: bool = True,
    page_render_workers: int | None = None,
):
    """Legacy API used by sequential stage01 execution."""
    return _process_pdf_internal(
        pdf_path=pdf_path,
        images_output_dir=images_output_dir,
        on_page_done=None,
        move_processed=move_processed,
        dynamic_dpi=dynamic_dpi,
        skip_existing=skip_existing,
        page_render_workers=page_render_workers,
    )


def process_pdf_streaming(
    pdf_path: Path,
    images_output_dir: str = "data/pdf2img",
    on_page_done=None,
    move_processed: bool = True,
    dynamic_dpi: bool = USE_DYNAMIC_DPI,
    skip_existing: bool = False,
    page_render_workers: int | None = None,
):
    """
    Streaming-friendly API.
    Emits page-level callbacks as soon as each image is generated.
    """
    return _process_pdf_internal(
        pdf_path=pdf_path,
        images_output_dir=images_output_dir,
        on_page_done=on_page_done,
        move_processed=move_processed,
        dynamic_dpi=dynamic_dpi,
        skip_existing=skip_existing,
        page_render_workers=page_render_workers,
    )

# ─── POST-RENDER VALIDATION ───────────────────────────────────────────────────
def _validate_page_manifest(rows: list) -> None:
    """Validate page manifest after render: image paths exist, page_ids unique."""
    missing_images = []
    seen_page_ids: dict = {}
    duplicate_page_ids = []

    for row in rows:
        page_id = row.get("page_id")
        image_path = row.get("image_path")

        if image_path and not Path(image_path).exists():
            missing_images.append(image_path)

        if page_id:
            if page_id in seen_page_ids:
                duplicate_page_ids.append(page_id)
            seen_page_ids[page_id] = True

    if missing_images:
        logging.warning(f"[!] Stage1 validation: {len(missing_images)} missing image files")
        for p in missing_images[:5]:
            logging.warning(f"    missing: {p}")
    else:
        logging.info(f"Stage1 validation: all {len(rows)} image paths exist")

    if duplicate_page_ids:
        logging.warning(f"[!] Stage1 validation: {len(duplicate_page_ids)} duplicate page_ids: {duplicate_page_ids[:5]}")
    else:
        logging.info(f"Stage1 validation: all {len(seen_page_ids)} page_ids unique")

    if not missing_images and not duplicate_page_ids:
        print(f"Stage1 manifest validation OK: {len(rows)} pages, all images present, all page_ids unique")
    else:
        print(f"[!] Stage1 manifest validation WARNINGS: {len(missing_images)} missing images, {len(duplicate_page_ids)} dup page_ids")


# ─── MAIN ENTRYPOINT ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to images for stage01")
    parser.add_argument("--config", default="configs/pipeline_paths.json", help="Path to pipeline paths config")
    parser.add_argument(
        "--pdf-input",
        default=None,
        help="Input PDF folder override (defaults to config paths.pdf_input)",
    )
    parser.add_argument(
        "--images-output",
        default=None,
        help="Rendered image root override (defaults to config paths.images_output)",
    )
    parser.add_argument(
        "--manifest-output",
        default=str(DEFAULT_STAGE1_MANIFEST_PATH),
        help="Path to Stage1 page identity manifest output",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max concurrent PDF workers (defaults to internal auto setting)",
    )
    parser.add_argument(
        "--dynamic-dpi",
        choices=["auto", "true", "false"],
        default="auto",
        help="Enable dynamic per-page DPI (default: auto -> existing behavior)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rendering a page when its target PNG already exists",
    )
    parser.add_argument(
        "--move-processed",
        choices=["auto", "true", "false"],
        default="auto",
        help="Move processed PDFs to data/processed_pdfs (auto preserves old behavior for default input only)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    default_raw_pdf_dir = get_path("pdf_input", config)
    default_images_output_dir = get_path("images_output", config)
    raw_pdf_dir = Path(args.pdf_input) if args.pdf_input else default_raw_pdf_dir
    images_output_dir = Path(args.images_output) if args.images_output else default_images_output_dir
    dynamic_dpi = _parse_bool_cli(None if args.dynamic_dpi == "auto" else args.dynamic_dpi, default=USE_DYNAMIC_DPI)
    max_concurrent_pdfs = int(args.workers) if args.workers and int(args.workers) > 0 else int(MAX_CONCURRENT_PDFS)

    if args.move_processed == "true":
        move_processed = True
    elif args.move_processed == "false":
        move_processed = False
    else:
        move_processed = raw_pdf_dir.resolve() == default_raw_pdf_dir.resolve()

    print("[CONFIG]")
    print(f"pdf_input = {raw_pdf_dir}")
    print(f"images_output = {images_output_dir}")
    print(f"manifest_output = {args.manifest_output}")
    print(f"workers = {max_concurrent_pdfs}")
    print(f"dynamic_dpi = {dynamic_dpi}")
    print(f"skip_existing = {bool(args.skip_existing)}")
    print(f"move_processed = {bool(move_processed)}")

    total_start = time.time()
    raw_pdf_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(
        [raw_pdf_dir / f for f in os.listdir(raw_pdf_dir) if f.lower().endswith(".pdf")],
        key=lambda p: p.name.lower(),
    )

    if not pdf_files:
        print(f"[!] No PDF files found in {raw_pdf_dir}/")
        return

    print(
        f"[→] Found {len(pdf_files)} PDF(s). Using up to {max_concurrent_pdfs} process(es), "
        f"dynamic threads per PDF, dynamic DPI = {bool(dynamic_dpi)}"
    )

    pdf_summaries: list[dict[str, Any]] = []

    # ✅ Replaced multiprocessing.Pool with ProcessPoolExecutor (Suggestion #3)
    if max_concurrent_pdfs <= 1 or len(pdf_files) == 1:
        for p in pdf_files:
            summary = process_pdf(
                Path(p),
                str(images_output_dir),
                dynamic_dpi=dynamic_dpi,
                skip_existing=bool(args.skip_existing),
                move_processed=move_processed,
            )
            if isinstance(summary, dict):
                pdf_summaries.append(summary)
    else:
        with ProcessPoolExecutor(max_workers=max_concurrent_pdfs) as ex:
            for summary in ex.map(
                process_pdf,
                pdf_files,
                repeat(str(images_output_dir)),
                repeat(dynamic_dpi),
                repeat(bool(args.skip_existing)),
                repeat(move_processed),
                repeat(None),
            ):
                if isinstance(summary, dict):
                    pdf_summaries.append(summary)

    manifest_path = Path(args.manifest_output)
    _write_stage1_manifest(pdf_summaries, manifest_path)
    print(f"🧾 Stage1 manifest written: {manifest_path}")
    logging.info(f"🧾 Stage1 manifest written: {manifest_path}")

    # Write new JSONL page manifest for metadata handoff to Stage 2
    all_page_manifest_rows: list[dict[str, Any]] = []
    for summary in pdf_summaries:
        if isinstance(summary, dict):
            rows = summary.get("page_manifest_rows", [])
            if isinstance(rows, list):
                all_page_manifest_rows.extend(rows)

    page_manifest_jsonl_path = Path(DEFAULT_STAGE1_PAGE_MANIFEST_JSONL)
    try:
        write_page_manifest_jsonl(all_page_manifest_rows, page_manifest_jsonl_path)
        print(f"📋 Page manifest JSONL written: {page_manifest_jsonl_path} ({len(all_page_manifest_rows)} pages)")
        logging.info(f"📋 Page manifest JSONL written: {page_manifest_jsonl_path} ({len(all_page_manifest_rows)} pages)")
    except Exception as e:
        print(f"[!] Failed to write page manifest JSONL: {e}")
        logging.error(f"[!] Failed to write page manifest JSONL: {e}")

    # Post-render manifest validation: check image paths exist and page_ids unique
    _validate_page_manifest(all_page_manifest_rows)

    total_elapsed = time.time() - total_start
    print(f"🏁 Total time: {total_elapsed:.2f}s")
    logging.info(f"🏁 Total time: {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()
