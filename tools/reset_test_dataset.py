#!/usr/bin/env python3
"""Reset benchmark dataset outputs for stage01-stage03 detector evaluation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_LABELS_DIR = (PROJECT_ROOT / "data" / "test_labels").resolve()
PROTECTED_BENCHMARK_IMAGES_DIR = (PROJECT_ROOT / "data" / "benchmark_images").resolve()

CLEAR_DIRS = [
    PROJECT_ROOT / "data" / "pdf2img",
    PROJECT_ROOT / "data" / "job_blocks_smart",
    PROJECT_ROOT / "data" / "job_blocks_refined",
    PROJECT_ROOT / "run_state" / "detections",
]

RAW_PDFS_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
PROCESSED_PDFS_DIR = PROJECT_ROOT / "data" / "processed_pdfs"

REQUIRED_PDFS = [
    "BS -Delhi - 18-03-2026.pdf",
    "ET Delhi 18-03.pdf",
    "TOI-Delhi 18-03.pdf",
]


def _ensure_safe_path(path: Path) -> None:
    resolved = path.resolve()
    if resolved == PROTECTED_LABELS_DIR or PROTECTED_LABELS_DIR in resolved.parents:
        raise RuntimeError(f"Refusing to modify protected labels path: {path}")
    if resolved == PROTECTED_BENCHMARK_IMAGES_DIR or PROTECTED_BENCHMARK_IMAGES_DIR in resolved.parents:
        raise RuntimeError(f"Refusing to modify frozen benchmark images path: {path}")


def _clear_generated_contents(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    _ensure_safe_path(target_dir)

    for child in target_dir.iterdir():
        _ensure_safe_path(child)
        # Keep repository placeholder files.
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def _move_processed_pdfs_back() -> int:
    RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PDFS_DIR.mkdir(parents=True, exist_ok=True)

    moved = 0
    for pdf_path in sorted(PROCESSED_PDFS_DIR.glob("*.pdf")):
        _ensure_safe_path(pdf_path)
        destination = RAW_PDFS_DIR / pdf_path.name
        if destination.exists():
            destination.unlink()
        shutil.move(str(pdf_path), str(destination))
        moved += 1
    return moved


def _validate_required_benchmark_pdfs() -> None:
    missing = [name for name in REQUIRED_PDFS if not (RAW_PDFS_DIR / name).is_file()]
    if missing:
        raise RuntimeError(
            "Benchmark dataset incomplete in data/raw_pdfs. Missing: "
            + ", ".join(missing)
        )


def reset_dataset() -> None:
    for target in CLEAR_DIRS:
        _clear_generated_contents(target)

    moved_count = _move_processed_pdfs_back()
    _validate_required_benchmark_pdfs()
    print(f"[reset] Cleared generated outputs in {len(CLEAR_DIRS)} directories")
    print(f"[reset] Moved {moved_count} processed PDF(s) back to data/raw_pdfs")
    print("[reset] Verified benchmark PDFs are present")
    if PROTECTED_BENCHMARK_IMAGES_DIR.is_dir():
        print(f"[reset] Preserved frozen benchmark images: {PROTECTED_BENCHMARK_IMAGES_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset stage01-stage03 benchmark dataset outputs")
    return parser.parse_args()


def main() -> int:
    _ = parse_args()
    reset_dataset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
