#!/usr/bin/env python3
"""
Pipeline metadata utilities for Stage1->Stage2 handoff.
Handles:
- Extraction of newspaper_name and issue_date from PDFs and filenames
- Deterministic ID generation (page_id, crop_id, doc_id)
- Normalized bounding box computation
- JSONL manifest reading/writing
"""

import json
import hashlib
import re
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
import fitz  # PyMuPDF


def extract_newspaper_and_date_from_pdf(pdf_path: Path) -> tuple[str, Optional[str]]:
    """
    Extract newspaper name and issue date from PDF metadata or filename.

    Returns:
        (newspaper_name, issue_date_iso_or_none)

    Strategy:
    1. Try PDF metadata (Subject, Title, Keywords)
    2. Fall back to filename pattern: NAME-DATE.pdf or NAME_DATE.pdf
    3. If date extraction fails, return None for date
    """
    pdf_path = Path(pdf_path)
    stem = pdf_path.stem

    # Try PDF metadata first
    try:
        with fitz.open(pdf_path) as doc:
            metadata = doc.metadata or {}
            subject = metadata.get("subject", "").strip()
            title = metadata.get("title", "").strip()
            keywords = metadata.get("keywords", "").strip()

            # Subject often contains "newspaper - YYYY-MM-DD" format
            if subject:
                parts = subject.split("-")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    date_str = "-".join(parts[1:]).strip()
                    parsed_date = _parse_date(date_str)
                    if name and parsed_date:
                        return (name, parsed_date)
                    if name:
                        return (name, None)
    except Exception:
        pass

    # Fall back to filename pattern
    # Try patterns: "Name-2026-04-07.pdf", "Name_2026-04-07.pdf", "Name-20260407.pdf"
    patterns = [
        r"^(.+?)[-_](\d{4}[-_]?\d{2}[-_]?\d{2})$",  # Name-YYYY-MM-DD or Name_YYYYMMDD
        r"^(.+?)[-_](\d{8})$",  # Name-YYYYMMDD
    ]

    for pattern in patterns:
        match = re.match(pattern, stem)
        if match:
            name = match.group(1).strip()
            date_str = match.group(2)
            parsed_date = _parse_date(date_str)
            if name:
                return (name, parsed_date)

    # Last resort: use filename stem as newspaper name
    return (stem, None)


def _parse_date(date_str: str) -> Optional[str]:
    """
    Try to parse various date formats and return ISO format (YYYY-MM-DD).
    Handles: YYYY-MM-DD, YYYY/MM/DD, YYYYMMDD, YYYY_MM_DD
    Returns None if parsing fails.
    """
    if not date_str:
        return None

    # Normalize separators
    normalized = date_str.replace("_", "-").replace("/", "-")

    # Try YYYY-MM-DD format
    try:
        dt = datetime.strptime(normalized, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass

    # Try YYYYMMDD format (no separators)
    if len(date_str) == 8 and date_str.isdigit():
        try:
            dt = datetime.strptime(date_str, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None


def generate_page_id(pdf_path: str, page_index0: int) -> str:
    """
    Generate deterministic page_id from PDF path and 0-based page index.
    Format: {pdf_hash_prefix}-p{page_index0}

    This allows reconstruction without metadata if needed.
    """
    pdf_stem = Path(pdf_path).stem
    # Create a short hash of the PDF path for uniqueness
    pdf_hash = hashlib.md5(str(Path(pdf_path).resolve()).encode()).hexdigest()[:8]
    return f"{pdf_hash}-{pdf_stem}-p{page_index0:06d}"


def generate_doc_id(pdf_path: str) -> str:
    """Generate deterministic doc_id from PDF path."""
    pdf_path = Path(pdf_path).resolve()
    pdf_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()[:12]
    return pdf_hash


def generate_crop_id(page_id: str, crop_index: int) -> str:
    """Generate deterministic crop_id from page_id and crop index."""
    return f"{page_id}-c{crop_index:04d}"


def compute_normalized_bbox(
    x1: int, y1: int, x2: int, y2: int,
    img_width: int, img_height: int
) -> dict[str, float]:
    """
    Compute normalized bounding box coordinates.

    Returns:
        {
            'x1_norm': float in [0, 1],
            'y1_norm': float in [0, 1],
            'x2_norm': float in [0, 1],
            'y2_norm': float in [0, 1],
            'center_x_norm': float in [0, 1],
            'center_y_norm': float in [0, 1],
            'area_norm': float in [0, 1]
        }
    """
    img_area = max(1, img_width * img_height)

    x1_norm = max(0.0, min(1.0, x1 / max(1, img_width)))
    y1_norm = max(0.0, min(1.0, y1 / max(1, img_height)))
    x2_norm = max(0.0, min(1.0, x2 / max(1, img_width)))
    y2_norm = max(0.0, min(1.0, y2 / max(1, img_height)))

    center_x_norm = (x1_norm + x2_norm) / 2.0
    center_y_norm = (y1_norm + y2_norm) / 2.0

    box_area = max(0, (x2 - x1) * (y2 - y1))
    area_norm = box_area / img_area

    return {
        "x1_norm": round(x1_norm, 6),
        "y1_norm": round(y1_norm, 6),
        "x2_norm": round(x2_norm, 6),
        "y2_norm": round(y2_norm, 6),
        "center_x_norm": round(center_x_norm, 6),
        "center_y_norm": round(center_y_norm, 6),
        "area_norm": round(area_norm, 6),
    }


def write_page_manifest_jsonl(
    pages: list[dict[str, Any]],
    output_path: Path
) -> None:
    """
    Write pages to JSONL manifest (one JSON object per line).
    Each page dict should contain all required metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for page in pages:
            f.write(json.dumps(page, ensure_ascii=False) + '\n')


def read_page_manifest_jsonl(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """
    Read JSONL page manifest and return dict keyed by page_id.

    Returns:
        {page_id: {metadata_dict}, ...}
    """
    manifest_path = Path(manifest_path)
    pages_by_id = {}

    if not manifest_path.exists():
        return pages_by_id

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                page = json.loads(line)
                page_id = page.get('page_id')
                if page_id:
                    pages_by_id[page_id] = page
            except json.JSONDecodeError:
                pass

    return pages_by_id


def write_crop_manifest_jsonl(
    crops: list[dict[str, Any]],
    output_path: Path
) -> None:
    """
    Write crops to JSONL manifest (one JSON object per line).
    Each crop dict should contain all required metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for crop in crops:
            f.write(json.dumps(crop, ensure_ascii=False) + '\n')


def read_crop_manifest_jsonl(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """
    Read JSONL crop manifest and return dict keyed by crop_id.

    Returns:
        {crop_id: {metadata_dict}, ...}
    """
    manifest_path = Path(manifest_path)
    crops_by_id = {}

    if not manifest_path.exists():
        return crops_by_id

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                crop = json.loads(line)
                crop_id = crop.get('crop_id')
                if crop_id:
                    crops_by_id[crop_id] = crop
            except json.JSONDecodeError:
                pass

    return crops_by_id
