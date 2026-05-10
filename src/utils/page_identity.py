"""Canonical page-identity helpers for PDF index vs printed page semantics."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_PAGE_IDENTITY_MAP_PATH = Path("run_state/page_identity_map.json")


def parse_rendered_page_key(page_key: str) -> tuple[str, int] | None:
    m = re.fullmatch(r"(.+)_p(\d+)\.png", str(page_key).strip())
    if not m:
        return None
    return m.group(1), int(m.group(2))


def fallback_page_identity(page_key: str) -> dict[str, Any]:
    parsed = parse_rendered_page_key(page_key)
    if not parsed:
        return {
            "newspaper": None,
            "pdf_page_index": None,
            "rendered_filename": page_key,
            "rendered_page_key": page_key,
            "printed_page_number": None,
            "printed_page_number_confidence": "none",
            "printed_page_number_source": "fallback_unparsed_key",
        }
    newspaper, page_index = parsed
    return {
        "newspaper": newspaper,
        "pdf_page_index": page_index,
        "rendered_filename": page_key,
        "rendered_page_key": page_key,
        "printed_page_number": None,
        "printed_page_number_confidence": "none",
        "printed_page_number_source": "fallback_from_filename",
    }


def load_page_identity_map(map_path: Path | None = None) -> dict[str, Any] | None:
    path = (map_path or DEFAULT_PAGE_IDENTITY_MAP_PATH)
    if not Path(path).is_file():
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    pages = payload.get("pages", []) if isinstance(payload, dict) else []
    if not isinstance(pages, list):
        return None
    index: dict[str, dict[str, Any]] = {}
    for row in pages:
        if not isinstance(row, dict):
            continue
        key = str(row.get("rendered_page_key") or row.get("rendered_filename") or "")
        if key:
            index[key] = row
    payload["_index_by_rendered_key"] = index
    return payload


def resolve_page_identity(page_key: str, page_identity_map: dict[str, Any] | None) -> dict[str, Any]:
    if not page_identity_map:
        return fallback_page_identity(page_key)
    idx = page_identity_map.get("_index_by_rendered_key", {})
    if isinstance(idx, dict):
        row = idx.get(page_key)
        if isinstance(row, dict):
            return row
    return fallback_page_identity(page_key)


def _run_tesseract_stdout(image_path: Path, psm: int = 11) -> str:
    if shutil.which("tesseract") is None:
        return ""
    cmd = [
        "tesseract",
        str(image_path),
        "stdout",
        "--oem",
        "1",
        "--psm",
        str(psm),
        "-l",
        "eng",
    ]
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)  # noqa: S603
    except Exception:
        return ""


def detect_printed_page_number_from_image(
    image_path: Path,
) -> dict[str, Any]:
    """Detect printed page number using conservative top-band OCR patterns.

    Returns nullable printed-page data; this function intentionally prefers "unknown"
    over weak guesses.
    """
    if not Path(image_path).is_file():
        return {
            "printed_page_number": None,
            "confidence": "none",
            "source": "image_missing",
            "evidence_excerpt": None,
        }

    if shutil.which("tesseract") is None:
        return {
            "printed_page_number": None,
            "confidence": "none",
            "source": "tesseract_missing",
            "evidence_excerpt": None,
        }

    try:
        img = Image.open(image_path).convert("L")
    except Exception:
        return {
            "printed_page_number": None,
            "confidence": "none",
            "source": "image_open_failed",
            "evidence_excerpt": None,
        }

    w, h = img.size
    top_band = img.crop((0, 0, w, max(1, int(h * 0.10))))
    bw = top_band.point(lambda px: 255 if px > 165 else 0)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        bw.save(tmp_path)
        text = _run_tesseract_stdout(tmp_path, psm=11)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    clean = " ".join(text.split())
    if not clean:
        return {
            "printed_page_number": None,
            "confidence": "none",
            "source": "ocr_empty",
            "evidence_excerpt": None,
        }

    # High-confidence TOI-style masthead pattern:
    # "THE TIMES OF INDIA, NEW DELHI 10 WEDNESDAY, MARCH ..."
    toi_match = re.search(
        r"NEW\s+DELHI\s*\|?\s*(\d{1,2})\s+WEDNESDAY",
        clean,
        flags=re.IGNORECASE,
    )
    if toi_match:
        page_num = int(toi_match.group(1))
        if 1 <= page_num <= 99:
            return {
                "printed_page_number": page_num,
                "confidence": "high",
                "source": "toi_header_new_delhi_number",
                "evidence_excerpt": clean[:220],
            }

    # Conservative ET fallback when page number is explicitly at line start:
    et_match = re.match(r"^\s*(\d{1,2})\s+THE\s+ECONOMIC\s+TIMES\b", clean, flags=re.IGNORECASE)
    if et_match:
        page_num = int(et_match.group(1))
        if 1 <= page_num <= 99:
            return {
                "printed_page_number": page_num,
                "confidence": "medium",
                "source": "et_header_leading_number",
                "evidence_excerpt": clean[:220],
            }

    return {
        "printed_page_number": None,
        "confidence": "none",
        "source": "no_high_confidence_pattern",
        "evidence_excerpt": clean[:220],
    }


def printed_page_to_rendered_keys(
    page_identity_map: dict[str, Any] | None,
    newspaper: str,
    printed_page_number: int,
) -> list[str]:
    if not page_identity_map:
        return []
    pages = page_identity_map.get("pages", [])
    if not isinstance(pages, list):
        return []
    out: list[str] = []
    for row in pages:
        if not isinstance(row, dict):
            continue
        if str(row.get("newspaper") or "") != str(newspaper):
            continue
        if row.get("printed_page_number") != int(printed_page_number):
            continue
        key = str(row.get("rendered_page_key") or row.get("rendered_filename") or "")
        if key:
            out.append(key)
    return sorted(set(out))
