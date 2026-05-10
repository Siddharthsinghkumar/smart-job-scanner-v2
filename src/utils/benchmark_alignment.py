"""Reusable benchmark alignment helpers for label/page-key reconciliation."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


def source_hint_from_label_file(label_file_name: str) -> str | None:
    lower = label_file_name.lower()
    if "et delhi" in lower:
        return "ET Delhi 18-03"
    if "toi-delhi" in lower:
        return "TOI-Delhi 18-03"
    if "bs -delhi" in lower:
        return "BS -Delhi - 18-03-2026"
    return None


def normalize_label_image_candidates(original_filename: str, source_hint: str | None) -> tuple[str, list[str], list[str]]:
    warnings: list[str] = []
    candidates: list[str] = []

    base = unquote(Path(original_filename).name)
    stripped = re.sub(r"^[0-9a-fA-F]{8}-", "", base)
    if stripped != base:
        warnings.append("label_file_has_hash_prefix")
    candidates.append(stripped)

    p_style = re.fullmatch(r"(.+)_p(\d+)\.png", stripped)
    if p_style:
        paper = p_style.group(1)
        page_no = int(p_style.group(2))
        if "_" in paper:
            warnings.append("label_uses_underscored_filename")
        candidates.append(f"{paper.replace('_', ' ')}_p{page_no}.png")

    generic_page = re.fullmatch(r"page_(\d+)\.png", stripped)
    if generic_page:
        warnings.append("generic_page_name")
        page_no = int(generic_page.group(1))
        if source_hint:
            candidates.append(f"{source_hint}_p{page_no}.png")

    dedup: list[str] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    chosen = dedup[0] if dedup else stripped
    return chosen, dedup, warnings


def choose_best_page_key(candidates: list[str], image_keys: set[str], detection_keys: set[str]) -> str | None:
    if not candidates:
        return None
    scored: list[tuple[int, str]] = []
    for c in candidates:
        score = 0
        if c in detection_keys:
            score += 2
        if c in image_keys:
            score += 1
        scored.append((score, c))
    scored.sort(key=lambda x: (-x[0], candidates.index(x[1])))
    return scored[0][1]


def compute_dim_scale(label_dims: tuple[int, int] | None, image_dims: tuple[int, int] | None) -> tuple[float, float] | None:
    if not label_dims or not image_dims:
        return None
    lw, lh = label_dims
    iw, ih = image_dims
    if lw <= 0 or lh <= 0:
        return None
    return (float(iw) / float(lw), float(ih) / float(lh))


def scale_bbox_xyxy(bbox: list[int], scale: tuple[float, float] | None) -> list[int]:
    if scale is None or not isinstance(bbox, list) or len(bbox) != 4:
        return bbox
    sx, sy = scale
    return [
        int(round(bbox[0] * sx)),
        int(round(bbox[1] * sy)),
        int(round(bbox[2] * sx)),
        int(round(bbox[3] * sy)),
    ]
