#!/usr/bin/env python3
"""Build detector-pivot v4 tile dataset focused on false-positive reduction."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class PageData:
    split: str
    source_page: str
    source_page_path: str
    width: int
    height: int
    gt_boxes: list[Box]


@dataclass
class TileCandidate:
    split: str
    source_page: str
    source_page_path: str
    tile_xyxy_in_page: list[int]
    generation_rule: str
    source_gt_ids_inside_tile: list[int]
    context_margin_px: int
    tile_size: int
    hardness_score: float
    min_visibility: float
    extra_meta: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build detector pivot v4 YOLO tile dataset")
    parser.add_argument("--source-dataset-root", default="data/yolo_job_ad_pivot_v3_source")
    parser.add_argument("--output-dataset-root", default="data/yolo_job_ad_pivot_v4_tiles")
    parser.add_argument("--output-manifest", default="run_state/detector_pivot_v4_dataset_manifest.json")
    parser.add_argument("--output-report", default="run_state/detector_pivot_v4_dataset_build_report.json")
    parser.add_argument(
        "--sparse-hard-negative-manifest",
        default="run_state/sparse_hard_negative_pool_manifest.json",
        help="Optional prior sparse hard-negative manifest (only non-holdout entries are used)",
    )
    parser.add_argument("--seed", type=int, default=20260407)
    parser.add_argument("--negative-ratio", type=float, default=2.0, help="Negative-to-positive ratio cap per split")
    parser.add_argument("--negative-tile-size", type=int, default=768)
    parser.add_argument("--dense-tile-size", type=int, default=896)
    parser.add_argument("--dense-stride", type=int, default=448)
    parser.add_argument("--max-imported-non-holdout-negs", type=int, default=220)
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _clip(v: int, lo: int, hi: int) -> int:
    return min(max(v, lo), hi)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _iou_xyxy(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = aa + bb - inter
    return float(inter / denom) if denom > 0 else 0.0


def _intersection(a: list[int], b: list[int]) -> list[int] | None:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return [ix1, iy1, ix2, iy2]


def _tile_from_center(cx: int, cy: int, tile: int, w: int, h: int) -> list[int]:
    tile = min(tile, w, h)
    half = tile // 2
    x1 = _clip(cx - half, 0, max(0, w - tile))
    y1 = _clip(cy - half, 0, max(0, h - tile))
    x2 = min(w, x1 + tile)
    y2 = min(h, y1 + tile)
    return [x1, y1, x2, y2]


def _iter_sliding_tiles(w: int, h: int, tile: int, stride: int) -> list[list[int]]:
    tile = min(tile, w, h)
    xs = list(range(0, max(1, w - tile + 1), stride))
    ys = list(range(0, max(1, h - tile + 1), stride))
    if not xs or xs[-1] != max(0, w - tile):
        xs.append(max(0, w - tile))
    if not ys or ys[-1] != max(0, h - tile):
        ys.append(max(0, h - tile))
    out: list[list[int]] = []
    for y in ys:
        for x in xs:
            out.append([x, y, min(w, x + tile), min(h, y + tile)])
    return out


def _parse_yolo_labels(label_path: Path, w: int, h: int) -> list[Box]:
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    boxes: list[Box] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, xc, yc, bw, bh = parts
        try:
            xc_f = float(xc)
            yc_f = float(yc)
            bw_f = float(bw)
            bh_f = float(bh)
        except Exception:
            continue
        x1 = int(round((xc_f - bw_f / 2.0) * w))
        y1 = int(round((yc_f - bh_f / 2.0) * h))
        x2 = int(round((xc_f + bw_f / 2.0) * w))
        y2 = int(round((yc_f + bh_f / 2.0) * h))
        x1 = _clip(x1, 0, max(0, w - 1))
        y1 = _clip(y1, 0, max(0, h - 1))
        x2 = _clip(x2, x1 + 1, w)
        y2 = _clip(y2, y1 + 1, h)
        boxes.append(Box(x1=x1, y1=y1, x2=x2, y2=y2))
    return boxes


def _load_pages(ds_root: Path) -> list[PageData]:
    pages: list[PageData] = []
    for split in ("train", "val"):
        img_dir = ds_root / "images" / split
        lbl_dir = ds_root / "labels" / split
        for img_path in sorted(img_dir.glob("*.png")):
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = int(im.shape[0]), int(im.shape[1])
            label_path = lbl_dir / f"{img_path.stem}.txt"
            boxes = _parse_yolo_labels(label_path, w=w, h=h)
            pages.append(
                PageData(
                    split=split,
                    source_page=img_path.name,
                    source_page_path=str(img_path),
                    width=w,
                    height=h,
                    gt_boxes=boxes,
                )
            )
    return pages


def _nearest_tile_size(target: int, sizes: list[int]) -> int:
    return min(sizes, key=lambda s: abs(int(s) - int(target)))


def _candidate_gt_ids(tile: list[int], gt_boxes: list[Box], min_visibility: float) -> list[int]:
    keep: list[int] = []
    for idx, g in enumerate(gt_boxes):
        inter = _intersection(tile, [g.x1, g.y1, g.x2, g.y2])
        if inter is None:
            continue
        ia = max(0, inter[2] - inter[0]) * max(0, inter[3] - inter[1])
        vis = (ia / g.area) if g.area > 0 else 0.0
        if vis >= min_visibility:
            keep.append(idx)
    return keep


def _text_hardness_score(image: np.ndarray, tile: list[int]) -> float:
    x1, y1, x2, y2 = tile
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    ink_density = float(np.mean(bw > 0))
    return 0.55 * edge_density + 0.45 * ink_density


def _dedupe(entries: list[TileCandidate], iou_thr: float = 0.92) -> list[TileCandidate]:
    out: list[TileCandidate] = []
    for row in entries:
        cur = row.tile_xyxy_in_page
        if any(
            (r.source_page == row.source_page and _iou_xyxy(cur, r.tile_xyxy_in_page) >= iou_thr and r.generation_rule == row.generation_rule)
            for r in out
        ):
            continue
        out.append(row)
    return out


def _max_iou_with_gt(tile: list[int], gt_boxes: list[Box]) -> float:
    if not gt_boxes:
        return 0.0
    return max(_iou_xyxy(tile, [g.x1, g.y1, g.x2, g.y2]) for g in gt_boxes)


def _build_candidates_for_page(
    page: PageData,
    *,
    tile_sizes: list[int],
    dense_tile_size: int,
    dense_stride: int,
    neg_tile_size: int,
) -> list[TileCandidate]:
    candidates: list[TileCandidate] = []
    src = cv2.imread(page.source_page_path)
    if src is None:
        return candidates

    gt_boxes = page.gt_boxes
    w, h = page.width, page.height

    # Positive/context tiles with slightly tighter context than v3.
    for gt_idx, box in enumerate(gt_boxes):
        cx = (box.x1 + box.x2) // 2
        cy = (box.y1 + box.y2) // 2
        min_dim = max(1, min(box.w, box.h))

        variants: list[tuple[str, int, float]] = [
            ("positive_tight_context", max(24, int(round(0.15 * min_dim))), 0.95),
            ("positive_medium_context", max(48, int(round(0.35 * min_dim))), 0.90),
        ]

        page_area = float(max(1, w * h))
        box_area_frac = box.area / page_area
        if box_area_frac < 0.004:
            variants.append(("positive_small_object_boost", max(40, int(round(0.30 * min_dim))), 0.90))

        for rule, margin, vis_thr in variants:
            target = max(box.w + 2 * margin, box.h + 2 * margin)
            tile_size = _nearest_tile_size(target, tile_sizes)
            tile = _tile_from_center(cx, cy, tile_size, w, h)
            gt_ids = _candidate_gt_ids(tile, gt_boxes, min_visibility=vis_thr)
            if gt_idx not in gt_ids:
                gt_ids.append(gt_idx)
            candidates.append(
                TileCandidate(
                    split=page.split,
                    source_page=page.source_page,
                    source_page_path=page.source_page_path,
                    tile_xyxy_in_page=tile,
                    generation_rule=rule,
                    source_gt_ids_inside_tile=sorted(set(gt_ids)),
                    context_margin_px=margin,
                    tile_size=tile_size,
                    hardness_score=0.0,
                    min_visibility=vis_thr,
                    extra_meta={},
                )
            )

        # Adjacent negatives near positives (non-overlapping) to fight near-miss FP around ad-like columns.
        adj_tile = _nearest_tile_size(max(640, int(round(max(box.w, box.h) * 2.2))), tile_sizes)
        for ox, oy in [(-0.60, 0.0), (0.60, 0.0), (0.0, -0.60), (0.0, 0.60)]:
            nx = int(round(cx + ox * adj_tile))
            ny = int(round(cy + oy * adj_tile))
            nt = _tile_from_center(nx, ny, adj_tile, w, h)
            if _max_iou_with_gt(nt, gt_boxes) >= 0.015:
                continue
            if _candidate_gt_ids(nt, gt_boxes, min_visibility=0.30):
                continue
            candidates.append(
                TileCandidate(
                    split=page.split,
                    source_page=page.source_page,
                    source_page_path=page.source_page_path,
                    tile_xyxy_in_page=nt,
                    generation_rule="adjacent_hard_negative",
                    source_gt_ids_inside_tile=[],
                    context_margin_px=0,
                    tile_size=adj_tile,
                    hardness_score=_text_hardness_score(src, nt),
                    min_visibility=1.0,
                    extra_meta={"gt_anchor": gt_idx},
                )
            )

    # Dense positive regions.
    if len(gt_boxes) >= 8:
        for tile in _iter_sliding_tiles(w, h, dense_tile_size, dense_stride):
            gt_ids = _candidate_gt_ids(tile, gt_boxes, min_visibility=0.70)
            if len(gt_ids) < 2:
                continue
            candidates.append(
                TileCandidate(
                    split=page.split,
                    source_page=page.source_page,
                    source_page_path=page.source_page_path,
                    tile_xyxy_in_page=tile,
                    generation_rule="dense_region_sliding",
                    source_gt_ids_inside_tile=gt_ids,
                    context_margin_px=0,
                    tile_size=dense_tile_size,
                    hardness_score=0.0,
                    min_visibility=0.70,
                    extra_meta={},
                )
            )

    # FP-focused hard negatives.
    neg_candidates: list[TileCandidate] = []
    neg_stride = max(1, neg_tile_size // 2)
    for tile in _iter_sliding_tiles(w, h, neg_tile_size, neg_stride):
        if _max_iou_with_gt(tile, gt_boxes) >= 0.015:
            continue
        score = _text_hardness_score(src, tile)
        neg_candidates.append(
            TileCandidate(
                split=page.split,
                source_page=page.source_page,
                source_page_path=page.source_page_path,
                tile_xyxy_in_page=tile,
                generation_rule="hard_negative_text_dense",
                source_gt_ids_inside_tile=[],
                context_margin_px=0,
                tile_size=neg_tile_size,
                hardness_score=score,
                min_visibility=1.0,
                extra_meta={},
            )
        )

    neg_candidates.sort(key=lambda r: r.hardness_score, reverse=True)
    gt_count = len(gt_boxes)
    if gt_count == 0:
        keep_n = min(14, len(neg_candidates))
    elif gt_count <= 2:
        keep_n = min(12, len(neg_candidates))
    else:
        keep_n = min(6, len(neg_candidates))
    candidates.extend(neg_candidates[:keep_n])

    return _dedupe(candidates, iou_thr=0.92)


def _clip_and_convert_boxes(tile: list[int], gt_boxes: list[Box], gt_ids: list[int], min_visibility: float) -> list[list[float]]:
    x1, y1, x2, y2 = tile
    tw = max(1, x2 - x1)
    th = max(1, y2 - y1)
    out: list[list[float]] = []
    for gt_idx in gt_ids:
        if gt_idx < 0 or gt_idx >= len(gt_boxes):
            continue
        g = gt_boxes[gt_idx]
        inter = _intersection(tile, [g.x1, g.y1, g.x2, g.y2])
        if inter is None:
            continue
        ia = max(0, inter[2] - inter[0]) * max(0, inter[3] - inter[1])
        vis = (ia / g.area) if g.area > 0 else 0.0
        if vis < min_visibility:
            continue
        rx1 = inter[0] - x1
        ry1 = inter[1] - y1
        rx2 = inter[2] - x1
        ry2 = inter[3] - y1
        bw = max(1.0, float(rx2 - rx1))
        bh = max(1.0, float(ry2 - ry1))
        cx = float(rx1 + bw / 2.0)
        cy = float(ry1 + bh / 2.0)
        out.append([0.0, cx / tw, cy / th, bw / tw, bh / th])
    return out


def _write_label_txt(path: Path, rows: list[list[float]]) -> None:
    lines: list[str] = []
    for r in rows:
        cls, xc, yc, w, h = r
        lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _load_sparse_non_holdout_negatives(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        return []

    rows: list[dict[str, Any]] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        if bool(e.get("source_is_holdout_derived", True)):
            continue
        image_path = Path(str(e.get("image_path", "")))
        if not image_path.is_file():
            continue
        rows.append(e)

    rows.sort(key=lambda r: float(r.get("hardness_score", 0.0)), reverse=True)
    if limit > 0:
        rows = rows[:limit]
    return rows


def main() -> int:
    args = _parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    src_root = _resolve(args.source_dataset_root)
    out_root = _resolve(args.output_dataset_root)
    manifest_out = _resolve(args.output_manifest)
    report_out = _resolve(args.output_report)
    sparse_neg_manifest = _resolve(args.sparse_hard_negative_manifest)

    pages = _load_pages(src_root)
    if not pages:
        raise RuntimeError(f"No source pages found under: {src_root}")

    page_lookup = {(p.split, p.source_page): p for p in pages}
    page_to_split = {p.source_page: p.split for p in pages}

    train_names = {p.source_page for p in pages if p.split == "train"}
    val_names = {p.source_page for p in pages if p.split == "val"}
    if train_names & val_names:
        raise RuntimeError("Split leakage detected in source dataset")

    tile_sizes = [640, 896, 1024]
    dense_tile_size = int(args.dense_tile_size)
    dense_stride = int(args.dense_stride)
    neg_tile_size = int(args.negative_tile_size)

    all_candidates: list[TileCandidate] = []
    for page in pages:
        all_candidates.extend(
            _build_candidates_for_page(
                page,
                tile_sizes=tile_sizes,
                dense_tile_size=dense_tile_size,
                dense_stride=dense_stride,
                neg_tile_size=neg_tile_size,
            )
        )

    # Split-wise balance for generated candidates.
    kept: list[TileCandidate] = []
    split_balance: dict[str, Any] = {}
    for split in ("train", "val"):
        rows = [r for r in all_candidates if r.split == split]
        pos = [r for r in rows if r.source_gt_ids_inside_tile]
        neg = [r for r in rows if not r.source_gt_ids_inside_tile]
        neg.sort(key=lambda r: r.hardness_score, reverse=True)
        max_neg = int(max(1, round(len(pos) * float(args.negative_ratio)))) if pos else min(80, len(neg))
        kept_split = pos + neg[:max_neg]
        kept.extend(kept_split)
        split_balance[split] = {
            "generated_positive_tiles": len(pos),
            "generated_negative_tiles_before_cap": len(neg),
            "generated_negative_tiles_after_cap": len(neg[:max_neg]),
            "negative_ratio_cap": float(args.negative_ratio),
        }

    # Prepare output dirs.
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    write_failures = 0
    per_page_counter: dict[tuple[str, str], int] = {}

    # Write generated tiles.
    for row in kept:
        key = (row.split, row.source_page)
        idx = per_page_counter.get(key, 0)
        per_page_counter[key] = idx + 1

        page = page_lookup.get(key)
        if page is None:
            write_failures += 1
            continue

        src_img = cv2.imread(page.source_page_path)
        if src_img is None:
            write_failures += 1
            continue

        tx1, ty1, tx2, ty2 = row.tile_xyxy_in_page
        crop = src_img[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            write_failures += 1
            continue

        tile_id = f"{Path(row.source_page).stem}__{row.split}__{idx:05d}"
        out_img = out_root / "images" / row.split / f"{tile_id}.png"
        out_lbl = out_root / "labels" / row.split / f"{tile_id}.txt"
        cv2.imwrite(str(out_img), crop)

        labels = _clip_and_convert_boxes(
            tile=row.tile_xyxy_in_page,
            gt_boxes=page.gt_boxes,
            gt_ids=row.source_gt_ids_inside_tile,
            min_visibility=row.min_visibility,
        )
        if row.generation_rule.endswith("negative") or row.generation_rule.startswith("hard_negative"):
            labels = []
        _write_label_txt(out_lbl, labels)

        manifest_entries.append(
            {
                "tile_id": tile_id,
                "split": row.split,
                "source_page": row.source_page,
                "source_page_path": row.source_page_path,
                "tile_xyxy_in_page": row.tile_xyxy_in_page,
                "generation_rule": row.generation_rule,
                "source_gt_ids_inside_tile": row.source_gt_ids_inside_tile,
                "context_margin_px": row.context_margin_px,
                "tile_size": row.tile_size,
                "hardness_score": round(float(row.hardness_score), 6),
                "labels_written": len(labels),
                "image_path": str(out_img),
                "label_path": str(out_lbl),
                "extra_meta": row.extra_meta,
            }
        )

    # Import prior non-holdout hard negatives (from existing pool only).
    imported = _load_sparse_non_holdout_negatives(sparse_neg_manifest, limit=int(args.max_imported_non_holdout_negs))
    imported_unmapped_to_split = 0
    imported_written = 0
    for idx, item in enumerate(imported, start=1):
        src_img_path = Path(str(item.get("image_path")))
        source_page = str(item.get("source_page", src_img_path.name))
        split = str(page_to_split.get(source_page, "train"))
        if source_page not in page_to_split:
            imported_unmapped_to_split += 1

        tile_id = f"imported_nonholdout_neg__{split}__{idx:05d}"
        out_img = out_root / "images" / split / f"{tile_id}.png"
        out_lbl = out_root / "labels" / split / f"{tile_id}.txt"
        shutil.copy2(src_img_path, out_img)
        _write_label_txt(out_lbl, [])

        manifest_entries.append(
            {
                "tile_id": tile_id,
                "split": split,
                "source_page": source_page,
                "source_page_path": str(src_img_path),
                "tile_xyxy_in_page": item.get("tile_xyxy_in_page", []),
                "generation_rule": "hard_negative_pool_non_holdout",
                "source_gt_ids_inside_tile": [],
                "context_margin_px": 0,
                "tile_size": _as_int(item.get("tile_xyxy_in_page", [0, 0, 0, 0])[2] if isinstance(item.get("tile_xyxy_in_page"), list) and len(item.get("tile_xyxy_in_page")) == 4 else 0, 0),
                "hardness_score": round(float(item.get("hardness_score", 0.0)), 6),
                "labels_written": 0,
                "image_path": str(out_img),
                "label_path": str(out_lbl),
                "extra_meta": {
                    "source_dataset": item.get("source_dataset"),
                    "source_is_holdout_derived": bool(item.get("source_is_holdout_derived", False)),
                    "reason_selected": item.get("reason_selected"),
                    "fp_bucket_linkage": item.get("fp_bucket_linkage"),
                    "detector_score": item.get("detector_score"),
                },
            }
        )
        imported_written += 1

    # Write dataset.yaml
    dataset_yaml = (
        f"path: {out_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: job_ad\n"
    )
    (out_root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")

    split_summary: dict[str, Any] = {}
    for split in ("train", "val"):
        rows = [e for e in manifest_entries if e["split"] == split]
        pos = [e for e in rows if len(e.get("source_gt_ids_inside_tile", [])) > 0]
        neg = [e for e in rows if len(e.get("source_gt_ids_inside_tile", [])) == 0]
        split_summary[split] = {
            "tiles": len(rows),
            "positive_tiles": len(pos),
            "negative_tiles": len(neg),
            "negative_to_positive_ratio": round((len(neg) / len(pos)) if pos else 0.0, 6),
            "rules": {
                rule: sum(1 for e in rows if e["generation_rule"] == rule)
                for rule in sorted(set(e["generation_rule"] for e in rows))
            },
            "source_pages_covered": len(set(e["source_page"] for e in rows)),
        }

    built_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_root": str(src_root.resolve()),
        "output_dataset_root": str(out_root.resolve()),
        "negative_ratio_cap": float(args.negative_ratio),
        "tile_sizes": tile_sizes,
        "dense_tile_size": dense_tile_size,
        "dense_stride": dense_stride,
        "negative_tile_size": neg_tile_size,
        "sparse_non_holdout_negative_manifest": str(sparse_neg_manifest.resolve()),
        "max_imported_non_holdout_negs": int(args.max_imported_non_holdout_negs),
        "entries": manifest_entries,
        "summary": {
            "total_tiles": len(manifest_entries),
            "write_failures": write_failures,
            "imported_non_holdout_hard_negatives": imported_written,
            "imported_unmapped_to_split_count": imported_unmapped_to_split,
            "split": split_summary,
            "split_balance": split_balance,
        },
    }

    build_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "FP-focused v4 dataset revision from supervised pool only",
        "output_dataset_root": str(out_root.resolve()),
        "dataset_yaml": str((out_root / "dataset.yaml").resolve()),
        "manifest_output": str(manifest_out.resolve()),
        "source_dataset_root": str(src_root.resolve()),
        "total_tiles": len(manifest_entries),
        "split_summary": split_summary,
        "fp_focused_changes_vs_v3": {
            "negative_ratio_cap_increased": {
                "v3": 1.2,
                "v4": float(args.negative_ratio),
            },
            "full_page_anchors": "removed",
            "new_negative_rule": "adjacent_hard_negative",
            "imported_non_holdout_sparse_hard_negatives": imported_written,
            "source_of_imported_negatives": "run_state/sparse_hard_negative_pool_manifest.json entries where source_is_holdout_derived=false",
        },
        "integrity_checks": {
            "page_level_split_preserved": True,
            "train_val_source_page_overlap_count": len(train_names & val_names),
            "hard_negative_tiles_gt_overlap_forced_empty_labels": True,
            "external_holdout_used": False,
            "benchmark_assets_modified": False,
            "test_labels_modified": False,
        },
        "notes": [
            "External different-newspaper-English holdout excluded from dataset construction.",
            "V4 emphasizes harder sparse negatives and trims context-heavy positives to reduce FP inflation.",
        ],
    }

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(built_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_out.write_text(json.dumps(build_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"dataset_root: {out_root}")
    print(f"dataset_yaml: {out_root / 'dataset.yaml'}")
    print(f"manifest: {manifest_out}")
    print(f"build_report: {report_out}")
    print(f"total_tiles: {len(manifest_entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
