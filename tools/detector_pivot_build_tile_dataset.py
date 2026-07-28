#!/usr/bin/env python3
"""Build detector-pivot v2 tile dataset (images + labels + traceability manifest)."""

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build detector pivot v2 YOLO tile dataset")
    parser.add_argument("--source-dataset-root", default="data/yolo_job_ad_pivot_v1", help="Source YOLO dataset root")
    parser.add_argument(
        "--strategy-json",
        default="run_state/detector_pivot_next_dataset_strategy.json",
        help="Strategy JSON path",
    )
    parser.add_argument(
        "--planning-manifest",
        default="run_state/detector_pivot_tile_dataset_manifest_v2.json",
        help="Planning scaffold manifest",
    )
    parser.add_argument(
        "--output-dataset-root",
        default="data/yolo_job_ad_pivot_v2_tiles",
        help="Output YOLO dataset root",
    )
    parser.add_argument(
        "--output-manifest",
        default="run_state/detector_pivot_tile_dataset_manifest_v2_built.json",
        help="Built tile manifest output JSON",
    )
    parser.add_argument(
        "--output-report",
        default="run_state/detector_pivot_tile_dataset_build_report.json",
        help="Build report output JSON",
    )
    parser.add_argument("--seed", type=int, default=20260404, help="Random seed")
    parser.add_argument("--negative-ratio", type=float, default=1.2, help="Negative-to-positive ratio cap per split")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid JSON object: {path}")
    return payload


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clip(v: int, lo: int, hi: int) -> int:
    return min(max(v, lo), hi)


def _iou(a: list[int], b: list[int]) -> float:
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
    # Inverted threshold approximates ink density on newspaper-like backgrounds.
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    ink_density = float(np.mean(bw > 0))
    return 0.55 * edge_density + 0.45 * ink_density


def _dedupe(entries: list[TileCandidate], iou_thr: float = 0.9) -> list[TileCandidate]:
    out: list[TileCandidate] = []
    for row in entries:
        cur = row.tile_xyxy_in_page
        if any(
            (r.source_page == row.source_page and _iou(cur, r.tile_xyxy_in_page) >= iou_thr and r.generation_rule == row.generation_rule)
            for r in out
        ):
            continue
        out.append(row)
    return out


def _build_candidates_for_page(page: PageData, tile_sizes: list[int], neg_tile: int) -> list[TileCandidate]:
    candidates: list[TileCandidate] = []
    page_path = Path(page.source_page_path)
    im = cv2.imread(str(page_path))
    if im is None:
        return candidates

    gt_boxes = page.gt_boxes
    w, h = page.width, page.height

    # Positive/context tiles.
    for gt_idx, box in enumerate(gt_boxes):
        cx = (box.x1 + box.x2) // 2
        cy = (box.y1 + box.y2) // 2
        min_dim = max(1, min(box.w, box.h))
        margins = [
            ("positive_tight_context", max(32, int(round(0.20 * min_dim))), 0.95),
            ("positive_broad_context", max(64, int(round(0.50 * min_dim))), 0.95),
        ]
        # Small-object boost tile.
        box_area_frac = (box.area / float(w * h)) if (w > 0 and h > 0) else 0.0
        if box_area_frac < 0.005:
            margins.append(("positive_small_object_boost", max(48, int(round(0.40 * min_dim))), 0.90))

        for rule, margin, vis_thr in margins:
            target = max(box.w + 2 * margin, box.h + 2 * margin)
            tile_size = _nearest_tile_size(target, tile_sizes)
            tile = _tile_from_center(cx, cy, tile_size, w, h)
            gt_ids = _candidate_gt_ids(tile, gt_boxes, min_visibility=vis_thr)
            if gt_idx not in gt_ids:
                gt_ids.append(gt_idx)
            gt_ids = sorted(set(gt_ids))
            candidates.append(
                TileCandidate(
                    split=page.split,
                    source_page=page.source_page,
                    source_page_path=page.source_page_path,
                    tile_xyxy_in_page=tile,
                    generation_rule=rule,
                    source_gt_ids_inside_tile=gt_ids,
                    context_margin_px=margin,
                    tile_size=tile_size,
                    hardness_score=0.0,
                    min_visibility=vis_thr,
                )
            )

    # Dense-region tiles.
    if len(gt_boxes) >= 10:
        dense_tile = _nearest_tile_size(896, tile_sizes)
        for tile in _iter_sliding_tiles(w=w, h=h, tile=dense_tile, stride=max(1, dense_tile // 2)):
            center_hits = 0
            for g in gt_boxes:
                cx = (g.x1 + g.x2) // 2
                cy = (g.y1 + g.y2) // 2
                if tile[0] <= cx < tile[2] and tile[1] <= cy < tile[3]:
                    center_hits += 1
            if center_hits < 2:
                continue
            gt_ids = _candidate_gt_ids(tile, gt_boxes, min_visibility=0.70)
            if not gt_ids:
                continue
            candidates.append(
                TileCandidate(
                    split=page.split,
                    source_page=page.source_page,
                    source_page_path=page.source_page_path,
                    tile_xyxy_in_page=tile,
                    generation_rule="dense_region_tile",
                    source_gt_ids_inside_tile=gt_ids,
                    context_margin_px=0,
                    tile_size=dense_tile,
                    hardness_score=0.0,
                    min_visibility=0.70,
                )
            )

    # Hard negatives.
    neg_stride = max(1, neg_tile // 2)
    neg_candidates: list[TileCandidate] = []
    for tile in _iter_sliding_tiles(w=w, h=h, tile=neg_tile, stride=neg_stride):
        overlaps = [_iou(tile, [g.x1, g.y1, g.x2, g.y2]) for g in gt_boxes]
        max_overlap = max(overlaps) if overlaps else 0.0
        if max_overlap >= 0.02:
            continue
        # Distance gate from nearest GT center.
        if gt_boxes:
            cx = (tile[0] + tile[2]) // 2
            cy = (tile[1] + tile[3]) // 2
            min_center_dist = min(
                math.hypot(cx - (g.x1 + g.x2) / 2.0, cy - (g.y1 + g.y2) / 2.0)
                for g in gt_boxes
            )
            if min_center_dist < 32:
                continue
        score = _text_hardness_score(im, tile)
        neg_candidates.append(
            TileCandidate(
                split=page.split,
                source_page=page.source_page,
                source_page_path=page.source_page_path,
                tile_xyxy_in_page=tile,
                generation_rule="hard_negative_text_dense",
                source_gt_ids_inside_tile=[],
                context_margin_px=0,
                tile_size=neg_tile,
                hardness_score=score,
                min_visibility=1.0,
            )
        )

    neg_candidates.sort(key=lambda r: r.hardness_score, reverse=True)
    if gt_boxes:
        keep_n = min(10, max(2, len(gt_boxes) // 2))
    else:
        keep_n = min(6, max(2, len(neg_candidates)))
    candidates.extend(neg_candidates[:keep_n])

    # Optional small retained full-page anchors (positive pages only).
    if gt_boxes:
        candidates.append(
            TileCandidate(
                split=page.split,
                source_page=page.source_page,
                source_page_path=page.source_page_path,
                tile_xyxy_in_page=[0, 0, w, h],
                generation_rule="full_page_anchor",
                source_gt_ids_inside_tile=list(range(len(gt_boxes))),
                context_margin_px=0,
                tile_size=min(w, h),
                hardness_score=0.0,
                min_visibility=0.95,
            )
        )

    return _dedupe(candidates, iou_thr=0.9)


def _clip_and_convert_boxes(
    tile: list[int],
    gt_boxes: list[Box],
    gt_ids: list[int],
    min_visibility: float,
) -> list[list[float]]:
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


def main() -> int:
    args = _parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    src_root = _resolve(args.source_dataset_root)
    strategy_path = _resolve(args.strategy_json)
    planning_manifest_path = _resolve(args.planning_manifest)
    out_root = _resolve(args.output_dataset_root)
    manifest_out = _resolve(args.output_manifest)
    report_out = _resolve(args.output_report)

    strategy = _load_json(strategy_path)
    planning = _load_json(planning_manifest_path)

    # Resolve tile sizing from strategy/planning with safe fallbacks.
    tile_sizes = [640, 896, 1024]
    try:
        tile_sizes = [int(x) for x in strategy["dataset_design"]["positive_tile_rules"]["base_tile_sizes"]]
    except Exception:
        try:
            tile_sizes = [int(x) for x in planning.get("tile_sizes", tile_sizes)]
        except Exception:
            pass
    tile_sizes = sorted(set([max(256, int(x)) for x in tile_sizes]))

    dense_tile_size = _as_int(strategy.get("dataset_design", {}).get("dense_page_rules", {}).get("sliding_window", {}).get("tile_size"), 896)
    dense_stride = _as_int(strategy.get("dataset_design", {}).get("dense_page_rules", {}).get("sliding_window", {}).get("stride"), dense_tile_size // 2)
    neg_tile = _as_int(planning.get("negative_tile_size"), 896)

    pages = _load_pages(src_root)
    split_pages = {"train": [p for p in pages if p.split == "train"], "val": [p for p in pages if p.split == "val"]}

    # Page-level split leak check.
    train_names = {p.source_page for p in split_pages["train"]}
    val_names = {p.source_page for p in split_pages["val"]}
    if train_names & val_names:
        raise RuntimeError("Split leakage detected: same source page exists in both train and val")

    all_candidates: list[TileCandidate] = []
    for page in pages:
        page_candidates = _build_candidates_for_page(page, tile_sizes=tile_sizes, neg_tile=neg_tile)

        # Add dense sliding tiles explicitly requested by strategy.
        if len(page.gt_boxes) >= 10:
            for tile in _iter_sliding_tiles(page.width, page.height, dense_tile_size, dense_stride):
                gt_ids = _candidate_gt_ids(tile, page.gt_boxes, min_visibility=0.70)
                if len(gt_ids) >= 2:
                    page_candidates.append(
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
                        )
                    )

        page_candidates = _dedupe(page_candidates, iou_thr=0.9)
        all_candidates.extend(page_candidates)

    # Balance negatives per split so easy/background negatives do not dominate.
    kept: list[TileCandidate] = []
    for split in ("train", "val"):
        rows = [r for r in all_candidates if r.split == split]
        pos = [r for r in rows if r.source_gt_ids_inside_tile]
        neg = [r for r in rows if not r.source_gt_ids_inside_tile]
        neg.sort(key=lambda r: r.hardness_score, reverse=True)
        max_neg = int(max(1, round(len(pos) * float(args.negative_ratio)))) if pos else min(40, len(neg))
        kept.extend(pos)
        kept.extend(neg[:max_neg])

    # Optional full-page anchor cap (<=10% by count) from strategy.
    max_anchor_frac = float(strategy.get("dataset_design", {}).get("optional_full_page_anchor_subset", {}).get("max_fraction_of_training_samples", 0.1))
    anchors = [r for r in kept if r.generation_rule == "full_page_anchor"]
    non_anchors = [r for r in kept if r.generation_rule != "full_page_anchor"]
    max_anchors = int(math.floor(max_anchor_frac * max(1, len(non_anchors))))
    if len(anchors) > max_anchors:
        anchors.sort(key=lambda r: len(r.source_gt_ids_inside_tile), reverse=True)
        anchors = anchors[:max_anchors]
    kept = non_anchors + anchors

    # Prepare output dirs.
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Build page lookup for fast GT access.
    page_lookup = {(p.split, p.source_page): p for p in pages}

    manifest_entries: list[dict[str, Any]] = []
    write_failures = 0

    per_page_counter: dict[tuple[str, str], int] = {}
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

        # Hard negatives must remain GT-free.
        if row.generation_rule.startswith("hard_negative"):
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
            }
        )

    # Write dataset.yaml
    dataset_yaml = (
        f"path: {out_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: job_ad\n"
    )
    (out_root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")

    # Summaries.
    split_summary: dict[str, Any] = {}
    for split in ("train", "val"):
        rows = [e for e in manifest_entries if e["split"] == split]
        pos = [e for e in rows if len(e["source_gt_ids_inside_tile"]) > 0]
        neg = [e for e in rows if len(e["source_gt_ids_inside_tile"]) == 0]
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
        "strategy_json": str(strategy_path.resolve()),
        "planning_manifest": str(planning_manifest_path.resolve()),
        "output_dataset_root": str(out_root.resolve()),
        "tile_sizes": tile_sizes,
        "dense_tile_size": dense_tile_size,
        "dense_stride": dense_stride,
        "negative_tile_size": neg_tile,
        "negative_ratio_cap": float(args.negative_ratio),
        "entries": manifest_entries,
        "summary": {
            "total_tiles": len(manifest_entries),
            "write_failures": write_failures,
            "split": split_summary,
        },
    }

    build_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dataset_root": str(out_root.resolve()),
        "dataset_yaml": str((out_root / "dataset.yaml").resolve()),
        "manifest_output": str(manifest_out.resolve()),
        "total_tiles": len(manifest_entries),
        "split_summary": split_summary,
        "integrity_checks": {
            "page_level_split_preserved": True,
            "train_val_source_page_overlap_count": len(train_names & val_names),
            "hard_negative_tiles_gt_overlap_forced_empty_labels": True,
            "benchmark_assets_modified": False,
            "test_labels_modified": False,
        },
        "notes": [
            "Tile generation includes positive context tiles, dense-region sliding tiles, hard negatives, and capped full-page anchors.",
            "Negative sampling is hardness-ranked and ratio-capped so easy blank negatives do not dominate.",
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
