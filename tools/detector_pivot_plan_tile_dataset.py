#!/usr/bin/env python3
"""Create a traceable tile manifest for the next detector-pivot dataset (planning scaffold)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BBox:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan tile dataset manifest for detector pivot")
    parser.add_argument("--dataset-root", default="data/yolo_job_ad_pivot_v1", help="YOLO dataset root")
    parser.add_argument(
        "--tile-sizes",
        nargs="+",
        type=int,
        default=[640, 896, 1024],
        help="Tile sizes to consider",
    )
    parser.add_argument("--negative-tile-size", type=int, default=896, help="Tile size for hard-negative planning")
    parser.add_argument(
        "--output",
        default="run_state/detector_pivot_tile_dataset_manifest_v2.json",
        help="Manifest JSON output path",
    )
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _clip(v: int, lo: int, hi: int) -> int:
    return min(max(v, lo), hi)


def _tile_from_center(cx: int, cy: int, tile: int, w: int, h: int) -> list[int]:
    half = tile // 2
    x1 = _clip(cx - half, 0, max(0, w - tile))
    y1 = _clip(cy - half, 0, max(0, h - tile))
    x2 = min(w, x1 + tile)
    y2 = min(h, y1 + tile)
    return [x1, y1, x2, y2]


def _parse_yolo_labels(label_path: Path, image_w: int, image_h: int) -> list[BBox]:
    text = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else ""
    if not text:
        return []
    boxes: list[BBox] = []
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
        x1 = int(round((xc_f - bw_f / 2.0) * image_w))
        y1 = int(round((yc_f - bh_f / 2.0) * image_h))
        x2 = int(round((xc_f + bw_f / 2.0) * image_w))
        y2 = int(round((yc_f + bh_f / 2.0) * image_h))
        x1 = _clip(x1, 0, image_w - 1)
        y1 = _clip(y1, 0, image_h - 1)
        x2 = _clip(x2, x1 + 1, image_w)
        y2 = _clip(y2, y1 + 1, image_h)
        boxes.append(BBox(x1, y1, x2, y2))
    return boxes


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
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    denom = aa + bb - inter
    return float(inter / denom) if denom > 0 else 0.0


def _dedupe_tiles(tiles: list[dict[str, Any]], iou_thr: float = 0.9) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for t in tiles:
        cur = t.get("tile_xyxy", [0, 0, 0, 0])
        if any(_iou(cur, k.get("tile_xyxy", [0, 0, 0, 0])) >= iou_thr for k in kept):
            continue
        kept.append(t)
    return kept


def main() -> int:
    args = parse_args()
    ds_root = _resolve(args.dataset_root)
    out_path = _resolve(args.output)

    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(ds_root.resolve()),
        "tile_sizes": [int(x) for x in args.tile_sizes],
        "negative_tile_size": int(args.negative_tile_size),
        "note": "Planning scaffold only: this creates a tile manifest, not image/label files.",
        "entries": [],
        "summary": {},
    }

    for split in ("train", "val"):
        img_dir = ds_root / "images" / split
        lbl_dir = ds_root / "labels" / split
        for img_path in sorted(img_dir.glob("*.png")):
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = int(im.shape[0]), int(im.shape[1])
            label_path = lbl_dir / f"{img_path.stem}.txt"
            boxes = _parse_yolo_labels(label_path, image_w=w, image_h=h)

            page_tiles: list[dict[str, Any]] = []
            if boxes:
                for gt_idx, b in enumerate(boxes):
                    cx = (b.x1 + b.x2) // 2
                    cy = (b.y1 + b.y2) // 2
                    min_dim = max(1, min(b.w, b.h))
                    margins = {
                        "tight": max(32, int(round(0.20 * min_dim))),
                        "broad": max(64, int(round(0.50 * min_dim))),
                    }
                    for rule_name, margin in margins.items():
                        expanded_w = b.w + 2 * margin
                        expanded_h = b.h + 2 * margin
                        target_tile = min([int(t) for t in args.tile_sizes], key=lambda t: abs(t - max(expanded_w, expanded_h)))
                        tile = _tile_from_center(cx, cy, target_tile, w, h)
                        page_tiles.append(
                            {
                                "source_split": split,
                                "source_page": img_path.name,
                                "tile_xyxy": tile,
                                "tile_size": target_tile,
                                "generation_rule": f"positive_{rule_name}",
                                "source_gt_indices": [gt_idx],
                                "context_margin_px": margin,
                            }
                        )

                # One hard-negative candidate from a corner if far from all GT.
                neg_size = int(args.negative_tile_size)
                neg_candidate = [max(0, w - neg_size), max(0, h - neg_size), w, h]
                if all(_iou(neg_candidate, [b.x1, b.y1, b.x2, b.y2]) < 0.02 for b in boxes):
                    page_tiles.append(
                        {
                            "source_split": split,
                            "source_page": img_path.name,
                            "tile_xyxy": neg_candidate,
                            "tile_size": neg_size,
                            "generation_rule": "hard_negative_corner",
                            "source_gt_indices": [],
                            "context_margin_px": 0,
                        }
                    )
            else:
                # Background page: include a center and corner negative tile.
                neg_size = int(args.negative_tile_size)
                page_tiles.extend(
                    [
                        {
                            "source_split": split,
                            "source_page": img_path.name,
                            "tile_xyxy": _tile_from_center(w // 2, h // 2, neg_size, w, h),
                            "tile_size": neg_size,
                            "generation_rule": "hard_negative_center",
                            "source_gt_indices": [],
                            "context_margin_px": 0,
                        },
                        {
                            "source_split": split,
                            "source_page": img_path.name,
                            "tile_xyxy": [0, 0, min(w, neg_size), min(h, neg_size)],
                            "tile_size": neg_size,
                            "generation_rule": "hard_negative_corner",
                            "source_gt_indices": [],
                            "context_margin_px": 0,
                        },
                    ]
                )

            deduped = _dedupe_tiles(page_tiles, iou_thr=0.9)
            for i, row in enumerate(deduped):
                row["tile_id"] = f"{img_path.stem}__{split}__t{i:04d}"
            manifest["entries"].extend(deduped)

    entries = manifest["entries"]
    pos = [e for e in entries if e.get("source_gt_indices")]
    neg = [e for e in entries if not e.get("source_gt_indices")]

    manifest["summary"] = {
        "total_tiles": len(entries),
        "positive_tiles": len(pos),
        "negative_tiles": len(neg),
        "negative_to_positive_ratio": round((len(neg) / len(pos)) if pos else 0.0, 6),
        "rules": {
            "positive_tight": sum(1 for e in entries if e.get("generation_rule") == "positive_tight"),
            "positive_broad": sum(1 for e in entries if e.get("generation_rule") == "positive_broad"),
            "hard_negative_center": sum(1 for e in entries if e.get("generation_rule") == "hard_negative_center"),
            "hard_negative_corner": sum(1 for e in entries if e.get("generation_rule") == "hard_negative_corner"),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"manifest: {out_path}")
    print(f"total_tiles: {manifest['summary']['total_tiles']}")
    print(f"positive_tiles: {manifest['summary']['positive_tiles']}")
    print(f"negative_tiles: {manifest['summary']['negative_tiles']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
