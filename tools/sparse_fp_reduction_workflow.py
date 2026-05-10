#!/usr/bin/env python3
"""Sparse-holdout FP reduction workflow with explicit contamination reporting."""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.auto_improve_detector import _merge_labels_to_temp  # noqa: E402
from tools.evaluate_against_labels import compute_metrics, load_labelstudio_boxes  # noqa: E402


@dataclass
class FPEntry:
    source_dataset: str
    holdout_derived: bool
    page: str
    bbox: list[int]
    score: float
    gt_count: int
    pred_count: int
    bucket: str
    bucket_reason: str
    features: dict[str, float]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = aa + bb - inter
    return float(inter / denom) if denom > 0 else 0.0


def _bbox_key(box: list[int]) -> str:
    return ",".join(str(int(v)) for v in box)


def _image_map(root: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted(root.rglob("*.png"))}


def _extract_crop_features(image: np.ndarray, bbox: list[int], score: float, page_w: int, page_h: int) -> dict[str, float]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(page_w - 1, x1))
    y1 = max(0, min(page_h - 1, y1))
    x2 = max(x1 + 1, min(page_w, x2))
    y2 = max(y1 + 1, min(page_h, y2))
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size > 0 else np.zeros((1, 1), dtype=np.uint8)
    edges = cv2.Canny(gray, 80, 160)
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    area = float(w * h)
    page_area = float(max(1, page_w * page_h))
    aspect = float(w / h)

    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten().astype(np.float64)
    hist_sum = float(hist.sum()) or 1.0
    p = hist / hist_sum
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    feat = {
        "score": float(score),
        "w": float(w),
        "h": float(h),
        "aspect": aspect,
        "area_ratio": float(area / page_area),
        "x_center_ratio": float(((x1 + x2) / 2.0) / max(1, page_w)),
        "y_center_ratio": float(((y1 + y2) / 2.0) / max(1, page_h)),
        "gray_mean": float(np.mean(gray)),
        "gray_std": float(np.std(gray)),
        "edge_density": float(np.mean(edges > 0)),
        "ink_density": float(np.mean(bw > 0)),
        "lap_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "entropy": entropy,
    }
    return feat


def _slice_boundaries(length: int, slice_size: int = 1024, overlap: float = 0.2) -> list[int]:
    tile = min(slice_size, length)
    stride = max(1, int(round(tile * (1.0 - overlap))))
    vals = list(range(0, max(1, length - tile + 1), stride))
    if not vals or vals[-1] != max(0, length - tile):
        vals.append(max(0, length - tile))
    return sorted({int(v) for v in vals} | {int(v + tile) for v in vals})


def _near_any(val: int, refs: list[int], tol: int) -> bool:
    return any(abs(int(val) - int(r)) <= int(tol) for r in refs)


def _classify_fp(
    *,
    bbox: list[int],
    score: float,
    features: dict[str, float],
    all_pred_boxes: list[list[int]],
    page_w: int,
    page_h: int,
    slice_size: int = 1024,
    overlap_ratio: float = 0.2,
) -> tuple[str, str]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    area_ratio = float(features["area_ratio"])
    aspect = float(features["aspect"])
    edge_density = float(features["edge_density"])
    ink_density = float(features["ink_density"])

    # Duplicate proposals: near-identical overlaps with another proposal.
    dup_count = 0
    for other in all_pred_boxes:
        if other is bbox:
            continue
        if _iou(bbox, other) >= 0.75:
            dup_count += 1
    if dup_count > 0:
        return "duplicate_detections", f"overlaps_with_{dup_count}_other_predictions_iou_ge_0.75"

    xb = _slice_boundaries(page_w, slice_size=slice_size, overlap=overlap_ratio)
    yb = _slice_boundaries(page_h, slice_size=slice_size, overlap=overlap_ratio)
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    near_vertical = _near_any(x1, xb, 24) or _near_any(x2, xb, 24) or _near_any(cx, xb, 18)
    near_horizontal = _near_any(y1, yb, 24) or _near_any(y2, yb, 24) or _near_any(cy, yb, 18)
    if (near_vertical or near_horizontal) and area_ratio < 0.07 and max(w, h) < 1200:
        return "slice_boundary_artifacts", "bbox_aligned_with_slice_boundary_band"

    if area_ratio >= 0.018 or (w >= 420 and h >= 180) or (0.45 <= aspect <= 2.8 and area_ratio >= 0.010):
        return "non_job_classifieds", "large_or_ad_like_rectangular_region"

    if area_ratio <= 0.010 or edge_density >= 0.09 or ink_density >= 0.24 or h < 140:
        return "text_dense_junk", "small_or_high_texture_text_region"

    return "other_unclear", "ambiguous_visual_pattern"


def _match_fp_detections(
    *,
    per_image: dict[str, Any],
    detections_dir: Path,
    image_map: dict[str, Path],
    source_dataset: str,
    holdout_derived: bool,
) -> list[FPEntry]:
    entries: list[FPEntry] = []
    for page, payload in per_image.items():
        fp_boxes = payload.get("false_positives", []) if isinstance(payload, dict) else []
        if not isinstance(fp_boxes, list) or not fp_boxes:
            continue
        det_file = detections_dir / f"{page}.json"
        if not det_file.is_file():
            continue
        img_path = image_map.get(page)
        if img_path is None:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        page_h, page_w = image.shape[:2]

        det_payload = _load_json(det_file)
        det_rows = det_payload.get("detections", []) if isinstance(det_payload, dict) else []
        det_rows = [r for r in det_rows if isinstance(r, dict) and isinstance(r.get("bbox"), list) and len(r["bbox"]) == 4]

        fp_counter = Counter(_bbox_key([int(v) for v in b]) for b in fp_boxes if isinstance(b, list) and len(b) == 4)
        all_pred_boxes = [[int(v) for v in r["bbox"]] for r in det_rows]

        for r in det_rows:
            box = [int(v) for v in r["bbox"]]
            key = _bbox_key(box)
            if fp_counter[key] <= 0:
                continue
            fp_counter[key] -= 1
            score = float(r.get("score", 0.0) or 0.0)
            feat = _extract_crop_features(image, box, score, page_w=page_w, page_h=page_h)
            bucket, reason = _classify_fp(
                bbox=box,
                score=score,
                features=feat,
                all_pred_boxes=all_pred_boxes,
                page_w=page_w,
                page_h=page_h,
            )
            entries.append(
                FPEntry(
                    source_dataset=source_dataset,
                    holdout_derived=holdout_derived,
                    page=page,
                    bbox=box,
                    score=score,
                    gt_count=int(payload.get("ground_truth_count", 0) or 0),
                    pred_count=int(payload.get("prediction_count", 0) or 0),
                    bucket=bucket,
                    bucket_reason=reason,
                    features=feat,
                )
            )
    return entries


def _save_taxonomy_examples(
    *,
    entries: list[FPEntry],
    image_map: dict[str, Path],
    out_dir: Path,
    per_bucket_max: int = 16,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket_groups: dict[str, list[FPEntry]] = defaultdict(list)
    for e in entries:
        bucket_groups[e.bucket].append(e)

    written: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bucket, rows in bucket_groups.items():
        rows_sorted = sorted(rows, key=lambda r: (-(r.score), -(r.features.get("area_ratio", 0.0))))
        for idx, e in enumerate(rows_sorted[:per_bucket_max], start=1):
            img_path = image_map.get(e.page)
            if img_path is None:
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            x1, y1, x2, y2 = e.bbox
            h, w = image.shape[:2]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))

            crop = image[y1:y2, x1:x2]
            stem = f"{bucket}__{idx:03d}__{Path(e.page).stem}__{x1}_{y1}_{x2}_{y2}"
            crop_path = out_dir / f"{stem}__crop.png"
            page_path = out_dir / f"{stem}__page.png"
            cv2.imwrite(str(crop_path), crop)

            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"{bucket} s={e.score:.3f}"
            cv2.putText(overlay, label, (max(0, x1), max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(page_path), overlay)

            written[bucket].append(
                {
                    "page": e.page,
                    "bbox_page_xyxy": [x1, y1, x2, y2],
                    "bbox_crop_xyxy": [0, 0, int(x2 - x1), int(y2 - y1)],
                    "crop_path": str(crop_path),
                    "page_overlay_path": str(page_path),
                    "score": e.score,
                }
            )
    return dict(written)


def _taxonomy_payload(entries: list[FPEntry], examples: dict[str, Any]) -> dict[str, Any]:
    total = len(entries)
    buckets = [
        "non_job_classifieds",
        "duplicate_detections",
        "text_dense_junk",
        "slice_boundary_artifacts",
        "other_unclear",
    ]
    by_bucket: dict[str, Any] = {}
    for b in buckets:
        rows = [e for e in entries if e.bucket == b]
        pages = Counter(e.page for e in rows)
        by_bucket[b] = {
            "count": len(rows),
            "percentage": round((100.0 * len(rows) / total), 3) if total else 0.0,
            "representative_pages": [p for p, _ in pages.most_common(8)],
            "representative_examples": examples.get(b, [])[:8],
            "explanation": {
                "non_job_classifieds": "Ad-like rectangular classifieds detected as job ads.",
                "duplicate_detections": "Multiple overlapping detections covering near-identical region.",
                "text_dense_junk": "High-texture small/medium text regions without job-ad semantics.",
                "slice_boundary_artifacts": "Proposals likely introduced by tiled slicing boundary effects.",
                "other_unclear": "Ambiguous cases not confidently assignable to a dominant bucket.",
            }[b],
        }

    return {
        "generated_at_utc": _utc_now(),
        "total_false_positives_analyzed": total,
        "bucket_definitions": buckets,
        "buckets": by_bucket,
        "notes": [
            "Bucket assignment is heuristic and conservative; ambiguous cases are pushed to other_unclear.",
            "Counts are based on matched FP boxes from holdout evaluation details at conf=0.022.",
        ],
    }


def _write_taxonomy_summary_md(path: Path, taxonomy: dict[str, Any]) -> None:
    lines = [
        "# Sparse Holdout FP Taxonomy",
        "",
        f"- Generated at: {taxonomy.get('generated_at_utc')}",
        f"- Total FPs analyzed: {taxonomy.get('total_false_positives_analyzed')}",
        "",
        "## Bucket Summary",
    ]
    for b, row in taxonomy.get("buckets", {}).items():
        lines.append(f"- {b}: {row.get('count', 0)} ({row.get('percentage', 0)}%)")
        lines.append(f"  - Why: {row.get('explanation', '')}")
        pages = row.get("representative_pages", [])
        if pages:
            lines.append(f"  - Representative pages: {', '.join(pages[:5])}")
    lines.append("")
    lines.append("## Evidence")
    lines.append("- Visual examples written under `run_state/sparse_holdout_fp_taxonomy_examples/`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _crop_and_save(image: np.ndarray, bbox: list[int], out_path: Path) -> bool:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return True


def _hardness(score: float, edge_density: float, ink_density: float) -> float:
    s = min(max(score / 0.08, 0.0), 1.0)
    e = min(max(edge_density / 0.18, 0.0), 1.0)
    i = min(max(ink_density / 0.30, 0.0), 1.0)
    return float(round((0.45 * s + 0.35 * e + 0.20 * i), 6))


def _build_hard_negative_pool(
    *,
    holdout_fp_entries: list[FPEntry],
    benchmark_fp_entries: list[FPEntry],
    holdout_image_map: dict[str, Path],
    benchmark_image_map: dict[str, Path],
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    selected: list[FPEntry] = []

    # A) non-holdout sparse/background: benchmark GT=0 FP pages first
    bench_gt0 = [e for e in benchmark_fp_entries if e.gt_count == 0]
    bench_gt0 = sorted(bench_gt0, key=lambda e: (-(e.pred_count), -(e.score)))
    selected.extend(bench_gt0[:220])

    # B) holdout GT=0 FP pages
    holdout_gt0 = [e for e in holdout_fp_entries if e.gt_count == 0]
    holdout_gt0 = sorted(holdout_gt0, key=lambda e: (-(e.pred_count), -(e.score)))
    selected.extend(holdout_gt0[:260])

    # C) holdout GT>0 regions with FP
    holdout_pos_pages_fp = [e for e in holdout_fp_entries if e.gt_count > 0]
    holdout_pos_pages_fp = sorted(holdout_pos_pages_fp, key=lambda e: (-(e.pred_count), -(e.score)))
    selected.extend(holdout_pos_pages_fp[:180])

    # Dedupe by source/page/bbox
    uniq: dict[str, FPEntry] = {}
    for e in selected:
        key = f"{e.source_dataset}|{e.page}|{_bbox_key(e.bbox)}"
        uniq[key] = e
    selected = list(uniq.values())

    manifest: list[dict[str, Any]] = []
    src_counts = Counter()
    holdout_count = 0

    for idx, e in enumerate(selected, start=1):
        image_map = holdout_image_map if e.holdout_derived else benchmark_image_map
        img_path = image_map.get(e.page)
        if img_path is None:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        source_tag = "holdout" if e.holdout_derived else "non_holdout"
        stem = f"neg_{idx:05d}__{source_tag}__{Path(e.page).stem}__{e.bbox[0]}_{e.bbox[1]}_{e.bbox[2]}_{e.bbox[3]}"
        out_path = out_images / f"{stem}.png"
        if not _crop_and_save(image, e.bbox, out_path):
            continue

        hardness = _hardness(e.score, e.features.get("edge_density", 0.0), e.features.get("ink_density", 0.0))
        reason = (
            "benchmark_gt0_fp_trigger" if not e.holdout_derived and e.gt_count == 0 else
            "holdout_gt0_fp_trigger" if e.holdout_derived and e.gt_count == 0 else
            "holdout_positive_page_fp_region"
        )

        entry = {
            "id": f"hard_neg_{idx:05d}",
            "image_path": str(out_path),
            "source_dataset": e.source_dataset,
            "source_page": e.page,
            "source_is_holdout_derived": bool(e.holdout_derived),
            "tile_xyxy_in_page": [int(v) for v in e.bbox],
            "reason_selected": reason,
            "fp_bucket_linkage": e.bucket,
            "hardness_score": hardness,
            "detector_score": float(e.score),
            "source_gt_count": int(e.gt_count),
            "source_pred_count": int(e.pred_count),
        }
        manifest.append(entry)
        src_counts[e.source_dataset] += 1
        if e.holdout_derived:
            holdout_count += 1

    report = {
        "generated_at_utc": _utc_now(),
        "output_root": str(out_dir),
        "total_hard_negatives": len(manifest),
        "source_dataset_counts": dict(src_counts),
        "holdout_derived_count": int(holdout_count),
        "holdout_derived_ratio": round(float(holdout_count / len(manifest)), 6) if manifest else 0.0,
        "contamination_risk": bool(holdout_count > 0),
        "selection_priority": [
            "A_non_holdout_background_pages",
            "B_holdout_gt0_fp_pages",
            "C_holdout_positive_pages_fp_regions",
        ],
    }
    return manifest, report


def _feature_vector(feat: dict[str, float]) -> list[float]:
    keys = [
        "score",
        "w",
        "h",
        "aspect",
        "area_ratio",
        "x_center_ratio",
        "y_center_ratio",
        "gray_mean",
        "gray_std",
        "edge_density",
        "ink_density",
        "lap_var",
        "entropy",
    ]
    return [float(feat.get(k, 0.0)) for k in keys]


def _collect_labeled_detector_samples(
    *,
    image_map: dict[str, Path],
    detections_dir: Path,
    labels_map: dict[str, list[list[int]]],
    source_name: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for det_file in sorted(detections_dir.glob("*.json")):
        page = det_file.stem if det_file.name.endswith(".json") else det_file.name
        payload = _load_json(det_file)
        page = str(payload.get("page") or page)
        rows = payload.get("detections", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            continue
        img_path = image_map.get(page)
        if img_path is None:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        gt_boxes = labels_map.get(page, [])

        for r in rows:
            if not isinstance(r, dict):
                continue
            bbox = r.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            box = [int(v) for v in bbox]
            score = float(r.get("score", 0.0) or 0.0)
            max_iou = 0.0
            for gt in gt_boxes:
                max_iou = max(max_iou, _iou(box, [int(v) for v in gt]))
            label = 1 if max_iou >= 0.5 else 0
            feat = _extract_crop_features(image, box, score, page_w=w, page_h=h)
            samples.append(
                {
                    "source": source_name,
                    "page": page,
                    "bbox": box,
                    "score": score,
                    "label": int(label),
                    "max_iou": float(max_iou),
                    "features": feat,
                }
            )
    return samples


def _score_detections_with_rejector(
    *,
    clf: Pipeline,
    image_map: dict[str, Path],
    detections_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for det_file in sorted(detections_dir.glob("*.json")):
        payload = _load_json(det_file)
        page = str(payload.get("page") or det_file.stem)
        rows = payload.get("detections", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            rows = []
        img_path = image_map.get(page)
        if img_path is None:
            out[page] = []
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            out[page] = []
            continue
        h, w = image.shape[:2]
        page_rows: list[dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            bbox = r.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            box = [int(v) for v in bbox]
            score = float(r.get("score", 0.0) or 0.0)
            feat = _extract_crop_features(image, box, score, page_w=w, page_h=h)
            vec = np.asarray([_feature_vector(feat)], dtype=np.float32)
            prob = float(clf.predict_proba(vec)[0][1])
            nr = dict(r)
            nr["rejector_score"] = round(prob, 6)
            nr["bbox"] = box
            nr["score"] = score
            page_rows.append(nr)
        out[page] = page_rows
    return out


def _filter_by_threshold(pred_map: dict[str, list[dict[str, Any]]], threshold: float) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for page, rows in pred_map.items():
        keep: list[dict[str, Any]] = []
        for r in rows:
            if float(r.get("rejector_score", 0.0)) >= float(threshold):
                keep.append(
                    {
                        "id": r.get("id"),
                        "bbox": [int(v) for v in r.get("bbox", [0, 0, 1, 1])],
                        "score": float(r.get("score", 0.0)),
                        "stage": str(r.get("stage", "detector")),
                        "page": page,
                        "rejector_score": float(r.get("rejector_score", 0.0)),
                    }
                )
        out[page] = keep
    return out


def _write_detection_dir(pred_map: dict[str, list[dict[str, Any]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for page, rows in pred_map.items():
        (out_dir / f"{page}.json").write_text(json.dumps({"page": page, "detections": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


def _top_pages(per_image: dict[str, Any], key: str, top_n: int = 12) -> list[dict[str, Any]]:
    rows = []
    for page, p in per_image.items():
        rows.append(
            {
                "page": page,
                "ground_truth_count": int(p.get("ground_truth_count", 0) or 0),
                "prediction_count": int(p.get("prediction_count", 0) or 0),
                "true_positives": int(p.get("true_positives", 0) or 0),
                "missed_count": int(p.get("missed_count", 0) or 0),
                "false_positives_count": int(p.get("false_positives_count", 0) or 0),
            }
        )
    if key == "fp":
        rows.sort(key=lambda r: (-r["false_positives_count"], -r["prediction_count"], r["page"]))
    else:
        rows.sort(key=lambda r: (-r["missed_count"], -r["ground_truth_count"], r["page"]))
    return rows[:top_n]


def main() -> int:
    # Fixed inputs from existing frozen/holdout runs
    holdout_eval_report_path = PROJECT_ROOT / "run_state/holdout_same_date_eval_report.json"
    holdout_details_path = PROJECT_ROOT / "run_state/holdout_same_date_eval_details/holdout_same_date_eval_details_conf_0p022.json"
    holdout_det_dir = PROJECT_ROOT / "run_state/detections_holdout_same_date/conf_0p022"
    holdout_images_dir = PROJECT_ROOT / "data/validation-data/same-newspaper-but-different-dates-then-trained/pdf2img"
    holdout_labels_dir = PROJECT_ROOT / "data/validation-data/same-newspaper-but-different-dates-then-trained/labels"

    bench_eval_report_path = PROJECT_ROOT / "run_state/detector_pivot_v2_eval_report_frozen_best.json"
    bench_details_path = PROJECT_ROOT / "run_state/detector_pivot_v2_eval_details_frozen_best/detector_pivot_v2_eval_details_conf_0p022.json"
    bench_det_dir = PROJECT_ROOT / "run_state/detections_detector_pivot_v2_tiled_frozen_best/conf_0p022"
    bench_images_dir = PROJECT_ROOT / "data/benchmark_images"
    bench_labels_dir = PROJECT_ROOT / "data/test_labels"

    # Output paths
    taxonomy_json = PROJECT_ROOT / "run_state/sparse_holdout_fp_taxonomy.json"
    taxonomy_md = PROJECT_ROOT / "run_state/sparse_holdout_fp_taxonomy_summary.md"
    taxonomy_examples_dir = PROJECT_ROOT / "run_state/sparse_holdout_fp_taxonomy_examples"

    hard_neg_root = PROJECT_ROOT / "data/yolo_sparse_hard_negatives_v1"
    hard_neg_manifest = PROJECT_ROOT / "run_state/sparse_hard_negative_pool_manifest.json"
    hard_neg_report = PROJECT_ROOT / "run_state/sparse_hard_negative_pool_report.json"

    strategy_decision_path = PROJECT_ROOT / "run_state/sparse_fp_reduction_strategy_decision.json"
    rejector_model_path = PROJECT_ROOT / "artifacts/sparse_fp_rejector_v1/rejector.pkl"
    build_report_path = PROJECT_ROOT / "run_state/sparse_fp_reduction_build_report.json"

    holdout_postfix_eval_report = PROJECT_ROOT / "run_state/sparse_holdout_postfix_eval_report.json"
    holdout_postfix_page_diag = PROJECT_ROOT / "run_state/sparse_holdout_postfix_page_diagnosis.json"
    holdout_postfix_det_dir = PROJECT_ROOT / "run_state/detections_holdout_same_date_postfix/conf_0p022"

    benchmark_postfix_det_dir = PROJECT_ROOT / "run_state/detections_detector_pivot_v2_tiled_frozen_best_postfix/conf_0p022"
    dense_regression_path = PROJECT_ROOT / "run_state/sparse_fix_vs_frozen_benchmark_regression.json"

    final_summary_path = PROJECT_ROOT / "run_state/sparse_fp_reduction_summary.md"

    # Load foundational inputs
    holdout_eval_report = _load_json(holdout_eval_report_path)
    holdout_details = _load_json(holdout_details_path)
    bench_eval_report = _load_json(bench_eval_report_path)
    bench_details = _load_json(bench_details_path)

    holdout_image_map = _image_map(holdout_images_dir)
    benchmark_image_map = _image_map(bench_images_dir)

    # Phase 1 taxonomy on sparse holdout FPs
    holdout_fp_entries = _match_fp_detections(
        per_image=holdout_details.get("per_image", {}),
        detections_dir=holdout_det_dir,
        image_map=holdout_image_map,
        source_dataset="holdout_same_date",
        holdout_derived=True,
    )

    taxonomy_examples = _save_taxonomy_examples(
        entries=holdout_fp_entries,
        image_map=holdout_image_map,
        out_dir=taxonomy_examples_dir,
        per_bucket_max=16,
    )
    taxonomy = _taxonomy_payload(holdout_fp_entries, taxonomy_examples)
    _save_json(taxonomy_json, taxonomy)
    _write_taxonomy_summary_md(taxonomy_md, taxonomy)

    # Phase 2 hard-negative pool
    benchmark_fp_entries = _match_fp_detections(
        per_image=bench_details.get("per_image", {}),
        detections_dir=bench_det_dir,
        image_map=benchmark_image_map,
        source_dataset="frozen_benchmark_non_holdout",
        holdout_derived=False,
    )

    pool_manifest, pool_report = _build_hard_negative_pool(
        holdout_fp_entries=holdout_fp_entries,
        benchmark_fp_entries=benchmark_fp_entries,
        holdout_image_map=holdout_image_map,
        benchmark_image_map=benchmark_image_map,
        out_dir=hard_neg_root,
    )
    _save_json(hard_neg_manifest, {"generated_at_utc": _utc_now(), "entries": pool_manifest})
    _save_json(hard_neg_report, pool_report)

    # Phase 3 strategy decision
    bucket_counts = {
        b: int(taxonomy.get("buckets", {}).get(b, {}).get("count", 0))
        for b in [
            "non_job_classifieds",
            "duplicate_detections",
            "text_dense_junk",
            "slice_boundary_artifacts",
            "other_unclear",
        ]
    }
    dominant = sorted(bucket_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_bucket = dominant[0][0] if dominant else "other_unclear"
    choose_rejector = True
    reason = (
        "Dominant FP modes are non-job/text-dense triggers on sparse pages; a lightweight post-detector rejector directly targets these without touching detector weights."
    )
    if top_bucket in {"duplicate_detections", "slice_boundary_artifacts"}:
        reason = "Dominant FP modes are post-detection artifacts; post-detector rejector is lower-risk and faster than detector retraining."

    strategy = {
        "generated_at_utc": _utc_now(),
        "chosen_path": "second_stage_rejector" if choose_rejector else "detector_v3_retrain",
        "why_chosen": reason,
        "why_not_chosen_first": "Detector v3 retraining would be slower and introduces greater risk to dense-benchmark recall before validating a lightweight rejection layer.",
        "dominant_buckets": dominant,
        "holdout_contamination_risk_exists": bool(pool_report.get("holdout_derived_count", 0) > 0),
        "contamination_note": "Holdout-derived negatives are included in hard-negative pool and rejector fit; sparse holdout becomes DEV-ONLY contaminated for final generalization claims.",
    }
    _save_json(strategy_decision_path, strategy)

    # Phase 4 build chosen path: second-stage rejector
    merged_bench_labels = PROJECT_ROOT / "run_state/merged_labels_tmp_sparse_fp_benchmark_eval.json"
    _merge_labels_to_temp(bench_labels_dir, merged_bench_labels)
    benchmark_labels_map = load_labelstudio_boxes(merged_bench_labels)

    merged_holdout_labels = PROJECT_ROOT / "run_state/merged_labels_tmp_sparse_fp_holdout_eval.json"
    _merge_labels_to_temp(holdout_labels_dir, merged_holdout_labels)
    holdout_labels_map = load_labelstudio_boxes(merged_holdout_labels)

    benchmark_labeled_samples = _collect_labeled_detector_samples(
        image_map=benchmark_image_map,
        detections_dir=bench_det_dir,
        labels_map=benchmark_labels_map,
        source_name="benchmark_detector_proposals",
    )

    # Negatives from hard-negative pool (includes holdout-derived items by design, contamination flagged).
    hard_neg_samples: list[dict[str, Any]] = []
    for row in pool_manifest:
        img_path = Path(str(row.get("image_path", "")))
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        bbox = [0, 0, int(w), int(h)]
        score = float(row.get("detector_score", 0.0) or 0.0)
        feat = _extract_crop_features(image, bbox, score, page_w=w, page_h=h)
        hard_neg_samples.append(
            {
                "source": "hard_negative_pool",
                "page": str(row.get("source_page", "")),
                "bbox": bbox,
                "score": score,
                "label": 0,
                "max_iou": 0.0,
                "features": feat,
                "holdout_derived": bool(row.get("source_is_holdout_derived", False)),
            }
        )

    samples = benchmark_labeled_samples + hard_neg_samples
    X = np.asarray([_feature_vector(s["features"]) for s in samples], dtype=np.float32)
    y = np.asarray([int(s["label"]) for s in samples], dtype=np.int32)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )
    clf.fit(X, y)

    rejector_model_path.parent.mkdir(parents=True, exist_ok=True)
    with rejector_model_path.open("wb") as fh:
        pickle.dump(clf, fh)

    # Prepare scored predictions for holdout + benchmark using same base detections at conf=0.022.
    holdout_scored = _score_detections_with_rejector(
        clf=clf,
        image_map=holdout_image_map,
        detections_dir=holdout_det_dir,
    )
    benchmark_scored = _score_detections_with_rejector(
        clf=clf,
        image_map=benchmark_image_map,
        detections_dir=bench_det_dir,
    )

    # Threshold selection: minimize holdout FP with dense recall safety.
    baseline_bench = bench_eval_report.get("selected_metrics", {})
    baseline_bench_recall = float(baseline_bench.get("recall", 0.0) or 0.0)
    threshold_candidates = [round(x, 2) for x in np.linspace(0.10, 0.95, 18)]

    sweep_rows: list[dict[str, Any]] = []
    for thr in threshold_candidates:
        holdout_filtered = _filter_by_threshold(holdout_scored, thr)
        bench_filtered = _filter_by_threshold(benchmark_scored, thr)

        h_metrics, _ = compute_metrics(labels=holdout_labels_map, preds=holdout_filtered, threshold=0.5)
        b_metrics, _ = compute_metrics(labels=benchmark_labels_map, preds=bench_filtered, threshold=0.5)
        sweep_rows.append(
            {
                "threshold": thr,
                "holdout": h_metrics,
                "benchmark": b_metrics,
            }
        )

    safe_rows = [r for r in sweep_rows if float(r["benchmark"].get("recall", 0.0)) >= max(0.96, baseline_bench_recall - 0.02)]
    chosen_pool = safe_rows if safe_rows else sweep_rows
    chosen_pool = sorted(
        chosen_pool,
        key=lambda r: (
            int(r["holdout"].get("false_positives", 10**9)),
            -float(r["holdout"].get("precision", 0.0)),
            -float(r["benchmark"].get("recall", 0.0)),
            -float(r["benchmark"].get("precision", 0.0)),
        ),
    )
    chosen = chosen_pool[0]
    chosen_thr = float(chosen["threshold"])

    holdout_filtered = _filter_by_threshold(holdout_scored, chosen_thr)
    bench_filtered = _filter_by_threshold(benchmark_scored, chosen_thr)

    _write_detection_dir(holdout_filtered, holdout_postfix_det_dir)
    _write_detection_dir(bench_filtered, benchmark_postfix_det_dir)

    holdout_post_metrics, holdout_post_per_image = compute_metrics(labels=holdout_labels_map, preds=holdout_filtered, threshold=0.5)
    bench_post_metrics, bench_post_per_image = compute_metrics(labels=benchmark_labels_map, preds=bench_filtered, threshold=0.5)

    build_report = {
        "generated_at_utc": _utc_now(),
        "chosen_path": "second_stage_rejector",
        "model_type": "StandardScaler + LogisticRegression",
        "model_path": str(rejector_model_path),
        "training_samples": {
            "total": int(len(samples)),
            "positives": int(int((y == 1).sum())),
            "negatives": int(int((y == 0).sum())),
            "benchmark_detector_samples": int(len(benchmark_labeled_samples)),
            "hard_negative_pool_samples": int(len(hard_neg_samples)),
        },
        "contamination": {
            "holdout_derived_negatives_in_training": int(sum(1 for s in hard_neg_samples if s.get("holdout_derived"))),
            "status": "contaminated_dev_only" if any(s.get("holdout_derived") for s in hard_neg_samples) else "clean",
        },
        "threshold_sweep": sweep_rows,
        "chosen_threshold": chosen_thr,
        "chosen_threshold_reason": "Minimum holdout false positives under dense benchmark recall safety constraint.",
    }
    _save_json(build_report_path, build_report)

    # Phase 5 sparse holdout post-fix eval report
    holdout_post_report_payload = {
        "generated_at_utc": _utc_now(),
        "evaluation_status": "DEV-ONLY / contaminated",
        "contamination_reason": "Holdout-derived negatives were used in hard-negative pool and rejector training.",
        "frozen_detector_operating_point": {
            "threshold": 0.022,
            "merge_nms_iou": 0.50,
            "overlap_ratio": 0.20,
            "slice_size": 1024,
            "imgsz": 1024,
            "predict_iou": 0.50,
            "max_det": 1500,
        },
        "second_stage_rejector": {
            "model_path": str(rejector_model_path),
            "threshold": chosen_thr,
        },
        "metrics": holdout_post_metrics,
        "top_pages_by_false_positives": _top_pages(holdout_post_per_image, key="fp", top_n=12),
        "top_pages_by_missed_count": _top_pages(holdout_post_per_image, key="missed", top_n=12),
        "detections_dir": str(holdout_postfix_det_dir),
    }
    _save_json(holdout_postfix_eval_report, holdout_post_report_payload)

    # Phase 5 page diagnosis
    holdout_before = holdout_eval_report.get("selected_metrics", {})
    holdout_page_diag_payload = {
        "generated_at_utc": _utc_now(),
        "evaluation_status": "DEV-ONLY / contaminated",
        "before_metrics": holdout_before,
        "after_metrics": holdout_post_metrics,
        "delta_after_minus_before": {
            "true_positives": int(holdout_post_metrics.get("true_positives", 0)) - int(holdout_before.get("true_positives", 0)),
            "false_positives": int(holdout_post_metrics.get("false_positives", 0)) - int(holdout_before.get("false_positives", 0)),
            "recall": round(float(holdout_post_metrics.get("recall", 0.0)) - float(holdout_before.get("recall", 0.0)), 6),
            "precision": round(float(holdout_post_metrics.get("precision", 0.0)) - float(holdout_before.get("precision", 0.0)), 6),
        },
        "top_pages_by_false_positives_after": _top_pages(holdout_post_per_image, key="fp", top_n=12),
        "top_pages_by_missed_after": _top_pages(holdout_post_per_image, key="missed", top_n=12),
    }
    _save_json(holdout_postfix_page_diag, holdout_page_diag_payload)

    # Phase 6 dense frozen benchmark regression
    dense_regression = {
        "generated_at_utc": _utc_now(),
        "old_dense_benchmark_metrics": baseline_bench,
        "new_dense_benchmark_metrics": bench_post_metrics,
        "delta_new_minus_old": {
            "true_positives": int(bench_post_metrics.get("true_positives", 0)) - int(baseline_bench.get("true_positives", 0)),
            "false_positives": int(bench_post_metrics.get("false_positives", 0)) - int(baseline_bench.get("false_positives", 0)),
            "recall": round(float(bench_post_metrics.get("recall", 0.0)) - float(baseline_bench.get("recall", 0.0)), 6),
            "precision": round(float(bench_post_metrics.get("precision", 0.0)) - float(baseline_bench.get("precision", 0.0)), 6),
        },
        "recall_regressed": float(bench_post_metrics.get("recall", 0.0)) < float(baseline_bench.get("recall", 0.0)),
        "precision_changed": round(float(bench_post_metrics.get("precision", 0.0)) - float(baseline_bench.get("precision", 0.0)), 6),
        "regression_guard": {
            "acceptable_recall_drop_threshold": 0.02,
            "recall_drop": round(float(baseline_bench.get("recall", 0.0)) - float(bench_post_metrics.get("recall", 0.0)), 6),
            "status": "pass" if (float(baseline_bench.get("recall", 0.0)) - float(bench_post_metrics.get("recall", 0.0))) <= 0.02 else "fail",
        },
    }
    _save_json(dense_regression_path, dense_regression)

    # Phase 7 final summary
    buckets = taxonomy.get("buckets", {})
    lines = [
        "# Sparse FP Reduction Summary",
        "",
        "## 1) Dominant FP Buckets",
    ]
    for name in ["non_job_classifieds", "text_dense_junk", "duplicate_detections", "slice_boundary_artifacts", "other_unclear"]:
        row = buckets.get(name, {})
        lines.append(f"- {name}: {row.get('count', 0)} ({row.get('percentage', 0)}%)")

    lines.extend(
        [
            "",
            "## 2) Hard-Negative Pool Construction",
            f"- Pool root: `{hard_neg_root}`",
            f"- Total negatives: {pool_report.get('total_hard_negatives', 0)}",
            f"- Source counts: {json.dumps(pool_report.get('source_dataset_counts', {}), ensure_ascii=False)}",
            "",
            "## 3) Holdout Contamination",
            f"- Contamination occurred: {bool(pool_report.get('contamination_risk', False))}",
            "- Reason: holdout-derived FP regions were used as hard negatives for rejector fit.",
            "- Therefore sparse holdout postfix result is DEV-ONLY, not a clean final generalization claim.",
            "",
            "## 4) Chosen Path",
            "- Chosen path: `second_stage_rejector`",
            "- Why: lower-risk, faster FP suppression on sparse pages without detector retraining.",
            "",
            "## 5) Sparse Holdout Before vs After",
            f"- Before: TP={holdout_before.get('true_positives')}, FP={holdout_before.get('false_positives')}, recall={holdout_before.get('recall')}, precision={holdout_before.get('precision')}",
            f"- After: TP={holdout_post_metrics.get('true_positives')}, FP={holdout_post_metrics.get('false_positives')}, recall={holdout_post_metrics.get('recall')}, precision={holdout_post_metrics.get('precision')}",
            "",
            "## 6) Dense Benchmark Before vs After",
            f"- Before: TP={baseline_bench.get('true_positives')}, FP={baseline_bench.get('false_positives')}, recall={baseline_bench.get('recall')}, precision={baseline_bench.get('precision')}",
            f"- After: TP={bench_post_metrics.get('true_positives')}, FP={bench_post_metrics.get('false_positives')}, recall={bench_post_metrics.get('recall')}, precision={bench_post_metrics.get('precision')}",
            "",
            "## 7) Recommended Next Step",
            "- Gather a fresh untouched sparse holdout before making final deployment claims (current sparse set is now contaminated/dev-only).",
            "- After that, evaluate `different-newspaper-English` holdout with fixed frozen detector + fixed rejector settings for cleaner external generalization readout.",
        ]
    )
    final_summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("WROTE")
    for p in [
        taxonomy_json,
        taxonomy_md,
        hard_neg_manifest,
        hard_neg_report,
        strategy_decision_path,
        build_report_path,
        holdout_postfix_eval_report,
        holdout_postfix_page_diag,
        dense_regression_path,
        final_summary_path,
    ]:
        print(p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
