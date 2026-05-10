#!/usr/bin/env python3
"""Evaluate pipeline detections against Label Studio annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detections against Label Studio export")
    parser.add_argument("--labels", required=True, help="Path to Label Studio export JSON")
    parser.add_argument("--detections", required=True, help="Directory containing <page>.json detection files")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for a match")
    parser.add_argument(
        "--stage",
        choices=["all", "detector", "refined"],
        default="all",
        help="Which detection stage to evaluate",
    )
    parser.add_argument(
        "--output-details",
        default=None,
        help="Optional path to write per-image TP/FP/FN/low-IoU details as JSON",
    )
    return parser.parse_args()


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def basename_from_task(task: dict[str, Any]) -> str | None:
    data = task.get("data", {}) if isinstance(task, dict) else {}
    image_ref = data.get("image") or data.get("page") or data.get("file")
    if not image_ref:
        return None
    return Path(str(image_ref)).name


def percent_to_pixel_bbox(value: dict[str, Any], original_width: float, original_height: float) -> list[int]:
    x = to_float(value.get("x"))
    y = to_float(value.get("y"))
    w = to_float(value.get("width"))
    h = to_float(value.get("height"))

    x1 = int(round((x / 100.0) * original_width))
    y1 = int(round((y / 100.0) * original_height))
    x2 = int(round(((x + w) / 100.0) * original_width))
    y2 = int(round(((y + h) / 100.0) * original_height))
    return [x1, y1, x2, y2]


def load_labelstudio_boxes(labels_path: Path) -> dict[str, list[list[int]]]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", []) if isinstance(payload, dict) else payload
    if not isinstance(tasks, list):
        raise ValueError("Unsupported Label Studio export format")

    boxes_by_image: dict[str, list[list[int]]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        image_name = basename_from_task(task)
        if not image_name:
            continue

        ann_list = task.get("annotations", [])
        if not isinstance(ann_list, list):
            continue

        for ann in ann_list:
            if not isinstance(ann, dict):
                continue
            results = ann.get("result", [])
            if not isinstance(results, list):
                continue
            for item in results:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "rectanglelabels":
                    continue
                value = item.get("value", {})
                if not isinstance(value, dict):
                    continue

                orig_w = to_float(item.get("original_width", value.get("original_width")), 0.0)
                orig_h = to_float(item.get("original_height", value.get("original_height")), 0.0)
                if orig_w <= 0 or orig_h <= 0:
                    continue

                bbox = percent_to_pixel_bbox(value, orig_w, orig_h)
                boxes_by_image.setdefault(image_name, []).append(bbox)

    return boxes_by_image


def load_pipeline_detections(detection_dir: Path, stage_filter: str) -> dict[str, list[dict[str, Any]]]:
    detections: dict[str, list[dict[str, Any]]] = {}
    for file_path in sorted(detection_dir.glob("*.json")):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        page_name = str(payload.get("page") or file_path.stem)
        rows = payload.get("detections", [])
        if not isinstance(rows, list):
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue
            if stage_filter != "all" and row.get("stage") != stage_filter:
                continue
            bbox = row.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            detections.setdefault(page_name, []).append(
                {
                    "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "stage": row.get("stage"),
                    "score": to_float(row.get("score"), 0.0),
                    "id": row.get("id"),
                }
            )
    return detections


def iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def compute_metrics(
    labels: dict[str, list[list[int]]],
    preds: dict[str, list[dict[str, Any]]],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    tp = 0
    fp = 0
    fn = 0
    low_iou_total = 0
    total_gt = 0
    total_detected = 0
    per_image: dict[str, dict[str, Any]] = {}

    all_images = sorted(set(labels.keys()) | set(preds.keys()))
    for image_name in all_images:
        gt_boxes = labels.get(image_name, [])
        pred_rows = preds.get(image_name, [])
        matched_pred: set[int] = set()
        image_low_iou: list[dict[str, Any]] = []
        image_missed: list[list[int]] = []
        image_tp = 0

        for gt in gt_boxes:
            best_idx = -1
            best_iou = 0.0
            for idx, pred_row in enumerate(pred_rows):
                if idx in matched_pred:
                    continue
                score = iou(gt, pred_row["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_idx = idx

            if best_idx >= 0 and best_iou >= threshold:
                tp += 1
                image_tp += 1
                matched_pred.add(best_idx)
            else:
                fn += 1
                image_missed.append(gt)
                if best_idx >= 0 and 0 < best_iou < threshold:
                    image_low_iou.append(
                        {
                            "gt_bbox": gt,
                            "pred_bbox": pred_rows[best_idx]["bbox"],
                            "iou": round(best_iou, 6),
                            "pred_stage": pred_rows[best_idx].get("stage"),
                            "pred_score": pred_rows[best_idx].get("score"),
                        }
                    )
                    low_iou_total += 1

        image_fp = [pred_rows[i]["bbox"] for i in range(len(pred_rows)) if i not in matched_pred]
        image_fp_count = len(image_fp)
        image_missed_count = len(image_missed)
        fp += len(image_fp)
        total_gt += len(gt_boxes)
        total_detected += len(pred_rows)
        per_image[image_name] = {
            "missed_detections": image_missed,
            "false_positives": image_fp,
            "low_iou_matches": image_low_iou,
            "ground_truth_count": len(gt_boxes),
            "prediction_count": len(pred_rows),
            "true_positives": image_tp,
            "false_positives_count": image_fp_count,
            "missed_count": image_missed_count,
        }

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "false_positives": fp,
        "missed_detections": fn,
        "true_positives": tp,
        "low_iou_matches": low_iou_total,
        "total_images": len(all_images),
        "total_ground_truth_ads": total_gt,
        "total_detected_ads": total_detected,
        "label_images": len(labels),
    }

    # Totals sanity check: aggregate per-page counts should match global counts.
    check_tp = sum(v.get("true_positives", 0) for v in per_image.values())
    check_fp = sum(v.get("false_positives_count", 0) for v in per_image.values())
    check_fn = sum(v.get("missed_count", 0) for v in per_image.values())
    if check_tp != tp or check_fp != fp or check_fn != fn:
        raise RuntimeError("Per-image totals do not match global totals")

    return metrics, per_image


def write_eval_summary(per_image: dict[str, dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for page_name in sorted(per_image.keys()):
        row = per_image[page_name]
        lines.append(f"Page {page_name}:")
        lines.append(f"GT: {int(row.get('ground_truth_count', 0))}")
        lines.append(f"Detected: {int(row.get('prediction_count', 0))}")
        lines.append(f"TP: {int(row.get('true_positives', 0))}")
        lines.append(f"FP: {int(row.get('false_positives_count', 0))}")
        lines.append(f"Missed: {int(row.get('missed_count', 0))}")
        lines.append("")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels)
    detections_dir = Path(args.detections)

    if not labels_path.is_file():
        raise SystemExit(f"Labels file not found: {labels_path}")
    if not detections_dir.is_dir():
        raise SystemExit(f"Detections directory not found: {detections_dir}")

    labels = load_labelstudio_boxes(labels_path)
    preds = load_pipeline_detections(detections_dir, args.stage)
    metrics, details = compute_metrics(labels, preds, args.iou_threshold)
    write_eval_summary(details, Path("run_state/eval_summary.txt"))

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output_details:
        out_path = Path(args.output_details)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "per_image": details,
                    "iou_threshold": args.iou_threshold,
                    "stage_filter": args.stage,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
