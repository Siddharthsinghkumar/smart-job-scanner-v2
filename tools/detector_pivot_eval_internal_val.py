#!/usr/bin/env python3
"""Evaluate detector-pivot model on internal YOLO val split (no external holdout)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.detector_device import resolve_device_with_preflight  # noqa: E402
from tools.evaluate_against_labels import compute_metrics  # noqa: E402


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise SystemExit(f"PyYAML import failed: {exc}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid dataset YAML: {path}")
    return payload


def _clip(v: int, lo: int, hi: int) -> int:
    return min(max(v, lo), hi)


def _labels_path_from_image(image_path: Path, val_dir: Path, labels_dir: Path) -> Path:
    rel = image_path.relative_to(val_dir)
    return labels_dir / rel.with_suffix(".txt")


def _parse_yolo_label_file(path: Path, width: int, height: int) -> list[list[int]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    boxes: list[list[int]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            _, xc, yc, bw, bh = parts
            xc_f = float(xc)
            yc_f = float(yc)
            bw_f = float(bw)
            bh_f = float(bh)
        except Exception:
            continue

        x1 = int(round((xc_f - bw_f / 2.0) * width))
        y1 = int(round((yc_f - bh_f / 2.0) * height))
        x2 = int(round((xc_f + bw_f / 2.0) * width))
        y2 = int(round((yc_f + bh_f / 2.0) * height))

        x1 = _clip(x1, 0, max(0, width - 1))
        y1 = _clip(y1, 0, max(0, height - 1))
        x2 = _clip(x2, x1 + 1, width)
        y2 = _clip(y2, y1 + 1, height)
        boxes.append([x1, y1, x2, y2])
    return boxes


def _top_pages(per_image: dict[str, Any], field: str, top_n: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page, payload in per_image.items():
        rows.append(
            {
                "page": page,
                "ground_truth_count": int(payload.get("ground_truth_count", 0) or 0),
                "prediction_count": int(payload.get("prediction_count", 0) or 0),
                "true_positives": int(payload.get("true_positives", 0) or 0),
                "missed_count": int(payload.get("missed_count", 0) or 0),
                "false_positives_count": int(payload.get("false_positives_count", 0) or 0),
            }
        )
    rows.sort(key=lambda r: (-int(r.get(field, 0)), -r["ground_truth_count"], r["page"]))
    return rows[:top_n]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector-pivot on internal val split")
    parser.add_argument("--dataset-yaml", default="data/yolo_job_ad_pivot_v3_tiles/dataset.yaml")
    parser.add_argument("--model-path", default="artifacts/detector_pivot_yolo_v3_tiles/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-preflight-report", default="run_state/detector_pivot_v3_device_preflight_internal_val.json")
    parser.add_argument("--conf", type=float, default=0.022)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max-det", type=int, default=1500)
    parser.add_argument("--detections-dir", default="run_state/detections_detector_pivot_v3_internal_val")
    parser.add_argument("--report-output", default="run_state/detector_pivot_v3_internal_val_report.json")
    parser.add_argument("--details-output", default="run_state/detector_pivot_v3_internal_val_details.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_yaml = _resolve(args.dataset_yaml)
    model_path = _resolve(args.model_path)
    device_preflight_report = _resolve(args.device_preflight_report)
    detections_dir = _resolve(args.detections_dir)
    report_output = _resolve(args.report_output)
    details_output = _resolve(args.details_output)

    if not dataset_yaml.is_file():
        raise SystemExit(f"dataset yaml not found: {dataset_yaml}")
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")

    ds = _load_yaml(dataset_yaml)
    ds_root_raw = ds.get("path")
    ds_root = Path(str(ds_root_raw)).resolve() if ds_root_raw else dataset_yaml.parent.resolve()
    val_field = ds.get("val", "images/val")

    val_dir = Path(str(val_field))
    if not val_dir.is_absolute():
        val_dir = (ds_root / val_dir).resolve()

    labels_dir = (ds_root / "labels" / "val").resolve()
    if not val_dir.is_dir():
        raise SystemExit(f"val images dir not found: {val_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"val labels dir not found: {labels_dir}")

    preflight = resolve_device_with_preflight(
        requested_device=args.device,
        context="detector_pivot_v3_internal_val",
        preflight_report_path=device_preflight_report,
    )
    selected_device = str(preflight.get("selected_device", "cpu"))

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise SystemExit(f"ultralytics import failed: {exc}")

    model = YOLO(str(model_path))
    images = sorted(val_dir.rglob("*.png"))
    if not images:
        raise SystemExit(f"no val images found under: {val_dir}")

    detections_dir.mkdir(parents=True, exist_ok=True)
    labels: dict[str, list[list[int]]] = {}
    preds: dict[str, list[dict[str, Any]]] = {}

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        page_name = image_path.name

        label_path = _labels_path_from_image(image_path, val_dir=val_dir, labels_dir=labels_dir)
        labels[page_name] = _parse_yolo_label_file(label_path, width=w, height=h)

        result = model.predict(
            source=str(image_path),
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
            imgsz=int(args.imgsz),
            device=selected_device,
            verbose=False,
        )[0]

        rows: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy().tolist()
            confs = boxes.conf.detach().cpu().numpy().tolist() if getattr(boxes, "conf", None) is not None else [0.0] * len(xyxy)
            for idx, b in enumerate(xyxy):
                if not isinstance(b, (list, tuple)) or len(b) != 4:
                    continue
                x1 = _clip(int(round(float(b[0]))), 0, max(0, w - 1))
                y1 = _clip(int(round(float(b[1]))), 0, max(0, h - 1))
                x2 = _clip(int(round(float(b[2]))), x1 + 1, w)
                y2 = _clip(int(round(float(b[3]))), y1 + 1, h)
                rows.append(
                    {
                        "id": f"detector_v3_internal_{Path(page_name).stem}_{idx}",
                        "bbox": [x1, y1, x2, y2],
                        "score": round(float(confs[idx]), 6),
                        "stage": "detector",
                        "page": page_name,
                    }
                )

        (detections_dir / f"{page_name}.json").write_text(
            json.dumps({"page": page_name, "detections": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        preds[page_name] = rows

    metrics, per_image = compute_metrics(labels=labels, preds=preds, threshold=0.5)

    details_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "per_image": per_image,
        "labels_count": len(labels),
        "preds_count": len(preds),
        "params": {
            "conf": float(args.conf),
            "iou": float(args.iou),
            "imgsz": int(args.imgsz),
            "max_det": int(args.max_det),
        },
    }
    details_output.parent.mkdir(parents=True, exist_ok=True)
    details_output.write_text(json.dumps(details_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_yaml": str(dataset_yaml),
        "val_images_dir": str(val_dir),
        "val_labels_dir": str(labels_dir),
        "model_path": str(model_path),
        "requested_device": str(args.device),
        "selected_device": selected_device,
        "device_preflight": preflight,
        "evaluation_params": {
            "conf": float(args.conf),
            "iou": float(args.iou),
            "imgsz": int(args.imgsz),
            "max_det": int(args.max_det),
            "matching_iou_threshold": 0.5,
        },
        "metrics": metrics,
        "top_pages_by_missed_count": _top_pages(per_image, field="missed_count", top_n=12),
        "top_pages_by_false_positives": _top_pages(per_image, field="false_positives_count", top_n=12),
        "details_output": str(details_output),
        "detections_dir": str(detections_dir),
        "notes": [
            "Internal validation only on v3 train/dev corpus split.",
            "External different-newspaper-English holdout is excluded from this run.",
        ],
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"report: {report_output}")
    print(f"details: {details_output}")
    print(f"detections_dir: {detections_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
