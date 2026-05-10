#!/usr/bin/env python3
"""Evaluate detector-pivot v2 with tiled full-page inference on frozen benchmark."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.benchmark_images import (  # noqa: E402
    DEFAULT_BENCHMARK_IMAGES_DIR,
    DEFAULT_BENCHMARK_MANIFEST_PATH,
    assert_benchmark_manifest_valid,
)
from src.utils.detector_device import resolve_device_with_preflight  # noqa: E402
from tools.auto_improve_detector import _merge_labels_to_temp  # noqa: E402
from tools.evaluate_against_labels import compute_metrics, load_labelstudio_boxes  # noqa: E402


@dataclass
class Det:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _clip(v: int, lo: int, hi: int) -> int:
    return min(max(v, lo), hi)


def _iou(a: Det, b: Det) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    bb = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    denom = aa + bb - inter
    return float(inter / denom) if denom > 0 else 0.0


def _nms(dets: list[Det], iou_thr: float) -> list[Det]:
    if not dets:
        return []
    dets_sorted = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: list[Det] = []
    while dets_sorted:
        cur = dets_sorted.pop(0)
        kept.append(cur)
        dets_sorted = [d for d in dets_sorted if _iou(cur, d) < iou_thr]
    return kept


def _iter_tiles(w: int, h: int, slice_size: int, overlap_ratio: float) -> list[tuple[int, int, int, int]]:
    tile = min(slice_size, w, h)
    stride = max(1, int(round(tile * (1.0 - overlap_ratio))))
    xs = list(range(0, max(1, w - tile + 1), stride))
    ys = list(range(0, max(1, h - tile + 1), stride))
    if not xs or xs[-1] != max(0, w - tile):
        xs.append(max(0, w - tile))
    if not ys or ys[-1] != max(0, h - tile):
        ys.append(max(0, h - tile))
    out: list[tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            out.append((x, y, min(w, x + tile), min(h, y + tile)))
    return out


def _top_pages_by_missed(per_image: dict[str, Any], top_n: int = 12) -> list[dict[str, Any]]:
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
    rows.sort(key=lambda r: (-r["missed_count"], -r["ground_truth_count"], r["true_positives"], r["page"]))
    return rows[:top_n]


def _parse_thresholds(value: str) -> list[float]:
    out: list[float] = []
    for token in value.split(","):
        t = token.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        raise RuntimeError("At least one confidence threshold is required")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector pivot v2 tiled benchmark evaluation")
    parser.add_argument("--labels-dir", default="data/test_labels")
    parser.add_argument(
        "--images-dir",
        default=str(DEFAULT_BENCHMARK_IMAGES_DIR.relative_to(PROJECT_ROOT)),
    )
    parser.add_argument(
        "--benchmark-manifest",
        default=str(DEFAULT_BENCHMARK_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
    )
    parser.add_argument("--model-path", default="artifacts/detector_pivot_yolo_v2_tiles/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-preflight-report", default="run_state/detector_pivot_v2_device_preflight_eval.json")
    parser.add_argument("--thresholds", default="0.003,0.005,0.01")
    parser.add_argument("--slice-size", type=int, default=1024)
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--predict-iou", type=float, default=0.5)
    parser.add_argument("--merge-nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=1500)
    parser.add_argument("--detections-root", default="run_state/detections_detector_pivot_v2_tiled")
    parser.add_argument("--merged-labels-output", default="run_state/merged_labels_tmp_detector_pivot_v2_eval.json")
    parser.add_argument("--threshold-sweep-output", default="run_state/detector_pivot_v2_threshold_sweep.json")
    parser.add_argument("--eval-report-output", default="run_state/detector_pivot_v2_eval_report.json")
    parser.add_argument("--details-root", default="run_state/detector_pivot_v2_eval_details")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    labels_dir = _resolve(args.labels_dir)
    images_dir = _resolve(args.images_dir)
    benchmark_manifest = _resolve(args.benchmark_manifest)
    model_path = _resolve(args.model_path)
    device_preflight_report = _resolve(args.device_preflight_report)
    detections_root = _resolve(args.detections_root)
    merged_labels_output = _resolve(args.merged_labels_output)
    threshold_sweep_output = _resolve(args.threshold_sweep_output)
    eval_report_output = _resolve(args.eval_report_output)
    details_root = _resolve(args.details_root)

    if not labels_dir.is_dir():
        raise SystemExit(f"labels dir not found: {labels_dir}")
    if not images_dir.is_dir():
        raise SystemExit(f"images dir not found: {images_dir}")
    if not benchmark_manifest.is_file():
        raise SystemExit(f"benchmark manifest not found: {benchmark_manifest}")
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")

    validation = assert_benchmark_manifest_valid(benchmark_images_dir=images_dir, manifest_path=benchmark_manifest)
    preflight = resolve_device_with_preflight(
        requested_device=args.device,
        context="detector_pivot_v2_eval_tiled",
        preflight_report_path=device_preflight_report,
    )
    selected_device = str(preflight.get("selected_device", "cpu"))

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise SystemExit(f"ultralytics import failed: {exc}")

    model = YOLO(str(model_path))
    thresholds = _parse_thresholds(args.thresholds)

    merged_labels_output.parent.mkdir(parents=True, exist_ok=True)
    _merge_labels_to_temp(labels_dir, merged_labels_output)
    labels = load_labelstudio_boxes(merged_labels_output)

    images = sorted(images_dir.rglob("*.png"))
    if not images:
        raise SystemExit("no benchmark images found")

    sweep_results: dict[str, Any] = {}
    heavy_pages_reference: dict[str, Any] = {}

    for conf in thresholds:
        conf_key = f"{conf:.3f}"
        conf_dir = detections_root / f"conf_{conf_key.replace('.', 'p')}"
        conf_dir.mkdir(parents=True, exist_ok=True)
        preds_for_eval: dict[str, list[dict[str, Any]]] = {}

        for img_path in images:
            page_name = img_path.name
            image = cv2.imread(str(img_path))
            if image is None:
                preds_for_eval[page_name] = []
                continue
            h, w = image.shape[:2]
            tiles = _iter_tiles(w=w, h=h, slice_size=int(args.slice_size), overlap_ratio=float(args.overlap_ratio))
            all_dets: list[Det] = []

            for (x1, y1, x2, y2) in tiles:
                tile_img = image[y1:y2, x1:x2]
                result = model.predict(
                    source=tile_img,
                    conf=float(conf),
                    iou=float(args.predict_iou),
                    max_det=int(args.max_det),
                    imgsz=int(args.imgsz),
                    device=selected_device,
                    verbose=False,
                )[0]
                boxes = getattr(result, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy = boxes.xyxy.detach().cpu().numpy().tolist()
                confs = boxes.conf.detach().cpu().numpy().tolist() if getattr(boxes, "conf", None) is not None else [0.0] * len(xyxy)
                for i, b in enumerate(xyxy):
                    if not isinstance(b, (list, tuple)) or len(b) != 4:
                        continue
                    gx1 = _clip(int(round(float(b[0])) + x1), 0, w - 1)
                    gy1 = _clip(int(round(float(b[1])) + y1), 0, h - 1)
                    gx2 = _clip(int(round(float(b[2])) + x1), gx1 + 1, w)
                    gy2 = _clip(int(round(float(b[3])) + y1), gy1 + 1, h)
                    all_dets.append(Det(x1=gx1, y1=gy1, x2=gx2, y2=gy2, score=float(confs[i])))

            merged = _nms(all_dets, iou_thr=float(args.merge_nms_iou))
            rows: list[dict[str, Any]] = []
            for i, d in enumerate(merged):
                rows.append(
                    {
                        "id": f"detector_v2_tiled_{Path(page_name).stem}_{i}",
                        "bbox": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
                        "score": round(float(d.score), 6),
                        "stage": "detector",
                        "page": page_name,
                    }
                )

            (conf_dir / f"{page_name}.json").write_text(
                json.dumps({"page": page_name, "detections": rows}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            preds_for_eval[page_name] = rows

        metrics, per_image = compute_metrics(labels=labels, preds=preds_for_eval, threshold=0.5)
        top_pages = _top_pages_by_missed(per_image, top_n=12)

        details_payload = {
            "metrics": metrics,
            "per_image": per_image,
            "threshold": conf,
            "detections_dir": str(conf_dir),
        }
        details_root.mkdir(parents=True, exist_ok=True)
        details_path = details_root / f"detector_pivot_v2_eval_details_conf_{conf_key.replace('.', 'p')}.json"
        details_path.write_text(json.dumps(details_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        sweep_results[conf_key] = {
            "threshold": conf,
            "metrics": metrics,
            "top_pages_by_missed_count": top_pages,
            "detections_dir": str(conf_dir),
            "details_path": str(details_path),
        }

        if not heavy_pages_reference:
            heavy_pages_reference = {
                p["page"]: p
                for p in sorted(top_pages, key=lambda r: (-r["ground_truth_count"], r["page"]))[:3]
            }

    threshold_sweep_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "model_path": str(model_path),
            "images_dir": str(images_dir),
            "benchmark_manifest": str(benchmark_manifest),
            "labels_dir": str(labels_dir),
            "merged_labels_output": str(merged_labels_output),
            "requested_device": str(args.device),
            "selected_device": selected_device,
            "execution_mode": str(preflight.get("execution_mode", "cpu")),
            "slice_size": int(args.slice_size),
            "overlap_ratio": float(args.overlap_ratio),
            "imgsz": int(args.imgsz),
            "predict_iou": float(args.predict_iou),
            "merge_nms_iou": float(args.merge_nms_iou),
            "max_det": int(args.max_det),
            "thresholds": thresholds,
            "device_preflight_report": str(device_preflight_report),
        },
        "benchmark_validation": validation,
        "results_by_threshold": sweep_results,
    }

    # Select threshold by max TP then max precision.
    threshold_rows = list(sweep_results.items())
    threshold_rows.sort(
        key=lambda kv: (
            -int(kv[1]["metrics"].get("true_positives", 0)),
            -float(kv[1]["metrics"].get("precision", 0.0)),
            float(kv[0]),
        )
    )
    selected_key, selected_payload = threshold_rows[0]

    eval_report_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_threshold": float(selected_key),
        "selected_metrics": selected_payload["metrics"],
        "selected_top_pages_by_missed_count": selected_payload["top_pages_by_missed_count"],
        "heavy_pages_reference": heavy_pages_reference,
        "threshold_sweep_path": str(threshold_sweep_output),
        "selected_details_path": selected_payload["details_path"],
        "selected_detections_dir": selected_payload["detections_dir"],
        "device_preflight": preflight,
        "benchmark_validation": validation,
        "notes": [
            "Frozen benchmark assets were read-only inputs.",
            "Inference used tiled slicing with merge NMS to improve small-object recall on full pages.",
        ],
    }

    threshold_sweep_output.parent.mkdir(parents=True, exist_ok=True)
    eval_report_output.parent.mkdir(parents=True, exist_ok=True)
    threshold_sweep_output.write_text(json.dumps(threshold_sweep_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    eval_report_output.write_text(json.dumps(eval_report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"threshold_sweep_output: {threshold_sweep_output}")
    print(f"eval_report_output: {eval_report_output}")
    print(f"selected_threshold: {selected_key}")
    print(f"selected_metrics: {json.dumps(selected_payload['metrics'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
