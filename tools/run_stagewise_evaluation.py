#!/usr/bin/env python3
"""Run benchmark evaluation separately for detector and refined stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.auto_improve_detector import _merge_labels_to_temp
from tools.evaluate_against_labels import (
    compute_metrics,
    load_pipeline_detections,
    to_float,
)
from src.utils.benchmark_images import (
    DEFAULT_BENCHMARK_IMAGES_DIR,
    DEFAULT_BENCHMARK_MANIFEST_PATH,
    assert_benchmark_manifest_valid,
)
from src.utils.benchmark_alignment import compute_dim_scale, scale_bbox_xyxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-wise benchmark evaluation")
    parser.add_argument("--labels-dir", default="data/test_labels", help="Label JSON directory")
    parser.add_argument("--detections-dir", default="run_state/detections", help="Detections JSON directory")
    parser.add_argument(
        "--images-dir",
        default=str(DEFAULT_BENCHMARK_IMAGES_DIR.relative_to(PROJECT_ROOT)),
        help="Benchmark page-image directory",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default=str(DEFAULT_BENCHMARK_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        help="Benchmark image manifest path",
    )
    parser.add_argument(
        "--validate-benchmark-images",
        action="store_true",
        help="Validate benchmark image hashes/filenames against manifest before evaluation",
    )
    parser.add_argument("--output", default="run_state/stagewise_eval_report.json", help="Output report path")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold")
    return parser.parse_args()


def _run_eval(labels_file: Path, detections_dir: Path, stage: str, iou_threshold: float) -> tuple[dict, Path]:
    details_path = PROJECT_ROOT / "run_state" / f"eval_details_{stage}.json"
    cmd = [
        sys.executable,
        "tools/evaluate_against_labels.py",
        "--labels",
        str(labels_file),
        "--detections",
        str(detections_dir),
        "--stage",
        stage,
        "--iou-threshold",
        str(iou_threshold),
        "--output-details",
        str(details_path),
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Evaluation failed for stage={stage} rc={proc.returncode}\n{proc.stderr or proc.stdout}"
        )
    metrics = json.loads(proc.stdout)
    return metrics, details_path


def _load_label_boxes_with_dims(merged_labels_path: Path) -> dict[str, dict]:
    payload = json.loads(merged_labels_path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", []) if isinstance(payload, dict) else payload
    if not isinstance(tasks, list):
        return {}
    out: dict[str, dict] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        data = task.get("data", {})
        if not isinstance(data, dict):
            continue
        image_ref = str(data.get("image") or data.get("page") or data.get("file") or "")
        page = Path(image_ref).name
        if not page:
            continue

        row = out.setdefault(page, {"boxes": [], "dims": None})
        anns = task.get("annotations", [])
        if not isinstance(anns, list):
            continue
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            results = ann.get("result", [])
            if not isinstance(results, list):
                continue
            for item in results:
                if not isinstance(item, dict) or item.get("type") != "rectanglelabels":
                    continue
                value = item.get("value", {})
                if not isinstance(value, dict):
                    continue
                ow = to_float(item.get("original_width", value.get("original_width")), 0.0)
                oh = to_float(item.get("original_height", value.get("original_height")), 0.0)
                if ow <= 0 or oh <= 0:
                    continue
                x = to_float(value.get("x"))
                y = to_float(value.get("y"))
                w = to_float(value.get("width"))
                h = to_float(value.get("height"))
                x1 = int(round((x / 100.0) * ow))
                y1 = int(round((y / 100.0) * oh))
                x2 = int(round(((x + w) / 100.0) * ow))
                y2 = int(round(((y + h) / 100.0) * oh))
                row["boxes"].append([x1, y1, x2, y2])
                row["dims"] = (int(round(ow)), int(round(oh)))
    return out


def _dimension_reconciled_metrics(
    merged_labels_path: Path,
    images_dir: Path,
    detections_dir: Path,
    stage: str,
    iou_threshold: float,
) -> dict:
    labels_with_dims = _load_label_boxes_with_dims(merged_labels_path)
    preds = load_pipeline_detections(detections_dir, stage)
    image_index = {p.name: p for p in images_dir.rglob("*.png")}

    scaled_labels: dict[str, list[list[int]]] = {}
    scaled_pages = 0
    for page, row in labels_with_dims.items():
        boxes = row.get("boxes", [])
        label_dims = row.get("dims")
        img_path = image_index.get(page)
        if img_path is None:
            scaled_labels[page] = boxes
            continue
        try:
            import cv2

            img = cv2.imread(str(img_path))
            if img is None:
                scaled_labels[page] = boxes
                continue
            ih, iw = img.shape[:2]
            scale = compute_dim_scale(label_dims, (iw, ih))
            if scale and (abs(scale[0] - 1.0) > 1e-6 or abs(scale[1] - 1.0) > 1e-6):
                scaled_pages += 1
            scaled_labels[page] = [scale_bbox_xyxy(b, scale) for b in boxes]
        except Exception:
            scaled_labels[page] = boxes

    metrics, _ = compute_metrics(scaled_labels, preds, iou_threshold)
    metrics["dimension_reconciled_pages"] = scaled_pages
    return metrics


def main() -> int:
    args = parse_args()
    labels_dir = (PROJECT_ROOT / args.labels_dir).resolve()
    detections_dir = (PROJECT_ROOT / args.detections_dir).resolve()
    images_dir = (PROJECT_ROOT / args.images_dir).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()

    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")
    if not detections_dir.is_dir():
        raise SystemExit(f"Detections directory not found: {detections_dir}")
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    should_validate = bool(
        args.validate_benchmark_images
        or images_dir.resolve() == DEFAULT_BENCHMARK_IMAGES_DIR.resolve()
    )
    if should_validate:
        assert_benchmark_manifest_valid(
            benchmark_images_dir=images_dir,
            manifest_path=(PROJECT_ROOT / args.benchmark_manifest).resolve()
            if not Path(args.benchmark_manifest).is_absolute()
            else Path(args.benchmark_manifest).resolve(),
        )

    merged_labels_path = (PROJECT_ROOT / "run_state" / "merged_labels_tmp_stagewise.json").resolve()
    _merge_labels_to_temp(labels_dir, merged_labels_path)

    detector_metrics, detector_details_path = _run_eval(
        merged_labels_path, detections_dir, "detector", args.iou_threshold
    )
    refined_metrics, refined_details_path = _run_eval(
        merged_labels_path, detections_dir, "refined", args.iou_threshold
    )
    detector_reconciled = _dimension_reconciled_metrics(
        merged_labels_path, images_dir, detections_dir, "detector", args.iou_threshold
    )
    refined_reconciled = _dimension_reconciled_metrics(
        merged_labels_path, images_dir, detections_dir, "refined", args.iou_threshold
    )

    report = {
        "inputs": {
            "labels_dir": str(labels_dir),
            "merged_labels_file": str(merged_labels_path),
            "detections_dir": str(detections_dir),
            "images_dir": str(images_dir),
            "iou_threshold": args.iou_threshold,
        },
        "detector": detector_metrics,
        "refined": refined_metrics,
        "dimension_reconciled_proxy": {
            "detector": detector_reconciled,
            "refined": refined_reconciled,
        },
        "details_files": {
            "detector": str(detector_details_path),
            "refined": str(refined_details_path),
        },
        "delta_refined_minus_detector": {
            "precision": round(float(refined_metrics.get("precision", 0.0)) - float(detector_metrics.get("precision", 0.0)), 6),
            "recall": round(float(refined_metrics.get("recall", 0.0)) - float(detector_metrics.get("recall", 0.0)), 6),
            "true_positives": int(refined_metrics.get("true_positives", 0)) - int(detector_metrics.get("true_positives", 0)),
            "false_positives": int(refined_metrics.get("false_positives", 0)) - int(detector_metrics.get("false_positives", 0)),
            "missed_detections": int(refined_metrics.get("missed_detections", 0)) - int(detector_metrics.get("missed_detections", 0)),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
