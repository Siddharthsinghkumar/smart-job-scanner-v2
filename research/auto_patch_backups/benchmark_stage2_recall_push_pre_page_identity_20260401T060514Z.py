#!/usr/bin/env python3
"""Recall-first Stage2 benchmark: compare baseline detector vs v4 on TP-first objectives."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_STATE = PROJECT_ROOT / "run_state"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.auto_improve_detector import _merge_labels_to_temp
from tools.evaluate_against_labels import compute_metrics, load_labelstudio_boxes, load_pipeline_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TP-first Stage2 recall push")
    parser.add_argument("--config", default="configs/pipeline_paths.json", help="Pipeline config path")
    parser.add_argument("--baseline-detector", choices=["v1", "v2", "v3", "v4"], default="v1")
    parser.add_argument("--candidate-detector", choices=["v1", "v2", "v3", "v4"], default="v4")
    parser.add_argument("--labels-dir", default="data/test_labels")
    parser.add_argument("--images-dir", default="data/pdf2img")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--top-n-overlays", type=int, default=20)
    return parser.parse_args()


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr or proc.stdout}")
    return proc.stdout


def _copy_detections(dst: Path) -> None:
    src = RUN_STATE / "detections"
    if not src.is_dir():
        raise RuntimeError(f"Detections directory not found: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _evaluate(labels, detections_dir: Path, stage: str, iou_threshold: float):
    preds = load_pipeline_detections(detections_dir, stage)
    metrics, per_image = compute_metrics(labels, preds, iou_threshold)
    return metrics, per_image, preds


def _draw_boxes(img, boxes, color):
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def _draw_legend(img):
    entries = [
        ("GT", (0, 255, 0)),
        ("Detector", (0, 165, 255)),
        ("Passthrough", (255, 140, 0)),
        ("Normal", (0, 0, 255)),
    ]
    x = 20
    y = 30
    for text, color in entries:
        cv2.rectangle(img, (x, y - 12), (x + 16, y + 4), color, -1)
        cv2.putText(img, text, (x + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name)


def _run_version(
    detector_version: str,
    config_path: str,
    labels: dict[str, list[list[int]]],
    iou_threshold: float,
) -> dict[str, Any]:
    version_root = RUN_STATE / "stage2_recall_push_runs" / detector_version
    normal_det_dir = version_root / "normal_detections"
    passthrough_det_dir = version_root / "passthrough_detections"

    # Honest sequential truth run (normal Stage3).
    _run([sys.executable, "tools/reset_test_dataset.py"])
    _run(
        [
            sys.executable,
            "tools/run_detection_pipeline.py",
            "--config",
            config_path,
            "--mode",
            "sequential",
            "--detector-version",
            detector_version,
            "--stage3-mode",
            "normal",
        ]
    )
    _copy_detections(normal_det_dir)
    detector_metrics, detector_per_image, detector_map = _evaluate(labels, normal_det_dir, "detector", iou_threshold)
    normal_metrics, normal_per_image, normal_map = _evaluate(labels, normal_det_dir, "refined", iou_threshold)

    # Re-run only Stage3 as passthrough on same Stage2 outputs to isolate proposal coverage.
    _run(
        [
            sys.executable,
            "src/pipeline/stage03_block_refiner.py",
            "--config",
            config_path,
            "--stage3-mode",
            "passthrough",
        ]
    )
    _copy_detections(passthrough_det_dir)
    passthrough_metrics, passthrough_per_image, passthrough_map = _evaluate(
        labels,
        passthrough_det_dir,
        "refined",
        iou_threshold,
    )

    return {
        "detector_version": detector_version,
        "detector_metrics": detector_metrics,
        "passthrough_metrics": passthrough_metrics,
        "normal_refined_metrics": normal_metrics,
        "per_image": {
            "detector": detector_per_image,
            "passthrough": passthrough_per_image,
            "normal_refined": normal_per_image,
        },
        "maps": {
            "detector": detector_map,
            "passthrough": passthrough_map,
            "normal_refined": normal_map,
        },
        "detection_dirs": {
            "normal": str(normal_det_dir),
            "passthrough": str(passthrough_det_dir),
        },
    }


def _build_page_report(labels, version_result: dict[str, Any]) -> list[dict[str, Any]]:
    detector_map = version_result["maps"]["detector"]
    passthrough_map = version_result["maps"]["passthrough"]
    normal_map = version_result["maps"]["normal_refined"]
    detector_per = version_result["per_image"]["detector"]
    passthrough_per = version_result["per_image"]["passthrough"]
    normal_per = version_result["per_image"]["normal_refined"]

    pages = sorted(labels.keys())
    rows = []
    for page in pages:
        rows.append(
            {
                "page": page,
                "labeled_box_count": len(labels.get(page, [])),
                "detector_box_count": len(detector_map.get(page, [])),
                "passthrough_box_count": len(passthrough_map.get(page, [])),
                "normal_refined_box_count": len(normal_map.get(page, [])),
                "detector_tp": int(detector_per.get(page, {}).get("true_positives", 0) or 0),
                "passthrough_tp": int(passthrough_per.get(page, {}).get("true_positives", 0) or 0),
                "normal_refined_tp": int(normal_per.get(page, {}).get("true_positives", 0) or 0),
                "detector_missed": int(detector_per.get(page, {}).get("missed_count", 0) or 0),
                "passthrough_missed": int(passthrough_per.get(page, {}).get("missed_count", 0) or 0),
                "normal_refined_missed": int(normal_per.get(page, {}).get("missed_count", 0) or 0),
            }
        )
    return rows


def _render_overlays(
    labels,
    images_dir: Path,
    page_rows: list[dict[str, Any]],
    version_result: dict[str, Any],
    output_dir: Path,
    top_n: int,
) -> dict[str, Any]:
    detector_map = version_result["maps"]["detector"]
    passthrough_map = version_result["maps"]["passthrough"]
    normal_map = version_result["maps"]["normal_refined"]
    image_index = {p.name: p for p in images_dir.rglob("*.png")}

    ranked = sorted(
        [r for r in page_rows if int(r.get("labeled_box_count", 0)) > 0],
        key=lambda r: (
            int(r["labeled_box_count"]) - int(r["detector_tp"]),
            int(r["labeled_box_count"]) - int(r["passthrough_tp"]),
            int(r["labeled_box_count"]) - int(r["normal_refined_tp"]),
            int(r["labeled_box_count"]),
        ),
        reverse=True,
    )
    selected = ranked[: max(1, int(top_n))]

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for row in selected:
        page = row["page"]
        img_path = image_index.get(page)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gt_boxes = labels.get(page, [])
        detector_boxes = [d.get("bbox") for d in detector_map.get(page, []) if isinstance(d, dict) and isinstance(d.get("bbox"), list)]
        passthrough_boxes = [d.get("bbox") for d in passthrough_map.get(page, []) if isinstance(d, dict) and isinstance(d.get("bbox"), list)]
        normal_boxes = [d.get("bbox") for d in normal_map.get(page, []) if isinstance(d, dict) and isinstance(d.get("bbox"), list)]

        _draw_boxes(img, gt_boxes, (0, 255, 0))
        _draw_boxes(img, detector_boxes, (0, 165, 255))
        _draw_boxes(img, passthrough_boxes, (255, 140, 0))
        _draw_boxes(img, normal_boxes, (0, 0, 255))
        _draw_legend(img)
        cv2.putText(
            img,
            f"{page} | GT:{len(gt_boxes)} D:{len(detector_boxes)} P:{len(passthrough_boxes)} N:{len(normal_boxes)}",
            (20, max(30, img.shape[0] - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        out_path = output_dir / _safe_name(page)
        cv2.imwrite(str(out_path), img)
        rendered.append(page)

    return {
        "ranked_pages": ranked,
        "selected_pages": selected,
        "rendered_pages": rendered,
        "overlay_dir": str(output_dir),
    }


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "tp_delta": int(after.get("true_positives", 0) or 0) - int(before.get("true_positives", 0) or 0),
        "fn_delta": int(after.get("missed_detections", 0) or 0) - int(before.get("missed_detections", 0) or 0),
        "recall_delta": round(float(after.get("recall", 0.0)) - float(before.get("recall", 0.0)), 6),
        "fp_delta": int(after.get("false_positives", 0) or 0) - int(before.get("false_positives", 0) or 0),
    }


def main() -> int:
    args = parse_args()
    labels_dir = (PROJECT_ROOT / args.labels_dir).resolve()
    images_dir = (PROJECT_ROOT / args.images_dir).resolve()
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels dir not found: {labels_dir}")
    if not images_dir.is_dir():
        raise SystemExit(f"Images dir not found: {images_dir}")

    RUN_STATE.mkdir(parents=True, exist_ok=True)
    merged_labels_path = RUN_STATE / "merged_labels_tmp_stage2_recall_push.json"
    _merge_labels_to_temp(labels_dir, merged_labels_path)
    labels = load_labelstudio_boxes(merged_labels_path)

    baseline = _run_version(args.baseline_detector, args.config, labels, args.iou_threshold)
    candidate = _run_version(args.candidate_detector, args.config, labels, args.iou_threshold)

    total_labeled_ads = int(candidate["detector_metrics"].get("total_ground_truth_ads", 0) or 0)

    # Page report + overlays from candidate recall-push run.
    page_rows = _build_page_report(labels, candidate)
    page_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "detector_version": args.candidate_detector,
        "rows": page_rows,
    }
    page_report_path = RUN_STATE / "stage2_recall_push_page_report.json"
    page_report_path.write_text(json.dumps(page_report, indent=2, ensure_ascii=False), encoding="utf-8")

    overlay_dir = RUN_STATE / "stage2_recall_push_overlays"
    overlay_data = _render_overlays(
        labels=labels,
        images_dir=images_dir,
        page_rows=page_rows,
        version_result=candidate,
        output_dir=overlay_dir,
        top_n=args.top_n_overlays,
    )

    comparison = {
        "generated_at_utc": page_report["generated_at_utc"],
        "baseline_detector_version": args.baseline_detector,
        "candidate_detector_version": args.candidate_detector,
        "total_labeled_ads": total_labeled_ads,
        "baseline": {
            "detector": baseline["detector_metrics"],
            "passthrough": baseline["passthrough_metrics"],
            "normal_refined": baseline["normal_refined_metrics"],
        },
        "candidate": {
            "detector": candidate["detector_metrics"],
            "passthrough": candidate["passthrough_metrics"],
            "normal_refined": candidate["normal_refined_metrics"],
        },
        "headline": {
            "v1_detector_tp_over_155": f"{int(baseline['detector_metrics'].get('true_positives', 0) or 0)}/{total_labeled_ads}",
            "v4_detector_tp_over_155": f"{int(candidate['detector_metrics'].get('true_positives', 0) or 0)}/{total_labeled_ads}",
            "v1_passthrough_tp_over_155": f"{int(baseline['passthrough_metrics'].get('true_positives', 0) or 0)}/{total_labeled_ads}",
            "v4_passthrough_tp_over_155": f"{int(candidate['passthrough_metrics'].get('true_positives', 0) or 0)}/{total_labeled_ads}",
            "v1_normal_refined_tp_over_155": f"{int(baseline['normal_refined_metrics'].get('true_positives', 0) or 0)}/{total_labeled_ads}",
            "v4_normal_refined_tp_over_155": f"{int(candidate['normal_refined_metrics'].get('true_positives', 0) or 0)}/{total_labeled_ads}",
        },
        "deltas": {
            "detector": _delta(baseline["detector_metrics"], candidate["detector_metrics"]),
            "passthrough": _delta(baseline["passthrough_metrics"], candidate["passthrough_metrics"]),
            "normal_refined": _delta(baseline["normal_refined_metrics"], candidate["normal_refined_metrics"]),
        },
        "artifacts": {
            "page_report": str(page_report_path),
            "overlay_dir": str(overlay_dir),
            "overlay_shortlist_pages": overlay_data["selected_pages"],
            "baseline_detection_dirs": baseline["detection_dirs"],
            "candidate_detection_dirs": candidate["detection_dirs"],
        },
    }
    comparison_path = RUN_STATE / "stage2_recall_push_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    det_tp_before = int(baseline["detector_metrics"].get("true_positives", 0) or 0)
    det_tp_after = int(candidate["detector_metrics"].get("true_positives", 0) or 0)
    pass_tp_before = int(baseline["passthrough_metrics"].get("true_positives", 0) or 0)
    pass_tp_after = int(candidate["passthrough_metrics"].get("true_positives", 0) or 0)
    norm_tp_before = int(baseline["normal_refined_metrics"].get("true_positives", 0) or 0)
    norm_tp_after = int(candidate["normal_refined_metrics"].get("true_positives", 0) or 0)

    diagnosis = {
        "generated_at_utc": page_report["generated_at_utc"],
        "exact_tp_before_vs_after": {
            "detector": f"{det_tp_before}/{total_labeled_ads} -> {det_tp_after}/{total_labeled_ads}",
            "passthrough": f"{pass_tp_before}/{total_labeled_ads} -> {pass_tp_after}/{total_labeled_ads}",
            "normal_refined": f"{norm_tp_before}/{total_labeled_ads} -> {norm_tp_after}/{total_labeled_ads}",
        },
        "detector_tp_increased": det_tp_after > det_tp_before,
        "passthrough_tp_increased": pass_tp_after > pass_tp_before,
        "normal_refined_tp_increased": norm_tp_after > norm_tp_before,
        "should_relax_stage3_more_next": bool(det_tp_after > det_tp_before and norm_tp_after < det_tp_after),
        "next_move_if_tp_still_low": (
            "Push Stage2 recall further: lower min-area/min-size, increase max_detections, and keep overlapping candidates from multi-kernel/xy-cut proposals."
            if det_tp_after <= det_tp_before
            else "Keep v4 proposal density, then tune Stage3 recall_friendly as a secondary pass to preserve more detector true positives."
        ),
        "honesty_note": (
            "False positives may rise substantially in recall-first mode; this run prioritizes true-positive movement."
        ),
    }
    diagnosis_path = RUN_STATE / "stage2_recall_push_diagnosis.json"
    diagnosis_path.write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"page report: {page_report_path}")
    print(f"comparison: {comparison_path}")
    print(f"diagnosis: {diagnosis_path}")
    print(f"overlays: {overlay_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
