#!/usr/bin/env python3
"""Geometry and Stage3 passthrough-vs-normal diagnostic benchmark."""

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
from src.utils.benchmark_images import (
    DEFAULT_BENCHMARK_IMAGES_DIR,
    DEFAULT_BENCHMARK_MANIFEST_PATH,
    assert_benchmark_manifest_valid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage3 passthrough vs normal debug benchmark")
    parser.add_argument("--config", default="configs/benchmark_pipeline_paths.json", help="Pipeline paths config")
    parser.add_argument("--detector-version", choices=["v1", "v2", "v3", "v4"], default="v1", help="Stage2 detector version")
    parser.add_argument("--labels-dir", default="data/test_labels", help="Label JSON directory")
    parser.add_argument(
        "--images-dir",
        default=str(DEFAULT_BENCHMARK_IMAGES_DIR.relative_to(PROJECT_ROOT)),
        help="Frozen benchmark page image directory",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default=str(DEFAULT_BENCHMARK_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        help="Benchmark image manifest for immutability validation",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--top-n", type=int, default=20, help="Top-N worst-page overlays")
    return parser.parse_args()


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr or proc.stdout}")
    return proc.stdout


def _copy_detections(dst_dir: Path) -> None:
    src = RUN_STATE / "detections"
    if not src.is_dir():
        raise RuntimeError(f"detections dir missing: {src}")
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src, dst_dir)


def _evaluate(labels: dict[str, list[list[int]]], detections_dir: Path, stage: str, iou: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    preds = load_pipeline_detections(detections_dir, stage)
    metrics, per_image = compute_metrics(labels, preds, iou)
    return metrics, per_image, preds


def _draw_boxes(img, boxes: list[list[int]], color: tuple[int, int, int]) -> None:
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def _draw_legend(img) -> None:
    entries = [
        ("GT", (0, 255, 0)),
        ("Detector", (0, 165, 255)),
        ("Passthrough", (255, 140, 0)),
        ("Normal", (0, 0, 255)),
    ]
    x, y = 20, 30
    for name, color in entries:
        cv2.rectangle(img, (x, y - 12), (x + 16, y + 4), color, -1)
        cv2.putText(img, name, (x + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name)


def _diagnose(
    detector_metrics: dict[str, Any],
    passthrough_metrics: dict[str, Any],
    normal_metrics: dict[str, Any],
    geometry_summary: dict[str, Any],
    rejection_summary: dict[str, Any],
) -> dict[str, Any]:
    total_gt = int(detector_metrics.get("total_ground_truth_ads", 0) or 0)
    det_tp = int(detector_metrics.get("true_positives", 0) or 0)
    pass_tp = int(passthrough_metrics.get("true_positives", 0) or 0)
    norm_tp = int(normal_metrics.get("true_positives", 0) or 0)

    audited = max(1, int(geometry_summary.get("total_blocks_audited", 0) or 0))
    proj_oob = int(geometry_summary.get("blocks_with_projected_out_of_bounds_boxes", 0) or 0)
    proj_outside = int(geometry_summary.get("blocks_with_projected_outside_parent_boxes", 0) or 0)
    missing_det = int(geometry_summary.get("blocks_missing_detector_entries", 0) or 0)
    idx_mismatch = int(geometry_summary.get("blocks_with_block_index_mismatch", 0) or 0)

    projection_issue_rate = (proj_oob + proj_outside + idx_mismatch) / audited
    projection_broken = projection_issue_rate > 0.05 or idx_mismatch > 0

    pass_vs_det_tp_gap = abs(pass_tp - det_tp)
    pass_vs_det_fp_gap = abs(int(passthrough_metrics.get("false_positives", 0) or 0) - int(detector_metrics.get("false_positives", 0) or 0))
    passthrough_differs_from_detector = pass_vs_det_tp_gap > 0 or pass_vs_det_fp_gap > 0

    semantic_rejections = 0
    reason_counts = rejection_summary.get("reason_counts", {}) if isinstance(rejection_summary, dict) else {}
    for key in ("no_text", "under_letter_threshold", "graphic"):
        semantic_rejections += int(reason_counts.get(key, 0) or 0)
    total_rejections = int(rejection_summary.get("total_rejected_candidates", 0) or 0)
    semantic_rejection_ratio = (
        min(1.0, semantic_rejections / total_rejections) if total_rejections > 0 else 0.0
    )

    filtering_too_aggressive = (
        pass_tp > 0 and norm_tp <= max(0, int(pass_tp * 0.5))
    ) or (
        total_rejections > 0 and semantic_rejection_ratio >= 0.7 and norm_tp < pass_tp
    )

    if det_tp == 0:
        dominant_issue = "stage2_proposal_quality"
        confidence = "high"
    elif projection_broken and passthrough_differs_from_detector:
        dominant_issue = "stage3_projection_or_write_path"
        confidence = "medium"
    elif filtering_too_aggressive:
        dominant_issue = "stage3_semantic_filtering"
        confidence = "medium"
    else:
        dominant_issue = "mixed_or_unclear"
        confidence = "low"

    return {
        "answers": {
            "are_stage2_proposals_already_too_poor": {
                "value": det_tp == 0,
                "evidence": {
                    "detector_true_positives": det_tp,
                    "total_labeled_ads": total_gt,
                },
            },
            "is_stage3_bbox_projection_broken": {
                "value": bool(projection_broken),
                "evidence": {
                    "projection_issue_rate": round(projection_issue_rate, 6),
                    "projected_out_of_bounds": proj_oob,
                    "projected_outside_parent": proj_outside,
                    "block_index_mismatch": idx_mismatch,
                    "missing_detector_entries": missing_det,
                    "passthrough_differs_from_detector": passthrough_differs_from_detector,
                },
            },
            "is_stage3_filtering_too_aggressive": {
                "value": bool(filtering_too_aggressive),
                "evidence": {
                    "passthrough_tp": pass_tp,
                    "normal_tp": norm_tp,
                    "total_rejections": total_rejections,
                    "semantic_rejection_ratio": round(semantic_rejection_ratio, 6),
                },
            },
            "are_text_or_ocr_filters_killing_likely_ads": {
                "value": bool(total_rejections > 0 and semantic_rejection_ratio >= 0.7),
                "evidence": {
                    "no_text": int(reason_counts.get("no_text", 0) or 0),
                    "under_letter_threshold": int(reason_counts.get("under_letter_threshold", 0) or 0),
                    "graphic": int(reason_counts.get("graphic", 0) or 0),
                    "total_rejected_candidates": total_rejections,
                },
            },
            "exact_next_fix": {
                "value": (
                    "Fix Stage2 proposal quality first (raise detector recall on dense classified pages), "
                    "then rerun passthrough-vs-normal with the same frozen detector to isolate Stage3 behavior."
                    if dominant_issue == "stage2_proposal_quality"
                    else (
                        "Instrument and fix projection/write path inconsistencies first, then re-measure detector-vs-passthrough parity."
                        if dominant_issue == "stage3_projection_or_write_path"
                        else (
                            "Relax semantic rejection thresholds (text/letter filters) and re-measure normal refined recall against passthrough."
                            if dominant_issue == "stage3_semantic_filtering"
                            else "Re-run with per-page matched-pair inspection before changing logic."
                        )
                    )
                )
            },
        },
        "dominant_issue": dominant_issue,
        "confidence": confidence,
    }


def main() -> int:
    args = parse_args()
    RUN_STATE.mkdir(parents=True, exist_ok=True)

    labels_dir = (PROJECT_ROOT / args.labels_dir).resolve()
    images_dir = (PROJECT_ROOT / args.images_dir).resolve()
    benchmark_manifest = (PROJECT_ROOT / args.benchmark_manifest).resolve()
    if not labels_dir.is_dir():
        raise SystemExit(f"labels dir not found: {labels_dir}")
    if not images_dir.is_dir():
        raise SystemExit(f"images dir not found: {images_dir}")
    assert_benchmark_manifest_valid(
        benchmark_images_dir=images_dir,
        manifest_path=benchmark_manifest,
    )

    merged_labels_path = RUN_STATE / "merged_labels_tmp_stage3_debug.json"
    _merge_labels_to_temp(labels_dir, merged_labels_path)
    labels = load_labelstudio_boxes(merged_labels_path)

    debug_root = RUN_STATE / "stage3_debug_runs"
    passthrough_det_dir = debug_root / "passthrough_detections"
    normal_det_dir = debug_root / "normal_detections"

    print("[debug] reset + passthrough run")
    _run([sys.executable, "tools/reset_test_dataset.py"])
    _run(
        [
            sys.executable,
            "tools/run_detection_pipeline.py",
            "--config",
            args.config,
            "--mode",
            "sequential",
            "--detector-version",
            args.detector_version,
            "--stage3-mode",
            "passthrough",
            "--skip-stage1",
            "--validate-benchmark-images",
            "--benchmark-manifest",
            str(benchmark_manifest),
        ]
    )
    _copy_detections(passthrough_det_dir)

    print("[debug] reset + normal run with geometry/rejection logs")
    _run([sys.executable, "tools/reset_test_dataset.py"])
    _run(
        [
            sys.executable,
            "tools/run_detection_pipeline.py",
            "--config",
            args.config,
            "--mode",
            "sequential",
            "--detector-version",
            args.detector_version,
            "--stage3-mode",
            "normal",
            "--skip-stage1",
            "--validate-benchmark-images",
            "--benchmark-manifest",
            str(benchmark_manifest),
            "--geometry-audit",
            "--rejection-log",
            "--geometry-audit-output",
            "run_state/stage3_geometry_audit.json",
            "--geometry-audit-summary-output",
            "run_state/stage3_geometry_audit_summary.json",
            "--rejection-log-output",
            "run_state/stage3_rejection_log.json",
            "--rejection-summary-output",
            "run_state/stage3_rejection_summary.json",
        ]
    )
    _copy_detections(normal_det_dir)

    detector_metrics, detector_per_image, detector_map = _evaluate(
        labels, normal_det_dir, "detector", args.iou_threshold
    )
    passthrough_metrics, passthrough_per_image, passthrough_map = _evaluate(
        labels, passthrough_det_dir, "refined", args.iou_threshold
    )
    normal_metrics, normal_per_image, normal_map = _evaluate(
        labels, normal_det_dir, "refined", args.iou_threshold
    )

    geometry_summary_path = RUN_STATE / "stage3_geometry_audit_summary.json"
    rejection_summary_path = RUN_STATE / "stage3_rejection_summary.json"
    geometry_summary = (
        json.loads(geometry_summary_path.read_text(encoding="utf-8"))
        if geometry_summary_path.exists()
        else {}
    )
    rejection_summary = (
        json.loads(rejection_summary_path.read_text(encoding="utf-8"))
        if rejection_summary_path.exists()
        else {}
    )

    # Build worst-page ranking for overlay generation.
    geometry_rows = []
    geometry_audit_path = RUN_STATE / "stage3_geometry_audit.json"
    if geometry_audit_path.exists():
        geometry_payload = json.loads(geometry_audit_path.read_text(encoding="utf-8"))
        geometry_rows = geometry_payload.get("rows", []) if isinstance(geometry_payload, dict) else []
    geom_page_issues = Counter()
    for row in geometry_rows:
        if not isinstance(row, dict):
            continue
        page = str(row.get("page_name") or "")
        if not page:
            continue
        flags = row.get("status_flags", [])
        geom_page_issues[page] += len(flags)

    rejection_rows = []
    rejection_log_path = RUN_STATE / "stage3_rejection_log.json"
    if rejection_log_path.exists():
        rejection_payload = json.loads(rejection_log_path.read_text(encoding="utf-8"))
        rejection_rows = rejection_payload.get("rows", []) if isinstance(rejection_payload, dict) else []
    reject_page_counts = Counter()
    for row in rejection_rows:
        if not isinstance(row, dict):
            continue
        page = str(row.get("page_name") or "")
        if page:
            reject_page_counts[page] += 1

    overlay_dir = RUN_STATE / "stage3_debug_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    image_index = {p.name: p for p in images_dir.rglob("*.png")}
    pages = sorted(set(labels.keys()) | set(detector_map.keys()) | set(passthrough_map.keys()) | set(normal_map.keys()))
    worst_rows = []

    for page in pages:
        gt_boxes = labels.get(page, [])
        det_rows = detector_map.get(page, [])
        pass_rows = passthrough_map.get(page, [])
        norm_rows = normal_map.get(page, [])

        det_boxes = [r.get("bbox") for r in det_rows if isinstance(r, dict) and isinstance(r.get("bbox"), list)]
        pass_boxes = [r.get("bbox") for r in pass_rows if isinstance(r, dict) and isinstance(r.get("bbox"), list)]
        norm_boxes = [r.get("bbox") for r in norm_rows if isinstance(r, dict) and isinstance(r.get("bbox"), list)]

        det_tp = int(detector_per_image.get(page, {}).get("true_positives", 0) or 0)
        pass_tp = int(passthrough_per_image.get(page, {}).get("true_positives", 0) or 0)
        norm_tp = int(normal_per_image.get(page, {}).get("true_positives", 0) or 0)

        worst_score = (
            int(len(gt_boxes)) * 3
            + (30 if (len(det_boxes) > 0 and norm_tp == 0 and len(gt_boxes) > 0) else 0)
            + int(geom_page_issues.get(page, 0)) * 2
            + int(reject_page_counts.get(page, 0))
            + int(pass_tp - norm_tp) * 2
        )

        worst_rows.append(
            {
                "page": page,
                "label_count": len(gt_boxes),
                "detector_count": len(det_boxes),
                "passthrough_count": len(pass_boxes),
                "normal_count": len(norm_boxes),
                "detector_tp": det_tp,
                "passthrough_tp": pass_tp,
                "normal_tp": norm_tp,
                "geometry_issue_count": int(geom_page_issues.get(page, 0)),
                "rejection_count": int(reject_page_counts.get(page, 0)),
                "worst_score": worst_score,
                "image_exists": page in image_index,
            }
        )

    ranked = sorted(worst_rows, key=lambda r: (r["worst_score"], r["label_count"]), reverse=True)
    selected = ranked[: max(1, int(args.top_n))]

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
        det_boxes = [r.get("bbox") for r in detector_map.get(page, []) if isinstance(r, dict) and isinstance(r.get("bbox"), list)]
        pass_boxes = [r.get("bbox") for r in passthrough_map.get(page, []) if isinstance(r, dict) and isinstance(r.get("bbox"), list)]
        norm_boxes = [r.get("bbox") for r in normal_map.get(page, []) if isinstance(r, dict) and isinstance(r.get("bbox"), list)]

        _draw_boxes(img, gt_boxes, (0, 255, 0))
        _draw_boxes(img, det_boxes, (0, 165, 255))
        _draw_boxes(img, pass_boxes, (255, 140, 0))
        _draw_boxes(img, norm_boxes, (0, 0, 255))
        _draw_legend(img)
        cv2.putText(
            img,
            f"{page} | GT:{len(gt_boxes)} D:{len(det_boxes)} P:{len(pass_boxes)} N:{len(norm_boxes)}",
            (20, max(30, img.shape[0] - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_name = _safe_name(page)
        cv2.imwrite(str(overlay_dir / out_name), img)
        rendered.append(page)

    total_gt = int(detector_metrics.get("total_ground_truth_ads", 0) or 0)
    diagnosis = _diagnose(detector_metrics, passthrough_metrics, normal_metrics, geometry_summary, rejection_summary)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "detector_version": args.detector_version,
        "iou_threshold": float(args.iou_threshold),
        "detector": detector_metrics,
        "passthrough_refined": passthrough_metrics,
        "normal_refined": normal_metrics,
        "headline": {
            "detector_tp_over_total": f"{int(detector_metrics.get('true_positives', 0) or 0)}/{total_gt}",
            "passthrough_tp_over_total": f"{int(passthrough_metrics.get('true_positives', 0) or 0)}/{total_gt}",
            "normal_refined_tp_over_total": f"{int(normal_metrics.get('true_positives', 0) or 0)}/{total_gt}",
            "projection_appears_broken": diagnosis["answers"]["is_stage3_bbox_projection_broken"]["value"],
            "filtering_appears_too_aggressive": diagnosis["answers"]["is_stage3_filtering_too_aggressive"]["value"],
            "dominant_issue": diagnosis["dominant_issue"],
        },
        "geometry_audit_summary": geometry_summary,
        "rejection_summary": rejection_summary,
        "artifacts": {
            "passthrough_detections": str(passthrough_det_dir),
            "normal_detections": str(normal_det_dir),
            "geometry_audit": str(RUN_STATE / "stage3_geometry_audit.json"),
            "geometry_audit_summary": str(RUN_STATE / "stage3_geometry_audit_summary.json"),
            "rejection_log": str(RUN_STATE / "stage3_rejection_log.json"),
            "rejection_summary": str(RUN_STATE / "stage3_rejection_summary.json"),
            "overlay_dir": str(overlay_dir),
        },
    }

    report_path = RUN_STATE / "stage3_passthrough_vs_normal_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    worst_payload = {
        "generated_at_utc": report["generated_at_utc"],
        "ranked_pages": ranked,
        "selected_pages": selected,
        "rendered_pages": rendered,
        "overlay_dir": str(overlay_dir),
    }
    worst_path = RUN_STATE / "stage3_debug_worst_pages.json"
    worst_path.write_text(json.dumps(worst_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    diagnosis_payload = {
        "generated_at_utc": report["generated_at_utc"],
        "detector_version": args.detector_version,
        "detector_metrics": detector_metrics,
        "passthrough_metrics": passthrough_metrics,
        "normal_metrics": normal_metrics,
        "geometry_summary": geometry_summary,
        "rejection_summary": rejection_summary,
        **diagnosis,
    }
    diagnosis_path = RUN_STATE / "stage3_debug_diagnosis.json"
    diagnosis_path.write_text(json.dumps(diagnosis_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"report: {report_path}")
    print(f"worst pages: {worst_path}")
    print(f"diagnosis: {diagnosis_path}")
    print(f"overlays: {overlay_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
