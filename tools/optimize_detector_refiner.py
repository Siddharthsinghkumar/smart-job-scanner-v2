#!/usr/bin/env python3
"""Iterative evaluation + safe optimization loop for stage02/stage03."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_STATE = PROJECT_ROOT / "run_state"
EVAL_HISTORY = RUN_STATE / "eval_history.json"
EVAL_FAILURES_DIR = RUN_STATE / "eval_failures"
EXPERIMENTS_DIR = RUN_STATE / "experiments"
FINAL_REPORT = RUN_STATE / "final_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize detector/refiner using labeled test data")
    parser.add_argument("--labels", default="labeling_exports/ls_export.json")
    parser.add_argument("--detections", default="run_state/detections")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-runs", type=int, default=8)
    parser.add_argument("--python-bin", default=str(PROJECT_ROOT / "4_env" / "bin" / "python"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_once(
    run_id: str,
    labels_path: Path,
    detections_dir: Path,
    python_bin: str,
    iou_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    details_path = EVAL_FAILURES_DIR / f"{run_id}.json"
    cmd = [
        python_bin,
        "tools/evaluate_against_labels.py",
        "--labels",
        str(labels_path),
        "--detections",
        str(detections_dir),
        "--iou-threshold",
        str(iou_threshold),
        "--output-details",
        str(details_path),
    ]
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"Evaluation failed (rc={rc}): {err.strip() or out.strip()}")

    metrics = json.loads(out)
    details = load_json(details_path, {})

    # pattern extraction for failure analysis
    per_image = details.get("per_image", {}) if isinstance(details, dict) else {}
    small_missed = 0
    merged_ads = 0
    noisy_blocks = 0
    incorrect_regions = 0

    for image_data in per_image.values():
        missed = image_data.get("missed_detections", [])
        low_iou = image_data.get("low_iou_matches", [])
        fp = image_data.get("false_positives", [])

        for b in missed:
            if len(b) == 4:
                area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
                if area < 20000:
                    small_missed += 1

        for li in low_iou:
            gt = li.get("gt_bbox", [])
            pred = li.get("pred_bbox", [])
            if len(gt) == 4 and len(pred) == 4:
                gt_area = max(0, gt[2] - gt[0]) * max(0, gt[3] - gt[1])
                pred_area = max(0, pred[2] - pred[0]) * max(0, pred[3] - pred[1])
                if gt_area > 0 and pred_area > (1.8 * gt_area):
                    merged_ads += 1

        for b in fp:
            if len(b) == 4:
                area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
                if area < 5000:
                    noisy_blocks += 1
                else:
                    incorrect_regions += 1

    details["patterns"] = {
        "small_boxes_missed": small_missed,
        "merged_ads": merged_ads,
        "noisy_blocks": noisy_blocks,
        "incorrect_regions": incorrect_regions,
    }
    save_json(details_path, details)
    return metrics, details


def run_pipeline_once(python_bin: str, limit: int, env_overrides: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(env_overrides)
    state_dir = PROJECT_ROOT / "data" / "test_output" / "runtime" / "run_state"
    for marker in (
        "src_pipeline_stage02_block_detection.py.done",
        "src_pipeline_stage03_block_refiner.py.done",
    ):
        (state_dir / marker).unlink(missing_ok=True)

    cmd = [python_bin, "scripts/run_pipeline.py", "--limit", str(limit), "--debug"]
    rc, out, err = run_cmd(cmd, env=env)
    if rc != 0:
        raise RuntimeError(f"Pipeline failed (rc={rc}): {err.strip() or out.strip()}")


def should_stop(history: list[dict[str, Any]], threshold_met: bool, degraded: bool) -> tuple[bool, str | None]:
    if threshold_met:
        return True, "target_metrics_reached"
    if degraded:
        return True, "performance_degraded_after_change"
    if len(history) >= 3:
        recent = history[-3:]
        f1_vals = [float(r.get("f1", 0.0)) for r in recent]
        if max(f1_vals) - min(f1_vals) < 0.01:
            return True, "improvement_below_1_percent_last_3_runs"
    return False, None


def main() -> int:
    args = parse_args()

    labels_path = (PROJECT_ROOT / args.labels).resolve() if not Path(args.labels).is_absolute() else Path(args.labels)
    detections_dir = (PROJECT_ROOT / args.detections).resolve() if not Path(args.detections).is_absolute() else Path(args.detections)

    if not labels_path.exists():
        raise SystemExit(f"Labels file not found: {labels_path}")

    RUN_STATE.mkdir(parents=True, exist_ok=True)
    EVAL_FAILURES_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = load_json(EVAL_HISTORY, [])
    if not isinstance(history, list):
        history = []

    candidate_changes = [
        {"DETECTOR_MIN_AREA": "4500"},
        {"DETECTOR_KERNEL_SIZE": "5"},
        {"DETECTOR_MORPH_ITERATIONS": "3"},
        {"REFINER_MIN_GAP_WIDTH": "15"},
        {"REFINER_MIN_COL_WIDTH": "180"},
        {"REFINER_GRAPHIC_THRESHOLD": "0.10"},
        {"REFINER_MIN_CONFIDENCE": "35"},
    ]

    best_params: dict[str, str] = {}
    best_metrics: dict[str, Any] | None = None
    stop_reason: str | None = None

    runs_to_execute = min(args.max_runs, len(candidate_changes) + 1)

    for idx in range(runs_to_execute):
        run_id = f"run_{len(history) + 1:03d}"
        params = dict(best_params)
        notes = "baseline"

        if idx > 0:
            change = candidate_changes[idx - 1]
            params.update(change)
            notes = f"trial_change={change}"

        run_pipeline_once(args.python_bin, args.limit, params)
        metrics, _details = evaluate_once(
            run_id=run_id,
            labels_path=labels_path,
            detections_dir=detections_dir,
            python_bin=args.python_bin,
            iou_threshold=args.iou_threshold,
        )

        precision = float(metrics.get("precision", 0.0))
        recall = float(metrics.get("recall", 0.0))
        f1 = _f1(precision, recall)

        run_record = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "notes": notes,
        }

        history.append(run_record)
        save_json(EVAL_HISTORY, history)
        save_json(EXPERIMENTS_DIR / f"{run_id}.json", run_record)

        degraded = False
        if len(history) >= 2:
            prev_f1 = float(history[-2].get("f1", 0.0))
            degraded = f1 < prev_f1

        if best_metrics is None or f1 > _f1(float(best_metrics.get("precision", 0.0)), float(best_metrics.get("recall", 0.0))):
            best_metrics = {"precision": precision, "recall": recall}
            best_params = dict(params)

        threshold_met = precision >= 0.85 and recall >= 0.80
        should_end, reason = should_stop(history, threshold_met, degraded)
        if should_end:
            stop_reason = reason
            break

    if best_metrics is None:
        best_metrics = {"precision": 0.0, "recall": 0.0}

    final_report = {
        "best_params": best_params,
        "final_precision": round(float(best_metrics.get("precision", 0.0)), 6),
        "final_recall": round(float(best_metrics.get("recall", 0.0)), 6),
        "total_runs": len(history),
        "stop_reason": stop_reason or "max_runs_reached",
    }
    save_json(FINAL_REPORT, final_report)
    print(json.dumps(final_report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
