#!/usr/bin/env python3
"""Automated benchmark + safe parameter tuning loop for stage01-stage03 detection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_STATE_DIR = PROJECT_ROOT / "run_state"
BACKUP_DIR = PROJECT_ROOT / "research" / "auto_patch_backups"
DETECTION_PARAMS_PATH = PROJECT_ROOT / "configs" / "detection_params.json"
TEMP_MERGED_LABELS_PATH = RUN_STATE_DIR / "merged_labels_tmp.json"
BENCHMARK_RESULTS_PATH = RUN_STATE_DIR / "benchmark_results.json"
AUTO_IMPROVE_LOG_PATH = RUN_STATE_DIR / "auto_improve_log.txt"

SAFE_BOUNDS = {
    "contour_area_min": (200, 20000),
    "contour_area_max": (50000, 1000000),
    "aspect_ratio_min": (0.2, 1.5),
    "aspect_ratio_max": (1.5, 10.0),
    "block_merge_distance": (5, 120),
    "morphology_kernel_size": (3, 11),
}

DEFAULT_PARAMS = {
    "contour_area_min": 1500,
    "contour_area_max": 500000,
    "aspect_ratio_min": 0.5,
    "aspect_ratio_max": 5.0,
    "block_merge_distance": 40,
    "morphology_kernel_size": 5,
}

TARGET_METRIC = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe auto-improve loop for stage01-stage03 detector")
    parser.add_argument("--iterations", type=int, default=10, help="Maximum improvement iterations")
    parser.add_argument("--config", default="configs/pipeline_paths.json", help="Pipeline paths config")
    parser.add_argument("--labels-dir", default="data/test_labels", help="Directory of Label Studio JSON files")
    parser.add_argument(
        "--pipeline-mode",
        choices=["sequential", "streaming"],
        default="sequential",
        help="Execution mode for tools/run_detection_pipeline.py",
    )
    return parser.parse_args()


def _run_checked(cmd: list[str], capture_output: bool = False) -> str:
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=capture_output,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{stderr or stdout}")
    return proc.stdout if capture_output else ""


def _backup_file(path: Path, tag: str) -> Path | None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = BACKUP_DIR / f"{path.stem}_{tag}_{stamp}{path.suffix}"
    shutil.copy2(path, backup_path)
    return backup_path


def _normalize_label_image_name(image_name: str) -> str:
    name = Path(image_name).name
    name = re.sub(r"^[0-9a-fA-F]{8}-", "", name)

    page_match = re.fullmatch(r"page_(\d+)\.png", name)
    if page_match:
        page_no = int(page_match.group(1))
        return f"ET Delhi 18-03_p{page_no}.png"

    p_match = re.fullmatch(r"(.+)_p(\d+)\.png", name)
    if p_match:
        paper = p_match.group(1).replace("_", " ")
        page_no = int(p_match.group(2))
        return f"{paper}_p{page_no}.png"

    return name.replace("_", " ")


def _merge_labels_to_temp(labels_dir: Path, output_path: Path) -> Path:
    merged_tasks: list[dict[str, Any]] = []
    for label_file in sorted(labels_dir.glob("*.json")):
        payload = json.loads(label_file.read_text(encoding="utf-8"))
        tasks = payload.get("tasks", []) if isinstance(payload, dict) else payload
        if not isinstance(tasks, list):
            continue

        for raw_task in tasks:
            if not isinstance(raw_task, dict):
                continue
            task = json.loads(json.dumps(raw_task))
            data = task.setdefault("data", {})
            image_ref = data.get("image") or data.get("page") or data.get("file")
            if image_ref:
                normalized_name = _normalize_label_image_name(str(image_ref))
                data["image"] = f"/normalized/{normalized_name}"
            merged_tasks.append(task)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged_tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _nearest_safe_odd_kernel(value: float) -> int:
    allowed = [3, 5, 7, 9, 11]
    return min(allowed, key=lambda x: abs(x - float(value)))


def _normalize_and_clamp_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(DEFAULT_PARAMS)
    normalized.update(params)

    normalized["contour_area_min"] = int(
        round(_clamp(float(normalized["contour_area_min"]), *SAFE_BOUNDS["contour_area_min"]))
    )
    normalized["contour_area_max"] = int(
        round(_clamp(float(normalized["contour_area_max"]), *SAFE_BOUNDS["contour_area_max"]))
    )
    normalized["aspect_ratio_min"] = round(
        _clamp(float(normalized["aspect_ratio_min"]), *SAFE_BOUNDS["aspect_ratio_min"]),
        4,
    )
    normalized["aspect_ratio_max"] = round(
        _clamp(float(normalized["aspect_ratio_max"]), *SAFE_BOUNDS["aspect_ratio_max"]),
        4,
    )
    normalized["block_merge_distance"] = int(
        round(_clamp(float(normalized["block_merge_distance"]), *SAFE_BOUNDS["block_merge_distance"]))
    )
    normalized["morphology_kernel_size"] = _nearest_safe_odd_kernel(normalized["morphology_kernel_size"])

    if normalized["aspect_ratio_min"] >= normalized["aspect_ratio_max"]:
        normalized["aspect_ratio_min"] = round(
            _clamp(normalized["aspect_ratio_max"] - 0.1, *SAFE_BOUNDS["aspect_ratio_min"]),
            4,
        )
    if normalized["aspect_ratio_min"] >= normalized["aspect_ratio_max"]:
        normalized["aspect_ratio_max"] = round(
            _clamp(normalized["aspect_ratio_min"] + 0.1, *SAFE_BOUNDS["aspect_ratio_max"]),
            4,
        )

    if normalized["contour_area_min"] >= normalized["contour_area_max"]:
        normalized["contour_area_min"] = max(
            SAFE_BOUNDS["contour_area_min"][0],
            min(normalized["contour_area_max"] - 100, SAFE_BOUNDS["contour_area_min"][1]),
        )
    if normalized["contour_area_min"] >= normalized["contour_area_max"]:
        normalized["contour_area_max"] = min(
            SAFE_BOUNDS["contour_area_max"][1],
            max(normalized["contour_area_min"] + 100, SAFE_BOUNDS["contour_area_max"][0]),
        )

    return normalized


def _load_detection_params() -> dict[str, Any]:
    if not DETECTION_PARAMS_PATH.exists():
        return dict(DEFAULT_PARAMS)
    try:
        loaded = json.loads(DETECTION_PARAMS_PATH.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return dict(DEFAULT_PARAMS)
        return _normalize_and_clamp_params(loaded)
    except Exception:
        return dict(DEFAULT_PARAMS)


def _write_detection_params(params: dict[str, Any], tag: str) -> None:
    _backup_file(DETECTION_PARAMS_PATH, tag)
    normalized = _normalize_and_clamp_params(params)
    DETECTION_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETECTION_PARAMS_PATH.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _propose_candidate_params(base: dict[str, Any], iteration: int) -> dict[str, Any]:
    moves = [
        ("contour_area_min", -300),
        ("contour_area_min", +300),
        ("contour_area_max", -20000),
        ("contour_area_max", +20000),
        ("aspect_ratio_min", -0.05),
        ("aspect_ratio_min", +0.05),
        ("aspect_ratio_max", -0.15),
        ("aspect_ratio_max", +0.15),
        ("block_merge_distance", -5),
        ("block_merge_distance", +5),
        ("morphology_kernel_size", -2),
        ("morphology_kernel_size", +2),
    ]
    key, delta = moves[(iteration - 1) % len(moves)]
    candidate = dict(base)
    candidate[key] = float(candidate[key]) + float(delta)
    return _normalize_and_clamp_params(candidate)


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _score(metrics: dict[str, Any]) -> tuple[float, float, float]:
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    return (min(precision, recall), _f1(precision, recall), precision + recall)


def _extract_key_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "precision",
        "recall",
        "true_positives",
        "false_positives",
        "missed_detections",
        "total_detected_ads",
        "total_ground_truth_ads",
    ]
    return {k: metrics.get(k) for k in keys}


def _append_iteration_log(iteration: int, metrics: dict[str, Any]) -> None:
    RUN_STATE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Iteration {iteration}",
        f"precision: {float(metrics.get('precision', 0.0)):.6f}",
        f"recall: {float(metrics.get('recall', 0.0)):.6f}",
        f"true positives: {int(metrics.get('true_positives', 0) or 0)}",
        f"false positives: {int(metrics.get('false_positives', 0) or 0)}",
        f"missed detections: {int(metrics.get('missed_detections', 0) or 0)}",
        f"total detected ads: {int(metrics.get('total_detected_ads', 0) or 0)}",
        f"total ground truth ads: {int(metrics.get('total_ground_truth_ads', 0) or 0)}",
        "",
    ]
    with AUTO_IMPROVE_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    args = parse_args()

    RUN_STATE_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    labels_dir = (PROJECT_ROOT / args.labels_dir).resolve()
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    # Reset iteration log each run for clarity.
    AUTO_IMPROVE_LOG_PATH.write_text("", encoding="utf-8")

    current_params = _load_detection_params()
    _write_detection_params(current_params, "initial_baseline")

    best_params = dict(current_params)
    best_metrics: dict[str, Any] | None = None
    best_score: tuple[float, float, float] | None = None
    iteration_results: list[dict[str, Any]] = []
    stop_reason = "max_iterations_reached"

    for iteration in range(1, int(args.iterations) + 1):
        print(f"\n[auto-improve] Iteration {iteration}/{args.iterations}")
        _run_checked([sys.executable, "tools/reset_test_dataset.py"])
        _run_checked(
            [
                sys.executable,
                "tools/run_detection_pipeline.py",
                "--config",
                args.config,
                "--mode",
                args.pipeline_mode,
            ]
        )

        merged_labels = _merge_labels_to_temp(labels_dir, TEMP_MERGED_LABELS_PATH)
        eval_stdout = _run_checked(
            [
                sys.executable,
                "tools/evaluate_against_labels.py",
                "--labels",
                str(merged_labels),
                "--detections",
                "run_state/detections",
                "--stage",
                "refined",
            ],
            capture_output=True,
        )
        metrics = json.loads(eval_stdout.strip())
        key_metrics = _extract_key_metrics(metrics)
        current_score = _score(metrics)

        print(
            "[auto-improve] metrics "
            f"precision={float(metrics.get('precision', 0.0)):.6f}, "
            f"recall={float(metrics.get('recall', 0.0)):.6f}, "
            f"fp={int(metrics.get('false_positives', 0) or 0)}, "
            f"missed={int(metrics.get('missed_detections', 0) or 0)}"
        )
        _append_iteration_log(iteration, metrics)

        improved = best_score is None or current_score > best_score
        if improved:
            best_score = current_score
            best_metrics = dict(metrics)
            best_params = dict(current_params)

        iteration_results.append(
            {
                "iteration": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "params": dict(current_params),
                "metrics": key_metrics,
                "score": {
                    "min_precision_recall": round(current_score[0], 6),
                    "f1": round(current_score[1], 6),
                },
                "improved": improved,
            }
        )

        precision = float(metrics.get("precision", 0.0))
        recall = float(metrics.get("recall", 0.0))
        if precision >= TARGET_METRIC and recall >= TARGET_METRIC:
            stop_reason = "target_precision_recall_reached"
            break

        next_params = _propose_candidate_params(best_params, iteration)
        _write_detection_params(next_params, f"iteration_{iteration}")
        current_params = dict(next_params)

    # Persist best params as final detector config.
    _write_detection_params(best_params, "final_best")
    final_metrics = best_metrics or {}

    summary = {
        "stop_reason": stop_reason,
        "iterations_run": len(iteration_results),
        "best_params": best_params,
        "best_metrics": _extract_key_metrics(final_metrics),
        "history": iteration_results,
    }
    BENCHMARK_RESULTS_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[auto-improve] Finished. Results saved to {BENCHMARK_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
