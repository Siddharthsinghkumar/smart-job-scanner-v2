#!/usr/bin/env python3
"""Run honest sequential Stage2 detector benchmark for v1/v2/v3 and write comparison artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_STATE = PROJECT_ROOT / "run_state"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.benchmark_images import (
    DEFAULT_BENCHMARK_IMAGES_DIR,
    DEFAULT_BENCHMARK_MANIFEST_PATH,
    assert_benchmark_manifest_valid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Stage2 detector versions v1/v2/v3")
    parser.add_argument(
        "--config",
        default="configs/benchmark_pipeline_paths.json",
        help="Pipeline paths config (benchmark-safe config should point images_output at data/benchmark_images)",
    )
    parser.add_argument(
        "--benchmark-images-dir",
        default=str(DEFAULT_BENCHMARK_IMAGES_DIR.relative_to(PROJECT_ROOT)),
        help="Frozen benchmark images directory",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default=str(DEFAULT_BENCHMARK_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        help="Benchmark image manifest for hash/filename validation",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for stagewise evaluation")
    parser.add_argument("--top-n-overlays", type=int, default=20, help="Worst pages to render as overlays")
    return parser.parse_args()


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr or proc.stdout}")
    return proc.stdout


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _version_paths(version: str) -> dict[str, Path]:
    assert version in {"v1", "v2", "v3"}
    return {
        "stagewise": RUN_STATE / ("stage2_v3_stagewise_eval_report.json" if version == "v3" else f"stage2_{version}_stagewise_eval_report.json"),
        "audit": RUN_STATE / f"stage2_{version}_label_alignment_audit.json",
        "worst": RUN_STATE / ("stage2_v3_worst_pages.json" if version == "v3" else f"stage2_{version}_worst_pages.json"),
        "overlay_dir": RUN_STATE / ("stage2_v3_overlays" if version == "v3" else f"stage2_{version}_overlays"),
    }


def _collect_coverage(audit: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in audit.get("label_pages", []) if isinstance(r, dict)]
    labeled_rows = [r for r in rows if int(r.get("labeled_box_count", 0)) > 0]
    detector_nonzero = sum(1 for r in labeled_rows if int(r.get("detector_box_count", 0)) > 0)
    refined_nonzero = sum(1 for r in labeled_rows if int(r.get("refined_box_count", 0)) > 0)
    pages_with_printed_number = sum(1 for r in rows if isinstance(r.get("printed_page_number"), int))

    dim_mismatch = []
    for r in labeled_rows:
        scale = r.get("label_to_pipeline_scale", {})
        sx = scale.get("sx") if isinstance(scale, dict) else None
        sy = scale.get("sy") if isinstance(scale, dict) else None
        if sx is None or sy is None:
            continue
        try:
            fsx = float(sx)
            fsy = float(sy)
        except Exception:
            continue
        if abs(fsx - 1.0) > 1e-6 or abs(fsy - 1.0) > 1e-6:
            dim_mismatch.append(
                {
                    "page": r.get("normalized_page_key"),
                    "label_dimensions": r.get("label_image_dimensions"),
                    "pipeline_dimensions": r.get("pipeline_image_dimensions"),
                    "scale": {"sx": round(fsx, 6), "sy": round(fsy, 6)},
                }
            )

    top_labeled = sorted(
        [
            {
                "page": r.get("normalized_page_key"),
                "pdf_page_index": r.get("pdf_page_index"),
                "printed_page_number": r.get("printed_page_number"),
                "labeled_box_count": int(r.get("labeled_box_count", 0)),
                "detector_box_count": int(r.get("detector_box_count", 0)),
                "refined_box_count": int(r.get("refined_box_count", 0)),
            }
            for r in labeled_rows
        ],
        key=lambda x: x["labeled_box_count"],
        reverse=True,
    )[:12]

    return {
        "labeled_pages_count": len(labeled_rows),
        "pages_with_detected_printed_number": pages_with_printed_number,
        "detector_nonzero_pages": detector_nonzero,
        "refined_nonzero_pages": refined_nonzero,
        "top_labeled_pages": top_labeled,
        "dimension_mismatch_pages": dim_mismatch,
    }


def _run_for_version(
    version: str,
    config_path: str,
    iou_threshold: float,
    top_n: int,
    benchmark_images_dir: str,
    benchmark_manifest: str,
) -> dict[str, Any]:
    paths = _version_paths(version)

    print(f"\n[benchmark] ===== {version} =====")
    _run([sys.executable, "tools/reset_test_dataset.py"])

    start = time.time()
    _run(
        [
            sys.executable,
            "tools/run_detection_pipeline.py",
            "--config",
            config_path,
            "--mode",
            "sequential",
            "--detector-version",
            version,
            "--skip-stage1",
            "--validate-benchmark-images",
            "--benchmark-manifest",
            benchmark_manifest,
        ]
    )
    runtime_sec = round(time.time() - start, 2)

    _run(
        [
            sys.executable,
            "tools/run_stagewise_evaluation.py",
            "--labels-dir",
            "data/test_labels",
            "--detections-dir",
            "run_state/detections",
            "--images-dir",
            benchmark_images_dir,
            "--validate-benchmark-images",
            "--benchmark-manifest",
            benchmark_manifest,
            "--iou-threshold",
            str(iou_threshold),
            "--output",
            str(paths["stagewise"].relative_to(PROJECT_ROOT)),
        ]
    )

    _run(
        [
            sys.executable,
            "tools/audit_label_alignment.py",
            "--labels-dir",
            "data/test_labels",
            "--images-dir",
            benchmark_images_dir,
            "--detections-dir",
            "run_state/detections",
            "--page-identity-map",
            "run_state/page_identity_map.json",
            "--output",
            str(paths["audit"].relative_to(PROJECT_ROOT)),
            "--validate-benchmark-images",
            "--benchmark-manifest",
            benchmark_manifest,
        ]
    )

    _run(
        [
            sys.executable,
            "tools/render_benchmark_overlays.py",
            "--labels-dir",
            "data/test_labels",
            "--images-dir",
            benchmark_images_dir,
            "--detections-dir",
            "run_state/detections",
            "--audit-json",
            str(paths["audit"].relative_to(PROJECT_ROOT)),
            "--page-identity-map",
            "run_state/page_identity_map.json",
            "--output-dir",
            str(paths["overlay_dir"].relative_to(PROJECT_ROOT)),
            "--shortlist-output",
            str(paths["worst"].relative_to(PROJECT_ROOT)),
            "--top-n",
            str(int(top_n)),
            "--scale-gt-when-dim-mismatch",
            "--validate-benchmark-images",
            "--benchmark-manifest",
            benchmark_manifest,
        ]
    )

    stagewise = _load_json(paths["stagewise"])
    audit = _load_json(paths["audit"])
    worst = _load_json(paths["worst"])
    coverage = _collect_coverage(audit)

    return {
        "detector": stagewise.get("detector", {}),
        "refined": stagewise.get("refined", {}),
        "labeled_page_coverage": {
            "labeled_pages_count": coverage["labeled_pages_count"],
            "pages_with_detected_printed_number": coverage["pages_with_detected_printed_number"],
            "detector_nonzero_pages": coverage["detector_nonzero_pages"],
            "refined_nonzero_pages": coverage["refined_nonzero_pages"],
            "top_labeled_pages": coverage["top_labeled_pages"],
        },
        "dimension_mismatch_pages": coverage["dimension_mismatch_pages"],
        "representative_worst_pages": worst.get("ranked_pages", [])[:10],
        "worst_pages_file": str(paths["worst"].relative_to(PROJECT_ROOT)),
        "overlay_dir": str(paths["overlay_dir"].relative_to(PROJECT_ROOT)),
        "pipeline_runtime_seconds_observed": runtime_sec,
    }


def _decision(comparison: dict[str, Any]) -> dict[str, Any]:
    v1 = comparison["metrics"]["v1"]
    v2 = comparison["metrics"]["v2"]
    v3 = comparison["metrics"]["v3"]

    def _score(row: dict[str, Any]) -> tuple[float, float, int, int]:
        det = row.get("detector", {})
        ref = row.get("refined", {})
        return (
            float(det.get("recall", 0.0)),
            float(det.get("precision", 0.0)),
            int(ref.get("true_positives", 0)),
            -int(det.get("false_positives", 0)),
        )

    v3_better_than_v1 = _score(v3) > _score(v1)
    v3_better_than_v2 = _score(v3) > _score(v2)
    v3_best = v3_better_than_v1 and v3_better_than_v2

    return {
        "v3_beats_v1": v3_better_than_v1,
        "v3_beats_v2": v3_better_than_v2,
        "v3_is_best_of_three": v3_best,
        "touch_stage3_next": False,
        "reason": "Stage 2 remains the priority optimization surface; Stage 3 should only be revisited after stable Stage2 gains.",
    }


def main() -> int:
    args = parse_args()
    RUN_STATE.mkdir(parents=True, exist_ok=True)
    assert_benchmark_manifest_valid(
        benchmark_images_dir=args.benchmark_images_dir,
        manifest_path=args.benchmark_manifest,
    )

    all_metrics: dict[str, Any] = {}
    for version in ("v1", "v2", "v3"):
        all_metrics[version] = _run_for_version(
            version,
            args.config,
            args.iou_threshold,
            args.top_n_overlays,
            args.benchmark_images_dir,
            args.benchmark_manifest,
        )

    total_labeled_ads = int(all_metrics["v3"].get("detector", {}).get("total_ground_truth_ads", 0) or 0)

    comparison = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_scope": "Sequential pipeline reset + Stage1->Stage3 run for detector versions v1, v2, v3 on current 3-paper English development dataset.",
        "total_labeled_ads": total_labeled_ads,
        "metrics": all_metrics,
        "generalization_boundary": {
            "development_dataset_note": "Current benchmark is a development set (3 English papers).",
            "future_holdout_note": "Future holdout must be 4 unseen English papers + 4 unseen Hindi papers.",
            "tuning_rule": "Do not tune on that holdout before first frozen-model evaluation.",
        },
    }
    comparison["decision"] = _decision(comparison)

    comparison_path = RUN_STATE / "stage2_v1_v2_v3_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    # Design summary (implementation + benchmark-readiness)
    design_summary = {
        "generated_at_utc": comparison["generated_at_utc"],
        "objective": "Add Stage2 YOLO-based v3 path beside v1/v2 with honest A/B/C benchmark compatibility.",
        "new_components": [
            "src/vision/block_detector_v3_yolo.py",
            "src/pipeline/stage02_block_detection_v3.py",
            "tools/prepare_yolo_dataset.py",
            "tools/train_stage2_yolo.py",
            "tools/predict_stage2_yolo.py",
            "tools/benchmark_stage2_v1_v2_v3.py",
            "configs/stage2_yolo_v3.yaml",
            "configs/detection_params_v3.json",
        ],
        "runner_wiring": {
            "script": "tools/run_detection_pipeline.py",
            "detector_flag": "--detector-version v1|v2|v3",
            "default": "v1",
        },
        "contract_files": {
            "interface_summary": "run_state/stage2_v3_interface_summary.json",
            "dataset_report": "run_state/stage2_v3_dataset_report.json",
            "split_manifest": "run_state/stage2_v3_split_manifest.json",
            "training_report": "run_state/stage2_v3_training_report.json",
            "stagewise_eval_v3": "run_state/stage2_v3_stagewise_eval_report.json",
            "comparison": "run_state/stage2_v1_v2_v3_comparison.json",
        },
        "holdout_policy": comparison["generalization_boundary"],
    }
    (RUN_STATE / "stage2_v3_design_summary.json").write_text(
        json.dumps(design_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Final recommendation based on measured benchmark metrics.
    v1 = all_metrics["v1"]
    v2 = all_metrics["v2"]
    v3 = all_metrics["v3"]
    final_reco = {
        "generated_at_utc": comparison["generated_at_utc"],
        "detector_metrics": {
            "v1": v1.get("detector", {}),
            "v2": v2.get("detector", {}),
            "v3": v3.get("detector", {}),
        },
        "refined_metrics": {
            "v1": v1.get("refined", {}),
            "v2": v2.get("refined", {}),
            "v3": v3.get("refined", {}),
        },
        "verdict": {
            "v3_beats_v1_on_detector": comparison["decision"]["v3_beats_v1"],
            "v3_beats_v2_on_detector": comparison["decision"]["v3_beats_v2"],
            "refined_improves_with_v3": int(v3.get("refined", {}).get("true_positives", 0)) > max(
                int(v1.get("refined", {}).get("true_positives", 0)),
                int(v2.get("refined", {}).get("true_positives", 0)),
            ),
            "recommend_v3_as_experimental_main": bool(comparison["decision"]["v3_is_best_of_three"]),
            "touch_stage3_next": False,
        },
        "next_step_before_holdout": "Freeze current best v3 checkpoint, rerun one no-tuning reproducibility pass on the 3-paper dev set, then evaluate once on the future 4-English + 4-Hindi unseen holdout.",
        "generalization_boundary": comparison["generalization_boundary"],
    }
    (RUN_STATE / "stage2_v3_final_recommendation.json").write_text(
        json.dumps(final_reco, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"comparison report: {comparison_path}")
    print(f"v3 stagewise report: {RUN_STATE / 'stage2_v3_stagewise_eval_report.json'}")
    print(f"v3 worst pages: {RUN_STATE / 'stage2_v3_worst_pages.json'}")
    print(f"v3 overlays dir: {RUN_STATE / 'stage2_v3_overlays'}")
    print(f"design summary: {RUN_STATE / 'stage2_v3_design_summary.json'}")
    print(f"final recommendation: {RUN_STATE / 'stage2_v3_final_recommendation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
