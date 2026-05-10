#!/usr/bin/env python3
"""Stage01-stage03 pipeline runner (sequential or streaming) for benchmark evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage01-stage03 detection pipeline")
    parser.add_argument(
        "--config",
        default="configs/pipeline_paths.json",
        help="Path to pipeline paths config",
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "streaming"],
        default="sequential",
        help="Pipeline execution mode (default: sequential)",
    )
    parser.add_argument(
        "--detector-version",
        choices=["v1", "v2", "v3"],
        default="v1",
        help="Stage2 detector path for sequential mode (default: v1)",
    )
    parser.add_argument(
        "--stage3-mode",
        choices=["normal", "passthrough"],
        default="normal",
        help="Stage3 mode for sequential pipeline (default: normal)",
    )
    parser.add_argument(
        "--geometry-audit",
        action="store_true",
        help="Enable Stage3 geometry-audit logging",
    )
    parser.add_argument(
        "--rejection-log",
        action="store_true",
        help="Enable Stage3 rejection-reason logging",
    )
    parser.add_argument(
        "--geometry-audit-output",
        default="run_state/stage3_geometry_audit.json",
        help="Stage3 geometry audit JSON path",
    )
    parser.add_argument(
        "--geometry-audit-summary-output",
        default="run_state/stage3_geometry_audit_summary.json",
        help="Stage3 geometry audit summary JSON path",
    )
    parser.add_argument(
        "--rejection-log-output",
        default="run_state/stage3_rejection_log.json",
        help="Stage3 rejection log JSON path",
    )
    parser.add_argument(
        "--rejection-summary-output",
        default="run_state/stage3_rejection_summary.json",
        help="Stage3 rejection summary JSON path",
    )
    return parser.parse_args()


def run_stage(stage_script: str, config_path: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, stage_script, "--config", config_path]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[pipeline] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)  # noqa: S603
    if proc.returncode != 0:
        raise RuntimeError(f"Stage failed ({proc.returncode}): {stage_script}")


def main() -> int:
    args = parse_args()
    if args.mode == "streaming":
        run_stage("tools/streaming_pipeline_runner.py", args.config)
        print("[pipeline] Streaming Stage1->Stage3 completed successfully")
        return 0

    if args.detector_version == "v2":
        stage2_script = "src/pipeline/stage02_block_detection_v2.py"
    elif args.detector_version == "v3":
        stage2_script = "src/pipeline/stage02_block_detection_v3.py"
    else:
        stage2_script = "src/pipeline/stage02_block_detection.py"
    stages = [
        "src/pipeline/stage01_pdf_to_images.py",
        stage2_script,
        "src/pipeline/stage03_block_refiner.py",
    ]

    stage3_args = [
        "--stage3-mode",
        args.stage3_mode,
    ]
    if args.geometry_audit:
        stage3_args.extend(
            [
                "--geometry-audit",
                "--geometry-audit-output",
                args.geometry_audit_output,
                "--geometry-audit-summary-output",
                args.geometry_audit_summary_output,
            ]
        )
    if args.rejection_log:
        stage3_args.extend(
            [
                "--rejection-log",
                "--rejection-log-output",
                args.rejection_log_output,
                "--rejection-summary-output",
                args.rejection_summary_output,
            ]
        )

    for stage in stages:
        extra_args = stage3_args if stage.endswith("stage03_block_refiner.py") else None
        run_stage(stage, args.config, extra_args=extra_args)

    print("[pipeline] Stage01->Stage03 completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
