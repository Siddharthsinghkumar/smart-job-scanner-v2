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
        choices=["v1", "v2"],
        default="v1",
        help="Stage2 detector path for sequential mode (default: v1)",
    )
    return parser.parse_args()


def run_stage(stage_script: str, config_path: str) -> None:
    cmd = [sys.executable, stage_script, "--config", config_path]
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

    stage2_script = (
        "src/pipeline/stage02_block_detection_v2.py"
        if args.detector_version == "v2"
        else "src/pipeline/stage02_block_detection.py"
    )
    stages = [
        "src/pipeline/stage01_pdf_to_images.py",
        stage2_script,
        "src/pipeline/stage03_block_refiner.py",
    ]

    for stage in stages:
        run_stage(stage, args.config)

    print("[pipeline] Stage01->Stage03 completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
