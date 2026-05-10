#!/usr/bin/env python3
"""Stage01-stage03 pipeline runner (sequential or streaming) for benchmark evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.benchmark_images import (
    DEFAULT_BENCHMARK_IMAGES_DIR,
    DEFAULT_BENCHMARK_MANIFEST_PATH,
    assert_benchmark_manifest_valid,
)
from src.utils.pipeline_config import get_path, load_config


def _resolve_from_project(path: str | Path) -> Path:
    raw = Path(path)
    return raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


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
        choices=["v1", "v2", "v3", "v4"],
        default="v1",
        help="Stage2 detector path for sequential mode (default: v1)",
    )
    parser.add_argument(
        "--stage3-mode",
        choices=["normal", "passthrough", "recall_friendly"],
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
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage1 and run Stage2->Stage3 only (required for frozen benchmark image runs)",
    )
    parser.add_argument(
        "--validate-benchmark-images",
        action="store_true",
        help="Validate benchmark image manifest before running",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default=str(DEFAULT_BENCHMARK_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        help="Benchmark image manifest path used for immutability validation",
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
    config = load_config(args.config)
    images_output_dir = _resolve_from_project(get_path("images_output", config))
    benchmark_images_dir = DEFAULT_BENCHMARK_IMAGES_DIR.resolve()
    uses_frozen_benchmark_images = images_output_dir == benchmark_images_dir

    if uses_frozen_benchmark_images and args.mode == "streaming":
        raise RuntimeError(
            "Streaming mode includes Stage1 and cannot run against frozen benchmark images. "
            "Use --mode sequential --skip-stage1 with benchmark config."
        )
    if uses_frozen_benchmark_images and not args.skip_stage1:
        raise RuntimeError(
            "Refusing to run Stage1 with frozen benchmark images. "
            "Use --skip-stage1 so detector experiments cannot regenerate benchmark pages."
        )

    should_validate = bool(args.validate_benchmark_images or uses_frozen_benchmark_images)
    if should_validate:
        validation = assert_benchmark_manifest_valid(
            benchmark_images_dir=images_output_dir,
            manifest_path=args.benchmark_manifest,
        )
        print(
            "[pipeline] benchmark image validation passed "
            f"(manifest_rows={int(validation.get('manifest_rows', 0) or 0)} "
            f"actual_image_files={int(validation.get('actual_image_files', 0) or 0)})"
        )

    if args.mode == "streaming":
        run_stage("tools/streaming_pipeline_runner.py", args.config)
        print("[pipeline] Streaming Stage1->Stage3 completed successfully")
        return 0

    if args.detector_version == "v2":
        stage2_script = "src/pipeline/stage02_block_detection_v2.py"
    elif args.detector_version == "v3":
        stage2_script = "src/pipeline/stage02_block_detection_v3.py"
    elif args.detector_version == "v4":
        stage2_script = "src/pipeline/stage02_block_detection_v4.py"
    else:
        stage2_script = "src/pipeline/stage02_block_detection.py"
    stages = [stage2_script, "src/pipeline/stage03_block_refiner.py"]
    if not args.skip_stage1:
        stages.insert(0, "src/pipeline/stage01_pdf_to_images.py")

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

    if args.skip_stage1:
        print("[pipeline] Stage2->Stage3 completed successfully (Stage1 skipped)")
    else:
        print("[pipeline] Stage01->Stage03 completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
