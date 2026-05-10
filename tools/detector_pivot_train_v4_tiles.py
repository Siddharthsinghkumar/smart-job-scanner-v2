#!/usr/bin/env python3
"""Thin reproducible entrypoint for detector-pivot v4 tile training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.detector_device import resolve_device_with_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train detector pivot v4 tile baseline")
    parser.add_argument("--device", default="auto", help="auto | cpu | 0 | cuda:0")
    parser.add_argument("--config", default="configs/detector_pivot_yolo_v4_tiles.yaml")
    parser.add_argument(
        "--output-report",
        default="run_state/detector_pivot_v4_training_report.json",
        help="Training report JSON output",
    )
    parser.add_argument(
        "--weights-out",
        default="artifacts/detector_pivot_yolo_v4_tiles/best.pt",
        help="Stable best checkpoint output",
    )
    parser.add_argument(
        "--last-weights-out",
        default="artifacts/detector_pivot_yolo_v4_tiles/last.pt",
        help="Stable last checkpoint output",
    )
    parser.add_argument(
        "--device-preflight-report",
        default="run_state/detector_pivot_v4_device_preflight_train.json",
        help="Device preflight JSON output",
    )
    parser.add_argument(
        "--require-gpu-if-available",
        action="store_true",
        help="Fail if CUDA is available but selected device is not GPU",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    preflight = resolve_device_with_preflight(
        requested_device=args.device,
        context="detector_pivot_train_v4_tiles",
        preflight_report_path=(PROJECT_ROOT / args.device_preflight_report),
    )
    selected_device = str(preflight.get("selected_device", "cpu"))
    cuda_available = bool(preflight.get("torch", {}).get("cuda_is_available", False))
    execution_mode = str(preflight.get("execution_mode", "cpu"))

    if args.require_gpu_if_available and cuda_available and execution_mode != "gpu":
        raise SystemExit(
            "GPU safety check failed: CUDA is available but selected execution mode is not GPU. "
            f"requested={args.device} selected={selected_device}"
        )

    cmd = [
        sys.executable,
        "tools/detector_pivot_train.py",
        "--config",
        args.config,
        "--device",
        args.device,
        "--output-report",
        args.output_report,
        "--weights-out",
        args.weights_out,
        "--last-weights-out",
        args.last_weights_out,
        "--device-preflight-report",
        args.device_preflight_report,
    ]
    if args.dry_run:
        print(" ".join(cmd))
        print(
            "preflight_selected_device="
            f"{selected_device} execution_mode={execution_mode} cuda_available={cuda_available}"
        )
        return 0

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)  # noqa: S603
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
