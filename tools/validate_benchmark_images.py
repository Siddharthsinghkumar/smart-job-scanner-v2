#!/usr/bin/env python3
"""Validate frozen benchmark images against benchmark manifest and write a report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.benchmark_images import (
    DEFAULT_BENCHMARK_IMAGES_DIR,
    DEFAULT_BENCHMARK_MANIFEST_PATH,
    validate_benchmark_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate frozen benchmark images")
    parser.add_argument(
        "--benchmark-images-dir",
        default=str(DEFAULT_BENCHMARK_IMAGES_DIR.relative_to(PROJECT_ROOT)),
        help="Frozen benchmark image root",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_BENCHMARK_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        help="Benchmark manifest JSON path",
    )
    parser.add_argument(
        "--output",
        default="run_state/benchmark_image_validation_report.json",
        help="Validation report output JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_benchmark_manifest(
        benchmark_images_dir=args.benchmark_images_dir,
        manifest_path=args.manifest,
    )

    output_path = (PROJECT_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"validation_passed={int(bool(report.get('validation_passed')))}")
    print(f"manifest_rows={int(report.get('manifest_rows', 0) or 0)}")
    print(f"actual_image_files={int(report.get('actual_image_files', 0) or 0)}")
    print(f"report={output_path}")

    if not report.get("validation_passed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
