#!/usr/bin/env python3
"""Prepare a derived one-class YOLO dataset for the detector pivot from frozen benchmark assets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.benchmark_images import (  # noqa: E402
    DEFAULT_BENCHMARK_IMAGES_DIR,
    DEFAULT_BENCHMARK_MANIFEST_PATH,
    assert_benchmark_manifest_valid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare detector-pivot YOLO dataset from frozen benchmark labels")
    parser.add_argument("--labels-dir", default="data/test_labels", help="Source label JSON directory (read-only)")
    parser.add_argument(
        "--images-dir",
        default=str(DEFAULT_BENCHMARK_IMAGES_DIR.relative_to(PROJECT_ROOT)),
        help="Frozen benchmark image root",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default=str(DEFAULT_BENCHMARK_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        help="Frozen benchmark manifest path",
    )
    parser.add_argument("--output-root", default="data/yolo_job_ad_pivot_v1", help="Derived YOLO dataset root")
    parser.add_argument(
        "--mapping-output",
        default="run_state/detector_pivot_yolo_mapping.json",
        help="Task-to-page mapping output JSON",
    )
    parser.add_argument(
        "--dataset-report-output",
        default="run_state/detector_pivot_dataset_report.json",
        help="Conversion report output JSON",
    )
    parser.add_argument(
        "--split-output",
        default="run_state/detector_pivot_split_manifest.json",
        help="Split manifest output JSON",
    )
    parser.add_argument(
        "--dataset-manifest-output",
        default="run_state/detector_pivot_dataset_manifest.json",
        help="Detector pivot dataset manifest output JSON",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=20260403, help="Split seed")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def main() -> int:
    args = parse_args()
    labels_dir = _resolve(args.labels_dir)
    images_dir = _resolve(args.images_dir)
    benchmark_manifest = _resolve(args.benchmark_manifest)
    output_root = _resolve(args.output_root)
    mapping_output = _resolve(args.mapping_output)
    dataset_report_output = _resolve(args.dataset_report_output)
    split_output = _resolve(args.split_output)
    dataset_manifest_output = _resolve(args.dataset_manifest_output)

    if not labels_dir.is_dir():
        raise SystemExit(f"labels directory not found: {labels_dir}")
    if not images_dir.is_dir():
        raise SystemExit(f"images directory not found: {images_dir}")
    if not benchmark_manifest.is_file():
        raise SystemExit(f"benchmark manifest not found: {benchmark_manifest}")

    validation = assert_benchmark_manifest_valid(
        benchmark_images_dir=images_dir,
        manifest_path=benchmark_manifest,
    )

    prepare_cmd = [
        sys.executable,
        "tools/prepare_yolo_dataset.py",
        "--labels-dir",
        str(labels_dir),
        "--images-dir",
        str(images_dir),
        "--benchmark-manifest",
        str(benchmark_manifest),
        "--validate-benchmark-images",
        "--output-root",
        str(output_root),
        "--mapping-output",
        str(mapping_output),
        "--report-output",
        str(dataset_report_output),
        "--split-output",
        str(split_output),
        "--val-ratio",
        str(float(args.val_ratio)),
        "--seed",
        str(int(args.seed)),
    ]
    proc = subprocess.run(prepare_cmd, cwd=str(PROJECT_ROOT), check=False)  # noqa: S603
    if proc.returncode != 0:
        raise SystemExit(f"dataset conversion failed with exit code {proc.returncode}")

    conversion_report = _load_json(dataset_report_output)
    split_manifest = _load_json(split_output)
    dataset_yaml = Path(str(conversion_report.get("dataset_yaml", output_root / "dataset.yaml"))).resolve()

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pivot": {
            "name": "detector_pivot_yolo_v1",
            "class_names": ["job_ad"],
            "class_map": {"0": "job_ad"},
        },
        "sources": {
            "labels_dir": _to_rel(labels_dir),
            "frozen_images_dir": _to_rel(images_dir),
            "benchmark_manifest": _to_rel(benchmark_manifest),
        },
        "benchmark_validation": {
            "passed": bool(validation.get("validation_passed", False)),
            "manifest_rows": int(validation.get("manifest_rows", 0) or 0),
            "actual_image_files": int(validation.get("actual_image_files", 0) or 0),
            "hash_mismatch_count": int(validation.get("hash_mismatch_count", 0) or 0),
            "missing_file_count": int(validation.get("missing_file_count", 0) or 0),
            "extra_file_count": int(validation.get("extra_file_count", 0) or 0),
        },
        "derived_dataset": {
            "output_root": _to_rel(output_root),
            "dataset_yaml": _to_rel(dataset_yaml),
            "mapping_file": _to_rel(mapping_output),
            "conversion_report_file": _to_rel(dataset_report_output),
            "split_manifest_file": _to_rel(split_output),
        },
        "summary": {
            "total_pages": int(split_manifest.get("total_pages", 0) or 0),
            "train_pages": int(split_manifest.get("train_pages", 0) or 0),
            "val_pages": int(split_manifest.get("val_pages", 0) or 0),
            "total_boxes": int(conversion_report.get("summary", {}).get("total_boxes", 0) or 0),
            "pages_with_boxes": int(conversion_report.get("summary", {}).get("pages_with_boxes", 0) or 0),
            "pages_without_boxes": int(conversion_report.get("summary", {}).get("pages_without_boxes", 0) or 0),
            "unmatched_pages": int(conversion_report.get("summary", {}).get("total_pages_unmatched", 0) or 0),
            "conversion_errors": int(conversion_report.get("summary", {}).get("conversion_errors", 0) or 0),
        },
        "split_by_newspaper": split_manifest.get("papers", {}),
        "reproducibility": {
            "seed": int(args.seed),
            "val_ratio": float(args.val_ratio),
            "conversion_command": prepare_cmd,
        },
        "safety_notes": [
            "Source labels were read-only and not modified.",
            "Frozen benchmark image files were validated before conversion.",
            "Derived YOLO dataset is a copy/mapping layer with traceability back to frozen images.",
        ],
    }

    dataset_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    dataset_manifest_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"dataset conversion report: {dataset_report_output}")
    print(f"dataset split manifest: {split_output}")
    print(f"detector pivot dataset manifest: {dataset_manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
