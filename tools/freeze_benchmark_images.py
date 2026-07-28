#!/usr/bin/env python3
"""Create/refresh frozen benchmark images and build benchmark manifest."""

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
    DEFAULT_PAGE_IDENTITY_MAP_PATH,
    DEFAULT_STAGE1_MANIFEST_PATH,
    assert_benchmark_manifest_valid,
    build_benchmark_manifest,
    sync_frozen_benchmark_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze benchmark images and build canonical benchmark manifest")
    parser.add_argument("--source-images-dir", default="data/pdf2img", help="Source image root to snapshot")
    parser.add_argument(
        "--benchmark-images-dir",
        default=str(DEFAULT_BENCHMARK_IMAGES_DIR.relative_to(PROJECT_ROOT)),
        help="Frozen benchmark image root",
    )
    parser.add_argument(
        "--manifest-output",
        default=str(DEFAULT_BENCHMARK_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        help="Benchmark manifest output JSON path",
    )
    parser.add_argument(
        "--stage1-manifest",
        default=str(DEFAULT_STAGE1_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        help="Stage1 manifest JSON path",
    )
    parser.add_argument(
        "--page-identity-map",
        default=str(DEFAULT_PAGE_IDENTITY_MAP_PATH.relative_to(PROJECT_ROOT)),
        help="Page identity map JSON path",
    )
    parser.add_argument("--force-sync", action="store_true", help="Replace existing frozen benchmark image files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sync_report = sync_frozen_benchmark_images(
        source_images_dir=args.source_images_dir,
        benchmark_images_dir=args.benchmark_images_dir,
        force_sync=bool(args.force_sync),
    )
    manifest_payload = build_benchmark_manifest(
        benchmark_images_dir=args.benchmark_images_dir,
        manifest_output_path=args.manifest_output,
        stage1_manifest_path=args.stage1_manifest,
        page_identity_map_path=args.page_identity_map,
    )
    validation = assert_benchmark_manifest_valid(
        benchmark_images_dir=args.benchmark_images_dir,
        manifest_path=args.manifest_output,
    )

    output = {
        "sync": sync_report,
        "manifest_summary": manifest_payload.get("summary", {}),
        "validation": {
            "validation_passed": bool(validation.get("validation_passed")),
            "manifest_rows": int(validation.get("manifest_rows", 0)),
            "actual_image_files": int(validation.get("actual_image_files", 0)),
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
