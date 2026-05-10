#!/usr/bin/env python3
"""Run Stage2 v3 YOLO inference and write Stage2-compatible artifacts + detection metadata."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.stage02_block_detection_v3 import (
    _load_detection_params_v3,
    _write_detector_metadata,
    detect_page_blocks_v3,
)
from src.utils.benchmark_images import (
    DEFAULT_BENCHMARK_IMAGES_DIR,
    DEFAULT_BENCHMARK_MANIFEST_PATH,
    assert_benchmark_manifest_valid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Stage2 v3 YOLO detections on full-page images")
    parser.add_argument(
        "--images-dir",
        default=str(DEFAULT_BENCHMARK_IMAGES_DIR.relative_to(PROJECT_ROOT)),
        help="Input benchmark page image root directory",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default=str(DEFAULT_BENCHMARK_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        help="Benchmark image manifest path",
    )
    parser.add_argument(
        "--validate-benchmark-images",
        action="store_true",
        help="Validate benchmark image hashes/filenames against manifest before inference",
    )
    parser.add_argument("--blocks-output", default="data/job_blocks_smart", help="Output block crop root")
    parser.add_argument("--detections-output", default="run_state/detections", help="Output detections JSON directory")
    parser.add_argument("--model-path", default=None, help="YOLO model checkpoint override")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="YOLO confidence threshold override")
    parser.add_argument("--iou-threshold", type=float, default=None, help="YOLO IoU threshold override")
    parser.add_argument("--max-detections", type=int, default=None, help="YOLO max detections override")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference image size override")
    parser.add_argument("--device", default=None, help="YOLO device override")
    parser.add_argument("--debug", action="store_true", help="Enable debug overlay generation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_dir = (PROJECT_ROOT / args.images_dir).resolve()
    blocks_output = (PROJECT_ROOT / args.blocks_output).resolve()
    detections_output = (PROJECT_ROOT / args.detections_output).resolve()

    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    should_validate = bool(
        args.validate_benchmark_images
        or images_dir.resolve() == DEFAULT_BENCHMARK_IMAGES_DIR.resolve()
    )
    if should_validate:
        assert_benchmark_manifest_valid(
            benchmark_images_dir=images_dir,
            manifest_path=(PROJECT_ROOT / args.benchmark_manifest).resolve()
            if not Path(args.benchmark_manifest).is_absolute()
            else Path(args.benchmark_manifest).resolve(),
        )

    detector_params = _load_detection_params_v3()
    overrides = {
        "model_path": args.model_path,
        "confidence_threshold": args.confidence_threshold,
        "iou_threshold": args.iou_threshold,
        "max_detections": args.max_detections,
        "imgsz": args.imgsz,
        "device": args.device,
    }
    detector_params.update({k: v for k, v in overrides.items() if v is not None})

    detections_output.mkdir(parents=True, exist_ok=True)

    all_images = [str(p) for p in sorted(images_dir.rglob("*.png"))]
    if not all_images:
        print(f"No images found under {images_dir}")
        return 0

    counts: Counter[str] = Counter()
    for img_path in all_images:
        result = detect_page_blocks_v3(img_path, str(blocks_output), detector_params, debug=bool(args.debug))
        status = str(result.get("status", "failed"))
        counts[status] += 1
        if status == "processed":
            _write_detector_metadata(str(result.get("page_name")), result.get("detections", []), detections_output)

    print(f"images_processed: {len(all_images)}")
    print(f"processed: {counts['processed']}")
    print(f"skipped: {counts['skipped']}")
    print(f"failed: {counts['failed']}")
    print(f"detections_output: {detections_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
