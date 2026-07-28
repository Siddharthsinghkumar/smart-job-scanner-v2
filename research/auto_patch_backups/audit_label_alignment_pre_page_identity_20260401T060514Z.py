#!/usr/bin/env python3
"""Audit benchmark page-key alignment between labels, images, and detections."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.benchmark_alignment import (
    choose_best_page_key,
    compute_dim_scale,
    normalize_label_image_candidates,
    source_hint_from_label_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit label/image/detection page alignment")
    parser.add_argument("--labels-dir", default="data/test_labels", help="Directory containing label JSON exports")
    parser.add_argument("--images-dir", default="data/pdf2img", help="Directory containing generated page images")
    parser.add_argument("--detections-dir", default="run_state/detections", help="Directory containing detections JSON")
    parser.add_argument("--output", default="run_state/label_alignment_audit.json", help="Audit output JSON path")
    return parser.parse_args()


def _load_tasks(label_file: Path) -> list[dict[str, Any]]:
    payload = json.loads(label_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        tasks = payload.get("tasks", [])
    else:
        tasks = payload
    return [t for t in tasks if isinstance(t, dict)] if isinstance(tasks, list) else []


def _task_image_ref(task: dict[str, Any]) -> str:
    data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
    return str(data.get("image") or data.get("page") or data.get("file") or "")


def _task_label_box_stats(task: dict[str, Any]) -> tuple[int, tuple[int, int] | None]:
    count = 0
    dims: tuple[int, int] | None = None

    annotations = task.get("annotations", [])
    if not isinstance(annotations, list):
        return 0, None

    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        results = ann.get("result", [])
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict) or item.get("type") != "rectanglelabels":
                continue
            count += 1
            ow = item.get("original_width")
            oh = item.get("original_height")
            try:
                if ow is not None and oh is not None:
                    dims = (int(round(float(ow))), int(round(float(oh))))
            except Exception:
                pass

    return count, dims


def _load_detection_index(detections_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for jf in sorted(detections_dir.glob("*.json")):
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        page = str(payload.get("page") or jf.name[:-5])
        rows = payload.get("detections", [])
        if not isinstance(rows, list):
            rows = []
        detector_count = 0
        refined_count = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            stage = row.get("stage")
            if stage == "detector":
                detector_count += 1
            elif stage == "refined":
                refined_count += 1
        out[page] = {
            "path": str(jf),
            "detector_count": detector_count,
            "refined_count": refined_count,
            "total_rows": len(rows),
        }
    return out


def main() -> int:
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir)
    detections_dir = Path(args.detections_dir)
    output_path = Path(args.output)

    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not detections_dir.is_dir():
        raise SystemExit(f"Detections directory not found: {detections_dir}")

    image_index = {p.name: str(p) for p in sorted(images_dir.rglob("*.png"))}
    det_index = _load_detection_index(detections_dir)
    image_keys = set(image_index.keys())
    det_keys = set(det_index.keys())

    page_rows: list[dict[str, Any]] = []
    per_key_task_refs: dict[str, list[str]] = defaultdict(list)
    suspicious_counter: dict[str, int] = defaultdict(int)

    label_files = sorted(labels_dir.glob("*.json"))
    for label_file in label_files:
        hint = source_hint_from_label_file(label_file.name)
        tasks = _load_tasks(label_file)
        for task in tasks:
            task_id = task.get("id")
            task_ref = f"{label_file.name}::task_{task_id}"
            image_ref = _task_image_ref(task)
            original_filename = Path(unquote(image_ref)).name if image_ref else ""
            label_box_count, label_dims = _task_label_box_stats(task)

            normalized_key, candidates, warnings = normalize_label_image_candidates(original_filename, hint)
            chosen_key = choose_best_page_key(candidates, image_keys, det_keys) or normalized_key

            image_exists = chosen_key in image_keys
            det_exists = chosen_key in det_keys
            detector_count = int(det_index.get(chosen_key, {}).get("detector_count", 0))
            refined_count = int(det_index.get(chosen_key, {}).get("refined_count", 0))
            image_dims = None
            dim_scale = None
            if image_exists:
                try:
                    import cv2

                    img = cv2.imread(image_index[chosen_key])
                    if img is not None:
                        ih, iw = img.shape[:2]
                        image_dims = (iw, ih)
                        dim_scale = compute_dim_scale(label_dims, image_dims)
                except Exception:
                    pass

            if not image_exists and not det_exists:
                warnings.append("label_page_unmatched_to_image_and_detection")
            if original_filename.startswith("page_"):
                warnings.append("et_style_generic_name")

            for w in warnings:
                suspicious_counter[w] += 1

            per_key_task_refs[chosen_key].append(task_ref)

            row = {
                "source_label_file": label_file.name,
                "task_id": task_id,
                "source_hint": hint,
                "original_image_ref": image_ref,
                "original_image_filename": original_filename,
                "normalized_page_key": chosen_key,
                "candidate_page_keys": candidates,
                "matching_page_image_exists": image_exists,
                "matching_page_image_path": image_index.get(chosen_key),
                "matching_detection_json_exists": det_exists,
                "matching_detection_json_path": det_index.get(chosen_key, {}).get("path"),
                "label_image_dimensions": {
                    "width": label_dims[0] if label_dims else None,
                    "height": label_dims[1] if label_dims else None,
                },
                "pipeline_image_dimensions": {
                    "width": image_dims[0] if image_dims else None,
                    "height": image_dims[1] if image_dims else None,
                },
                "label_to_pipeline_scale": {
                    "sx": round(dim_scale[0], 6) if dim_scale else None,
                    "sy": round(dim_scale[1], 6) if dim_scale else None,
                },
                "labeled_box_count": label_box_count,
                "detector_box_count": detector_count,
                "refined_box_count": refined_count,
                "warnings": sorted(set(warnings)),
            }
            page_rows.append(row)

    duplicate_keys = {
        key: refs for key, refs in sorted(per_key_task_refs.items()) if len(refs) > 1
    }

    label_keys = {row["normalized_page_key"] for row in page_rows}
    unmatched_label_rows = [
        row for row in page_rows
        if not row["matching_page_image_exists"] or not row["matching_detection_json_exists"]
    ]

    outputs_without_labels_images = sorted(image_keys - label_keys)
    outputs_without_labels_detections = sorted(det_keys - label_keys)

    summary = {
        "total_label_files": len(label_files),
        "total_label_pages": len(page_rows),
        "matched_pages_both_image_and_detection": sum(
            1
            for row in page_rows
            if row["matching_page_image_exists"] and row["matching_detection_json_exists"]
        ),
        "unmatched_pages": len(unmatched_label_rows),
        "pages_with_labeled_boxes": sum(1 for row in page_rows if int(row["labeled_box_count"]) > 0),
        "total_labeled_boxes": sum(int(row["labeled_box_count"]) for row in page_rows),
        "suspicious_naming_patterns_found": dict(sorted(suspicious_counter.items())),
        "duplicate_normalized_keys": len(duplicate_keys),
        "image_outputs_without_label_task_count": len(outputs_without_labels_images),
        "detection_outputs_without_label_task_count": len(outputs_without_labels_detections),
    }

    report = {
        "summary": summary,
        "label_pages": page_rows,
        "duplicate_normalized_keys": duplicate_keys,
        "unmatched_label_pages": unmatched_label_rows,
        "outputs_without_labels": {
            "image_page_keys": outputs_without_labels_images,
            "detection_page_keys": outputs_without_labels_detections,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"total labeled pages: {summary['total_label_pages']}")
    print(f"matched pages: {summary['matched_pages_both_image_and_detection']}")
    print(f"unmatched pages: {summary['unmatched_pages']}")
    print(f"suspicious naming patterns found: {summary['suspicious_naming_patterns_found']}")
    print(f"saved audit: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
