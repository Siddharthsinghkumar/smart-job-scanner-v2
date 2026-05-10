#!/usr/bin/env python3
"""Prepare Stage2 v3 YOLO dataset from existing benchmark labels without modifying source labels."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.benchmark_alignment import (
    choose_best_page_key,
    compute_dim_scale,
    normalize_label_image_candidates,
    scale_bbox_xyxy,
    source_hint_from_label_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO Stage2 v3 dataset from Label Studio exports")
    parser.add_argument("--labels-dir", default="data/test_labels", help="Label JSON directory")
    parser.add_argument("--images-dir", default="data/pdf2img", help="Generated page image directory")
    parser.add_argument("--output-root", default="data/yolo_stage2_v3", help="YOLO dataset output root")
    parser.add_argument("--mapping-output", default="run_state/stage2_v3_yolo_mapping.json", help="Mapping output JSON")
    parser.add_argument("--report-output", default="run_state/stage2_v3_dataset_report.json", help="Dataset report output JSON")
    parser.add_argument("--split-output", default="run_state/stage2_v3_split_manifest.json", help="Split manifest output JSON")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=20260331, help="Split seed")
    return parser.parse_args()


def _load_tasks(label_file: Path) -> list[dict[str, Any]]:
    payload = json.loads(label_file.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", []) if isinstance(payload, dict) else payload
    return [t for t in tasks if isinstance(t, dict)] if isinstance(tasks, list) else []


def _task_image_ref(task: dict[str, Any]) -> str:
    data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
    return str(data.get("image") or data.get("page") or data.get("file") or "")


def _pct_to_xyxy(value: dict[str, Any], original_width: float, original_height: float) -> list[int]:
    x = float(value.get("x", 0.0))
    y = float(value.get("y", 0.0))
    w = float(value.get("width", 0.0))
    h = float(value.get("height", 0.0))
    x1 = int(round((x / 100.0) * original_width))
    y1 = int(round((y / 100.0) * original_height))
    x2 = int(round(((x + w) / 100.0) * original_width))
    y2 = int(round(((y + h) / 100.0) * original_height))
    return [x1, y1, x2, y2]


def _extract_task_label_boxes(task: dict[str, Any]) -> tuple[list[list[int]], tuple[int, int] | None]:
    boxes: list[list[int]] = []
    label_dims: tuple[int, int] | None = None

    ann_list = task.get("annotations", [])
    if not isinstance(ann_list, list):
        return boxes, label_dims

    for ann in ann_list:
        if not isinstance(ann, dict):
            continue
        results = ann.get("result", [])
        if not isinstance(results, list):
            continue

        for item in results:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "rectanglelabels":
                continue

            value = item.get("value", {})
            if not isinstance(value, dict):
                continue

            labels = value.get("rectanglelabels", [])
            if isinstance(labels, list) and labels and "job_ad" not in {str(x) for x in labels}:
                continue

            try:
                ow = float(item.get("original_width", value.get("original_width", 0.0)))
                oh = float(item.get("original_height", value.get("original_height", 0.0)))
            except Exception:
                continue

            if ow <= 0 or oh <= 0:
                continue

            label_dims = (int(round(ow)), int(round(oh)))
            boxes.append(_pct_to_xyxy(value, ow, oh))

    return boxes, label_dims


def _clip_xyxy(bbox: list[int], image_w: int, image_h: int) -> list[int] | None:
    if len(bbox) != 4:
        return None
    x1 = max(0, min(image_w - 1, int(bbox[0])))
    y1 = max(0, min(image_h - 1, int(bbox[1])))
    x2 = max(0, min(image_w, int(bbox[2])))
    y2 = max(0, min(image_h, int(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _paper_key(page_name: str) -> str:
    m = re.fullmatch(r"(.+)_p\d+\.png", page_name)
    return m.group(1) if m else "unknown"


def _safe_filename(page_name: str) -> str:
    stem = Path(page_name).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    if not safe:
        safe = "page"
    return f"{safe}.png"


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _build_split(page_keys: list[str], val_ratio: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    by_paper: dict[str, list[str]] = defaultdict(list)
    for key in sorted(page_keys):
        by_paper[_paper_key(key)].append(key)

    split: dict[str, str] = {}
    for paper, keys in sorted(by_paper.items()):
        local = list(keys)
        rng.shuffle(local)

        if len(local) <= 1:
            split[local[0]] = "train"
            continue

        val_count = int(round(len(local) * val_ratio))
        val_count = max(1, min(len(local) - 1, val_count))

        val_keys = set(local[:val_count])
        for k in local:
            split[k] = "val" if k in val_keys else "train"

    if not any(v == "val" for v in split.values()) and split:
        # Last-resort safety for very small datasets.
        last_key = sorted(split.keys())[-1]
        split[last_key] = "val"

    return split


def _yolo_line_from_bbox(bbox: list[int], image_w: int, image_h: int) -> str:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return "0 {:.6f} {:.6f} {:.6f} {:.6f}".format(cx / image_w, cy / image_h, bw / image_w, bh / image_h)


def main() -> int:
    args = parse_args()

    labels_dir = (PROJECT_ROOT / args.labels_dir).resolve()
    images_dir = (PROJECT_ROOT / args.images_dir).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    mapping_output = (PROJECT_ROOT / args.mapping_output).resolve()
    report_output = (PROJECT_ROOT / args.report_output).resolve()
    split_output = (PROJECT_ROOT / args.split_output).resolve()

    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    image_index = {p.name: p for p in sorted(images_dir.rglob("*.png"))}
    image_keys = set(image_index.keys())

    # Aggregate per-page records to avoid train/val leakage for same page.
    per_page: dict[str, dict[str, Any]] = {}
    unmatched_pages: list[dict[str, Any]] = []
    dimension_mismatch_pages: list[dict[str, Any]] = []
    conversion_errors: list[dict[str, Any]] = []

    label_files = sorted(labels_dir.glob("*.json"))
    for label_file in label_files:
        hint = source_hint_from_label_file(label_file.name)
        tasks = _load_tasks(label_file)
        for task in tasks:
            task_id = task.get("id")
            image_ref = _task_image_ref(task)
            original_filename = Path(unquote(image_ref)).name if image_ref else ""
            normalized_key, candidates, warnings = normalize_label_image_candidates(original_filename, hint)
            chosen_key = choose_best_page_key(candidates, image_keys, set()) or normalized_key

            image_path = image_index.get(chosen_key)
            if image_path is None:
                unmatched_pages.append(
                    {
                        "label_file": label_file.name,
                        "task_id": task_id,
                        "original_image_ref": image_ref,
                        "original_image_filename": original_filename,
                        "normalized_candidates": candidates,
                        "chosen_key": chosen_key,
                        "warnings": sorted(set(warnings + ["page_not_found_in_generated_images"])),
                    }
                )
                continue

            boxes, label_dims = _extract_task_label_boxes(task)

            img = cv2.imread(str(image_path))
            if img is None:
                conversion_errors.append(
                    {
                        "label_file": label_file.name,
                        "task_id": task_id,
                        "page": chosen_key,
                        "error": f"failed_to_read_image:{image_path}",
                    }
                )
                continue

            image_h, image_w = img.shape[:2]
            scale = compute_dim_scale(label_dims, (image_w, image_h)) if label_dims else None
            reconciled = bool(scale and (abs(scale[0] - 1.0) > 1e-6 or abs(scale[1] - 1.0) > 1e-6))
            if reconciled:
                dimension_mismatch_pages.append(
                    {
                        "label_file": label_file.name,
                        "task_id": task_id,
                        "page": chosen_key,
                        "label_dimensions": {
                            "width": label_dims[0] if label_dims else None,
                            "height": label_dims[1] if label_dims else None,
                        },
                        "image_dimensions": {
                            "width": image_w,
                            "height": image_h,
                        },
                        "scale": {
                            "sx": round(scale[0], 6) if scale else None,
                            "sy": round(scale[1], 6) if scale else None,
                        },
                        "reconciliation": "scaled_label_boxes_to_image_dimensions",
                    }
                )

            converted_boxes: list[list[int]] = []
            for b in boxes:
                bb = scale_bbox_xyxy(b, scale) if scale else b
                clipped = _clip_xyxy(bb, image_w, image_h)
                if clipped is not None:
                    converted_boxes.append(clipped)

            row = per_page.setdefault(
                chosen_key,
                {
                    "page": chosen_key,
                    "paper": _paper_key(chosen_key),
                    "image_src": str(image_path),
                    "image_width": image_w,
                    "image_height": image_h,
                    "boxes": [],
                    "sources": [],
                    "dimension_reconciled": False,
                    "normalization_warnings": set(),
                },
            )
            row["boxes"].extend(converted_boxes)
            row["dimension_reconciled"] = bool(row["dimension_reconciled"] or reconciled)
            row["normalization_warnings"].update(warnings)
            row["sources"].append(
                {
                    "label_file": label_file.name,
                    "task_id": task_id,
                    "original_image_ref": image_ref,
                    "original_image_filename": original_filename,
                    "normalized_candidates": candidates,
                    "chosen_key": chosen_key,
                    "label_dimensions": {
                        "width": label_dims[0] if label_dims else None,
                        "height": label_dims[1] if label_dims else None,
                    },
                    "source_box_count": len(boxes),
                    "converted_box_count": len(converted_boxes),
                    "dimension_reconciled": reconciled,
                }
            )

    all_page_keys = sorted(per_page.keys())
    split_map = _build_split(all_page_keys, float(args.val_ratio), int(args.seed))

    train_img_dir = output_root / "images" / "train"
    val_img_dir = output_root / "images" / "val"
    train_lbl_dir = output_root / "labels" / "train"
    val_lbl_dir = output_root / "labels" / "val"

    _ensure_clean_dir(train_img_dir)
    _ensure_clean_dir(val_img_dir)
    _ensure_clean_dir(train_lbl_dir)
    _ensure_clean_dir(val_lbl_dir)

    mappings: list[dict[str, Any]] = []
    split_manifest = {
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "total_pages": len(all_page_keys),
        "train_pages": 0,
        "val_pages": 0,
        "papers": {},
        "pages": [],
    }

    paper_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0})

    used_names: set[str] = set()
    for idx, page_key in enumerate(all_page_keys, start=1):
        row = per_page[page_key]
        split = split_map.get(page_key, "train")
        out_img_dir = train_img_dir if split == "train" else val_img_dir
        out_lbl_dir = train_lbl_dir if split == "train" else val_lbl_dir

        base_name = _safe_filename(page_key)
        stem = Path(base_name).stem
        candidate = stem
        suffix = 1
        while f"{candidate}.png" in used_names:
            suffix += 1
            candidate = f"{stem}_{suffix}"
        safe_png_name = f"{candidate}.png"
        used_names.add(safe_png_name)

        dst_img = out_img_dir / safe_png_name
        dst_lbl = out_lbl_dir / f"{Path(safe_png_name).stem}.txt"

        shutil.copy2(row["image_src"], dst_img)

        yolo_lines = [
            _yolo_line_from_bbox(b, int(row["image_width"]), int(row["image_height"]))
            for b in row["boxes"]
        ]
        dst_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        paper = str(row["paper"])
        paper_counts[paper][split] += 1

        split_manifest["pages"].append(
            {
                "page": page_key,
                "paper": paper,
                "split": split,
                "yolo_image": str(dst_img),
                "yolo_label": str(dst_lbl),
                "box_count": len(row["boxes"]),
                "dimension_reconciled": bool(row["dimension_reconciled"]),
            }
        )

        for src in row["sources"]:
            mappings.append(
                {
                    "label_file": src["label_file"],
                    "task_id": src["task_id"],
                    "original_image_ref": src["original_image_ref"],
                    "original_image_filename": src["original_image_filename"],
                    "normalized_candidates": src["normalized_candidates"],
                    "chosen_page_key": page_key,
                    "image_src": row["image_src"],
                    "yolo_image": str(dst_img),
                    "yolo_label": str(dst_lbl),
                    "split": split,
                    "source_box_count": src["source_box_count"],
                    "converted_box_count": src["converted_box_count"],
                    "dimension_reconciled": src["dimension_reconciled"],
                }
            )

    split_manifest["train_pages"] = sum(1 for p in split_manifest["pages"] if p["split"] == "train")
    split_manifest["val_pages"] = sum(1 for p in split_manifest["pages"] if p["split"] == "val")
    split_manifest["papers"] = dict(sorted(paper_counts.items()))

    dataset_yaml_path = output_root / "dataset.yaml"
    dataset_yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: job_ad",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = {
        "inputs": {
            "labels_dir": str(labels_dir),
            "images_dir": str(images_dir),
            "output_root": str(output_root),
            "seed": int(args.seed),
            "val_ratio": float(args.val_ratio),
        },
        "summary": {
            "label_files": len(label_files),
            "total_pages_matched": len(all_page_keys),
            "total_pages_unmatched": len(unmatched_pages),
            "train_pages": split_manifest["train_pages"],
            "val_pages": split_manifest["val_pages"],
            "pages_with_boxes": sum(1 for row in per_page.values() if len(row["boxes"]) > 0),
            "pages_without_boxes": sum(1 for row in per_page.values() if len(row["boxes"]) == 0),
            "total_boxes": sum(len(row["boxes"]) for row in per_page.values()),
            "dimension_mismatch_pages_reconciled": len({r["page"] for r in dimension_mismatch_pages}),
            "conversion_errors": len(conversion_errors),
        },
        "papers": split_manifest["papers"],
        "unmatched_pages": unmatched_pages,
        "dimension_mismatch_pages": dimension_mismatch_pages,
        "conversion_errors": conversion_errors,
        "dataset_yaml": str(dataset_yaml_path),
        "mapping_file": str(mapping_output),
    }

    mapping_output.parent.mkdir(parents=True, exist_ok=True)
    mapping_output.write_text(json.dumps(mappings, indent=2, ensure_ascii=False), encoding="utf-8")

    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    split_output.parent.mkdir(parents=True, exist_ok=True)
    split_output.write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"prepared pages: {len(all_page_keys)}")
    print(f"train pages: {split_manifest['train_pages']}")
    print(f"val pages: {split_manifest['val_pages']}")
    print(f"total boxes: {report['summary']['total_boxes']}")
    print(f"dataset yaml: {dataset_yaml_path}")
    print(f"report: {report_output}")
    print(f"split manifest: {split_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
