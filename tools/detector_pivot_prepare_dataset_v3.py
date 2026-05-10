#!/usr/bin/env python3
"""Prepare detector-pivot v3 source YOLO dataset from expanded 6-issue supervised corpus."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare detector pivot v3 source dataset")
    parser.add_argument("--labels-dir", default="data/test_labels")
    parser.add_argument("--candidate-image-roots", default="data/pdf2img,data/benchmark_images")
    parser.add_argument("--output-root", default="data/yolo_job_ad_pivot_v3_source")
    parser.add_argument("--audit-output", default="run_state/detector_pivot_v3_corpus_audit.json")
    parser.add_argument("--split-output", default="run_state/detector_pivot_v3_source_split_manifest.json")
    parser.add_argument("--report-output", default="run_state/detector_pivot_v3_source_dataset_report.json")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260407)
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", []) if isinstance(payload, dict) else payload
    if not isinstance(tasks, list):
        return []
    return [t for t in tasks if isinstance(t, dict)]


_HASH_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{8}-")
_PAGE_SUFFIX_RE = re.compile(r" (p\d+\.png)$")


def _normalize_label_image_name(ref: str) -> str:
    name = unquote(Path(str(ref)).name)
    name = _HASH_PREFIX_RE.sub("", name)
    name = name.replace("_", " ")
    # Restore expected page-name convention: "<issue>_pN.png"
    name = _PAGE_SUFFIX_RE.sub(r"_\1", name)
    return name


def _extract_boxes(task: dict[str, Any]) -> tuple[list[list[int]], tuple[int, int] | None]:
    boxes: list[list[int]] = []
    label_dims: tuple[int, int] | None = None
    anns = task.get("annotations", [])
    if not isinstance(anns, list):
        return boxes, label_dims

    for ann in anns:
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
            x = float(value.get("x", 0.0))
            y = float(value.get("y", 0.0))
            w = float(value.get("width", 0.0))
            h = float(value.get("height", 0.0))
            x1 = int(round((x / 100.0) * ow))
            y1 = int(round((y / 100.0) * oh))
            x2 = int(round(((x + w) / 100.0) * ow))
            y2 = int(round(((y + h) / 100.0) * oh))
            boxes.append([x1, y1, x2, y2])

    return boxes, label_dims


def _clip_box(box: list[int], image_w: int, image_h: int) -> list[int] | None:
    if len(box) != 4:
        return None
    x1 = max(0, min(image_w - 1, int(box[0])))
    y1 = max(0, min(image_h - 1, int(box[1])))
    x2 = max(0, min(image_w, int(box[2])))
    y2 = max(0, min(image_h, int(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _safe_name(page_name: str) -> str:
    stem = Path(page_name).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return f"{safe or 'page'}.png"


def _yolo_line(box: list[int], image_w: int, image_h: int) -> str:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"0 {cx / image_w:.6f} {cy / image_h:.6f} {bw / image_w:.6f} {bh / image_h:.6f}"


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))

    labels_dir = _resolve(args.labels_dir)
    candidate_roots = [_resolve(x.strip()) for x in str(args.candidate_image_roots).split(",") if x.strip()]
    output_root = _resolve(args.output_root)
    audit_output = _resolve(args.audit_output)
    split_output = _resolve(args.split_output)
    report_output = _resolve(args.report_output)

    if not labels_dir.is_dir():
        raise SystemExit(f"labels dir not found: {labels_dir}")
    if not candidate_roots:
        raise SystemExit("no candidate image roots provided")

    # Resolve issue directories under allowed roots only (explicitly avoid validation-data holdouts).
    issue_dirs: dict[str, set[str]] = {}
    issue_dir_paths: dict[str, Path] = {}
    for root in candidate_roots:
        if not root.is_dir():
            continue
        for d in sorted([x for x in root.iterdir() if x.is_dir()]):
            if "validation-data" in str(d).replace("\\", "/"):
                continue
            pages = {p.name for p in d.glob("*.png")}
            if not pages:
                continue
            issue_dirs[str(d)] = pages
            issue_dir_paths[str(d)] = d

    label_files = sorted(labels_dir.glob("*.json"))
    issue_audit_rows: list[dict[str, Any]] = []
    issue_resolution: dict[str, str] = {}
    unresolved_issues: list[str] = []

    for lf in label_files:
        tasks = _load_tasks(lf)
        refs: list[str] = []
        for t in tasks:
            data = t.get("data", {}) if isinstance(t.get("data"), dict) else {}
            image_ref = data.get("image") or data.get("page") or data.get("file")
            if image_ref:
                refs.append(_normalize_label_image_name(str(image_ref)))
        ref_set = set(refs)

        overlaps: list[tuple[int, int, str]] = []
        for issue_dir, pages in issue_dirs.items():
            ov = len(ref_set & pages)
            if ov > 0:
                overlaps.append((ov, len(ref_set), issue_dir))
        overlaps.sort(key=lambda x: (-x[0], x[2]))

        if overlaps:
            best_dir = overlaps[0][2]
            issue_resolution[lf.name] = best_dir
            missing = sorted(ref_set - issue_dirs[best_dir])
            issue_audit_rows.append(
                {
                    "label_file": lf.name,
                    "task_count": len(tasks),
                    "unique_normalized_page_refs": len(ref_set),
                    "resolved_image_root": best_dir,
                    "overlap_count": int(overlaps[0][0]),
                    "missing_page_refs_in_resolved_root_count": len(missing),
                    "missing_page_refs_in_resolved_root_sample": missing[:20],
                    "candidate_overlaps_top5": [
                        {"image_root": row[2], "overlap": int(row[0]), "total_refs": int(row[1])}
                        for row in overlaps[:5]
                    ],
                }
            )
        else:
            unresolved_issues.append(lf.name)
            issue_audit_rows.append(
                {
                    "label_file": lf.name,
                    "task_count": len(tasks),
                    "unique_normalized_page_refs": len(ref_set),
                    "resolved_image_root": None,
                    "overlap_count": 0,
                    "missing_page_refs_in_resolved_root_count": len(ref_set),
                    "missing_page_refs_in_resolved_root_sample": sorted(ref_set)[:20],
                    "candidate_overlaps_top5": [],
                }
            )

    # Build per-page canonical records.
    per_page: dict[str, dict[str, Any]] = {}
    conversion_errors: list[dict[str, Any]] = []
    unmatched_tasks: list[dict[str, Any]] = []

    for lf in label_files:
        resolved_root = issue_resolution.get(lf.name)
        if not resolved_root:
            continue
        root_pages = issue_dirs.get(resolved_root, set())
        tasks = _load_tasks(lf)
        for t in tasks:
            data = t.get("data", {}) if isinstance(t.get("data"), dict) else {}
            image_ref = data.get("image") or data.get("page") or data.get("file")
            if not image_ref:
                continue
            page_name = _normalize_label_image_name(str(image_ref))
            if page_name not in root_pages:
                unmatched_tasks.append(
                    {
                        "label_file": lf.name,
                        "task_id": t.get("id"),
                        "image_ref": str(image_ref),
                        "normalized_page_name": page_name,
                        "resolved_image_root": resolved_root,
                    }
                )
                continue
            image_path = issue_dir_paths[resolved_root] / page_name
            img = cv2.imread(str(image_path))
            if img is None:
                conversion_errors.append(
                    {
                        "label_file": lf.name,
                        "task_id": t.get("id"),
                        "page": page_name,
                        "error": f"failed_to_read_image:{image_path}",
                    }
                )
                continue
            image_h, image_w = int(img.shape[0]), int(img.shape[1])

            boxes, label_dims = _extract_boxes(t)
            if label_dims:
                sx = image_w / max(1.0, float(label_dims[0]))
                sy = image_h / max(1.0, float(label_dims[1]))
            else:
                sx = 1.0
                sy = 1.0

            converted: list[list[int]] = []
            for b in boxes:
                scaled = [
                    int(round(b[0] * sx)),
                    int(round(b[1] * sy)),
                    int(round(b[2] * sx)),
                    int(round(b[3] * sy)),
                ]
                clipped = _clip_box(scaled, image_w=image_w, image_h=image_h)
                if clipped is not None:
                    converted.append(clipped)

            key = page_name
            row = per_page.setdefault(
                key,
                {
                    "page": page_name,
                    "issue_label_file": lf.name,
                    "resolved_image_root": resolved_root,
                    "image_path": str(image_path),
                    "image_w": image_w,
                    "image_h": image_h,
                    "boxes": [],
                    "task_ids": [],
                },
            )
            row["boxes"].extend(converted)
            row["task_ids"].append(t.get("id"))

    # Deduplicate same box repeats per page.
    for row in per_page.values():
        uniq = []
        seen = set()
        for b in row["boxes"]:
            k = tuple(int(v) for v in b)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(b)
        row["boxes"] = uniq

    # Split (issue-stratified pragmatic split with positive-page coverage where possible).
    by_issue: dict[str, list[str]] = defaultdict(list)
    for page_name, row in per_page.items():
        by_issue[row["issue_label_file"]].append(page_name)

    split_map: dict[str, str] = {}
    rng = random.Random(int(args.seed))
    split_debug: dict[str, Any] = {}

    for issue, pages in sorted(by_issue.items()):
        pages = sorted(pages)
        if len(pages) == 1:
            split_map[pages[0]] = "train"
            split_debug[issue] = {"total_pages": 1, "train_pages": 1, "val_pages": 0, "note": "single_page_issue"}
            continue

        val_target = int(round(len(pages) * float(args.val_ratio)))
        val_target = max(1, min(len(pages) - 1, val_target))
        pos_pages = [p for p in pages if len(per_page[p]["boxes"]) > 0]
        val_pages: set[str] = set()

        pool = list(pages)
        rng.shuffle(pool)

        if pos_pages and val_target > 0:
            pos_pool = sorted(pos_pages)
            rng.shuffle(pos_pool)
            val_pages.add(pos_pool[0])

        for p in pool:
            if len(val_pages) >= val_target:
                break
            val_pages.add(p)

        # Keep at least one positive page in train when issue has >1 positive page.
        if len(pos_pages) > 1 and all(p in val_pages for p in pos_pages):
            move_back = sorted(pos_pages)[0]
            if move_back in val_pages:
                val_pages.remove(move_back)
                for p in pool:
                    if p not in val_pages and p != move_back:
                        val_pages.add(p)
                        break

        for p in pages:
            split_map[p] = "val" if p in val_pages else "train"

        split_debug[issue] = {
            "total_pages": len(pages),
            "positive_pages": len(pos_pages),
            "train_pages": int(sum(1 for p in pages if split_map[p] == "train")),
            "val_pages": int(sum(1 for p in pages if split_map[p] == "val")),
            "train_positive_pages": int(sum(1 for p in pages if split_map[p] == "train" and len(per_page[p]["boxes"]) > 0)),
            "val_positive_pages": int(sum(1 for p in pages if split_map[p] == "val" and len(per_page[p]["boxes"]) > 0)),
        }

    # Write YOLO source dataset.
    train_img_dir = output_root / "images" / "train"
    val_img_dir = output_root / "images" / "val"
    train_lbl_dir = output_root / "labels" / "train"
    val_lbl_dir = output_root / "labels" / "val"
    _clean_dir(train_img_dir)
    _clean_dir(val_img_dir)
    _clean_dir(train_lbl_dir)
    _clean_dir(val_lbl_dir)

    mapping_rows: list[dict[str, Any]] = []
    for page_name, row in sorted(per_page.items()):
        split = split_map.get(page_name, "train")
        out_name = _safe_name(page_name)
        out_img = (train_img_dir if split == "train" else val_img_dir) / out_name
        out_lbl = (train_lbl_dir if split == "train" else val_lbl_dir) / f"{Path(out_name).stem}.txt"
        shutil.copy2(row["image_path"], out_img)

        lines = [_yolo_line(b, image_w=int(row["image_w"]), image_h=int(row["image_h"])) for b in row["boxes"]]
        out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        mapping_rows.append(
            {
                "page": page_name,
                "output_image": str(out_img),
                "output_label": str(out_lbl),
                "split": split,
                "box_count": len(row["boxes"]),
                "issue_label_file": row["issue_label_file"],
                "resolved_image_root": row["resolved_image_root"],
                "source_image_path": row["image_path"],
            }
        )

    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {output_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: job_ad\n",
        encoding="utf-8",
    )

    # Audit output (PHASE 1)
    missing_mappings = [row for row in issue_audit_rows if (not row.get("resolved_image_root")) or row.get("missing_page_refs_in_resolved_root_count", 0) > 0]
    roots_used = sorted({row["resolved_image_root"] for row in issue_audit_rows if row.get("resolved_image_root")})
    audit_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "labeled_issue_files_detected": [p.name for p in label_files],
        "issue_resolution": issue_audit_rows,
        "missing_mappings": missing_mappings,
        "unresolved_issue_files": unresolved_issues,
        "unmatched_tasks_count": len(unmatched_tasks),
        "unmatched_tasks_sample": unmatched_tasks[:20],
        "conversion_errors_count": len(conversion_errors),
        "conversion_errors_sample": conversion_errors[:20],
        "recommendation": {
            "image_roots_to_use_for_v3": roots_used,
            "labels_root": str(labels_dir),
            "external_holdout_excluded": "data/validation-data/different-newspaper-english",
            "notes": [
                "Use resolved roots only for these 6 label files.",
                "Do not include any validation-data holdout pages in training/dev splits.",
            ],
        },
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Source split manifest/report for downstream build traceability.
    split_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "total_pages": len(per_page),
        "train_pages": int(sum(1 for s in split_map.values() if s == "train")),
        "val_pages": int(sum(1 for s in split_map.values() if s == "val")),
        "issue_split_debug": split_debug,
        "page_splits": [{"page": p, "split": split_map[p], "issue_label_file": per_page[p]["issue_label_file"]} for p in sorted(per_page.keys())],
    }
    split_output.parent.mkdir(parents=True, exist_ok=True)
    split_output.write_text(json.dumps(split_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "output_root": str(output_root),
        "dataset_yaml": str(dataset_yaml),
        "summary": {
            "label_files": len(label_files),
            "resolved_issues": len(issue_resolution),
            "unresolved_issues": len(unresolved_issues),
            "total_pages_converted": len(per_page),
            "pages_with_boxes": int(sum(1 for x in per_page.values() if len(x["boxes"]) > 0)),
            "pages_without_boxes": int(sum(1 for x in per_page.values() if len(x["boxes"]) == 0)),
            "total_boxes": int(sum(len(x["boxes"]) for x in per_page.values())),
            "train_pages": split_payload["train_pages"],
            "val_pages": split_payload["val_pages"],
        },
        "artifacts": {
            "audit_output": str(audit_output),
            "split_output": str(split_output),
            "mapping_rows_count": len(mapping_rows),
        },
        "safety": {
            "labels_read_only": True,
            "benchmark_images_read_only": True,
            "external_holdout_used": False,
        },
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"audit_output: {audit_output}")
    print(f"split_output: {split_output}")
    print(f"report_output: {report_output}")
    print(f"dataset_yaml: {dataset_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

