#!/usr/bin/env python3
"""Render benchmark overlays for GT vs detector vs refined boxes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.auto_improve_detector import _merge_labels_to_temp
from tools.evaluate_against_labels import load_labelstudio_boxes, load_pipeline_detections
from src.utils.benchmark_alignment import compute_dim_scale, scale_bbox_xyxy
from src.utils.page_identity import load_page_identity_map, resolve_page_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render benchmark overlay pages")
    parser.add_argument("--labels-dir", default="data/test_labels", help="Label JSON directory")
    parser.add_argument("--images-dir", default="data/pdf2img", help="Page image directory")
    parser.add_argument("--detections-dir", default="run_state/detections", help="Detections JSON directory")
    parser.add_argument("--audit-json", default="run_state/label_alignment_audit.json", help="Audit JSON path")
    parser.add_argument(
        "--page-identity-map",
        default="run_state/page_identity_map.json",
        help="Optional canonical page-identity mapping JSON",
    )
    parser.add_argument("--output-dir", default="run_state/benchmark_overlays", help="Overlay output directory")
    parser.add_argument("--shortlist-output", default="run_state/benchmark_worst_pages.json", help="Worst-page output JSON")
    parser.add_argument("--top-n", type=int, default=20, help="Number of overlays to render")
    parser.add_argument(
        "--scale-gt-when-dim-mismatch",
        action="store_true",
        help="Scale GT boxes to pipeline image dimensions when label and image dimensions differ",
    )
    return parser.parse_args()


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", name)


def _draw_boxes(img, boxes, color, label):
    for b in boxes:
        if len(b) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


def _draw_legend(img):
    entries = [
        ("GT", (0, 255, 0)),
        ("Detector", (0, 165, 255)),
        ("Refined", (0, 0, 255)),
    ]
    x = 20
    y = 30
    for text, color in entries:
        cv2.rectangle(img, (x, y - 12), (x + 16, y + 4), color, -1)
        cv2.putText(img, text, (x + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26


def main() -> int:
    args = parse_args()
    labels_dir = (PROJECT_ROOT / args.labels_dir).resolve()
    images_dir = (PROJECT_ROOT / args.images_dir).resolve()
    detections_dir = (PROJECT_ROOT / args.detections_dir).resolve()
    audit_json_path = (PROJECT_ROOT / args.audit_json).resolve()
    page_identity_map_path = (PROJECT_ROOT / args.page_identity_map).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    shortlist_output = (PROJECT_ROOT / args.shortlist_output).resolve()

    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not detections_dir.is_dir():
        raise SystemExit(f"Detections directory not found: {detections_dir}")
    if not audit_json_path.is_file():
        raise SystemExit(f"Audit JSON not found: {audit_json_path}")

    merged_labels_path = (PROJECT_ROOT / "run_state" / "merged_labels_tmp_overlays.json").resolve()
    _merge_labels_to_temp(labels_dir, merged_labels_path)

    gt_map = load_labelstudio_boxes(merged_labels_path)
    det_map = load_pipeline_detections(detections_dir, "detector")
    ref_map = load_pipeline_detections(detections_dir, "refined")
    audit = json.loads(audit_json_path.read_text(encoding="utf-8"))
    page_identity_map = load_page_identity_map(page_identity_map_path)
    audit_rows = audit.get("label_pages", [])
    audit_by_page = {row.get("normalized_page_key"): row for row in audit_rows if isinstance(row, dict)}

    image_index = {p.name: p for p in sorted(images_dir.rglob("*.png"))}

    stats: list[dict[str, Any]] = []
    for page in sorted(set(gt_map.keys()) | set(det_map.keys()) | set(ref_map.keys())):
        page_identity = resolve_page_identity(page, page_identity_map)
        gt_count = len(gt_map.get(page, []))
        det_count = len(det_map.get(page, []))
        ref_count = len(ref_map.get(page, []))
        stats.append(
            {
                "page": page,
                "newspaper": page_identity.get("newspaper"),
                "pdf_page_index": page_identity.get("pdf_page_index"),
                "printed_page_number": page_identity.get("printed_page_number"),
                "image_exists": page in image_index,
                "label_count": gt_count,
                "detector_count": det_count,
                "refined_count": ref_count,
                "label_minus_refined": gt_count - ref_count,
                "label_minus_detector": gt_count - det_count,
                "zero_refined_with_labels": bool(gt_count > 0 and ref_count == 0),
            }
        )

    unmatched_label_pages = audit.get("unmatched_label_pages", [])
    unmatched_by_name_issue = [
        row for row in unmatched_label_pages
        if any("name" in w or "hash_prefix" in w for w in row.get("warnings", []))
    ]

    # Rank pages with strongest evidence of failure first.
    ranked = sorted(
        [s for s in stats if s["label_count"] > 0],
        key=lambda s: (
            1 if s["zero_refined_with_labels"] else 0,
            s["label_minus_refined"],
            s["label_count"],
        ),
        reverse=True,
    )

    selected = ranked[: max(1, int(args.top_n))]
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_pages: list[str] = []

    for row in selected:
        page = row["page"]
        img_path = image_index.get(page)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gt_boxes = gt_map.get(page, [])
        det_boxes = [r["bbox"] for r in det_map.get(page, []) if isinstance(r, dict) and isinstance(r.get("bbox"), list)]
        ref_boxes = [r["bbox"] for r in ref_map.get(page, []) if isinstance(r, dict) and isinstance(r.get("bbox"), list)]
        scaled_gt_applied = False
        if args.scale_gt_when_dim_mismatch:
            audit_row = audit_by_page.get(page, {})
            if isinstance(audit_row, dict):
                ld = audit_row.get("label_image_dimensions", {})
                pd = audit_row.get("pipeline_image_dimensions", {})
                label_dims = (
                    int(ld["width"]),
                    int(ld["height"]),
                ) if isinstance(ld, dict) and ld.get("width") and ld.get("height") else None
                pipeline_dims = (
                    int(pd["width"]),
                    int(pd["height"]),
                ) if isinstance(pd, dict) and pd.get("width") and pd.get("height") else None
                scale = compute_dim_scale(label_dims, pipeline_dims)
                if scale and (abs(scale[0] - 1.0) > 1e-6 or abs(scale[1] - 1.0) > 1e-6):
                    gt_boxes = [scale_bbox_xyxy(b, scale) for b in gt_boxes]
                    scaled_gt_applied = True

        _draw_boxes(img, gt_boxes, (0, 255, 0), "GT")
        _draw_boxes(img, det_boxes, (0, 165, 255), "Detector")
        _draw_boxes(img, ref_boxes, (0, 0, 255), "Refined")
        _draw_legend(img)
        if scaled_gt_applied:
            cv2.putText(
                img,
                "GT scaled to pipeline image dims",
                (20, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            img,
            (
                f"{page} | PDF:{row.get('pdf_page_index')} PRINT:{row.get('printed_page_number')} "
                f"| GT:{len(gt_boxes)} DET:{len(det_boxes)} REF:{len(ref_boxes)}"
            ),
            (20, max(30, img.shape[0] - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_name = _safe_name(page)
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), img)
        rendered_pages.append(page)

    highest_label_zero_refined = sorted(
        [s for s in stats if s["label_count"] > 0 and s["refined_count"] == 0],
        key=lambda s: (s["label_count"], s["label_minus_refined"]),
        reverse=True,
    )[:30]

    largest_label_refined_gap = sorted(
        [s for s in stats if s["label_count"] > 0],
        key=lambda s: s["label_minus_refined"],
        reverse=True,
    )[:30]

    shortlist = {
        "page_identity_map_loaded": bool(page_identity_map),
        "page_identity_map_path": str(page_identity_map_path),
        "ranked_pages": ranked[:50],
        "highest_labeled_with_zero_refined": highest_label_zero_refined,
        "largest_label_refined_gap": largest_label_refined_gap,
        "unmatched_pages_with_naming_issues": unmatched_by_name_issue,
        "rendered_pages": rendered_pages,
        "overlay_dir": str(output_dir),
    }

    shortlist_output.parent.mkdir(parents=True, exist_ok=True)
    shortlist_output.write_text(json.dumps(shortlist, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"rendered overlays: {len(rendered_pages)}")
    print(f"overlay dir: {output_dir}")
    print(f"worst pages saved: {shortlist_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
