#!/usr/bin/env python3
"""Build canonical page-identity map for rendered pages, labels, and printed-page semantics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.benchmark_alignment import (  # noqa: E402
    choose_best_page_key,
    normalize_label_image_candidates,
    source_hint_from_label_file,
)
from src.utils.page_identity import detect_printed_page_number_from_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical page identity map")
    parser.add_argument("--processed-pdfs-dir", default="data/processed_pdfs")
    parser.add_argument("--images-dir", default="data/pdf2img")
    parser.add_argument("--labels-dir", default="data/test_labels")
    parser.add_argument("--detections-dir", default="run_state/detections")
    parser.add_argument("--output", default="run_state/page_identity_map.json")
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip printed-page OCR detection and leave printed_page_number null",
    )
    return parser.parse_args()


def _safe_pdf_page_count(pdf_path: Path) -> int | None:
    if not pdf_path.is_file():
        return None
    try:
        out = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True, stderr=subprocess.DEVNULL)  # noqa: S603
    except Exception:
        return None
    for line in out.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                return None
    return None


def _load_detection_keys(detections_dir: Path) -> set[str]:
    out: set[str] = set()
    if not detections_dir.is_dir():
        return out
    for jf in sorted(detections_dir.glob("*.json")):
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        page_key = str(payload.get("page") or jf.stem)
        if page_key:
            out.add(page_key)
    return out


def _load_label_aliases(
    labels_dir: Path,
    image_keys: set[str],
    detection_keys: set[str],
) -> dict[str, dict[str, Any]]:
    aliases_by_rendered: dict[str, dict[str, Any]] = {}

    for label_file in sorted(labels_dir.glob("*.json")):
        payload = json.loads(label_file.read_text(encoding="utf-8"))
        tasks = payload if isinstance(payload, list) else payload.get("tasks", [])
        if not isinstance(tasks, list):
            continue
        hint = source_hint_from_label_file(label_file.name)
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_id = task.get("id")
            data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
            image_ref = str(data.get("image") or data.get("page") or data.get("file") or "")
            original_filename = Path(unquote(image_ref)).name if image_ref else ""

            normalized_key, candidates, warnings = normalize_label_image_candidates(original_filename, hint)
            chosen_key = choose_best_page_key(candidates, image_keys, detection_keys) or normalized_key

            row = aliases_by_rendered.setdefault(
                chosen_key,
                {
                    "label_task_filenames": [],
                    "label_task_refs": [],
                    "label_source_files": [],
                    "label_aliases": set(),
                    "label_warnings": set(),
                },
            )
            row["label_task_filenames"].append(original_filename)
            row["label_task_refs"].append(f"{label_file.name}::task_{task_id}")
            row["label_source_files"].append(label_file.name)
            row["label_aliases"].add(original_filename)
            row["label_aliases"].add(normalized_key)
            for c in candidates:
                row["label_aliases"].add(str(c))
            for w in warnings:
                row["label_warnings"].add(str(w))

    # Deduplicate + sort.
    cleaned: dict[str, dict[str, Any]] = {}
    for key, row in aliases_by_rendered.items():
        cleaned[key] = {
            "label_task_filenames": sorted(set(str(x) for x in row["label_task_filenames"] if str(x))),
            "label_task_refs": sorted(set(str(x) for x in row["label_task_refs"] if str(x))),
            "label_source_files": sorted(set(str(x) for x in row["label_source_files"] if str(x))),
            "label_aliases": sorted(set(str(x) for x in row["label_aliases"] if str(x))),
            "label_warnings": sorted(set(str(x) for x in row["label_warnings"] if str(x))),
        }
    return cleaned


def main() -> int:
    args = parse_args()
    processed_pdfs_dir = (PROJECT_ROOT / args.processed_pdfs_dir).resolve()
    images_dir = (PROJECT_ROOT / args.images_dir).resolve()
    labels_dir = (PROJECT_ROOT / args.labels_dir).resolve()
    detections_dir = (PROJECT_ROOT / args.detections_dir).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()

    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    image_paths = sorted(images_dir.rglob("*.png"))
    image_keys = {p.name for p in image_paths}
    detection_keys = _load_detection_keys(detections_dir)
    label_aliases = _load_label_aliases(labels_dir, image_keys, detection_keys)

    pages: list[dict[str, Any]] = []
    per_paper_counts: dict[str, int] = {}

    for paper_dir in sorted([p for p in images_dir.iterdir() if p.is_dir()]):
        newspaper = paper_dir.name
        source_pdf = (processed_pdfs_dir / f"{newspaper}.pdf").resolve()
        source_pdf_str = str(source_pdf) if source_pdf.is_file() else None
        source_pdf_pages = _safe_pdf_page_count(source_pdf) if source_pdf.is_file() else None

        rendered = sorted(paper_dir.glob("*.png"), key=lambda p: int(Path(p.name).stem.rsplit("_p", 1)[1]))
        per_paper_counts[newspaper] = len(rendered)

        for img in rendered:
            page_idx = int(Path(img.name).stem.rsplit("_p", 1)[1])
            rendered_key = img.name

            printed = {
                "printed_page_number": None,
                "confidence": "none",
                "source": "ocr_skipped" if args.skip_ocr else "not_detected",
                "evidence_excerpt": None,
            }
            if not args.skip_ocr:
                printed = detect_printed_page_number_from_image(img)

            alias_row = label_aliases.get(
                rendered_key,
                {
                    "label_task_filenames": [],
                    "label_task_refs": [],
                    "label_source_files": [],
                    "label_aliases": [rendered_key],
                    "label_warnings": [],
                },
            )

            row = {
                "newspaper": newspaper,
                "source_pdf": source_pdf_str,
                "source_pdf_page_count": source_pdf_pages,
                "pdf_page_index": page_idx,
                "rendered_filename": rendered_key,
                "rendered_page_key": rendered_key,
                "printed_page_number": printed.get("printed_page_number"),
                "printed_page_number_confidence": printed.get("confidence"),
                "printed_page_number_source": printed.get("source"),
                "printed_page_number_evidence": printed.get("evidence_excerpt"),
                "printed_minus_pdf_index": (
                    int(printed.get("printed_page_number")) - page_idx
                    if isinstance(printed.get("printed_page_number"), int)
                    else None
                ),
                "label_task_filenames": alias_row.get("label_task_filenames", []),
                "label_task_refs": alias_row.get("label_task_refs", []),
                "label_source_files": alias_row.get("label_source_files", []),
                "label_aliases": sorted(set(alias_row.get("label_aliases", []) + [rendered_key])),
                "label_warnings": alias_row.get("label_warnings", []),
            }
            pages.append(row)

    by_rendered_key = {row["rendered_page_key"]: row for row in pages}
    by_newspaper_printed: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in pages:
        pp = row.get("printed_page_number")
        if isinstance(pp, int):
            by_newspaper_printed[row["newspaper"]][str(pp)].append(row["rendered_page_key"])

    summary = {
        "newspaper_count": len(per_paper_counts),
        "total_pages": len(pages),
        "pages_with_printed_number": sum(1 for r in pages if isinstance(r.get("printed_page_number"), int)),
        "per_newspaper_rendered_pages": dict(sorted(per_paper_counts.items())),
        "per_newspaper_pages_with_printed_number": {
            paper: sum(
                1
                for r in pages
                if r.get("newspaper") == paper and isinstance(r.get("printed_page_number"), int)
            )
            for paper in sorted(per_paper_counts.keys())
        },
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "processed_pdfs_dir": str(processed_pdfs_dir),
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "detections_dir": str(detections_dir),
            "skip_ocr": bool(args.skip_ocr),
        },
        "summary": summary,
        "pages": pages,
        "indexes": {
            "by_rendered_key": by_rendered_key,
            "by_newspaper_and_printed_page": {
                paper: dict(sorted(inner.items(), key=lambda kv: int(kv[0])))
                for paper, inner in sorted(by_newspaper_printed.items())
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved map: {output_path}")
    print(f"total pages: {summary['total_pages']}")
    print(f"pages with printed number: {summary['pages_with_printed_number']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
