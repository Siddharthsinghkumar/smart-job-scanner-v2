#!/usr/bin/env python3
"""Run Stage1 repeatedly on the same PDFs and prove filename/hash stability."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pipeline.stage01_pdf_to_images import process_pdf_streaming  # noqa: E402


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed_pdfs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "run_state" / "stage1_stability_runs"
DEFAULT_MANIFEST = PROJECT_ROOT / "run_state" / "stage1_page_identity_manifest.json"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "run_state" / "stage1_rerun_stability_report.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "run_state" / "stage1_rerun_stability_report.md"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _flatten_files(pdf_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pdf_row in pdf_summaries:
        pdf_stem = str(pdf_row.get("pdf_stem", ""))
        pages = pdf_row.get("pages", [])
        if not isinstance(pages, list):
            continue
        for page in pages:
            if not isinstance(page, dict):
                continue
            rendered_filename = str(page.get("rendered_filename", ""))
            rel_path = f"{pdf_stem}/{rendered_filename}" if pdf_stem and rendered_filename else rendered_filename
            rows.append(
                {
                    "relative_path": rel_path,
                    "newspaper": page.get("newspaper"),
                    "source_pdf": page.get("source_pdf"),
                    "pdf_page_index": page.get("pdf_page_index"),
                    "rendered_filename": rendered_filename,
                    "rendered_file_path": page.get("rendered_file_path"),
                    "image_sha256": page.get("image_sha256"),
                    "printed_page_number": page.get("printed_page_number"),
                    "printed_page_number_confidence": page.get("printed_page_number_confidence"),
                    "printed_page_number_source": page.get("printed_page_number_source"),
                }
            )
    return sorted(rows, key=lambda r: str(r.get("relative_path", "")))


def _build_manifest(pdf_summaries: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    for pdf_row in pdf_summaries:
        page_rows = pdf_row.get("pages", [])
        if isinstance(page_rows, list):
            for row in page_rows:
                if isinstance(row, dict):
                    pages.append(row)

    pages = sorted(
        pages,
        key=lambda r: (
            str(r.get("source_pdf", "")),
            int(r.get("pdf_page_index", 10**9)),
            str(r.get("rendered_filename", "")),
        ),
    )

    payload = {
        "generated_at_utc": _utc_now(),
        "naming_policy": {
            "identity_basis": "pdf_page_index",
            "filename_format": "<pdf_stem>_p<index>.png",
            "printed_page_number_used_in_filename": False,
            "printed_page_number_role": "metadata_only",
        },
        "summary": {
            "total_pdfs": len(pdf_summaries),
            "total_rendered_pages": len(pages),
            "total_failed_pages": int(sum(int(p.get("failed_pages", 0)) for p in pdf_summaries)),
        },
        "pdfs": pdf_summaries,
        "pages": pages,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _run_once(run_number: int, pdf_paths: list[Path], run_output_root: Path) -> dict[str, Any]:
    images_output_dir = run_output_root / f"run_{run_number}" / "pdf2img"
    if images_output_dir.exists():
        # Clean only this run output tree.
        for p in sorted(images_output_dir.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
    images_output_dir.mkdir(parents=True, exist_ok=True)

    pdf_summaries: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        summary = process_pdf_streaming(
            pdf_path=pdf_path,
            images_output_dir=str(images_output_dir),
            on_page_done=None,
            move_processed=False,
        )
        if isinstance(summary, dict):
            pdf_summaries.append(summary)

    files = _flatten_files(pdf_summaries)
    hash_index = {str(row.get("relative_path", "")): row.get("image_sha256") for row in files}

    return {
        "run_number": int(run_number),
        "images_output_dir": str(images_output_dir),
        "total_pdfs_processed": len(pdf_summaries),
        "total_pages_rendered": int(sum(int(p.get("rendered_pages", 0)) for p in pdf_summaries)),
        "total_failed_pages": int(sum(int(p.get("failed_pages", 0)) for p in pdf_summaries)),
        "pdfs": pdf_summaries,
        "output_filenames": [str(row.get("relative_path", "")) for row in files],
        "per_file_hashes": hash_index,
        "files": files,
    }


def _compare_against_baseline(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    base_names = set(baseline.get("output_filenames", []))
    curr_names = set(current.get("output_filenames", []))

    base_hashes = baseline.get("per_file_hashes", {})
    curr_hashes = current.get("per_file_hashes", {})

    missing_files = sorted(base_names - curr_names)
    extra_files = sorted(curr_names - base_names)

    shared = sorted(base_names & curr_names)
    hash_mismatches = []
    for rel_path in shared:
        b_hash = base_hashes.get(rel_path)
        c_hash = curr_hashes.get(rel_path)
        if b_hash != c_hash:
            hash_mismatches.append(
                {
                    "relative_path": rel_path,
                    "baseline_sha256": b_hash,
                    "current_sha256": c_hash,
                }
            )

    counts_match = int(baseline.get("total_pages_rendered", 0)) == int(current.get("total_pages_rendered", 0))
    filenames_match = not missing_files and not extra_files
    hashes_match = len(hash_mismatches) == 0

    return {
        "baseline_run": int(baseline.get("run_number", 1)),
        "current_run": int(current.get("run_number", 0)),
        "page_count_match": counts_match,
        "filenames_match": filenames_match,
        "hashes_match": hashes_match,
        "all_match_exactly": bool(counts_match and filenames_match and hashes_match),
        "missing_files": missing_files,
        "extra_files": extra_files,
        "hash_mismatches": hash_mismatches,
    }


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Stage1 Rerun Stability Report")
    lines.append("")
    lines.append(f"Generated (UTC): {report.get('generated_at_utc')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Runs executed: {report.get('runs_executed')}")
    lines.append(f"- Total PDFs per run: {report.get('expected_total_pdfs')}")
    lines.append(f"- Expected pages per run: {report.get('expected_total_pages')}")
    lines.append(f"- All runs match exactly: {report.get('all_runs_match_exactly')}")
    lines.append("")

    lines.append("## Run Stats")
    lines.append("")
    lines.append("| run | total_pdfs | total_pages | failed_pages |")
    lines.append("|---|---:|---:|---:|")
    for row in report.get("runs", []):
        lines.append(
            "| {run} | {pdfs} | {pages} | {failed} |".format(
                run=row.get("run_number"),
                pdfs=row.get("total_pdfs_processed"),
                pages=row.get("total_pages_rendered"),
                failed=row.get("total_failed_pages"),
            )
        )
    lines.append("")

    lines.append("## Baseline Comparisons")
    lines.append("")
    lines.append("| baseline_run | current_run | page_count_match | filenames_match | hashes_match | all_match_exactly |")
    lines.append("|---:|---:|---|---|---|---|")
    for comp in report.get("comparisons_vs_run1", []):
        lines.append(
            "| {b} | {c} | {pcm} | {fnm} | {hm} | {allm} |".format(
                b=comp.get("baseline_run"),
                c=comp.get("current_run"),
                pcm=comp.get("page_count_match"),
                fnm=comp.get("filenames_match"),
                hm=comp.get("hashes_match"),
                allm=comp.get("all_match_exactly"),
            )
        )

    mismatches: list[str] = []
    for comp in report.get("comparisons_vs_run1", []):
        if comp.get("all_match_exactly"):
            continue
        run_no = comp.get("current_run")
        for rel_path in comp.get("missing_files", []):
            mismatches.append(f"- Run {run_no}: missing file vs baseline: {rel_path}")
        for rel_path in comp.get("extra_files", []):
            mismatches.append(f"- Run {run_no}: extra file vs baseline: {rel_path}")
        for row in comp.get("hash_mismatches", []):
            mismatches.append(
                f"- Run {run_no}: hash mismatch {row.get('relative_path')} | baseline={row.get('baseline_sha256')} current={row.get('current_sha256')}"
            )

    lines.append("")
    lines.append("## Detailed Mismatches")
    lines.append("")
    if mismatches:
        lines.extend(mismatches)
    else:
        lines.append("- None. Filename sets, page counts, and per-file hashes matched across all runs.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prove Stage1 rerun stability using repeated rendering")
    parser.add_argument("--pdf-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing source PDFs")
    parser.add_argument("--runs", type=int, default=3, help="Number of reruns to execute")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root for per-run render outputs")
    parser.add_argument("--manifest-output", default=str(DEFAULT_MANIFEST), help="Stage1 identity manifest path")
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON), help="Stability report JSON path")
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD), help="Stability report markdown path")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = (PROJECT_ROOT / pdf_dir).resolve()
    if not pdf_dir.is_dir():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")

    pdf_paths = sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.name.lower())
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {pdf_dir}")

    runs = max(1, int(args.runs))
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()

    run_records: list[dict[str, Any]] = []
    for run_number in range(1, runs + 1):
        run_records.append(_run_once(run_number, pdf_paths, output_root))

    _build_manifest(run_records[0].get("pdfs", []), Path(args.manifest_output) if Path(args.manifest_output).is_absolute() else (PROJECT_ROOT / args.manifest_output).resolve())

    comparisons: list[dict[str, Any]] = []
    baseline = run_records[0]
    for current in run_records[1:]:
        comparisons.append(_compare_against_baseline(baseline, current))

    all_runs_match = all(bool(c.get("all_match_exactly")) for c in comparisons) if comparisons else True

    report_payload = {
        "generated_at_utc": _utc_now(),
        "pdf_source_dir": str(pdf_dir),
        "runs_executed": int(runs),
        "expected_total_pdfs": int(run_records[0].get("total_pdfs_processed", 0)),
        "expected_total_pages": int(run_records[0].get("total_pages_rendered", 0)),
        "all_runs_match_exactly": bool(all_runs_match),
        "naming_semantics": {
            "identity_basis": "pdf_page_index",
            "printed_page_number_used_in_filename": False,
            "printed_page_number_role": "metadata_only",
        },
        "runs": run_records,
        "comparisons_vs_run1": comparisons,
    }

    report_json = Path(args.report_json)
    if not report_json.is_absolute():
        report_json = (PROJECT_ROOT / report_json).resolve()
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report_md = Path(args.report_md)
    if not report_md.is_absolute():
        report_md = (PROJECT_ROOT / report_md).resolve()
    _write_markdown(report_payload, report_md)

    print(f"[ok] manifest: {args.manifest_output}")
    print(f"[ok] report json: {args.report_json}")
    print(f"[ok] report md: {args.report_md}")


if __name__ == "__main__":
    main()
