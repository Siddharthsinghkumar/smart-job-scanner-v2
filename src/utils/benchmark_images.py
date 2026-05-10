"""Benchmark image freezing, manifest building, and immutability validation helpers."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_IMAGES_DIR = PROJECT_ROOT / "data" / "benchmark_images"
DEFAULT_BENCHMARK_MANIFEST_PATH = PROJECT_ROOT / "run_state" / "benchmark_image_manifest.json"
DEFAULT_STAGE1_MANIFEST_PATH = PROJECT_ROOT / "run_state" / "stage1_page_identity_manifest.json"
DEFAULT_PAGE_IDENTITY_MAP_PATH = PROJECT_ROOT / "run_state" / "page_identity_map.json"

_PAGE_NAME_RE = re.compile(r"^(?P<paper>.+)_p(?P<idx>\d+)\.png$")


def _resolve_project_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def _to_project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _parse_page_identity(rendered_filename: str) -> tuple[str | None, int | None]:
    m = _PAGE_NAME_RE.fullmatch(str(rendered_filename).strip())
    if not m:
        return None, None
    return m.group("paper"), int(m.group("idx"))


def _stage1_index(stage1_manifest_path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(stage1_manifest_path)
    if not isinstance(payload, dict):
        return {}
    pages = payload.get("pages", [])
    if not isinstance(pages, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in pages:
        if not isinstance(row, dict):
            continue
        key = str(row.get("rendered_filename") or row.get("rendered_page_key") or "")
        if key:
            out[key] = row
    return out


def _page_identity_index(page_identity_map_path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(page_identity_map_path)
    if not isinstance(payload, dict):
        return {}
    pages = payload.get("pages", [])
    if not isinstance(pages, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in pages:
        if not isinstance(row, dict):
            continue
        key = str(row.get("rendered_page_key") or row.get("rendered_filename") or "")
        if key:
            out[key] = row
    return out


def sync_frozen_benchmark_images(
    *,
    source_images_dir: str | Path = "data/pdf2img",
    benchmark_images_dir: str | Path = DEFAULT_BENCHMARK_IMAGES_DIR,
    force_sync: bool = False,
) -> dict[str, Any]:
    source_dir = _resolve_project_path(source_images_dir)
    target_dir = _resolve_project_path(benchmark_images_dir)
    if not source_dir.is_dir():
        raise RuntimeError(f"Source images directory not found: {source_dir}")

    source_pngs = sorted(source_dir.rglob("*.png"))
    if not source_pngs:
        raise RuntimeError(f"No PNG files found under source images directory: {source_dir}")

    target_has_pngs = target_dir.is_dir() and any(target_dir.rglob("*.png"))
    copied = 0
    action = "kept_existing"

    if force_sync and target_dir.exists():
        shutil.rmtree(target_dir)
        target_has_pngs = False

    if not target_has_pngs:
        action = "copied_from_source"
        target_dir.mkdir(parents=True, exist_ok=True)
        for src in source_pngs:
            rel = src.relative_to(source_dir)
            dst = target_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    target_png_count = sum(1 for _ in target_dir.rglob("*.png")) if target_dir.is_dir() else 0
    return {
        "action": action,
        "source_images_dir": _to_project_relative(source_dir),
        "benchmark_images_dir": _to_project_relative(target_dir),
        "source_png_count": len(source_pngs),
        "copied_png_count": int(copied),
        "benchmark_png_count": int(target_png_count),
        "force_sync": bool(force_sync),
    }


def build_benchmark_manifest(
    *,
    benchmark_images_dir: str | Path = DEFAULT_BENCHMARK_IMAGES_DIR,
    manifest_output_path: str | Path = DEFAULT_BENCHMARK_MANIFEST_PATH,
    stage1_manifest_path: str | Path = DEFAULT_STAGE1_MANIFEST_PATH,
    page_identity_map_path: str | Path = DEFAULT_PAGE_IDENTITY_MAP_PATH,
) -> dict[str, Any]:
    images_dir = _resolve_project_path(benchmark_images_dir)
    output_path = _resolve_project_path(manifest_output_path)
    stage1_path = _resolve_project_path(stage1_manifest_path)
    page_identity_path = _resolve_project_path(page_identity_map_path)

    if not images_dir.is_dir():
        raise RuntimeError(f"Benchmark images directory not found: {images_dir}")

    stage1_rows = _stage1_index(stage1_path)
    identity_rows = _page_identity_index(page_identity_path)
    image_paths = sorted(images_dir.rglob("*.png"))
    rows: list[dict[str, Any]] = []

    for image_path in image_paths:
        rendered_filename = image_path.name
        paper_from_name, page_idx_from_name = _parse_page_identity(rendered_filename)
        stage1_row = stage1_rows.get(rendered_filename, {})
        identity_row = identity_rows.get(rendered_filename, {})

        source_pdf = (
            stage1_row.get("source_pdf")
            or identity_row.get("source_pdf")
            or (f"data/raw_pdfs/{paper_from_name}.pdf" if paper_from_name else None)
        )
        pdf_page_index = (
            stage1_row.get("pdf_page_index")
            if stage1_row.get("pdf_page_index") is not None
            else identity_row.get("pdf_page_index")
        )
        if pdf_page_index is None:
            pdf_page_index = page_idx_from_name

        printed_page_number = identity_row.get("printed_page_number")
        if printed_page_number is None and "printed_page_number" in stage1_row:
            printed_page_number = stage1_row.get("printed_page_number")

        printed_conf = (
            identity_row.get("printed_page_number_confidence")
            or stage1_row.get("printed_page_number_confidence")
            or "none"
        )
        printed_source = (
            identity_row.get("printed_page_number_source")
            or stage1_row.get("printed_page_number_source")
            or "not_available"
        )

        rows.append(
            {
                "newspaper": identity_row.get("newspaper") or stage1_row.get("newspaper") or paper_from_name,
                "source_pdf": str(source_pdf) if source_pdf else None,
                "pdf_page_index": int(pdf_page_index) if pdf_page_index is not None else None,
                "rendered_filename": rendered_filename,
                "rendered_file_path": _to_project_relative(image_path),
                "image_sha256": _sha256_file(image_path),
                "printed_page_number": int(printed_page_number) if isinstance(printed_page_number, int) else None,
                "printed_page_number_confidence": str(printed_conf),
                "printed_page_number_source": str(printed_source),
            }
        )

    rows = sorted(
        rows,
        key=lambda r: (
            str(r.get("source_pdf") or ""),
            int(r.get("pdf_page_index") or 10**9),
            str(r.get("rendered_filename") or ""),
        ),
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_images_dir": _to_project_relative(images_dir),
        "stage1_manifest_path": _to_project_relative(stage1_path),
        "page_identity_map_path": _to_project_relative(page_identity_path),
        "summary": {
            "total_images": len(rows),
            "pages_with_printed_page_number": sum(1 for r in rows if isinstance(r.get("printed_page_number"), int)),
        },
        "pages": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def validate_benchmark_manifest(
    *,
    benchmark_images_dir: str | Path = DEFAULT_BENCHMARK_IMAGES_DIR,
    manifest_path: str | Path = DEFAULT_BENCHMARK_MANIFEST_PATH,
) -> dict[str, Any]:
    images_dir = _resolve_project_path(benchmark_images_dir)
    manifest_file = _resolve_project_path(manifest_path)
    payload = _load_json(manifest_file)

    required_fields = {
        "source_pdf",
        "pdf_page_index",
        "rendered_filename",
        "rendered_file_path",
        "image_sha256",
        "printed_page_number",
    }

    schema_issues: list[str] = []
    duplicate_rendered_filenames: list[str] = []
    missing_files: list[str] = []
    extra_files: list[str] = []
    hash_mismatches: list[dict[str, str]] = []

    rows: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        schema_issues.append("manifest_not_valid_json_object")
    else:
        manifest_rows = payload.get("pages", [])
        if not isinstance(manifest_rows, list):
            schema_issues.append("manifest_pages_not_a_list")
        else:
            rows = [r for r in manifest_rows if isinstance(r, dict)]

    seen_names: set[str] = set()
    manifest_name_set: set[str] = set()
    manifest_path_set: set[str] = set()
    for idx, row in enumerate(rows):
        missing_required = sorted(required_fields - set(row.keys()))
        if missing_required:
            schema_issues.append(f"row_{idx}_missing_fields:{','.join(missing_required)}")
            continue
        rendered_filename = str(row.get("rendered_filename") or "")
        rendered_file_path = str(row.get("rendered_file_path") or "")
        if not rendered_filename:
            schema_issues.append(f"row_{idx}_missing_rendered_filename")
            continue
        if rendered_filename in seen_names:
            duplicate_rendered_filenames.append(rendered_filename)
        seen_names.add(rendered_filename)
        manifest_name_set.add(rendered_filename)
        if rendered_file_path:
            manifest_path_set.add(rendered_file_path)

        candidate_path = _resolve_project_path(rendered_file_path) if rendered_file_path else (images_dir / rendered_filename)
        if not candidate_path.is_file():
            missing_files.append(rendered_filename)
            continue
        actual_hash = _sha256_file(candidate_path)
        expected_hash = str(row.get("image_sha256") or "")
        if actual_hash != expected_hash:
            hash_mismatches.append(
                {
                    "rendered_filename": rendered_filename,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                }
            )

    actual_files = sorted(images_dir.rglob("*.png")) if images_dir.is_dir() else []
    actual_name_set = {p.name for p in actual_files}
    actual_rel_set = {_to_project_relative(p) for p in actual_files}
    extra_files = sorted(actual_name_set - manifest_name_set)
    missing_from_disk_by_set = sorted(manifest_name_set - actual_name_set)
    if missing_from_disk_by_set:
        for name in missing_from_disk_by_set:
            if name not in missing_files:
                missing_files.append(name)
    missing_files = sorted(set(missing_files))

    filename_set_matches = manifest_name_set == actual_name_set
    path_set_has_drift = bool(manifest_path_set and manifest_path_set != actual_rel_set)
    validation_passed = not (
        schema_issues
        or duplicate_rendered_filenames
        or missing_files
        or extra_files
        or hash_mismatches
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_images_dir": _to_project_relative(images_dir),
        "manifest_path": _to_project_relative(manifest_file),
        "manifest_rows": len(rows),
        "actual_image_files": len(actual_files),
        "filename_set_matches_manifest": bool(filename_set_matches),
        "manifest_rendered_paths_match_disk_paths": not path_set_has_drift,
        "validation_passed": bool(validation_passed),
        "schema_issues": schema_issues,
        "duplicate_rendered_filenames": sorted(set(duplicate_rendered_filenames)),
        "missing_files": missing_files,
        "extra_files": extra_files,
        "hash_mismatches": hash_mismatches,
    }
    return report


def assert_benchmark_manifest_valid(
    *,
    benchmark_images_dir: str | Path = DEFAULT_BENCHMARK_IMAGES_DIR,
    manifest_path: str | Path = DEFAULT_BENCHMARK_MANIFEST_PATH,
) -> dict[str, Any]:
    report = validate_benchmark_manifest(
        benchmark_images_dir=benchmark_images_dir,
        manifest_path=manifest_path,
    )
    if report.get("validation_passed"):
        return report

    msg_parts = ["Benchmark image validation failed."]
    if report.get("schema_issues"):
        msg_parts.append(f"schema_issues={len(report['schema_issues'])}")
    if report.get("missing_files"):
        msg_parts.append(f"missing_files={len(report['missing_files'])}")
    if report.get("extra_files"):
        msg_parts.append(f"extra_files={len(report['extra_files'])}")
    if report.get("hash_mismatches"):
        msg_parts.append(f"hash_mismatches={len(report['hash_mismatches'])}")
    raise RuntimeError(" ".join(msg_parts))
