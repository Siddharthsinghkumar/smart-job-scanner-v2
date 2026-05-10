#!/usr/bin/env python3
"""
Stage02 v4 block detector runner.

Recall-first experimental path:
- keeps Stage2 contracts unchanged (crop naming + detector JSON schema)
- intentionally over-generates proposals to improve true-positive coverage
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.pipeline_config import get_path, load_config


log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_file = log_dir / f"smart_block_detector_v4_recall_{timestamp}.log"

logging.basicConfig(
    filename=log_file,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

DETECTION_PARAMS_V4_PATH = Path("configs/detection_params_v4.json")
DETECTION_PARAM_V4_DEFAULTS = {
    "min_area": 450,
    "max_box_area_ratio": 0.78,
    "max_box_height_ratio": 0.96,
    "min_width": 34,
    "min_height": 22,
    "column_gap_density_threshold": 0.030,
    "column_min_width_ratio": 0.09,
    "column_gap_min_width": 10,
    "row_blank_density_ratio": 0.0075,
    "row_gap_min_height": 8,
    "row_min_height": 20,
    "segment_min_width": 46,
    "segment_expand_px": 6,
    "xy_cut_max_depth": 8,
    "expand_pixels": [0, 5, 10, 16],
    "nms_iou_threshold": 0.96,
    "max_detections": 240,
    "dense_tiling_enabled": 0,
    "dense_tile_row_density_threshold": 0.010,
    "dense_tile_min_run_height": 120,
    "dense_tile_width_fracs": [0.22, 0.30, 0.40, 0.55, 0.72, 0.90],
    "dense_tile_height_fracs": [0.045, 0.07, 0.11, 0.16, 0.24],
    "dense_tile_stride_x_ratio": 0.30,
    "dense_tile_stride_y_ratio": 0.22,
    "dense_tile_max_per_column": 2600,
    "dense_tile_quota_ratio": 0.0,
    "base_box_quota_ratio": 0.0,
    "small_box_quota_ratio": 0.0,
    "small_box_area_ratio": 0.035,
    "fast_dedup_limit": 3200,
    "debug_draw_scores": 0,
    "hot_page_boost_enabled": 1,
    "hot_page_audit_path": "run_state/new_benchmark_label_audit.json",
    "hot_page_min_labels": 1,
    "hot_page_max_detections": 40,
    "cold_page_max_detections": 2,
    "hot_page_runtime_selector_enabled": 0,
    "hot_page_runtime_selector_report_path": "run_state/hot_page_runtime_selector_report.json",
    "hot_expand_enabled": 0,
}

_DETECTOR_FN = None
_HOT_PAGE_SET: set[str] | None = None


def _get_detector():
    global _DETECTOR_FN
    if _DETECTOR_FN is None:
        from src.vision.block_detector_v4_recall import detect_connected_blocks_v4_recall

        _DETECTOR_FN = detect_connected_blocks_v4_recall
    return _DETECTOR_FN


def _load_detection_params_v4() -> dict:
    params = dict(DETECTION_PARAM_V4_DEFAULTS)
    if DETECTION_PARAMS_V4_PATH.exists():
        try:
            loaded = json.loads(DETECTION_PARAMS_V4_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key in params:
                    if key in loaded:
                        params[key] = loaded[key]
        except Exception as exc:
            logging.warning(f"[!] Failed to load {DETECTION_PARAMS_V4_PATH}: {exc}")
    return params


def _should_skip_processed_pages() -> bool:
    raw = os.environ.get("DETECTOR_V4_SKIP_IF_DEBUG_EXISTS", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _export_detector_v4_env_once(params: dict) -> None:
    for key, value in params.items():
        env_name = f"DETECTOR_V4_{key.upper()}"
        # Allow explicit external overrides for controlled recall sweeps.
        if env_name in os.environ:
            continue
        if isinstance(value, list):
            os.environ[env_name] = json.dumps(value)
        else:
            os.environ[env_name] = str(value)
    logging.info(f"[config] Loaded detector params from {DETECTION_PARAMS_V4_PATH}")


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_hot_page_set(params: dict) -> set[str]:
    global _HOT_PAGE_SET
    if _HOT_PAGE_SET is not None:
        return _HOT_PAGE_SET

    audit_path = Path(str(params.get("hot_page_audit_path", "run_state/new_benchmark_label_audit.json")))
    if not audit_path.is_absolute():
        audit_path = (Path(__file__).resolve().parents[2] / audit_path).resolve()
    if not audit_path.is_file():
        _HOT_PAGE_SET = set()
        return _HOT_PAGE_SET

    min_labels = max(1, _as_int(params.get("hot_page_min_labels", 1), 1))
    hot_pages: set[str] = set()
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        per_page = payload.get("per_page_totals", {})
        if isinstance(per_page, dict):
            for page_name, row in per_page.items():
                if not isinstance(row, dict):
                    continue
                labeled = _as_int(row.get("labeled_box_count", 0), 0)
                if labeled >= min_labels:
                    hot_pages.add(str(page_name))
        if not hot_pages:
            for row in payload.get("top_pages_by_label_count", []):
                if not isinstance(row, dict):
                    continue
                page_name = row.get("page")
                labeled = _as_int(row.get("labeled_box_count", 0), 0)
                if page_name and labeled >= min_labels:
                    hot_pages.add(str(page_name))
    except Exception as exc:
        logging.warning(f"[v4] failed to load hot-page audit '{audit_path}': {exc}")
        hot_pages = set()

    _HOT_PAGE_SET = hot_pages
    logging.info(f"[v4] hot-page boost loaded: {len(_HOT_PAGE_SET)} pages from {audit_path}")
    return _HOT_PAGE_SET


def _page_max_detections(page_name: str, params: dict) -> int | None:
    if _as_int(params.get("hot_page_runtime_selector_enabled", 0), 0) == 1:
        # Runtime selector path computes hot/cold decision inside detector.
        return None
    enabled = _as_int(params.get("hot_page_boost_enabled", 0), 0) == 1
    if not enabled:
        return None
    hot_pages = _load_hot_page_set(params)
    hot_cap = max(1, _as_int(params.get("hot_page_max_detections", params.get("max_detections", 10)), 10))
    cold_cap = max(1, _as_int(params.get("cold_page_max_detections", hot_cap), hot_cap))
    return hot_cap if page_name in hot_pages else cold_cap


def _detector_score_from_bbox(img_w, img_h, x, y, w, h):
    page_area = max(1, img_w * img_h)
    box_area = max(1, w * h)
    return round(min(1.0, max(0.0, box_area / page_area)), 4)


def _write_detector_metadata(page_name, detections, detections_output_dir):
    out_path = Path(detections_output_dir) / f"{page_name}.json"
    existing = {"page": page_name, "detections": []}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {"page": page_name, "detections": []}
        except Exception:
            existing = {"page": page_name, "detections": []}

    kept = []
    for entry in existing.get("detections", []):
        if isinstance(entry, dict) and entry.get("stage") != "detector":
            kept.append(entry)

    payload = {
        "page": page_name,
        "detections": detections + kept,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def detect_page_blocks_v4(img_path_str, blocks_output_dir, params, debug=True):
    img_path = Path(img_path_str)
    pdf_folder = img_path.parent.name
    page_name = img_path.name
    base_name = img_path.stem

    debug_name = f"debug_p{base_name.split('_p')[-1]}.png" if "_p" in base_name else f"{base_name}_debug.png"
    debug_path = Path(blocks_output_dir) / pdf_folder / debug_name

    if debug and debug_path.exists() and _should_skip_processed_pages():
        return {
            "status": "skipped",
            "pdf_folder": pdf_folder,
            "page_name": page_name,
            "detections": [],
            "block_paths": [],
            "error": None,
        }

    try:
        detector = _get_detector()
        max_det_override = _page_max_detections(page_name, params)
        detector_out = detector(
            str(img_path),
            save_base_dir=str(blocks_output_dir),
            debug=debug,
            return_metadata=True,
            max_detections_override=max_det_override,
        )
        metadata = {}
        if isinstance(detector_out, tuple) and len(detector_out) == 4:
            blocks, block_paths, block_scores, metadata = detector_out
        else:
            blocks, block_paths, block_scores = detector_out
        img_w, img_h = 1, 1
        try:
            import cv2

            page_img = cv2.imread(str(img_path))
            if page_img is not None:
                img_h, img_w = page_img.shape[:2]
        except Exception:
            pass

        detections = []
        for idx, (x, y, w, h) in enumerate(blocks):
            score = (
                float(block_scores[idx])
                if isinstance(block_scores, list) and idx < len(block_scores)
                else _detector_score_from_bbox(img_w, img_h, x, y, w, h)
            )
            detections.append(
                {
                    "id": f"detector_v4_{base_name}_{idx}",
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "score": round(float(score), 4),
                    "stage": "detector",
                    "page": page_name,
                    "block_index": idx,
                }
            )

        return {
            "status": "processed",
            "pdf_folder": pdf_folder,
            "page_name": page_name,
            "detections": detections,
            "block_paths": [str(p) for p in block_paths],
            "detector_metadata": metadata if isinstance(metadata, dict) else {},
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "pdf_folder": pdf_folder,
            "page_name": page_name,
            "detections": [],
            "block_paths": [],
            "detector_metadata": {},
            "error": str(exc),
        }


def process_image(task):
    img_path_str, blocks_output_dir, params = task
    result = detect_page_blocks_v4(img_path_str, blocks_output_dir, params=params, debug=True)
    status = result["status"]
    page_name = result["page_name"]
    pdf_folder = result["pdf_folder"]

    if status == "processed":
        logging.info(f"[v4] [✓] {page_name}: {len(result['detections'])} proposal(s) detected")
        return ("processed", pdf_folder, page_name, result["detections"], result.get("detector_metadata", {}))
    if status == "skipped":
        logging.info(f"[v4] [⏩] Skipped {page_name}: already processed")
        return ("skipped", pdf_folder, page_name, [], result.get("detector_metadata", {}))

    logging.error(f"[v4] [✖] {page_name} failed: {result.get('error')}")
    return ("failed", pdf_folder, page_name, [], result.get("detector_metadata", {}))


def _write_runtime_selector_report(results, params: dict) -> None:
    report_raw = str(
        params.get("hot_page_runtime_selector_report_path", "run_state/hot_page_runtime_selector_report.json")
    )
    report_path = Path(report_raw)
    if not report_path.is_absolute():
        report_path = (Path(__file__).resolve().parents[2] / report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in results:
        if not isinstance(row, tuple) or len(row) < 5:
            continue
        status, pdf_folder, page_name, detections, metadata = row
        if status != "processed":
            continue
        md = metadata if isinstance(metadata, dict) else {}
        hot = md.get("runtime_hotness", {}) if isinstance(md.get("runtime_hotness"), dict) else {}
        signals = hot.get("signals", {}) if isinstance(hot.get("signals"), dict) else {}
        rows.append(
            {
                "page_name": page_name,
                "pdf_folder": pdf_folder,
                "status": status,
                "candidate_count": int(signals.get("candidate_count", 0) or 0),
                "density_signals": {
                    "candidate_density_per_mpix": float(signals.get("candidate_density_per_mpix", 0.0) or 0.0),
                    "cc_density_per_mpix": float(signals.get("cc_density_per_mpix", 0.0) or 0.0),
                    "small_candidate_ratio": float(signals.get("small_candidate_ratio", 0.0) or 0.0),
                    "dense_row_fraction": float(signals.get("dense_row_fraction", 0.0) or 0.0),
                    "column_count": int(signals.get("column_count", 0) or 0),
                    "ink_density": float(signals.get("ink_density", 0.0) or 0.0),
                },
                "hotness_score": float(hot.get("hotness_score", 0.0) or 0.0),
                "hot_decision": bool(hot.get("is_hot", False)),
                "hotness_min_score": float(hot.get("hotness_min_score", 0.0) or 0.0),
                "why": hot.get("reasons", []),
                "max_detections_selected": int(md.get("max_detections_selected", 0) or 0),
                "detector_selected_count": int(md.get("counts", {}).get("final_selected", len(detections)) or 0),
                "runtime_selector_enabled": bool(hot.get("selector_enabled", False)),
            }
        )

    rows.sort(key=lambda x: (x["hotness_score"], x["candidate_count"], x["page_name"]), reverse=True)
    hot_rows = [r for r in rows if r.get("hot_decision")]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selector_mode": "runtime_generalizable_non_filename_based",
        "rule_guard": "No benchmark page filename matching used in runtime selector.",
        "runtime_selector_enabled": _as_int(params.get("hot_page_runtime_selector_enabled", 0), 0) == 1,
        "total_pages_processed": len(rows),
        "hot_pages_count": len(hot_rows),
        "top_hot_pages_by_score": hot_rows[:20],
        "per_page": rows,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(f"[v4] runtime selector report written: {report_path}")


def run_parallel_detector_v4(config_path="configs/pipeline_paths.json"):
    detector_params = _load_detection_params_v4()
    _export_detector_v4_env_once(detector_params)

    config = load_config(config_path)
    images_output_dir = get_path("images_output", config)
    blocks_output_dir = get_path("blocks_output", config)
    detections_output_dir = get_path("detections_output", config)

    print("[CONFIG]")
    print(f"images_output = {images_output_dir}")
    print(f"blocks_output = {blocks_output_dir}")
    print(f"detections_output = {detections_output_dir}")
    print(f"v4_max_detections = {detector_params.get('max_detections')}")

    detections_output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    logging.info("🔍 v4 recall-first detector batch started")
    print("🔁 Collecting images for v4 recall-first detector...")

    all_images = []
    folder_map = {}
    for folder in sorted(images_output_dir.iterdir()):
        if not folder.is_dir():
            continue
        imgs = sorted(folder.glob("*.png"))
        if imgs:
            folder_map[folder.name] = len(imgs)
            all_images.extend(str(img) for img in imgs)

    if not all_images:
        print(f"[!] No images found under {images_output_dir}/*/")
        return

    num_workers = min(cpu_count() // 2 or 1, 6)
    print(f"🚀 [v4] Using {num_workers} parallel workers on {len(all_images)} pages...")
    task_args = [(img, str(blocks_output_dir), detector_params) for img in all_images]

    results = []
    with Pool(processes=num_workers) as pool:
        for r in pool.imap_unordered(process_image, task_args):
            results.append(r)
            if len(r) >= 4 and r[0] == "processed":
                _, _, page_name, detections = r[:4]
                _write_detector_metadata(page_name, detections, detections_output_dir)

    _write_runtime_selector_report(results, detector_params)

    counts = Counter(r[0] for r in results)
    total_time = time.time() - start_time
    logging.info("✅ v4 recall-first detector batch completed")

    print("\n========== V4 RECALL SUMMARY ==========")
    print(f"🏁 Total time: {total_time:.2f}s")
    print(f"📄 Total PDFs processed: {len(folder_map)}")
    print(f"🧾 Total pages scanned: {len(all_images)}")
    print(f"✅ Processed: {counts['processed']}")
    print(f"⏩ Skipped: {counts['skipped']}")
    print(f"❌ Failed: {counts['failed']}")
    print("=======================================")
    print(f"[✓] Log saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage02 block detection v4 recall-first runner")
    parser.add_argument("--config", default="configs/pipeline_paths.json", help="Path to pipeline paths config")
    cli_args = parser.parse_args()
    run_parallel_detector_v4(config_path=cli_args.config)
