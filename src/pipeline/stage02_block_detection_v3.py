#!/usr/bin/env python3
"""
Stage02 v3 block detector runner (v3.1).
Fixed manifest path reconstruction and coordinate normalization.
"""

from __future__ import annotations

import argparse
import json
import logging
from src.utils.logging_utils import configure_logging
logger = configure_logging("stage02_block_detection_v3")
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.pipeline_config import get_path, load_config


log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_file = log_dir / f"smart_block_detector_v3_yolo_{timestamp}.log"

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

DETECTION_PARAMS_V3_PATH = Path("configs/detection_params_v3.json")
DETECTION_PARAM_V3_DEFAULTS = {
    "model_path": "artifacts/stage2_yolo_v3/best.pt",
    "confidence_threshold": 0.2,
    "iou_threshold": 0.5,
    "max_detections": 300,
    "imgsz": 1280,
    "device": "cpu",
}


_DETECTOR_FN = None


def _get_detector():
    global _DETECTOR_FN
    if _DETECTOR_FN is None:
        from src.vision.block_detector_v3_yolo import detect_connected_blocks_v3_yolo
        _DETECTOR_FN = detect_connected_blocks_v3_yolo
    return _DETECTOR_FN


def _load_detection_params_v3() -> dict[str, Any]:
    params = DETECTION_PARAM_V3_DEFAULTS.copy()
    if DETECTION_PARAMS_V3_PATH.exists():
        try:
            with open(DETECTION_PARAMS_V3_PATH) as f:
                params.update(json.load(f))
        except Exception as e:
            logging.warning(f"Failed to load v3 params from {DETECTION_PARAMS_V3_PATH}: {e}")

    params["confidence_threshold"] = float(params.get("confidence_threshold", 0.2))
    params["iou_threshold"] = float(params.get("iou_threshold", 0.5))
    params["max_detections"] = max(1, int(params.get("max_detections", 300)))
    return params


def _write_detector_metadata(page_name: str, detections: list[dict[str, Any]], detections_output_dir: Path) -> None:
    out_path = Path(detections_output_dir) / f"{page_name}.json"
    existing: dict[str, Any] = {"page": page_name, "detections": []}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
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


def detect_page_blocks_v3(
    img_path_str: str,
    blocks_output_dir: str,
    detector_params: dict[str, Any],
    debug: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    img_path = Path(img_path_str)
    pdf_folder = img_path.parent.name
    page_name = img_path.name
    base_name = img_path.stem

    debug_name = f"debug_p{base_name.split('_p')[-1]}.png" if "_p" in base_name else f"{base_name}_debug.png"
    debug_path = Path(blocks_output_dir) / pdf_folder / debug_name

    if not force and debug and debug_path.exists():
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
        blocks, block_paths, block_scores = detector(
            str(img_path),
            save_base_dir=str(blocks_output_dir),
            debug=debug,
            model_path=detector_params["model_path"],
            conf_threshold=detector_params["confidence_threshold"],
            iou_threshold=detector_params["iou_threshold"],
            max_det=detector_params["max_detections"],
            imgsz=detector_params["imgsz"],
            device=detector_params["device"],
        )

        detections = []
        for idx, (x, y, w, h) in enumerate(blocks):
            score = float(block_scores[idx]) if idx < len(block_scores) else 0.0
            detections.append(
                {
                    "id": f"detector_v3_{base_name}_{idx}",
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "score": round(max(0.0, min(1.0, score)), 4),
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
            "block_paths": block_paths,
            "error": None,
        }

    except Exception as e:
        logging.error(f"Failed to process {img_path}: {e}")
        return {
            "status": "failed",
            "pdf_folder": pdf_folder,
            "page_name": page_name,
            "detections": [],
            "block_paths": [],
            "error": str(e),
        }


def run_detector_v3(
    config_path: str = "configs/pipeline_paths.json",
    overrides: dict[str, Any] | None = None,
    force: bool = False,
) -> None:
    detector_params = _load_detection_params_v3()
    if overrides:
        detector_params.update({k: v for k, v in overrides.items() if v is not None})

    config = load_config(config_path)
    pdf_input_dir = get_path("pdf_input", config)
    images_output_dir = get_path("images_output", config)
    blocks_output_dir = get_path("blocks_output", config)
    detections_output_dir = get_path("detections_output", config)

    print("[CONFIG]")
    print(f"images_output = {images_output_dir}")
    print(f"blocks_output = {blocks_output_dir}")
    print(f"detections_output = {detections_output_dir}")
    print(f"v3_model_path = {detector_params['model_path']}")
    print(f"v3_confidence_threshold = {detector_params['confidence_threshold']}")

    detections_output_dir.mkdir(parents=True, exist_ok=True)

    all_images: list[str] = []
    folder_map: dict[str, int] = {}
    for folder in sorted(images_output_dir.iterdir()):
        if not folder.is_dir(): continue
        imgs = sorted(folder.glob("*.png"))
        if imgs:
            folder_map[folder.name] = len(imgs)
            all_images.extend(str(img) for img in imgs)

    if not all_images:
        print(f"[!] No images found under {images_output_dir}/*/")
        return

    start_time = time.time()
    counts: Counter[str] = Counter()
    print(f"🚀 [v3] Running YOLO detector on {len(all_images)} pages (correctness-first mode)...")

    for img_path in all_images:
        result = detect_page_blocks_v3(img_path, str(blocks_output_dir), detector_params, debug=True, force=force)
        status = str(result.get("status", "failed"))
        counts[status] += 1
        page_name = result.get("page_name", "unknown")

        if status == "processed":
            _write_detector_metadata(str(page_name), result.get("detections", []), detections_output_dir)
            logging.info(f"[v3] [✓] {page_name}: {len(result.get('detections', []))} block(s) detected")
        elif status == "skipped":
            logging.info(f"[v3] [⏩] Skipped {page_name}: already processed")
        else:
            logging.error(f"[v3] [✖] {page_name} failed: {result.get('error')}")

    # Final Manifest Consolidation
    import os
    import cv2
    from src.pipeline.pipeline_metadata import write_crop_manifest_jsonl
    all_crops = {}
    print(f"DEBUG: Consolidating from {detections_output_dir}")
    for pg_file in detections_output_dir.glob("*.json"):
        try:
            data = json.loads(pg_file.read_text())
            pg_name = data["page"]
            # Robustly find the subfolder
            pdf_name = pg_name.split("_p")[0]
            # Page might be "UHT Delhi 07-04_p1.png" -> folder is "UHT Delhi 07-04"
            img_path = str(images_output_dir / pdf_name / pg_name)
            
            if not os.path.exists(img_path):
                print(f"DEBUG: Image not found at {img_path}")
                continue

            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            for i, det in enumerate(data["detections"]):
                cid = det["id"]
                all_crops[cid] = {
                    "crop_id": cid,
                    "page_image_path": img_path,
                    "bbox_xyxy_norm": [det["bbox"][0]/w, det["bbox"][1]/h, det["bbox"][2]/w, det["bbox"][3]/h],
                    "status": "ok",
                    "pdf_path": str(pdf_input_dir / f"{pdf_name}.pdf"),
                    "page_index0": int(pg_name.split("_p")[-1].replace(".png","")) - 1,
                    "newspaper_name": pdf_name
                }
        except Exception as e:
            print(f"Error on {pg_file}: {e}")
    
    print(f"DEBUG: Consolidated {len(all_crops)} crops.")
    with open("run_state/crop_manifest.jsonl", "w") as f:
        for c in all_crops.values():
            f.write(json.dumps(c) + "\n")

    total_time = time.time() - start_time
    print("\n========== V3 SUMMARY ==========")
    print(f"🏁 Total time: {total_time:.2f}s")
    print(f"📄 Total PDFs processed: {len(folder_map)}")
    print(f"🧾 Total pages scanned: {len(all_images)}")
    print(f"✅ Processed: {counts['processed']}")
    print(f"⏩ Skipped: {counts['skipped']}")
    print(f"❌ Failed: {counts['failed']}")
    print("================================")
    print(f"[✓] Log saved to {log_file}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage02 block detection v3 (YOLO) runner")
    parser.add_argument("--config", default="configs/pipeline_paths.json", help="Path to pipeline paths config")
    parser.add_argument("--model-path", default=None, help="Override YOLO model checkpoint path")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="YOLO confidence threshold override")
    parser.add_argument("--iou-threshold", type=float, default=None, help="YOLO IoU NMS threshold override")
    parser.add_argument("--max-detections", type=int, default=None, help="Maximum detections per image")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference image size")
    parser.add_argument("--device", default=None, help="YOLO device string, e.g. cpu or 0")
    parser.add_argument("--force", action="store_true", help="Force processing")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_args()
    run_detector_v3(
        config_path=cli_args.config,
        overrides={
            "model_path": cli_args.model_path,
            "confidence_threshold": cli_args.confidence_threshold,
            "iou_threshold": cli_args.iou_threshold,
            "max_detections": cli_args.max_detections,
            "imgsz": cli_args.imgsz,
            "device": cli_args.device,
        },
        force=cli_args.force,
    )
