"""Stage2 v3 detector: Ultralytics YOLO-based full-page job-ad block detector."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - import failure handled at runtime
    YOLO = None


DETECTION_V3_PARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "detection_params_v3.json"
DETECTION_V3_PARAM_DEFAULTS = {
    "model_path": "artifacts/stage2_yolo_v3/best.pt",
    "confidence_threshold": 0.2,
    "iou_threshold": 0.5,
    "max_detections": 300,
    "imgsz": 1280,
    "device": "cpu",
}

_MODEL_CACHE: dict[str, Any] = {}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_str(value: Any, default: str) -> str:
    if value is None:
        return str(default)
    return str(value)


def _load_detection_params_v3() -> dict[str, Any]:
    params = dict(DETECTION_V3_PARAM_DEFAULTS)
    if DETECTION_V3_PARAMS_PATH.exists():
        try:
            loaded = json.loads(DETECTION_V3_PARAMS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                params.update(loaded)
        except Exception:
            pass

    # Env overrides are intentionally explicit for reproducible benchmark runs.
    params["model_path"] = _as_str(os.getenv("DETECTOR_V3_MODEL_PATH", params["model_path"]), params["model_path"])
    params["confidence_threshold"] = _as_float(
        os.getenv("DETECTOR_V3_CONFIDENCE_THRESHOLD", params["confidence_threshold"]),
        float(params["confidence_threshold"]),
    )
    params["iou_threshold"] = _as_float(
        os.getenv("DETECTOR_V3_IOU_THRESHOLD", params["iou_threshold"]),
        float(params["iou_threshold"]),
    )
    params["max_detections"] = _as_int(
        os.getenv("DETECTOR_V3_MAX_DETECTIONS", params["max_detections"]),
        int(params["max_detections"]),
    )
    params["imgsz"] = _as_int(os.getenv("DETECTOR_V3_IMGSZ", params["imgsz"]), int(params["imgsz"]))
    params["device"] = _as_str(os.getenv("DETECTOR_V3_DEVICE", params["device"]), str(params["device"]))

    params["confidence_threshold"] = min(1.0, max(0.0, float(params["confidence_threshold"])))
    params["iou_threshold"] = min(1.0, max(0.0, float(params["iou_threshold"])))
    params["max_detections"] = max(1, int(params["max_detections"]))
    params["imgsz"] = max(64, int(params["imgsz"]))
    return params


def _get_model(model_path: str):
    if YOLO is None:
        raise RuntimeError(
            "ultralytics is not installed. Install it before running Stage2 v3 (e.g. pip install ultralytics)."
        )

    resolved = str(Path(model_path).resolve())
    if resolved not in _MODEL_CACHE:
        if not Path(resolved).exists():
            raise FileNotFoundError(f"YOLO model checkpoint not found: {resolved}")
        _MODEL_CACHE[resolved] = YOLO(resolved)
    return _MODEL_CACHE[resolved]


def _normalize_box(x1: float, y1: float, x2: float, y2: float, image_w: int, image_h: int) -> tuple[int, int, int, int] | None:
    ix1 = max(0, min(image_w - 1, int(round(x1))))
    iy1 = max(0, min(image_h - 1, int(round(y1))))
    ix2 = max(0, min(image_w, int(round(x2))))
    iy2 = max(0, min(image_h, int(round(y2))))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def detect_connected_blocks_v3_yolo(
    image_path: str,
    save_base_dir: str = "data/job_blocks_smart",
    debug: bool = True,
    *,
    model_path: str | None = None,
    conf_threshold: float | None = None,
    iou_threshold: float | None = None,
    max_det: int | None = None,
    imgsz: int | None = None,
    device: str | None = None,
):
    """Run YOLO detection on one full-page image and emit Stage2-style block artifacts."""
    image_path = Path(image_path)
    pdf_folder = image_path.parent.name
    base_name = image_path.stem

    save_dir = Path(save_base_dir) / pdf_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[v3] [!] Failed to load image: {image_path}")
        return [], [], []

    image_h, image_w = image.shape[:2]
    params = _load_detection_params_v3()
    used_model_path = str(model_path or params["model_path"])
    used_conf = float(conf_threshold if conf_threshold is not None else params["confidence_threshold"])
    used_iou = float(iou_threshold if iou_threshold is not None else params["iou_threshold"])
    used_max_det = int(max_det if max_det is not None else params["max_detections"])
    used_imgsz = int(imgsz if imgsz is not None else params["imgsz"])
    used_device = str(device or params["device"])

    model = _get_model(used_model_path)
    results = model.predict(
        source=str(image_path),
        conf=used_conf,
        iou=used_iou,
        max_det=used_max_det,
        imgsz=used_imgsz,
        device=used_device,
        verbose=False,
    )

    raw: list[tuple[int, int, int, int, float]] = []
    if results:
        pred = results[0]
        boxes = getattr(pred, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy().tolist()
            confs = boxes.conf.detach().cpu().numpy().tolist() if getattr(boxes, "conf", None) is not None else []
            for idx, row in enumerate(xyxy):
                if not isinstance(row, (list, tuple)) or len(row) != 4:
                    continue
                score = float(confs[idx]) if idx < len(confs) else 0.0
                normalized = _normalize_box(row[0], row[1], row[2], row[3], image_w, image_h)
                if normalized is None:
                    continue
                x1, y1, x2, y2 = normalized
                raw.append((x1, y1, x2, y2, max(0.0, min(1.0, score))))

    # Deterministic order preserving Stage2 expectations for block_index semantics.
    raw.sort(key=lambda b: (b[1], b[0], -b[4]))

    blocks: list[tuple[int, int, int, int]] = []
    cropped_paths: list[str] = []
    scores: list[float] = []
    for idx, (x1, y1, x2, y2, score) in enumerate(raw):
        bw = int(x2 - x1)
        bh = int(y2 - y1)
        if bw <= 0 or bh <= 0:
            continue
        crop = image[y1:y2, x1:x2]
        out_path = save_dir / f"{base_name}_block{len(blocks)}.png"
        cv2.imwrite(str(out_path), crop)
        blocks.append((int(x1), int(y1), int(bw), int(bh)))
        scores.append(round(float(score), 4))
        cropped_paths.append(str(out_path))

    if debug:
        dbg = image.copy()
        for i, (x, y, bw, bh) in enumerate(blocks):
            cv2.rectangle(dbg, (x, y), (x + bw, y + bh), (64, 255, 64), 2)
            cv2.putText(
                dbg,
                f"{i}:{scores[i]:.2f}",
                (x, max(14, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if "_p" in base_name:
            pnum = base_name.split("_p")[-1]
            debug_name = f"debug_p{pnum}.png"
        else:
            debug_name = f"{base_name}_debug.png"
        debug_path = save_dir / debug_name
        cv2.imwrite(str(debug_path), dbg)
        print(f"[v3] Debug image saved: {debug_path}")

    print(f"[v3] Detected {len(blocks)} YOLO blocks from page {base_name}")
    return blocks, cropped_paths, scores


if __name__ == "__main__":
    import sys

    img = sys.argv[1] if len(sys.argv) > 1 else "data/pdf2img/sample/page1.png"
    detect_connected_blocks_v3_yolo(img)
