"""Stage2 v4 recall-first detector.

Design goal: maximize proposal coverage, accept higher false positives.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np


V4_DEFAULTS = {
    "min_area": 2600,
    "min_width": 72,
    "min_height": 44,
    "max_area_ratio": 0.52,
    "column_gap_density_threshold": 0.02,
    "column_gap_min_width": 8,
    "column_min_width_ratio": 0.07,
    "row_blank_density_ratio": 0.004,
    "row_gap_min_height": 8,
    "row_min_height": 40,
    "segment_min_width": 70,
    "segment_expand_px": 8,
    "combo_span_max": 3,
    "dense_cc_min_area": 420,
    "dense_cc_max_area_ratio": 0.24,
    "min_fill_density": 0.002,
    "max_fill_density": 0.985,
    "dedup_iou": 0.985,
    "max_boxes": 180,
    "header_exclude_ratio": 0.0,
    "footer_exclude_ratio": 0.0,
}

V4_PARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "detection_params_v4.json"


def _as_int(v, default):
    try:
        return int(v)
    except Exception:
        return int(default)


def _as_float(v, default):
    try:
        return float(v)
    except Exception:
        return float(default)


def _load_params_once() -> dict:
    params = dict(V4_DEFAULTS)
    if V4_PARAMS_PATH.exists():
        try:
            loaded = json.loads(V4_PARAMS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key in params:
                    if key in loaded:
                        params[key] = loaded[key]
        except Exception:
            pass

    for key in list(params.keys()):
        env_key = f"DETECTOR_V4_{key.upper()}"
        if env_key in os.environ:
            params[key] = os.environ[env_key]

    int_keys = [
        "min_area",
        "min_width",
        "min_height",
        "column_gap_min_width",
        "row_gap_min_height",
        "row_min_height",
        "segment_min_width",
        "segment_expand_px",
        "combo_span_max",
        "dense_cc_min_area",
        "max_boxes",
    ]
    for key in int_keys:
        params[key] = _as_int(params[key], V4_DEFAULTS[key])

    float_keys = [
        "max_area_ratio",
        "column_gap_density_threshold",
        "column_min_width_ratio",
        "row_blank_density_ratio",
        "dense_cc_max_area_ratio",
        "min_fill_density",
        "max_fill_density",
        "dedup_iou",
        "header_exclude_ratio",
        "footer_exclude_ratio",
    ]
    for key in float_keys:
        params[key] = _as_float(params[key], V4_DEFAULTS[key])

    params["min_area"] = max(400, params["min_area"])
    params["min_width"] = max(28, params["min_width"])
    params["min_height"] = max(20, params["min_height"])
    params["max_area_ratio"] = float(np.clip(params["max_area_ratio"], 0.12, 0.95))
    params["column_gap_density_threshold"] = float(np.clip(params["column_gap_density_threshold"], 0.001, 0.15))
    params["column_min_width_ratio"] = float(np.clip(params["column_min_width_ratio"], 0.03, 0.50))
    params["row_blank_density_ratio"] = float(np.clip(params["row_blank_density_ratio"], 0.0005, 0.08))
    params["dense_cc_max_area_ratio"] = float(np.clip(params["dense_cc_max_area_ratio"], 0.02, 0.9))
    params["min_fill_density"] = float(np.clip(params["min_fill_density"], 0.0, 0.2))
    params["max_fill_density"] = float(np.clip(params["max_fill_density"], 0.1, 1.0))
    params["dedup_iou"] = float(np.clip(params["dedup_iou"], 0.8, 0.9999))
    params["header_exclude_ratio"] = float(np.clip(params["header_exclude_ratio"], 0.0, 0.2))
    params["footer_exclude_ratio"] = float(np.clip(params["footer_exclude_ratio"], 0.0, 0.2))
    params["max_boxes"] = max(20, params["max_boxes"])
    params["combo_span_max"] = max(1, min(6, params["combo_span_max"]))
    return params


P = _load_params_once()


def _smooth_1d(arr: np.ndarray, ksize: int) -> np.ndarray:
    ksize = max(3, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    arr2 = arr.reshape(-1, 1).astype(np.float32)
    return cv2.GaussianBlur(arr2, (1, ksize), 0).reshape(-1)


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if bool(v) and start is None:
            start = i
        elif (not bool(v)) and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask)))
    return out


def _threshold(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_img = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        15,
    )
    return bin_img


def _detect_columns(bin_img: np.ndarray) -> list[tuple[int, int]]:
    h, w = bin_img.shape
    ink = (bin_img > 0).astype(np.uint8)
    col_density = ink.sum(axis=0).astype(np.float32) / max(1.0, float(h))
    col_density = _smooth_1d(col_density, max(11, int(w * 0.015)))

    gap_mask = col_density < P["column_gap_density_threshold"]
    gaps = _runs(gap_mask, P["column_gap_min_width"])

    min_col_w = max(int(w * P["column_min_width_ratio"]), 90)
    cols: list[tuple[int, int]] = []
    left = 0
    for g0, g1 in gaps:
        if g0 - left >= min_col_w:
            cols.append((left, g0))
        left = g1
    if w - left >= min_col_w:
        cols.append((left, w))

    if not cols:
        return [(0, w)]

    out = []
    for x0, x1 in cols:
        tx0 = max(0, x0 + 1)
        tx1 = min(w, x1 - 1)
        if tx1 - tx0 >= 40:
            out.append((tx0, tx1))
    return out or [(0, w)]


def _row_segments(col_bin: np.ndarray) -> list[tuple[int, int]]:
    h, w = col_bin.shape
    if h <= 0 or w <= 0:
        return []
    ink = (col_bin > 0).astype(np.uint8)
    row_density = ink.sum(axis=1).astype(np.float32) / max(1.0, float(w))
    row_density = _smooth_1d(row_density, max(9, int(h * 0.009)))

    blank_mask = row_density < P["row_blank_density_ratio"]
    gaps = _runs(blank_mask, P["row_gap_min_height"])

    segs: list[tuple[int, int]] = []
    y0 = 0
    for g0, g1 in gaps:
        if g0 - y0 >= P["row_min_height"]:
            segs.append((y0, g0))
        y0 = g1
    if h - y0 >= P["row_min_height"]:
        segs.append((y0, h))
    return segs


def _tighten_to_ink(bin_img: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    patch = bin_img[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    ys, xs = np.where(patch > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    ex = int(P["segment_expand_px"])
    nx1 = max(0, x1 + int(xs.min()) - ex)
    ny1 = max(0, y1 + int(ys.min()) - ex)
    nx2 = min(bin_img.shape[1], x1 + int(xs.max()) + 1 + ex)
    ny2 = min(bin_img.shape[0], y1 + int(ys.max()) + 1 + ex)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return nx1, ny1, nx2, ny2


def _split_row_projection(row_bin: np.ndarray, x0: int, y0: int) -> list[tuple[int, int, int, int]]:
    h, w = row_bin.shape
    if h <= 0 or w <= 0:
        return []
    col_density = (row_bin > 0).sum(axis=0).astype(np.float32) / max(1.0, float(h))
    active = col_density > max(P["row_blank_density_ratio"] * 1.8, 0.0025)
    runs = _runs(active, P["segment_min_width"])
    out = []
    for rx0, rx1 in runs:
        box = _tighten_to_ink(row_bin, (rx0, 0, rx1, h))
        if box is None:
            continue
        bx1, by1, bx2, by2 = box
        out.append((x0 + bx1, y0 + by1, x0 + bx2, y0 + by2))
    return out


def _dense_cc_boxes(bin_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = bin_img.shape
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
    merged = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel1, iterations=1)
    merged = cv2.dilate(merged, kernel2, iterations=1)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    out = []
    max_area = int(P["dense_cc_max_area_ratio"] * w * h)
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < P["dense_cc_min_area"]:
            continue
        if area > max_area:
            continue
        out.append((int(x), int(y), int(x + bw), int(y + bh)))
    return out


def _box_area(b: tuple[int, int, int, int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = _box_area(a) + _box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def _valid_box(bin_img: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    h, w = bin_img.shape
    if x2 <= x1 or y2 <= y1:
        return False
    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh
    if bw < P["min_width"] or bh < P["min_height"]:
        return False
    if area < P["min_area"]:
        return False
    if area > int(P["max_area_ratio"] * w * h):
        return False

    top_y = int(P["header_exclude_ratio"] * h)
    bottom_y = int((1.0 - P["footer_exclude_ratio"]) * h)
    if y2 <= top_y or y1 >= bottom_y:
        return False

    patch = bin_img[y1:y2, x1:x2]
    fill = float(np.count_nonzero(patch)) / float(max(1, patch.size))
    if fill < P["min_fill_density"] or fill > P["max_fill_density"]:
        return False
    return True


def _score(bin_img: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    h, w = bin_img.shape
    area = _box_area(box)
    patch = bin_img[y1:y2, x1:x2]
    fill = float(np.count_nonzero(patch)) / float(max(1, patch.size))
    area_frac = area / float(max(1, w * h))
    s = 0.2 + 0.45 * min(1.0, fill / 0.09) + 0.35 * min(1.0, area_frac / 0.12)
    return float(np.clip(s, 0.0, 1.0))


def _dedup_keep_overlaps(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return boxes
    out: list[tuple[int, int, int, int]] = []
    for b in sorted(boxes, key=lambda x: (x[1], x[0], x[2] - x[0], x[3] - x[1])):
        duplicated = False
        for k in out:
            if _iou(b, k) >= P["dedup_iou"]:
                duplicated = True
                break
        if not duplicated:
            out.append(b)
    return out


def detect_connected_blocks_v4_recall(image_path, save_base_dir="data/job_blocks_smart", debug=True):
    image_path = Path(image_path)
    pdf_folder = image_path.parent.name
    base_name = image_path.stem

    save_dir = Path(save_base_dir) / pdf_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        # Rare transient read failures can happen under high parallel I/O.
        for _ in range(3):
            time.sleep(0.05)
            image = cv2.imread(str(image_path))
            if image is not None:
                break
    if image is None:
        print(f"[v4] [!] Failed to load image: {image_path}")
        return [], [], []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bin_img = _threshold(gray)

    h, w = bin_img.shape
    candidates: list[tuple[int, int, int, int]] = []

    # 1) Column + row segmentation proposals (high recall on classifieds layouts).
    columns = _detect_columns(bin_img)
    for cx0, cx1 in columns:
        col_bin = bin_img[:, cx0:cx1]
        segs = _row_segments(col_bin)

        for idx, (ry0, ry1) in enumerate(segs):
            tight = _tighten_to_ink(col_bin, (0, ry0, cx1 - cx0, ry1))
            if tight is not None:
                tx1, ty1, tx2, ty2 = tight
                candidates.append((cx0 + tx1, ty1, cx0 + tx2, ty2))

            row_patch = col_bin[ry0:ry1, :]
            candidates.extend(_split_row_projection(row_patch, cx0, ry0))

            for span in range(2, int(P["combo_span_max"]) + 1):
                if idx + span > len(segs):
                    continue
                cy0 = segs[idx][0]
                cy1 = segs[idx + span - 1][1]
                tight2 = _tighten_to_ink(col_bin, (0, cy0, cx1 - cx0, cy1))
                if tight2 is not None:
                    tx1, ty1, tx2, ty2 = tight2
                    candidates.append((cx0 + tx1, ty1, cx0 + tx2, ty2))

    # 2) Dense connected-component regions for coarse grouping recall.
    for box in _dense_cc_boxes(bin_img):
        tight = _tighten_to_ink(bin_img, box)
        if tight is not None:
            candidates.append(tight)

    # 3) Keep overlaps/nesting, only remove near-identical duplicates.
    filtered = [b for b in candidates if _valid_box(bin_img, b)]
    filtered = _dedup_keep_overlaps(filtered)

    if len(filtered) > int(P["max_boxes"]):
        ranked = sorted(filtered, key=lambda b: _score(bin_img, b), reverse=True)
        filtered = ranked[: int(P["max_boxes"])]

    filtered = sorted(filtered, key=lambda b: (b[1], b[0], b[3] - b[1], b[2] - b[0]))

    blocks_xywh: list[tuple[int, int, int, int]] = []
    cropped_paths: list[str] = []
    scores: list[float] = []

    for i, (x1, y1, x2, y2) in enumerate(filtered):
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        out_path = save_dir / f"{base_name}_block{i}.png"
        cv2.imwrite(str(out_path), crop)
        blocks_xywh.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        cropped_paths.append(str(out_path))
        scores.append(round(_score(bin_img, (x1, y1, x2, y2)), 4))

    if debug:
        dbg = image.copy()
        for idx, (x, y, bw, bh) in enumerate(blocks_xywh):
            cv2.rectangle(dbg, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            if idx < 100:
                cv2.putText(dbg, str(idx), (x + 2, max(10, y + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 220, 30), 1, cv2.LINE_AA)

        if "_p" in base_name:
            pnum = base_name.split("_p")[-1]
            debug_name = f"debug_p{pnum}.png"
        else:
            debug_name = f"{base_name}_debug.png"

        cv2.imwrite(str(save_dir / debug_name), dbg)

    print(f"[v4] [✓] Detected {len(blocks_xywh)} recall-first proposals for {base_name}")
    return blocks_xywh, cropped_paths, scores


if __name__ == "__main__":
    import sys

    img = sys.argv[1] if len(sys.argv) > 1 else "data/pdf2img/sample/page1.png"
    detect_connected_blocks_v4_recall(img)
