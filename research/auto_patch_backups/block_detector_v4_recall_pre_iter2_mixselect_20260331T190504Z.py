"""Stage2 v4 detector: recall-first proposal generator for job-ad regions."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np


DETECTION_V4_DEFAULTS = {
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
    "debug_draw_scores": 0,
}

DETECTION_V4_PARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "detection_params_v4.json"


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


def _as_int_list(v, default):
    if isinstance(v, list):
        out = []
        for item in v:
            try:
                out.append(int(item))
            except Exception:
                continue
        if out:
            return out
    return list(default)


def _load_params_once() -> dict:
    params = dict(DETECTION_V4_DEFAULTS)
    if DETECTION_V4_PARAMS_PATH.exists():
        try:
            loaded = json.loads(DETECTION_V4_PARAMS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key in params:
                    if key in loaded:
                        params[key] = loaded[key]
        except Exception:
            pass

    for key in list(params.keys()):
        env_name = f"DETECTOR_V4_{key.upper()}"
        if env_name in os.environ:
            params[key] = os.environ[env_name]

    int_keys = [
        "min_area",
        "min_width",
        "min_height",
        "column_gap_min_width",
        "row_gap_min_height",
        "row_min_height",
        "segment_min_width",
        "segment_expand_px",
        "xy_cut_max_depth",
        "max_detections",
        "debug_draw_scores",
    ]
    for key in int_keys:
        params[key] = _as_int(params[key], DETECTION_V4_DEFAULTS[key])

    float_keys = [
        "max_box_area_ratio",
        "max_box_height_ratio",
        "column_gap_density_threshold",
        "column_min_width_ratio",
        "row_blank_density_ratio",
        "nms_iou_threshold",
    ]
    for key in float_keys:
        params[key] = _as_float(params[key], DETECTION_V4_DEFAULTS[key])

    params["expand_pixels"] = _as_int_list(params.get("expand_pixels"), DETECTION_V4_DEFAULTS["expand_pixels"])
    params["min_area"] = max(100, params["min_area"])
    params["min_width"] = max(20, params["min_width"])
    params["min_height"] = max(12, params["min_height"])
    params["column_gap_min_width"] = max(4, params["column_gap_min_width"])
    params["row_gap_min_height"] = max(4, params["row_gap_min_height"])
    params["row_min_height"] = max(10, params["row_min_height"])
    params["segment_min_width"] = max(24, params["segment_min_width"])
    params["segment_expand_px"] = max(0, params["segment_expand_px"])
    params["xy_cut_max_depth"] = max(2, params["xy_cut_max_depth"])
    params["max_detections"] = max(10, params["max_detections"])
    params["max_box_area_ratio"] = float(np.clip(params["max_box_area_ratio"], 0.1, 0.95))
    params["max_box_height_ratio"] = float(np.clip(params["max_box_height_ratio"], 0.2, 1.0))
    params["column_gap_density_threshold"] = float(np.clip(params["column_gap_density_threshold"], 0.003, 0.20))
    params["column_min_width_ratio"] = float(np.clip(params["column_min_width_ratio"], 0.05, 0.6))
    params["row_blank_density_ratio"] = float(np.clip(params["row_blank_density_ratio"], 0.001, 0.08))
    params["nms_iou_threshold"] = float(np.clip(params["nms_iou_threshold"], 0.80, 0.99))
    params["expand_pixels"] = sorted(set(max(0, int(x)) for x in params["expand_pixels"]))
    params["debug_draw_scores"] = 1 if params["debug_draw_scores"] else 0
    return params


P = _load_params_once()


def _smooth_1d(arr: np.ndarray, ksize: int) -> np.ndarray:
    ksize = max(3, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    arr2 = arr.reshape(-1, 1).astype(np.float32)
    sm = cv2.GaussianBlur(arr2, (1, ksize), 0).reshape(-1)
    return sm


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    out = []
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


def _detect_columns(bin_img: np.ndarray) -> list[tuple[int, int]]:
    h, w = bin_img.shape
    ink = (bin_img > 0).astype(np.uint8)
    col_density = ink.sum(axis=0).astype(np.float32) / max(1.0, float(h))
    col_density = _smooth_1d(col_density, max(9, int(w * 0.016)))

    gap_mask = col_density < P["column_gap_density_threshold"]
    gap_runs = _runs(gap_mask, P["column_gap_min_width"])
    min_col_w = max(int(w * P["column_min_width_ratio"]), 70)

    cols: list[tuple[int, int]] = []
    left = 0
    for g0, g1 in gap_runs:
        if g0 - left >= min_col_w:
            cols.append((left, g0))
        left = g1
    if w - left >= min_col_w:
        cols.append((left, w))
    return cols or [(0, w)]


def _row_segments(col_bin: np.ndarray) -> list[tuple[int, int]]:
    h, w = col_bin.shape
    if h == 0 or w == 0:
        return []
    ink = (col_bin > 0).astype(np.uint8)
    row_density = ink.sum(axis=1).astype(np.float32) / max(1.0, float(w))
    row_density = _smooth_1d(row_density, max(9, int(h * 0.008)))

    blank_mask = row_density < P["row_blank_density_ratio"]
    gaps = _runs(blank_mask, P["row_gap_min_height"])
    min_h = P["row_min_height"]
    segs: list[tuple[int, int]] = []
    y0 = 0
    for g0, g1 in gaps:
        if g0 - y0 >= min_h:
            segs.append((y0, g0))
        y0 = g1
    if h - y0 >= min_h:
        segs.append((y0, h))
    return segs


def _split_row_into_boxes(row_bin: np.ndarray, x0: int, y0: int) -> list[tuple[int, int, int, int]]:
    h, w = row_bin.shape
    if h <= 0 or w <= 0:
        return []

    col_density = (row_bin > 0).sum(axis=0).astype(np.float32) / max(1.0, float(h))
    active = col_density > max(P["row_blank_density_ratio"] * 1.4, 0.004)
    runs = _runs(active, P["segment_min_width"])
    if not runs:
        runs = [(0, w)]

    boxes: list[tuple[int, int, int, int]] = []
    for rx0, rx1 in runs:
        part = row_bin[:, rx0:rx1]
        ys, xs = np.where(part > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        px0 = max(0, int(rx0 + int(xs.min()) - P["segment_expand_px"]))
        px1 = min(w, int(rx0 + int(xs.max()) + 1 + P["segment_expand_px"]))
        py0 = max(0, int(ys.min() - P["segment_expand_px"]))
        py1 = min(h, int(ys.max()) + 1 + P["segment_expand_px"])
        boxes.append((x0 + px0, y0 + py0, x0 + px1, y0 + py1))
    return boxes


def _xy_cut_regions(
    bin_img: np.ndarray,
    x0: int,
    y0: int,
    depth: int = 0,
    max_depth: int = 8,
) -> list[tuple[int, int, int, int]]:
    h, w = bin_img.shape
    if h < max(2 * P["row_min_height"], 28) or w < max(2 * P["segment_min_width"], 64):
        return [(x0, y0, x0 + w, y0 + h)]
    if depth >= max_depth:
        return [(x0, y0, x0 + w, y0 + h)]

    ink = (bin_img > 0).astype(np.uint8)
    row_density = ink.sum(axis=1).astype(np.float32) / max(1.0, float(w))
    col_density = ink.sum(axis=0).astype(np.float32) / max(1.0, float(h))

    row_blank = row_density < max(P["row_blank_density_ratio"] * 0.7, 0.0013)
    col_blank = col_density < max(P["column_gap_density_threshold"] * 0.8, 0.0018)
    row_gaps = _runs(row_blank, max(P["row_gap_min_height"], 4))
    col_gaps = _runs(col_blank, max(P["column_gap_min_width"], 5))

    best_row = max(row_gaps, key=lambda g: g[1] - g[0], default=None)
    best_col = max(col_gaps, key=lambda g: g[1] - g[0], default=None)
    row_len = (best_row[1] - best_row[0]) if best_row else 0
    col_len = (best_col[1] - best_col[0]) if best_col else 0

    choose_row = row_len >= max(col_len, 8)
    choose_col = (not choose_row) and col_len >= 9

    if choose_row and best_row is not None:
        g0, g1 = best_row
        out: list[tuple[int, int, int, int]] = []
        if g0 >= P["row_min_height"]:
            out.extend(_xy_cut_regions(bin_img[:g0, :], x0, y0, depth + 1, max_depth))
        if (h - g1) >= P["row_min_height"]:
            out.extend(_xy_cut_regions(bin_img[g1:, :], x0, y0 + g1, depth + 1, max_depth))
        if out:
            return out

    if choose_col and best_col is not None:
        g0, g1 = best_col
        out = []
        if g0 >= P["segment_min_width"]:
            out.extend(_xy_cut_regions(bin_img[:, :g0], x0, y0, depth + 1, max_depth))
        if (w - g1) >= P["segment_min_width"]:
            out.extend(_xy_cut_regions(bin_img[:, g1:], x0 + g1, y0, depth + 1, max_depth))
        if out:
            return out

    return [(x0, y0, x0 + w, y0 + h)]


def _find_contour_candidates_multi(bin_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    kernels = [(5, 3), (9, 3), (13, 5), (17, 7), (23, 9)]
    out: list[tuple[int, int, int, int]] = []
    for kw, kh in kernels:
        merged = cv2.morphologyEx(
            bin_img,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)),
            iterations=1,
        )
        cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            out.append((x, y, x + w, y + h))
    return out


def _find_cc_candidates(bin_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats((bin_img > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return out
    for idx in range(1, n):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < max(12, P["min_area"] // 9):
            continue
        out.append((x, y, x + w, y + h))
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


def _clip_box(b: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x1 = min(max(0, int(b[0])), w - 1)
    y1 = min(max(0, int(b[1])), h - 1)
    x2 = min(max(x1 + 1, int(b[2])), w)
    y2 = min(max(y1 + 1, int(b[3])), h)
    return (x1, y1, x2, y2)


def _expand_box(b: tuple[int, int, int, int], pad: int, w: int, h: int) -> tuple[int, int, int, int]:
    return _clip_box((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad), w, h)


def _light_filter(box: tuple[int, int, int, int], page_w: int, page_h: int) -> bool:
    x1, y1, x2, y2 = box
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    area = bw * bh
    if bw < P["min_width"] or bh < P["min_height"]:
        return False
    if area < P["min_area"]:
        return False
    if area > int(page_w * page_h * P["max_box_area_ratio"]):
        return False
    if bh > int(page_h * P["max_box_height_ratio"]):
        return False
    return True


def _dedup_high_iou(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    boxes = sorted(set(boxes), key=lambda b: (b[1], b[0], _box_area(b)))
    if len(boxes) <= 1:
        return boxes
    kept: list[tuple[int, int, int, int]] = []
    for b in boxes:
        duplicate = False
        for k in kept:
            if _iou(b, k) >= P["nms_iou_threshold"]:
                duplicate = True
                break
        if not duplicate:
            kept.append(b)
    return kept


def _fill_density(bin_img: np.ndarray, b: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b
    patch = bin_img[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float((patch > 0).sum() / float(patch.size))


def _score_box(bin_img: np.ndarray, b: tuple[int, int, int, int]) -> float:
    h, w = bin_img.shape
    area_ratio = _box_area(b) / max(1.0, float(h * w))
    dens = _fill_density(bin_img, b)
    size_term = np.exp(-abs(area_ratio - 0.03) / 0.05)
    dens_term = np.exp(-abs(dens - 0.11) / 0.16)
    return float(np.clip(0.55 * size_term + 0.45 * dens_term, 0.0, 1.0))


def detect_connected_blocks_v4_recall(image_path, save_base_dir="data/job_blocks_smart", debug=True):
    image_path = Path(image_path)
    pdf_folder = image_path.parent.name
    base_name = image_path.stem
    save_dir = Path(save_base_dir) / pdf_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[v4] [!] Failed to load: {image_path}")
        return [], [], []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    page_h, page_w = gray.shape

    ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 13)
    _, ot = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_img = cv2.bitwise_or(ad, ot)
    bin_img = cv2.morphologyEx(
        bin_img,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    candidates: list[tuple[int, int, int, int]] = []
    columns = _detect_columns(bin_img)
    for x0, x1 in columns:
        col_bin = bin_img[:, x0:x1]
        for y0, y1 in _row_segments(col_bin):
            row_bin = col_bin[y0:y1, :]
            candidates.extend(_split_row_into_boxes(row_bin, x0, y0))
        candidates.extend(_xy_cut_regions(col_bin, x0, 0, depth=0, max_depth=P["xy_cut_max_depth"]))

    candidates.extend(_find_contour_candidates_multi(bin_img))
    candidates.extend(_find_cc_candidates(bin_img))

    expanded: list[tuple[int, int, int, int]] = []
    for box in candidates:
        clipped = _clip_box(box, page_w, page_h)
        expanded.append(clipped)
        for pad in P["expand_pixels"]:
            if pad <= 0:
                continue
            expanded.append(_expand_box(clipped, pad, page_w, page_h))

    filtered = [b for b in expanded if _light_filter(b, page_w, page_h)]
    filtered = _dedup_high_iou(filtered)
    filtered = sorted(filtered, key=lambda b: (b[1], b[0], _box_area(b)))
    if len(filtered) > P["max_detections"]:
        scored = [(_score_box(bin_img, b), b) for b in filtered]
        scored.sort(key=lambda x: x[0], reverse=True)
        filtered = [b for _, b in scored[: P["max_detections"]]]
        filtered = sorted(filtered, key=lambda b: (b[1], b[0], _box_area(b)))

    blocks = []
    cropped_paths = []
    scores = []
    for idx, (x1, y1, x2, y2) in enumerate(filtered):
        bw = int(x2 - x1)
        bh = int(y2 - y1)
        if bw <= 0 or bh <= 0:
            continue
        blocks.append((int(x1), int(y1), bw, bh))
        scores.append(_score_box(bin_img, (x1, y1, x2, y2)))
        crop = image[y1:y2, x1:x2]
        out_path = save_dir / f"{base_name}_block{len(blocks) - 1}.png"
        cv2.imwrite(str(out_path), crop)
        cropped_paths.append(str(out_path))

    if debug:
        dbg = image.copy()
        for i, (x, y, bw, bh) in enumerate(blocks):
            cv2.rectangle(dbg, (x, y), (x + bw, y + bh), (0, 200, 255), 2)
            if P["debug_draw_scores"]:
                cv2.putText(
                    dbg,
                    f"{i}:{scores[i]:.2f}",
                    (x, max(14, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        debug_name = f"debug_p{base_name.split('_p')[-1]}.png" if "_p" in base_name else f"{base_name}_debug.png"
        debug_path = save_dir / debug_name
        cv2.imwrite(str(debug_path), dbg)
        print(f"[v4] Debug image saved: {debug_path}")

    print(f"[v4] Detected {len(blocks)} recall-first proposals from page {base_name}")
    return blocks, cropped_paths, scores


if __name__ == "__main__":
    import sys

    img = sys.argv[1] if len(sys.argv) > 1 else "data/pdf2img/sample/page1.png"
    detect_connected_blocks_v4_recall(img)
