"""Stage2 v2 detector: layout-aware CV detector for ad-level block localization."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np


DETECTION_V2_DEFAULTS = {
    "min_area": 1200,
    "max_area": 250000,
    "min_width": 55,
    "min_height": 28,
    "max_box_area_ratio": 0.22,
    "max_box_height_ratio": 0.42,
    "column_gap_density_threshold": 0.04,
    "column_min_width_ratio": 0.14,
    "column_gap_min_width": 14,
    "row_blank_density_ratio": 0.012,
    "row_gap_min_height": 12,
    "row_min_height": 26,
    "segment_min_width": 70,
    "segment_expand_px": 4,
    "merge_iou_threshold": 0.62,
    "merge_gap_px": 6,
    "min_fill_density": 0.012,
    "max_fill_density": 0.52,
    "header_exclude_ratio": 0.04,
    "footer_exclude_ratio": 0.03,
    "debug_draw_scores": 1,
}

DETECTION_V2_PARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "detection_params_v2.json"


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
    params = dict(DETECTION_V2_DEFAULTS)
    if DETECTION_V2_PARAMS_PATH.exists():
        try:
            loaded = json.loads(DETECTION_V2_PARAMS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key in params:
                    if key in loaded:
                        params[key] = loaded[key]
        except Exception:
            pass

    # Optional env overrides for controlled experimentation.
    for key in list(params.keys()):
        env_name = f"DETECTOR_V2_{key.upper()}"
        if env_name in os.environ:
            params[key] = os.environ[env_name]

    int_keys = [
        "min_area",
        "max_area",
        "min_width",
        "min_height",
        "column_gap_min_width",
        "row_gap_min_height",
        "row_min_height",
        "segment_min_width",
        "segment_expand_px",
        "merge_gap_px",
        "debug_draw_scores",
    ]
    for key in int_keys:
        params[key] = _as_int(params[key], DETECTION_V2_DEFAULTS[key])

    float_keys = [
        "max_box_area_ratio",
        "max_box_height_ratio",
        "column_gap_density_threshold",
        "column_min_width_ratio",
        "row_blank_density_ratio",
        "merge_iou_threshold",
        "min_fill_density",
        "max_fill_density",
        "header_exclude_ratio",
        "footer_exclude_ratio",
    ]
    for key in float_keys:
        params[key] = _as_float(params[key], DETECTION_V2_DEFAULTS[key])

    params["min_area"] = max(100, params["min_area"])
    params["max_area"] = max(params["min_area"] + 1, params["max_area"])
    params["min_width"] = max(20, params["min_width"])
    params["min_height"] = max(15, params["min_height"])
    params["column_gap_min_width"] = max(6, params["column_gap_min_width"])
    params["row_gap_min_height"] = max(4, params["row_gap_min_height"])
    params["row_min_height"] = max(15, params["row_min_height"])
    params["segment_min_width"] = max(30, params["segment_min_width"])
    params["segment_expand_px"] = max(0, params["segment_expand_px"])
    params["merge_gap_px"] = max(0, params["merge_gap_px"])
    params["max_box_area_ratio"] = float(np.clip(params["max_box_area_ratio"], 0.05, 0.8))
    params["max_box_height_ratio"] = float(np.clip(params["max_box_height_ratio"], 0.1, 0.95))
    params["column_gap_density_threshold"] = float(np.clip(params["column_gap_density_threshold"], 0.005, 0.20))
    params["column_min_width_ratio"] = float(np.clip(params["column_min_width_ratio"], 0.08, 0.5))
    params["row_blank_density_ratio"] = float(np.clip(params["row_blank_density_ratio"], 0.002, 0.08))
    params["merge_iou_threshold"] = float(np.clip(params["merge_iou_threshold"], 0.30, 0.95))
    params["min_fill_density"] = float(np.clip(params["min_fill_density"], 0.001, 0.25))
    params["max_fill_density"] = float(np.clip(params["max_fill_density"], 0.08, 0.98))
    params["header_exclude_ratio"] = float(np.clip(params["header_exclude_ratio"], 0.0, 0.2))
    params["footer_exclude_ratio"] = float(np.clip(params["footer_exclude_ratio"], 0.0, 0.2))
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
    col_density = _smooth_1d(col_density, max(15, int(w * 0.02)))

    gap_mask = col_density < P["column_gap_density_threshold"]
    gap_runs = _runs(gap_mask, P["column_gap_min_width"])
    min_col_w = max(int(w * P["column_min_width_ratio"]), 120)

    cols: list[tuple[int, int]] = []
    left = 0
    for g0, g1 in gap_runs:
        if g0 - left >= min_col_w:
            cols.append((left, g0))
        left = g1
    if w - left >= min_col_w:
        cols.append((left, w))

    if not cols:
        return [(0, w)]

    # Trim tiny margins from columns to reduce separator noise.
    out: list[tuple[int, int]] = []
    for x0, x1 in cols:
        tx0 = min(max(0, x0 + 1), w - 1)
        tx1 = max(tx0 + 1, min(w, x1 - 1))
        out.append((tx0, tx1))
    return out


def _row_segments(col_bin: np.ndarray) -> list[tuple[int, int]]:
    h, w = col_bin.shape
    if h == 0 or w == 0:
        return []
    ink = (col_bin > 0).astype(np.uint8)
    row_density = ink.sum(axis=1).astype(np.float32) / max(1.0, float(w))
    row_density = _smooth_1d(row_density, max(11, int(h * 0.01)))

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
    active = col_density > max(P["row_blank_density_ratio"] * 1.7, 0.008)
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
        py1 = min(h, int(ys.max() + 1 + P["segment_expand_px"]))
        boxes.append((x0 + px0, y0 + py0, x0 + px1, y0 + py1))
    return boxes


def _find_contour_candidates(bin_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    # Conservative close to group nearby words into ad chunks without swallowing full page.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    merged = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        out.append((x, y, x + w, y + h))
    return out


def _xy_cut_regions(
    bin_img: np.ndarray,
    x0: int,
    y0: int,
    depth: int = 0,
    max_depth: int = 6,
) -> list[tuple[int, int, int, int]]:
    h, w = bin_img.shape
    if h < max(2 * P["row_min_height"], 40) or w < max(2 * P["segment_min_width"], 80):
        return [(x0, y0, x0 + w, y0 + h)]
    if depth >= max_depth:
        return [(x0, y0, x0 + w, y0 + h)]

    ink = (bin_img > 0).astype(np.uint8)
    row_density = ink.sum(axis=1).astype(np.float32) / max(1.0, float(w))
    col_density = ink.sum(axis=0).astype(np.float32) / max(1.0, float(h))

    row_blank = row_density < max(P["row_blank_density_ratio"] * 0.6, 0.0015)
    col_blank = col_density < max(P["column_gap_density_threshold"] * 0.75, 0.002)
    row_gaps = _runs(row_blank, max(P["row_gap_min_height"], 5))
    col_gaps = _runs(col_blank, max(P["column_gap_min_width"], 6))

    best_row = max(row_gaps, key=lambda g: g[1] - g[0], default=None)
    best_col = max(col_gaps, key=lambda g: g[1] - g[0], default=None)
    row_len = (best_row[1] - best_row[0]) if best_row else 0
    col_len = (best_col[1] - best_col[0]) if best_col else 0

    # Prefer horizontal splits; classifieds are typically stacked vertically in columns.
    choose_row = row_len >= max(col_len, max(P["row_gap_min_height"] + 2, 8))
    choose_col = (not choose_row) and col_len >= max(P["column_gap_min_width"] + 2, 10)

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


def _tighten_to_ink(bin_img: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    patch = bin_img[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    ys, xs = np.where(patch > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    nx1 = x1 + int(xs.min())
    nx2 = x1 + int(xs.max()) + 1
    ny1 = y1 + int(ys.min())
    ny2 = y1 + int(ys.max()) + 1
    return (nx1, ny1, nx2, ny2)


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


def _should_merge(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    if _iou(a, b) >= P["merge_iou_threshold"]:
        return True
    x_overlap = min(a[2], b[2]) - max(a[0], b[0])
    y_overlap = min(a[3], b[3]) - max(a[1], b[1])
    x_gap = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    y_gap = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return (x_overlap > 0 and y_gap <= P["merge_gap_px"]) or (y_overlap > 0 and x_gap <= P["merge_gap_px"])


def _merge_boxes(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _dedup_and_merge(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    boxes = sorted(boxes, key=lambda b: (b[1], b[0], _box_area(b)))
    if len(boxes) < 2:
        return boxes

    changed = True
    while changed:
        changed = False
        used = [False] * len(boxes)
        merged: list[tuple[int, int, int, int]] = []
        for i, b in enumerate(boxes):
            if used[i]:
                continue
            cur = b
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if _should_merge(cur, boxes[j]):
                    cur = _merge_boxes(cur, boxes[j])
                    used[j] = True
                    changed = True
            merged.append(cur)
        boxes = sorted(merged, key=lambda b: (b[1], b[0], _box_area(b)))
    return boxes


def _fill_density(bin_img: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    patch = bin_img[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float((patch > 0).sum() / float(patch.size))


def _filter_box(box: tuple[int, int, int, int], bin_img: np.ndarray) -> bool:
    h, w = bin_img.shape
    x1, y1, x2, y2 = box
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    area = bw * bh
    if bw < P["min_width"] or bh < P["min_height"]:
        return False
    if area < P["min_area"] or area > P["max_area"]:
        return False
    if area > int(h * w * P["max_box_area_ratio"]):
        return False
    if bh > int(h * P["max_box_height_ratio"]):
        return False

    # Filter obvious page furniture/header/footer slabs.
    if y1 < int(h * P["header_exclude_ratio"]) and bh < int(h * 0.06):
        return False
    if y2 > int(h * (1.0 - P["footer_exclude_ratio"])) and bh < int(h * 0.05):
        return False

    aspect = bw / max(1.0, float(bh))
    if aspect < 0.18 or aspect > 8.5:
        return False

    dens = _fill_density(bin_img, box)
    if dens < P["min_fill_density"] or dens > P["max_fill_density"]:
        return False
    return True


def _score_box(bin_img: np.ndarray, gray_img: np.ndarray, box: tuple[int, int, int, int]) -> float:
    h, w = bin_img.shape
    x1, y1, x2, y2 = box
    area_ratio = _box_area(box) / max(1.0, float(h * w))
    dens = _fill_density(bin_img, box)
    patch = gray_img[y1:y2, x1:x2]
    if patch.size == 0:
        edge_density = 0.0
    else:
        edges = cv2.Canny(patch, 80, 160)
        edge_density = float((edges > 0).sum() / float(edges.size))

    size_term = np.exp(-abs(area_ratio - 0.02) / 0.03)  # favors ad-like regions
    dens_term = np.exp(-abs(dens - 0.10) / 0.12)
    edge_term = np.clip(edge_density * 3.2, 0.0, 1.0)
    score = 0.45 * size_term + 0.35 * dens_term + 0.20 * edge_term
    return float(np.clip(score, 0.0, 1.0))


def detect_connected_blocks_v2(image_path, save_base_dir="data/job_blocks_smart", debug=True):
    image_path = Path(image_path)
    pdf_folder = image_path.parent.name
    base_name = image_path.stem
    save_dir = Path(save_base_dir) / pdf_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[!] Failed to load: {image_path}")
        return [], [], []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Conservative text masks: OR of adaptive + otsu for robustness.
    ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 13)
    _, ot = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_img = cv2.bitwise_or(ad, ot)
    bin_img = cv2.morphologyEx(
        bin_img,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    columns = _detect_columns(bin_img)
    candidates: list[tuple[int, int, int, int]] = []
    for x0, x1 in columns:
        col_bin = bin_img[:, x0:x1]
        for y0, y1 in _row_segments(col_bin):
            row_bin = col_bin[y0:y1, :]
            candidates.extend(_split_row_into_boxes(row_bin, x0, y0))
        # Fallback for dense classifieds: whitespace XY-cut over full column.
        for cbox in _xy_cut_regions(col_bin, x0, 0, depth=0, max_depth=7):
            t = _tighten_to_ink(bin_img, cbox)
            if t is not None:
                candidates.append(t)

    # Add contour-based candidates to recover boxed ads and sparse layouts.
    candidates.extend(_find_contour_candidates(bin_img))

    merged = _dedup_and_merge(candidates)
    filtered = [b for b in merged if _filter_box(b, bin_img)]
    filtered = _dedup_and_merge(filtered)
    filtered = sorted(filtered, key=lambda b: (b[1], b[0]))

    # Final stage: deterministic scores and block artifact writing.
    blocks = []
    cropped_paths = []
    scores = []
    for idx, (x1, y1, x2, y2) in enumerate(filtered):
        bw = int(x2 - x1)
        bh = int(y2 - y1)
        if bw <= 0 or bh <= 0:
            continue
        blocks.append((int(x1), int(y1), bw, bh))
        scores.append(_score_box(bin_img, gray, (x1, y1, x2, y2)))
        crop = image[y1:y2, x1:x2]
        out_path = save_dir / f"{base_name}_block{len(blocks)-1}.png"
        cv2.imwrite(str(out_path), crop)
        cropped_paths.append(str(out_path))

    if debug:
        dbg = image.copy()
        for i, (x, y, bw, bh) in enumerate(blocks):
            cv2.rectangle(dbg, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            if P["debug_draw_scores"]:
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
        print(f"[v2] Debug image saved: {debug_path}")

    print(f"[v2] Detected {len(blocks)} smart blocks from page {base_name}")
    return blocks, cropped_paths, scores


if __name__ == "__main__":
    import sys

    img = sys.argv[1] if len(sys.argv) > 1 else "data/pdf2img/sample/page1.png"
    detect_connected_blocks_v2(img)
