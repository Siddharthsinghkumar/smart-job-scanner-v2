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
    "anchor_grid_enabled": 1,
    "anchor_grid_hot_only": 1,
    "anchor_grid_width_fracs": [0.08, 0.095, 0.11, 0.14, 0.18, 0.24, 0.30],
    "anchor_grid_height_fracs": [0.022, 0.03, 0.04, 0.055, 0.075, 0.10, 0.14, 0.20],
    "anchor_grid_stride_x_ratio": 0.12,
    "anchor_grid_stride_y_ratio": 0.12,
    "anchor_grid_density_min": 0.012,
    "anchor_grid_density_max": 0.45,
    "anchor_grid_max_boxes": 5000,
    "fast_dedup_limit": 3200,
    "debug_draw_scores": 0,
    "hot_page_boost_enabled": 1,
    "hot_page_runtime_selector_enabled": 0,
    "hot_page_hotness_min_score": 0.56,
    "hot_page_candidate_density_ref": 2600.0,
    "hot_page_cc_density_ref": 1400.0,
    "hot_page_small_box_ratio_ref": 0.48,
    "hot_page_dense_row_fraction_ref": 0.42,
    "hot_page_column_count_ref": 5.0,
    "hot_page_force_candidate_count": 1800,
    "hot_page_force_cc_count": 2400,
    "hot_page_force_small_box_ratio": 0.60,
    "hot_page_row_dense_threshold": 0.055,
    "hot_page_max_detections": 900,
    "cold_page_max_detections": 1,
    "hot_expand_enabled": 0,
    "hot_expand_sym_fracs": [0.04],
    "hot_expand_wide_fracs": [0.06],
    "hot_expand_tall_fracs": [0.06],
    "hot_expand_bias_frac": 0.05,
    "hot_expand_max_new_boxes": 2200,
    "hot_expand_dedup_iou": 0.985,
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


def _as_float_list(v, default):
    if isinstance(v, list):
        out = []
        for item in v:
            try:
                out.append(float(item))
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
        "dense_tiling_enabled",
        "dense_tile_min_run_height",
        "dense_tile_max_per_column",
        "anchor_grid_enabled",
        "anchor_grid_hot_only",
        "anchor_grid_max_boxes",
        "fast_dedup_limit",
        "debug_draw_scores",
        "hot_page_boost_enabled",
        "hot_page_runtime_selector_enabled",
        "hot_page_force_candidate_count",
        "hot_page_force_cc_count",
        "hot_page_max_detections",
        "cold_page_max_detections",
        "hot_expand_enabled",
        "hot_expand_max_new_boxes",
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
        "dense_tile_row_density_threshold",
        "dense_tile_stride_x_ratio",
        "dense_tile_stride_y_ratio",
        "dense_tile_quota_ratio",
        "base_box_quota_ratio",
        "small_box_quota_ratio",
        "small_box_area_ratio",
        "anchor_grid_stride_x_ratio",
        "anchor_grid_stride_y_ratio",
        "anchor_grid_density_min",
        "anchor_grid_density_max",
        "hot_page_hotness_min_score",
        "hot_page_candidate_density_ref",
        "hot_page_cc_density_ref",
        "hot_page_small_box_ratio_ref",
        "hot_page_dense_row_fraction_ref",
        "hot_page_column_count_ref",
        "hot_page_force_small_box_ratio",
        "hot_page_row_dense_threshold",
        "hot_expand_bias_frac",
        "hot_expand_dedup_iou",
    ]
    for key in float_keys:
        params[key] = _as_float(params[key], DETECTION_V4_DEFAULTS[key])

    params["expand_pixels"] = _as_int_list(params.get("expand_pixels"), DETECTION_V4_DEFAULTS["expand_pixels"])
    params["dense_tile_width_fracs"] = _as_float_list(
        params.get("dense_tile_width_fracs"), DETECTION_V4_DEFAULTS["dense_tile_width_fracs"]
    )
    params["dense_tile_height_fracs"] = _as_float_list(
        params.get("dense_tile_height_fracs"), DETECTION_V4_DEFAULTS["dense_tile_height_fracs"]
    )
    params["anchor_grid_width_fracs"] = _as_float_list(
        params.get("anchor_grid_width_fracs"), DETECTION_V4_DEFAULTS["anchor_grid_width_fracs"]
    )
    params["anchor_grid_height_fracs"] = _as_float_list(
        params.get("anchor_grid_height_fracs"), DETECTION_V4_DEFAULTS["anchor_grid_height_fracs"]
    )
    params["hot_expand_sym_fracs"] = _as_float_list(
        params.get("hot_expand_sym_fracs"), DETECTION_V4_DEFAULTS["hot_expand_sym_fracs"]
    )
    params["hot_expand_wide_fracs"] = _as_float_list(
        params.get("hot_expand_wide_fracs"), DETECTION_V4_DEFAULTS["hot_expand_wide_fracs"]
    )
    params["hot_expand_tall_fracs"] = _as_float_list(
        params.get("hot_expand_tall_fracs"), DETECTION_V4_DEFAULTS["hot_expand_tall_fracs"]
    )
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
    params["dense_tiling_enabled"] = 1 if params["dense_tiling_enabled"] else 0
    params["dense_tile_min_run_height"] = max(24, params["dense_tile_min_run_height"])
    params["dense_tile_max_per_column"] = max(80, params["dense_tile_max_per_column"])
    params["anchor_grid_enabled"] = 1 if params["anchor_grid_enabled"] else 0
    params["anchor_grid_hot_only"] = 1 if params["anchor_grid_hot_only"] else 0
    params["anchor_grid_max_boxes"] = max(200, params["anchor_grid_max_boxes"])
    params["fast_dedup_limit"] = max(600, params["fast_dedup_limit"])
    params["hot_page_boost_enabled"] = 1 if params["hot_page_boost_enabled"] else 0
    params["hot_page_runtime_selector_enabled"] = 1 if params["hot_page_runtime_selector_enabled"] else 0
    params["hot_page_force_candidate_count"] = max(10, params["hot_page_force_candidate_count"])
    params["hot_page_force_cc_count"] = max(10, params["hot_page_force_cc_count"])
    params["hot_page_max_detections"] = max(1, params["hot_page_max_detections"])
    params["cold_page_max_detections"] = max(1, params["cold_page_max_detections"])
    params["hot_expand_enabled"] = 1 if params["hot_expand_enabled"] else 0
    params["hot_expand_max_new_boxes"] = max(1, params["hot_expand_max_new_boxes"])
    params["max_box_area_ratio"] = float(np.clip(params["max_box_area_ratio"], 0.1, 0.95))
    params["max_box_height_ratio"] = float(np.clip(params["max_box_height_ratio"], 0.2, 1.0))
    params["column_gap_density_threshold"] = float(np.clip(params["column_gap_density_threshold"], 0.003, 0.20))
    params["column_min_width_ratio"] = float(np.clip(params["column_min_width_ratio"], 0.05, 0.6))
    params["row_blank_density_ratio"] = float(np.clip(params["row_blank_density_ratio"], 0.001, 0.08))
    params["nms_iou_threshold"] = float(np.clip(params["nms_iou_threshold"], 0.80, 0.99))
    params["dense_tile_row_density_threshold"] = float(np.clip(params["dense_tile_row_density_threshold"], 0.001, 0.20))
    params["dense_tile_stride_x_ratio"] = float(np.clip(params["dense_tile_stride_x_ratio"], 0.10, 0.95))
    params["dense_tile_stride_y_ratio"] = float(np.clip(params["dense_tile_stride_y_ratio"], 0.10, 0.95))
    params["dense_tile_quota_ratio"] = float(np.clip(params["dense_tile_quota_ratio"], 0.0, 0.95))
    params["base_box_quota_ratio"] = float(np.clip(params["base_box_quota_ratio"], 0.0, 0.95))
    params["small_box_quota_ratio"] = float(np.clip(params["small_box_quota_ratio"], 0.0, 0.95))
    params["small_box_area_ratio"] = float(np.clip(params["small_box_area_ratio"], 0.005, 0.30))
    params["anchor_grid_stride_x_ratio"] = float(np.clip(params["anchor_grid_stride_x_ratio"], 0.05, 0.95))
    params["anchor_grid_stride_y_ratio"] = float(np.clip(params["anchor_grid_stride_y_ratio"], 0.05, 0.95))
    params["anchor_grid_density_min"] = float(np.clip(params["anchor_grid_density_min"], 0.0005, 0.80))
    params["anchor_grid_density_max"] = float(np.clip(params["anchor_grid_density_max"], 0.005, 0.95))
    params["hot_page_hotness_min_score"] = float(np.clip(params["hot_page_hotness_min_score"], 0.05, 0.99))
    params["hot_page_candidate_density_ref"] = float(np.clip(params["hot_page_candidate_density_ref"], 100.0, 30000.0))
    params["hot_page_cc_density_ref"] = float(np.clip(params["hot_page_cc_density_ref"], 100.0, 30000.0))
    params["hot_page_small_box_ratio_ref"] = float(np.clip(params["hot_page_small_box_ratio_ref"], 0.05, 1.0))
    params["hot_page_dense_row_fraction_ref"] = float(np.clip(params["hot_page_dense_row_fraction_ref"], 0.05, 1.0))
    params["hot_page_column_count_ref"] = float(np.clip(params["hot_page_column_count_ref"], 1.0, 20.0))
    params["hot_page_force_small_box_ratio"] = float(np.clip(params["hot_page_force_small_box_ratio"], 0.05, 1.0))
    params["hot_page_row_dense_threshold"] = float(np.clip(params["hot_page_row_dense_threshold"], 0.005, 0.40))
    params["hot_expand_bias_frac"] = float(np.clip(params["hot_expand_bias_frac"], 0.005, 0.30))
    params["hot_expand_dedup_iou"] = float(np.clip(params["hot_expand_dedup_iou"], 0.85, 0.999))
    if params["anchor_grid_density_min"] >= params["anchor_grid_density_max"]:
        params["anchor_grid_density_min"] = max(0.0005, params["anchor_grid_density_max"] * 0.5)
    params["expand_pixels"] = sorted(set(max(0, int(x)) for x in params["expand_pixels"]))
    params["dense_tile_width_fracs"] = sorted(
        set(float(np.clip(x, 0.05, 1.0)) for x in params["dense_tile_width_fracs"])
    )
    params["dense_tile_height_fracs"] = sorted(
        set(float(np.clip(x, 0.02, 1.2)) for x in params["dense_tile_height_fracs"])
    )
    params["anchor_grid_width_fracs"] = sorted(
        set(float(np.clip(x, 0.03, 0.98)) for x in params["anchor_grid_width_fracs"])
    )
    params["anchor_grid_height_fracs"] = sorted(
        set(float(np.clip(x, 0.012, 0.95)) for x in params["anchor_grid_height_fracs"])
    )
    params["hot_expand_sym_fracs"] = sorted(
        set(float(np.clip(x, 0.005, 0.30)) for x in params["hot_expand_sym_fracs"])
    )
    params["hot_expand_wide_fracs"] = sorted(
        set(float(np.clip(x, 0.005, 0.30)) for x in params["hot_expand_wide_fracs"])
    )
    params["hot_expand_tall_fracs"] = sorted(
        set(float(np.clip(x, 0.005, 0.30)) for x in params["hot_expand_tall_fracs"])
    )
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


def _is_hot_profile(max_detections_override: int | None) -> bool:
    if max_detections_override is None:
        return False
    return int(max_detections_override) > int(P["max_detections"])


def _dense_classified_tiles(
    bin_img: np.ndarray,
    columns: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    if not P["dense_tiling_enabled"]:
        return []

    page_h, _ = bin_img.shape
    out: list[tuple[int, int, int, int]] = []
    for x0, x1 in columns:
        col_bin = bin_img[:, x0:x1]
        col_h, col_w = col_bin.shape
        if col_w < P["min_width"]:
            continue

        row_density = (col_bin > 0).sum(axis=1).astype(np.float32) / max(1.0, float(col_w))
        row_density = _smooth_1d(row_density, max(9, int(col_h * 0.010)))
        adaptive_thr = max(
            P["dense_tile_row_density_threshold"],
            float(np.percentile(row_density, 45) * 0.82),
        )
        dense_runs = _runs(row_density >= adaptive_thr, max(P["dense_tile_min_run_height"], P["min_height"]))
        if not dense_runs:
            dense_runs = [(0, col_h)]

        col_tiles: list[tuple[int, int, int, int]] = []
        for ry0, ry1 in dense_runs:
            run_h = int(ry1 - ry0)
            if run_h < P["min_height"]:
                continue

            # Keep run-level slabs so large stacked classifieds are not missed.
            col_tiles.append((x0, ry0, x1, ry1))
            if run_h >= 2 * P["min_height"]:
                mid = int(ry0 + run_h // 2)
                col_tiles.append((x0, ry0, x1, mid))
                col_tiles.append((x0, mid, x1, ry1))

            tile_ws = sorted(set(max(P["min_width"], int(round(col_w * f))) for f in P["dense_tile_width_fracs"]))
            tile_hs = set(max(P["min_height"], int(round(page_h * f))) for f in P["dense_tile_height_fracs"])
            tile_hs.update(
                {
                    max(P["min_height"], int(round(run_h * 0.30))),
                    max(P["min_height"], int(round(run_h * 0.45))),
                    max(P["min_height"], int(round(run_h * 0.65))),
                    max(P["min_height"], int(round(run_h * 0.85))),
                    run_h,
                }
            )
            tile_hs_sorted = sorted(min(run_h, int(v)) for v in tile_hs if int(v) >= P["min_height"])
            for tw in tile_ws:
                tw = min(int(tw), col_w)
                if tw < P["min_width"]:
                    continue
                sx = max(8, int(round(tw * P["dense_tile_stride_x_ratio"])))
                x_min = int(x0)
                x_max_start = int(x1 - tw)
                x_starts = list(range(x_min, x_max_start + 1, sx)) if x_max_start >= x_min else [x_min]
                if x_starts and x_starts[-1] != x_max_start:
                    x_starts.append(x_max_start)

                for th in tile_hs_sorted:
                    th = min(int(th), run_h)
                    if th < P["min_height"]:
                        continue
                    sy = max(8, int(round(th * P["dense_tile_stride_y_ratio"])))
                    y_min = int(ry0)
                    y_max_start = int(ry1 - th)
                    y_starts = list(range(y_min, y_max_start + 1, sy)) if y_max_start >= y_min else [y_min]
                    if y_starts and y_starts[-1] != y_max_start:
                        y_starts.append(y_max_start)

                    for xs in x_starts:
                        for ys in y_starts:
                            col_tiles.append((xs, ys, xs + tw, ys + th))

        if len(col_tiles) > P["dense_tile_max_per_column"]:
            scored = [(_score_box(bin_img, b), b) for b in col_tiles]
            scored.sort(key=lambda x: x[0], reverse=True)
            col_tiles = [b for _, b in scored[: P["dense_tile_max_per_column"]]]
        out.extend(col_tiles)
    return out


def _find_anchor_grid_candidates(
    bin_img: np.ndarray,
    columns: list[tuple[int, int]],
    is_hot_runtime: bool = False,
    max_detections_override: int | None = None,
) -> list[tuple[int, int, int, int]]:
    if not P["anchor_grid_enabled"]:
        return []
    if P["anchor_grid_hot_only"] and not (is_hot_runtime or _is_hot_profile(max_detections_override)):
        return []

    page_h, page_w = bin_img.shape
    if page_h <= 0 or page_w <= 0:
        return []

    integral = cv2.integral((bin_img > 0).astype(np.uint8), sdepth=cv2.CV_32S)
    out: list[tuple[int, int, int, int]] = []

    def _rect_ink(x1: int, y1: int, x2: int, y2: int) -> int:
        return int(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])

    def _emit_region(rx0: int, rx1: int) -> None:
        region_w = int(rx1 - rx0)
        if region_w < P["min_width"]:
            return
        for wf in P["anchor_grid_width_fracs"]:
            bw = max(P["min_width"], int(round(page_w * wf)))
            bw = min(region_w, bw)
            if bw < P["min_width"]:
                continue
            sx = max(8, int(round(bw * P["anchor_grid_stride_x_ratio"])))
            x_min = int(rx0)
            x_max_start = int(rx1 - bw)
            if x_max_start < x_min:
                x_starts = [x_min]
            else:
                x_starts = list(range(x_min, x_max_start + 1, sx))
                if x_starts and x_starts[-1] != x_max_start:
                    x_starts.append(x_max_start)

            for hf in P["anchor_grid_height_fracs"]:
                bh = max(P["min_height"], int(round(page_h * hf)))
                bh = min(page_h, bh)
                if bh < P["min_height"]:
                    continue
                sy = max(8, int(round(bh * P["anchor_grid_stride_y_ratio"])))
                y_max_start = int(page_h - bh)
                if y_max_start < 0:
                    y_starts = [0]
                else:
                    y_starts = list(range(0, y_max_start + 1, sy))
                    if y_starts and y_starts[-1] != y_max_start:
                        y_starts.append(y_max_start)

                area = float(bw * bh)
                if area <= 0.0:
                    continue
                for xs in x_starts:
                    xe = xs + bw
                    for ys in y_starts:
                        ye = ys + bh
                        density = _rect_ink(xs, ys, xe, ye) / area
                        if density < P["anchor_grid_density_min"] or density > P["anchor_grid_density_max"]:
                            continue
                        out.append((int(xs), int(ys), int(xe), int(ye)))

    # Whole-page anchors plus column-local anchors for dense classifieds.
    _emit_region(0, page_w)
    for x0, x1 in columns:
        _emit_region(int(x0), int(x1))

    unique = sorted(set(out), key=lambda b: (b[1], b[0], _box_area(b)))
    max_keep = int(P["anchor_grid_max_boxes"])
    if len(unique) <= max_keep:
        return unique
    if max_keep <= 1:
        return [unique[0]]

    idx = np.linspace(0, len(unique) - 1, num=max_keep, dtype=np.int32).tolist()
    return [unique[i] for i in idx]


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


def _dedup_high_iou_with_threshold(
    boxes: list[tuple[int, int, int, int]],
    iou_threshold: float,
) -> list[tuple[int, int, int, int]]:
    boxes = sorted(set(boxes), key=lambda b: (b[1], b[0], _box_area(b)))
    if len(boxes) <= 1:
        return boxes
    if len(boxes) > P["fast_dedup_limit"]:
        return boxes
    kept: list[tuple[int, int, int, int]] = []
    for b in boxes:
        duplicate = False
        for k in kept:
            if _iou(b, k) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(b)
    return kept


def _dedup_high_iou(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return _dedup_high_iou_with_threshold(boxes, float(P["nms_iou_threshold"]))


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


def _expand_filter_dedup(
    candidates: list[tuple[int, int, int, int]],
    page_w: int,
    page_h: int,
) -> list[tuple[int, int, int, int]]:
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
    return sorted(filtered, key=lambda b: (b[1], b[0], _box_area(b)))


def _runtime_hot_profile(
    bin_img: np.ndarray,
    columns: list[tuple[int, int]],
    base_candidates: list[tuple[int, int, int, int]],
    cc_candidates: list[tuple[int, int, int, int]],
) -> dict:
    page_h, page_w = bin_img.shape
    page_area = max(1.0, float(page_h * page_w))
    area_mpix = page_area / 1_000_000.0

    candidate_count = int(len(base_candidates))
    cc_count = int(len(cc_candidates))
    column_count = int(len(columns))
    small_thr = float(page_area * P["small_box_area_ratio"])
    small_count = int(sum(1 for b in base_candidates if _box_area(b) <= small_thr))
    small_ratio = float(small_count / max(1, candidate_count))

    ink_mask = (bin_img > 0).astype(np.uint8)
    ink_density = float(ink_mask.sum() / page_area)
    row_density = ink_mask.sum(axis=1).astype(np.float32) / max(1.0, float(page_w))
    dense_row_fraction = float((row_density >= float(P["hot_page_row_dense_threshold"])).mean())

    candidate_density = float(candidate_count / max(0.01, area_mpix))
    cc_density = float(cc_count / max(0.01, area_mpix))

    score_components = {
        "candidate_density": float(np.clip(candidate_density / P["hot_page_candidate_density_ref"], 0.0, 1.0)),
        "cc_density": float(np.clip(cc_density / P["hot_page_cc_density_ref"], 0.0, 1.0)),
        "small_box_ratio": float(np.clip(small_ratio / P["hot_page_small_box_ratio_ref"], 0.0, 1.0)),
        "dense_row_fraction": float(np.clip(dense_row_fraction / P["hot_page_dense_row_fraction_ref"], 0.0, 1.0)),
        "column_count": float(np.clip(column_count / P["hot_page_column_count_ref"], 0.0, 1.0)),
    }
    score_weights = {
        "candidate_density": 0.38,
        "cc_density": 0.20,
        "small_box_ratio": 0.18,
        "dense_row_fraction": 0.14,
        "column_count": 0.10,
    }
    score = float(sum(score_components[k] * score_weights[k] for k in score_components))
    forced = {
        "candidate_count": candidate_count >= int(P["hot_page_force_candidate_count"]),
        "cc_count": cc_count >= int(P["hot_page_force_cc_count"]),
        "small_box_ratio": small_ratio >= float(P["hot_page_force_small_box_ratio"]),
    }

    enabled = bool(P["hot_page_boost_enabled"]) and bool(P["hot_page_runtime_selector_enabled"])
    hot_by_score = score >= float(P["hot_page_hotness_min_score"])
    is_hot = bool(enabled and (hot_by_score or any(forced.values())))

    reasons: list[str] = []
    if not enabled:
        reasons.append("runtime_selector_disabled")
    else:
        if hot_by_score:
            reasons.append("hotness_score_above_threshold")
        for k, v in forced.items():
            if v:
                reasons.append(f"forced_by_{k}")
        if not reasons:
            top_comp = sorted(score_components.items(), key=lambda kv: kv[1], reverse=True)[:2]
            top_tag = ",".join(k for k, _ in top_comp)
            reasons.append(f"below_threshold_top_components={top_tag}")

    return {
        "selector_enabled": enabled,
        "is_hot": is_hot,
        "hotness_score": round(score, 6),
        "hotness_min_score": float(P["hot_page_hotness_min_score"]),
        "signals": {
            "candidate_count": candidate_count,
            "candidate_density_per_mpix": round(candidate_density, 3),
            "cc_count": cc_count,
            "cc_density_per_mpix": round(cc_density, 3),
            "small_candidate_ratio": round(small_ratio, 6),
            "dense_row_fraction": round(dense_row_fraction, 6),
            "column_count": column_count,
            "ink_density": round(ink_density, 6),
            "page_area_mpix": round(area_mpix, 4),
        },
        "score_components": {k: round(v, 6) for k, v in score_components.items()},
        "forced_rules": forced,
        "reasons": reasons,
    }


def _hot_page_expansion_variants(
    boxes: list[tuple[int, int, int, int]],
    page_w: int,
    page_h: int,
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for b in boxes:
        x1, y1, x2, y2 = b
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        for frac in P["hot_expand_sym_fracs"]:
            dx = max(1, int(round(bw * frac)))
            dy = max(1, int(round(bh * frac)))
            out.append(_clip_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), page_w, page_h))

        for frac in P["hot_expand_wide_fracs"]:
            dx = max(1, int(round(bw * frac)))
            dy = max(1, int(round(bh * frac * 0.18)))
            out.append(_clip_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), page_w, page_h))

        for frac in P["hot_expand_tall_fracs"]:
            dx = max(1, int(round(bw * frac * 0.18)))
            dy = max(1, int(round(bh * frac)))
            out.append(_clip_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), page_w, page_h))

        bfrac = float(P["hot_expand_bias_frac"])
        bdx = max(1, int(round(bw * bfrac)))
        bdy = max(1, int(round(bh * bfrac)))
        out.append(_clip_box((x1 - 2 * bdx, y1 - bdy // 3, x2 + bdx // 2, y2 + bdy // 3), page_w, page_h))
        out.append(_clip_box((x1 - bdx // 2, y1 - bdy // 3, x2 + 2 * bdx, y2 + bdy // 3), page_w, page_h))
        out.append(_clip_box((x1 - bdx // 3, y1 - 2 * bdy, x2 + bdx // 3, y2 + bdy // 2), page_w, page_h))
        out.append(_clip_box((x1 - bdx // 3, y1 - bdy // 2, x2 + bdx // 3, y2 + 2 * bdy), page_w, page_h))
    return out


def _select_with_recall_quotas(
    bin_img: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    base_boxes: list[tuple[int, int, int, int]],
    dense_tile_boxes: list[tuple[int, int, int, int]],
    max_detections_override: int | None = None,
) -> list[tuple[int, int, int, int]]:
    max_det = int(max_detections_override) if max_detections_override is not None else int(P["max_detections"])
    max_det = max(1, max_det)
    if len(boxes) <= max_det:
        return boxes

    page_h, page_w = bin_img.shape
    page_area = max(1, int(page_h * page_w))
    small_area_threshold = int(round(page_area * P["small_box_area_ratio"]))
    base_set = set(base_boxes)
    tile_set = set(dense_tile_boxes)

    scored = [(_score_box(bin_img, b), b) for b in boxes]
    scored.sort(key=lambda x: (x[0], -x[1][1], -x[1][0]), reverse=True)

    base_ranked = [b for _, b in scored if b in base_set]
    dense_ranked = [b for _, b in scored if b in tile_set]
    small_ranked = [b for _, b in scored if _box_area(b) <= small_area_threshold]

    selected: list[tuple[int, int, int, int]] = []
    seen = set()

    def _try_add(b: tuple[int, int, int, int]) -> bool:
        if b in seen:
            return False
        selected.append(b)
        seen.add(b)
        return True

    base_quota = min(len(base_ranked), int(round(max_det * P["base_box_quota_ratio"])))
    dense_quota = min(len(dense_ranked), int(round(max_det * P["dense_tile_quota_ratio"])))
    small_quota = min(
        len(small_ranked),
        max(0, int(round(max_det * P["small_box_quota_ratio"]))),
    )

    for b in base_ranked:
        if len(selected) >= base_quota:
            break
        _try_add(b)

    for b in dense_ranked:
        if len(selected) >= base_quota + dense_quota:
            break
        _try_add(b)

    added_small = 0
    for b in small_ranked:
        if len(selected) >= max_det or added_small >= small_quota:
            break
        if _try_add(b):
            added_small += 1

    for _, b in scored:
        if len(selected) >= max_det:
            break
        _try_add(b)

    return sorted(selected, key=lambda b: (b[1], b[0], _box_area(b)))


def detect_connected_blocks_v4_recall(
    image_path,
    save_base_dir="data/job_blocks_smart",
    debug=True,
    return_metadata: bool = False,
    max_detections_override: int | None = None,
):
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

    base_candidates: list[tuple[int, int, int, int]] = []
    columns = _detect_columns(bin_img)
    row_segment_count = 0
    for x0, x1 in columns:
        col_bin = bin_img[:, x0:x1]
        row_segments = _row_segments(col_bin)
        row_segment_count += len(row_segments)
        for y0, y1 in row_segments:
            row_bin = col_bin[y0:y1, :]
            base_candidates.extend(_split_row_into_boxes(row_bin, x0, y0))
        base_candidates.extend(_xy_cut_regions(col_bin, x0, 0, depth=0, max_depth=P["xy_cut_max_depth"]))

    base_candidates.extend(_find_contour_candidates_multi(bin_img))
    cc_candidates = _find_cc_candidates(bin_img)
    base_candidates.extend(cc_candidates)
    hot_profile = _runtime_hot_profile(bin_img, columns, base_candidates, cc_candidates)
    is_hot_page = bool(hot_profile.get("is_hot", False))

    if max_detections_override is not None:
        selected_max_detections = int(max(1, max_detections_override))
    elif bool(P["hot_page_boost_enabled"]) and bool(P["hot_page_runtime_selector_enabled"]):
        hot_cap = int(max(1, P["hot_page_max_detections"]))
        cold_cap = int(max(1, P["cold_page_max_detections"]))
        selected_max_detections = hot_cap if is_hot_page else cold_cap
    else:
        selected_max_detections = int(max(1, P["max_detections"]))

    dense_tile_candidates = _dense_classified_tiles(bin_img, columns)
    dense_tile_candidates.extend(
        _find_anchor_grid_candidates(
            bin_img,
            columns,
            is_hot_runtime=is_hot_page,
            max_detections_override=selected_max_detections,
        )
    )

    base_filtered = _expand_filter_dedup(base_candidates, page_w, page_h)
    dense_filtered = _expand_filter_dedup(dense_tile_candidates, page_w, page_h)

    filtered = _dedup_high_iou(base_filtered + dense_filtered)
    hot_expansion_added = 0
    if bool(P["hot_expand_enabled"]) and is_hot_page and filtered:
        hot_expanded = _hot_page_expansion_variants(filtered, page_w, page_h)
        hot_expanded = [b for b in hot_expanded if _light_filter(b, page_w, page_h)]
        hot_expanded = _dedup_high_iou_with_threshold(hot_expanded, float(P["hot_expand_dedup_iou"]))
        max_new = int(max(1, P["hot_expand_max_new_boxes"]))
        if len(hot_expanded) > max_new:
            scored_hot = [(_score_box(bin_img, b), b) for b in hot_expanded]
            scored_hot.sort(key=lambda x: x[0], reverse=True)
            hot_expanded = [b for _, b in scored_hot[:max_new]]
        hot_expansion_added = int(len(hot_expanded))
        filtered = _dedup_high_iou_with_threshold(filtered + hot_expanded, float(P["hot_expand_dedup_iou"]))

    pre_quota_count = int(len(filtered))
    filtered = _select_with_recall_quotas(
        bin_img,
        filtered,
        base_filtered,
        dense_filtered,
        max_detections_override=selected_max_detections,
    )

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
    metadata = {
        "selector_version": "runtime_hotness_v1",
        "runtime_hotness": hot_profile,
        "is_hot_page": is_hot_page,
        "max_detections_selected": int(selected_max_detections),
        "row_segment_count": int(row_segment_count),
        "counts": {
            "base_candidates_raw": int(len(base_candidates)),
            "cc_candidates_raw": int(len(cc_candidates)),
            "dense_tile_candidates_raw": int(len(dense_tile_candidates)),
            "base_filtered": int(len(base_filtered)),
            "dense_filtered": int(len(dense_filtered)),
            "pre_quota_candidates": int(pre_quota_count),
            "hot_expansion_added": int(hot_expansion_added),
            "final_selected": int(len(blocks)),
        },
    }
    if return_metadata:
        return blocks, cropped_paths, scores, metadata
    return blocks, cropped_paths, scores


if __name__ == "__main__":
    import sys

    img = sys.argv[1] if len(sys.argv) > 1 else "data/pdf2img/sample/page1.png"
    detect_connected_blocks_v4_recall(img)
