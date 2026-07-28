import cv2
import os
import json
from pathlib import Path


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _as_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


DETECTION_PARAM_DEFAULTS = {
    "contour_area_min": 1500,
    "contour_area_max": 500000,
    "aspect_ratio_min": 0.5,
    "aspect_ratio_max": 5.0,
    "block_merge_distance": 40,
    "morphology_kernel_size": 5,
}

DETECTION_PARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "detection_params.json"


def _load_detection_params_once():
    params = dict(DETECTION_PARAM_DEFAULTS)
    if DETECTION_PARAMS_PATH.exists():
        try:
            loaded = json.loads(DETECTION_PARAMS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key in params:
                    if key in loaded:
                        params[key] = loaded[key]
        except Exception:
            pass

    # Allow stage02 to export overrides to worker processes.
    params["contour_area_min"] = _as_int(
        os.getenv("DETECTOR_CONTOUR_AREA_MIN", params["contour_area_min"]),
        params["contour_area_min"],
    )
    params["contour_area_max"] = _as_int(
        os.getenv("DETECTOR_CONTOUR_AREA_MAX", params["contour_area_max"]),
        params["contour_area_max"],
    )
    params["aspect_ratio_min"] = _as_float(
        os.getenv("DETECTOR_ASPECT_RATIO_MIN", params["aspect_ratio_min"]),
        params["aspect_ratio_min"],
    )
    params["aspect_ratio_max"] = _as_float(
        os.getenv("DETECTOR_ASPECT_RATIO_MAX", params["aspect_ratio_max"]),
        params["aspect_ratio_max"],
    )
    params["block_merge_distance"] = _as_int(
        os.getenv("DETECTOR_BLOCK_MERGE_DISTANCE", params["block_merge_distance"]),
        params["block_merge_distance"],
    )
    params["morphology_kernel_size"] = _as_int(
        os.getenv("DETECTOR_MORPHOLOGY_KERNEL_SIZE", params["morphology_kernel_size"]),
        params["morphology_kernel_size"],
    )

    # Backward-compatible env overrides.
    params["contour_area_min"] = _as_int(os.getenv("DETECTOR_MIN_AREA", params["contour_area_min"]), params["contour_area_min"])
    params["aspect_ratio_min"] = _as_float(os.getenv("DETECTOR_ASPECT_MIN", params["aspect_ratio_min"]), params["aspect_ratio_min"])
    params["aspect_ratio_max"] = _as_float(os.getenv("DETECTOR_ASPECT_MAX", params["aspect_ratio_max"]), params["aspect_ratio_max"])
    params["morphology_kernel_size"] = _as_int(os.getenv("DETECTOR_KERNEL_SIZE", params["morphology_kernel_size"]), params["morphology_kernel_size"])

    params["contour_area_min"] = max(1, int(params["contour_area_min"]))
    params["contour_area_max"] = max(params["contour_area_min"] + 1, int(params["contour_area_max"]))
    params["aspect_ratio_min"] = max(0.01, float(params["aspect_ratio_min"]))
    params["aspect_ratio_max"] = max(params["aspect_ratio_min"] + 0.01, float(params["aspect_ratio_max"]))
    params["block_merge_distance"] = max(0, int(params["block_merge_distance"]))
    kernel_size = max(1, int(params["morphology_kernel_size"]))
    if kernel_size % 2 == 0:
        kernel_size += 1
    params["morphology_kernel_size"] = kernel_size
    return params


DETECTION_PARAMS = _load_detection_params_once()

MORPH_KERNEL_SIZE = int(DETECTION_PARAMS["morphology_kernel_size"])
CONTOUR_AREA_MIN = int(DETECTION_PARAMS["contour_area_min"])
CONTOUR_AREA_MAX = int(DETECTION_PARAMS["contour_area_max"])
ASPECT_RATIO_MIN = float(DETECTION_PARAMS["aspect_ratio_min"])
ASPECT_RATIO_MAX = float(DETECTION_PARAMS["aspect_ratio_max"])
BLOCK_MERGE_DISTANCE = int(DETECTION_PARAMS["block_merge_distance"])

MORPH_ITERATIONS = _env_int("DETECTOR_MORPH_ITERATIONS", 2)
MIN_WIDTH = _env_int("DETECTOR_MIN_WIDTH", 80)
MIN_HEIGHT = _env_int("DETECTOR_MIN_HEIGHT", 40)


def _should_merge(a, b, max_gap):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x_overlap = min(ax2, bx2) - max(ax1, bx1)
    y_overlap = min(ay2, by2) - max(ay1, by1)
    x_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
    y_gap = max(0, max(ay1, by1) - min(ay2, by2))

    return (x_overlap > 0 and y_gap <= max_gap) or (y_overlap > 0 and x_gap <= max_gap)


def _merge_boxes(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (
        min(ax1, bx1),
        min(ay1, by1),
        max(ax2, bx2),
        max(ay2, by2),
    )


def _merge_nearby_blocks(blocks, max_gap):
    if max_gap <= 0 or len(blocks) < 2:
        return blocks

    boxes = [(x, y, x + w, y + h) for (x, y, w, h) in blocks]
    changed = True

    while changed and len(boxes) > 1:
        changed = False
        used = [False] * len(boxes)
        merged = []

        for i, base in enumerate(boxes):
            if used[i]:
                continue

            current = base
            used[i] = True
            expanded = True
            while expanded:
                expanded = False
                for j, other in enumerate(boxes):
                    if used[j]:
                        continue
                    if _should_merge(current, other, max_gap):
                        current = _merge_boxes(current, other)
                        used[j] = True
                        expanded = True
                        changed = True

            merged.append(current)
        boxes = merged

    merged_blocks = []
    for x1, y1, x2, y2 in boxes:
        merged_blocks.append((x1, y1, x2 - x1, y2 - y1))
    return merged_blocks


def detect_connected_blocks(image_path, save_base_dir="data/job_blocks_smart", debug=True):
    image_path = Path(image_path)
    pdf_folder = image_path.parent.name
    base_name = image_path.stem

    # Organized output
    save_dir = Path(save_base_dir) / pdf_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[!] Failed to load: {image_path}")
        return [], []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 15
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    morphed = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=max(1, MORPH_ITERATIONS),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morphed, connectivity=8)

    blocks = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        aspect = w / h if h > 0 else 0

        if (
            w > MIN_WIDTH
            and h > MIN_HEIGHT
            and CONTOUR_AREA_MIN <= area <= CONTOUR_AREA_MAX
            and ASPECT_RATIO_MIN < aspect < ASPECT_RATIO_MAX
        ):
            blocks.append((x, y, w, h))

    blocks = _merge_nearby_blocks(blocks, BLOCK_MERGE_DISTANCE)
    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # top-bottom, left-right

    cropped_paths = []
    for i, (x, y, w, h) in enumerate(blocks):
        crop = image[y:y+h, x:x+w]
        out_path = save_dir / f"{base_name}_block{i}.png"
        cv2.imwrite(str(out_path), crop)
        cropped_paths.append(str(out_path))

    if debug:
        debug_img = image.copy()
        for (x, y, w, h) in blocks:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if "_p" in base_name:
            pnum = base_name.split("_p")[-1]
            debug_name = f"debug_p{pnum}.png"
        else:
            debug_name = f"{base_name}_debug.png"

        debug_path = save_dir / debug_name
        cv2.imwrite(str(debug_path), debug_img)
        print(f"[✓] Debug image saved: {debug_path}")

    print(f"[✓] Detected {len(blocks)} legacy smart blocks from page {base_name}")
    return blocks, cropped_paths


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "data/pdf2img/sample/page1.png"
    detect_connected_blocks(img)
