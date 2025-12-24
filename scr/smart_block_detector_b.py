import cv2
import numpy as np
from pathlib import Path

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

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morphed, connectivity=8)

    blocks = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        aspect = w / h if h > 0 else 0

        # Earlier filtering logic — slightly stricter than current
        if w > 80 and h > 40 and area > 5000 and 0.1 < aspect < 10:
            blocks.append((x, y, w, h))

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
