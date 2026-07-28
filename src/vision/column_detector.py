import cv2
import numpy as np
import os

def detect_columns(image_path, save_dir="data/column_images", debug=True):
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Load image and preprocess
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 15
    )

    # Sum white pixels vertically
    vertical_proj = np.sum(thresh, axis=0)  # shape: (width,)

    # Normalize
    norm = (vertical_proj - np.min(vertical_proj)) / (np.max(vertical_proj) - np.min(vertical_proj) + 1e-5)

    # Threshold: columns where pixel density is low (i.e., whitespace)
    is_gap = norm < 0.05

    # Scan across x-axis to detect solid blocks
    column_boxes = []
    in_column = False
    start = 0

    for x in range(len(is_gap)):
        if not is_gap[x] and not in_column:
            start = x
            in_column = True
        elif is_gap[x] and in_column:
            end = x
            if end - start > 60:  # Only keep columns wider than 60px
                column_boxes.append((start, end))
            in_column = False

    if not column_boxes:
        print("[!] No vertical columns detected, falling back to full width")
        column_boxes = [(0, image.shape[1])]

    # Save column crops
    cropped_paths = []
    for i, (x1, x2) in enumerate(column_boxes):
        col_img = image[:, x1:x2]
        out_path = os.path.join(save_dir, f"{base_name}_col{i}.png")
        cv2.imwrite(out_path, col_img)
        cropped_paths.append(out_path)

    # Save debug overlay
    if debug:
        debug_img = image.copy()
        for (x1, x2) in column_boxes:
            cv2.rectangle(debug_img, (x1, 0), (x2, image.shape[0]), (0, 255, 0), 3)
        debug_path = os.path.join(save_dir, f"{base_name}_debug.png")
        cv2.imwrite(debug_path, debug_img)
        print(f"[✓] Debug image saved: {debug_path}")

    print(f"Detected {len(column_boxes)} columns:")
    for i, path in enumerate(cropped_paths):
        print(f" - {column_boxes[i]} → {path}")

    return column_boxes, cropped_paths


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/pdf2img/page1.png"
    detect_columns(image_path)
