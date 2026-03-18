import cv2
import numpy as np
import os

def detect_job_blocks(image_path, save_dir="data/job_blocks", debug=True):
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Load image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to highlight borders
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 15
    )

    # Morph to close small white gaps (helps enclose boxes)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / h

        if w > 100 and h > 60 and w < image.shape[1] * 0.95:
            if area > 12000 and 0.25 < aspect < 4.0 and y > 300:
                blocks.append((x, y, w, h))


    # Sort blocks top-to-bottom, then left-to-right
    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

    # Save each block crop
    cropped_paths = []
    for i, (x, y, w, h) in enumerate(blocks):
        crop = image[y:y+h, x:x+w]
        out_path = os.path.join(save_dir, f"{base_name}_block{i}.png")
        cv2.imwrite(out_path, crop)
        cropped_paths.append(out_path)

    # Save debug overlay
    if debug:
        debug_img = image.copy()
        for (x, y, w, h) in blocks:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        debug_path = os.path.join(save_dir, f"{base_name}_debug.png")
        cv2.imwrite(debug_path, debug_img)
        print(f"[✓] Debug image saved: {debug_path}")

    print(f"[✓] Detected {len(blocks)} job blocks:")
    for i, path in enumerate(cropped_paths):
        print(f" - Block {i}: {blocks[i]} → {path}")

    return blocks, cropped_paths


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/pdf2img/page1.png"
    detect_job_blocks(img_path)
