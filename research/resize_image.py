import cv2
import os

def resize_image(input_path, output_path, max_width=1024):
    img = cv2.imread(input_path)
    h, w = img.shape[:2]

    # Scale down only if needed
    if w > max_width:
        scale = max_width / w
        new_dim = (int(w * scale), int(h * scale))
        resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, resized)
        print(f"[✓] Saved resized image to {output_path}")
    else:
        print("[i] Image already small enough, skipping resize.")

if __name__ == "__main__":
    input_file = "data/job_blocks/TH Delhi-30-06-2025_p1_debug.png"
    output_file = "data/job_blocks/TH Delhi-30-06-2025_p1_debug_small.png"
    resize_image(input_file, output_file)
