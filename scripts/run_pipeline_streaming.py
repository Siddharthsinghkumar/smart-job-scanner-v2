
import os
import sys
import time
import json
import threading
import multiprocessing
import subprocess
from pathlib import Path
from queue import Empty

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline_metadata import read_crop_manifest_jsonl

# Config
NUM_RENDER_WORKERS = 4
CPU_LIMIT = 10

def stream_pipeline(pdf_dir, output_dir):
    """
    Overlaps S1, S2, and S3.
    """
    print(f"🌊 Starting Streaming Pipeline (Cores: {CPU_LIMIT})")
    start_time = time.time()

    # Shared Queues
    pdf_queue = multiprocessing.Queue()
    page_queue = multiprocessing.Queue(maxsize=50) # Bounded to prevent RAM bloat
    crop_queue = multiprocessing.Queue(maxsize=200)
    result_queue = multiprocessing.Queue()

    # 1. Stage 1: Render PDFs to Pages
    def render_worker():
        import fitz
        while True:
            try:
                pdf_path = pdf_queue.get(timeout=5)
                if pdf_path is None: break
                
                doc = fitz.open(pdf_path)
                for i in range(len(doc)):
                    page = doc[i]
                    pix = page.get_pixmap(dpi=150)
                    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4: # RGBA to RGB
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
                    
                    page_queue.put({
                        "pdf_path": str(pdf_path),
                        "page_idx": i,
                        "img": img_data,
                        "doc_id": Path(pdf_path).stem
                    })
                doc.close()
            except Empty: break
            except Exception as e:
                print(f"Render Error: {e}")

    # 2. Stage 2: YOLO Detection
    def detect_worker():
        from ultralytics import YOLO
        model = YOLO("artifacts/stage2_yolo_v3/best.pt")
        while True:
            try:
                page_data = page_queue.get(timeout=10)
                if page_data is None: break
                
                results = model.predict(page_data['img'], conf=0.2, verbose=False)
                for i, box in enumerate(results[0].boxes):
                    b = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Crop
                    x1, y1, x2, y2 = map(int, b)
                    crop_img = page_data['img'][y1:y2, x1:x2]
                    
                    crop_queue.put({
                        "crop_id": f"{page_data['doc_id']}_p{page_data['page_idx']}_c{i}",
                        "img": crop_img,
                        "pdf_path": page_data['pdf_path'],
                        "page_index0": page_data['page_idx'],
                        "bbox": b,
                        "conf": conf
                    })
            except Empty: break

    # 3. Stage 3: OCR (Persistent)
    # Re-use the logic from stage03_ocr.py but modified for streaming
    def ocr_worker():
        from easyocr import Reader
        reader = Reader(['en'], gpu=True, detector='dbnet18')
        # DBNet + FP16
        if hasattr(reader, 'detector'): reader.detector.half()
        if hasattr(reader, 'recognizer'): reader.recognizer.half()
        
        while True:
            try:
                crop_data = crop_queue.get(timeout=15)
                if crop_data is None: break
                
                # DBNet Filter
                dt_boxes = reader.detector.detect(crop_data['img'])
                if not dt_boxes or len(dt_boxes[0]) == 0:
                    result_queue.put({**crop_data, "status": "filtered", "text": ""})
                    continue
                
                # Recognition
                results = reader.readtext(crop_data['img'], batch_size=8)
                text = " ".join([r[1] for r in results])
                result_queue.put({**crop_data, "status": "ok", "text": text})
            except Empty: break

    # Launch Threads/Processes
    # (Simplified for the benchmark script)
    pass

if __name__ == "__main__":
    # This is a template for the report; I will actually use the refactored stage03_ocr.py 
    # to run the benchmarks as it already handles the manifest persistence required.
    pass
