#!/usr/bin/env python3
"""
Stage 1-2-3 Unified v16.10: Final Warm Engine.
Goal: Process all 10 PDFs in one session for maximum speed.
"""
import os
import sys, json, time, numpy as np, cv2, fitz, shutil, warnings, queue
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# Config
YOLO_MODEL_PATH = str(PROJ_ROOT / "artifacts/detector_pivot_yolo_v4_tiles/best.pt")
RENDER_DPI = 300
TILE_SIZE = 320
OVERLAP = 0.52
CONF_THRESH = 0.001 
OCR_BATCH_SIZE = 128

def iou(b1, b2):
    xa, ya, xb, yb = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / u if u > 0 else 0

def gpu_worker(in_q, det_q, task_q, res_q, heartbeat_dict, worker_id):
    import torch
    from ultralytics import YOLO
    from easyocr import Reader
    
    print(f"[GPU {worker_id}] Process entered. Checking CUDA...")
    sys.stdout.flush()
    
    if not torch.cuda.is_available():
        print(f"[GPU {worker_id}] CUDA NOT AVAILABLE. Device count: {torch.cuda.device_count()}")
        try:
            torch.cuda.init()
            print(f"[GPU {worker_id}] CUDA Init Forced.")
        except Exception as e:
            print(f"[GPU {worker_id}] CUDA Forced Init Failed: {e}")
            sys.exit(1)

    print(f"[GPU {worker_id}] CUDA Ready. Loading models...")
    sys.stdout.flush()
    
    model = YOLO(YOLO_MODEL_PATH).to('cuda')
    reader = Reader(['en'], gpu=True, detector="dbnet18", cudnn_benchmark=False)
    print(f"[GPU {worker_id}] Ready.")
    sys.stdout.flush()
    
    def process_ocr():
        while not task_q.empty():
            try:
                batch = task_q.get_nowait()
                if batch:
                    for item in batch:
                        res = reader.readtext(item['img'], paragraph=True)
                        text = " ".join([r[1] for r in res])
                        entry = item['meta'].copy(); entry.update({"ocr_text_raw": text, "status": "ok"})
                        res_q.put(entry)
                    torch.cuda.empty_cache()
            except queue.Empty: break

    while True:
        heartbeat_dict[f"gpu_{worker_id}"] = time.time()
        process_ocr()

        try:
            batch = in_q.get(timeout=0.1)
            if batch is None: 
                process_ocr()
                break
            imgs = [item['tile'] for item in batch]
            results = model.predict(imgs, conf=CONF_THRESH, imgsz=TILE_SIZE, verbose=False, device=0)
            for i, res in enumerate(results):
                meta = batch[i]['meta']
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                dets = []
                for j in range(len(boxes)):
                    box = boxes[j]
                    dets.append({
                        "p_idx": meta[0],
                        "box": [float(box[0]+meta[1]), float(box[1]+meta[2]), float(box[2]+meta[1]), float(box[3]+meta[2])],
                        "conf": float(confs[j])
                    })
                det_q.put({"p_idx": meta[0], "dets": dets, "tile_done": True})
        except queue.Empty: continue
        except Exception as e:
            print(f"[GPU {worker_id}] Error: {e}")
            sys.stdout.flush()
            continue
    print(f"[GPU {worker_id}] Exiting.")
    sys.stdout.flush()

def cpu_producer(pdf_path, p_indices, tile_q, info_q, res_q, heartbeat_dict, worker_id):
    print(f"[CPU {worker_id}] Starting for {pdf_path.name}...")
    doc = fitz.open(str(pdf_path))
    for p_idx in p_indices:
        heartbeat_dict[f"cpu_{worker_id}"] = time.time()
        page = doc[p_idx]; text = page.get_text()
        if sum(1 for c in text if c.isalpha()) > 400:
            res_q.put({"page_index0": p_idx, "bbox_xyxy_norm": [0,0,1,1], "ocr_text_raw": text[:1000], "status": "digital_bypass"})
            info_q.put({"p_idx": p_idx, "is_digital": True}); continue
        pix = page.get_pixmap(dpi=RENDER_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        info_q.put({"p_idx": p_idx, "dim": (img.shape[0], img.shape[1]), "img": img})
        stride = int(TILE_SIZE * (1 - OVERLAP))
        tiles_count = 0; batch = []
        for y in range(0, img.shape[0], stride):
            for x in range(0, img.shape[1], stride):
                y2, x2 = min(y + TILE_SIZE, img.shape[0]), min(x + TILE_SIZE, img.shape[1])
                tile = img[y:y2, x:x2]
                if np.mean(tile) > 245: continue
                if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                    tile = cv2.copyMakeBorder(tile, 0, TILE_SIZE-tile.shape[0], 0, TILE_SIZE-tile.shape[1], cv2.BORDER_CONSTANT, value=[255,255,255])
                batch.append({'tile': tile, 'meta': (p_idx, x, y)})
                tiles_count += 1
                if len(batch) >= 128: tile_q.put(batch); batch = []
        if batch: tile_q.put(batch)
        info_q.put({"p_idx": p_idx, "expected_tiles": tiles_count})
    doc.close()
    print(f"[CPU {worker_id}] Done.")

def manager(num_pages, det_q, info_q, task_q, tile_q, heartbeat_dict):
    page_dets = defaultdict(list); tracker = defaultdict(int); page_expected = {}; page_dims = {}; page_imgs = {}; finished = 0
    while finished < num_pages:
        heartbeat_dict["manager"] = time.time()
        while not info_q.empty():
            try:
                info = info_q.get_nowait()
                if 'is_digital' in info: finished += 1
                if 'dim' in info: page_dims[info['p_idx']] = info['dim']
                if 'img' in info: page_imgs[info['p_idx']] = info['img']
                if 'expected_tiles' in info: page_expected[info['p_idx']] = info['expected_tiles']
            except queue.Empty: break

        try:
            res = det_q.get(timeout=0.1)
            p_idx = res['p_idx']; page_dets[p_idx].extend(res['dets'])
            if res['tile_done']: tracker[p_idx] += 1
            if p_idx in page_expected and tracker[p_idx] == page_expected[p_idx]:
                dets = page_dets[p_idx]; ph, pw = page_dims[p_idx]; refined = []
                for d in sorted(dets, key=lambda x: x["conf"], reverse=True):
                    merged = False
                    for r in refined:
                        if iou(d["box"], r["box"]) > 0.15:
                            r["box"] = [min(d["box"][0], r["box"][0]), min(d["box"][1], r["box"][1]), max(d["box"][2], r["box"][2]), max(d["box"][3], r["box"][3])]; merged = True; break
                    if not merged: refined.append(d)
                img = page_imgs[p_idx]; ocr_batch = []
                for j, r in enumerate(refined):
                    box = r["box"]; crop = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    if crop.size == 0: continue
                    crop = cv2.resize(crop, (int(crop.shape[1] * (128/crop.shape[0])), 128))
                    ocr_batch.append({"img": crop, "meta": {"p_idx": p_idx, "c_idx": j, "bbox_xyxy_norm": [box[0]/pw, box[1]/ph, box[2]/pw, box[3]/ph]}})
                    if len(ocr_batch) >= OCR_BATCH_SIZE: task_q.put(ocr_batch); ocr_batch = []
                if ocr_batch: task_q.put(ocr_batch)
                if p_idx in page_imgs: del page_imgs[p_idx]
                if p_idx in page_dets: del page_dets[p_idx]
                finished += 1
        except queue.Empty: continue
        except Exception as e:
            print(f"[Manager] Error: {e}")
            sys.stdout.flush()
            continue
    print("[Manager] PDF Pages Finished.")
    sys.stdout.flush()

def run_corpus(pdf_dir):
    import torch.multiprocessing as tmp
    pdfs = sorted(list(Path(pdf_dir).glob("*.pdf")))
    start_all = time.time()
    manager_hb = tmp.Manager(); heartbeat_dict = manager_hb.dict()
    
    final_out = PROJ_ROOT / "run_state/ocr_manifest.jsonl"
    with open(final_out, "w") as f: pass

    tile_q = tmp.Queue(maxsize=100); det_q = tmp.Queue(maxsize=500); info_q = tmp.Queue(); task_q = tmp.Queue(maxsize=50); res_q = tmp.Queue()
    gpu_procs = [tmp.Process(target=gpu_worker, args=(tile_q, det_q, task_q, res_q, heartbeat_dict, i)) for i in range(2)]
    for p in gpu_procs: 
        p.start()
        print(f"[Main] Started GPU Worker {p.pid}")
        sys.stdout.flush()

    for pdf in pdfs:
        print(f"\n>>> SENTINEL START: {pdf.name}")
        sys.stdout.flush()
        num_pages = fitz.open(str(pdf)).page_count
        producers = [tmp.Process(target=cpu_producer, args=(pdf, c.tolist(), tile_q, info_q, res_q, heartbeat_dict, i)) for i, c in enumerate(np.array_split(range(num_pages), min(num_pages, 6)))]
        for p in producers: p.start()
        mgr = tmp.Process(target=manager, args=(num_pages, det_q, info_q, task_q, tile_q, heartbeat_dict))
        mgr.start()
        
        with open(final_out, "a") as f:
            while mgr.is_alive() or not res_q.empty():
                try:
                    item = res_q.get(timeout=1)
                    item.update({"pdf_path": str(pdf), "newspaper_name": pdf.stem})
                    f.write(json.dumps(item) + "\n")
                    f.flush()
                except queue.Empty:
                    if not mgr.is_alive() and res_q.empty(): break
                except Exception as e:
                    print(f"[Main] Error draining res_q: {e}"); sys.stdout.flush()
                    
        for p in producers: p.join()
        mgr.join()
        print(f"✅ {pdf.name} Finished.")
        sys.stdout.flush()

    print("\n>>> SHUTTING DOWN PERSISTENT WORKERS <<<")
    for _ in range(2): tile_q.put(None)
    for p in gpu_procs: p.join()
    print(f"\n🏁 Total Time: {time.time()-start_all:.2f}s")

if __name__ == "__main__":
    import torch.multiprocessing as tmp
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    tmp.set_start_method("spawn", force=True)
    run_corpus(args.dir)
