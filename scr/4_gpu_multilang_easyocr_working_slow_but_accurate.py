#!/usr/bin/env python3
"""
gpu_multilang_easyocr_working_slow_but_accurate.py

GPU-dominant OCR pipeline with dedicated local queue for GPU and simplified CPU workers.
Eliminates queue contention and startup delays.
"""

import os
import re
import torch
import easyocr
import logging
import cv2
import numpy as np
import gc
import threading
import queue
import time
import psutil
import multiprocessing as mp
import signal
import hashlib
import tempfile
import traceback
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple, Dict
from contextlib import contextmanager

# ───── NVIDIA ML Import with Fallback (DEFERRED INIT) ─────
try:
    import pynvml as nvidia_smi
    HAS_NVIDIA = True
except ImportError:
    try:
        import nvidia_ml_py as nvidia_smi
        HAS_NVIDIA = True
    except ImportError:
        HAS_NVIDIA = False
        nvidia_smi = None
        # Defer logger configuration to avoid early use

# ───── Config ─────
DEFAULT_LANGS = ["en", "hi"]
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
GPU_WORKERS = 1  # ✅ Single GPU worker - EasyOCR uses single CUDA context
CPU_WORKERS = 4  # ✅ Reduced for stability - 4 cores instead of 8
CACHE_DIR = Path("cache_blocks")
GPU_MEMORY_LIMIT_MB = 3800  # ✅ Fully utilize 3050 Ti VRAM

# ───── New Directory Structure ─────
PROGRESS_DIR = Path("data/progress")
BLOCK_TEXT_DIR_ROOT = Path("data/block_texts")
PAGE_TEXT_DIR_ROOT = Path("data/page_texts")
for d in (PROGRESS_DIR, BLOCK_TEXT_DIR_ROOT, PAGE_TEXT_DIR_ROOT):
    d.mkdir(parents=True, exist_ok=True)

# ───── Utility Helpers ─────
def atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False, mode="w", encoding="utf-8") as tf:
        tf.write(text or "")
        tmp = Path(tf.name)
    os.replace(str(tmp), str(path))

def atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False, mode="w", encoding="utf-8") as tf:
        json.dump(data, tf, ensure_ascii=False, indent=2)
        tmp = Path(tf.name)
    os.replace(str(tmp), str(path))

def load_progress(newspaper_name: str) -> dict:
    p = PROGRESS_DIR / f"{newspaper_name}_progress.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_progress(newspaper_name: str, data: dict):
    p = PROGRESS_DIR / f"{newspaper_name}_progress.json"
    atomic_write_json(p, data)

def reset_progress_and_outputs(newspaper_name: str):
    """Delete progress & outputs for a newspaper to force fresh processing"""
    # Delete progress file
    progress_file = PROGRESS_DIR / f"{newspaper_name}_progress.json"
    if progress_file.exists():
        progress_file.unlink()
        logger.info(f"🧹 Deleted progress file: {progress_file}")
    
    # Delete block texts
    block_text_dir = BLOCK_TEXT_DIR_ROOT / newspaper_name
    if block_text_dir.exists():
        shutil.rmtree(block_text_dir)
        logger.info(f"🧹 Deleted block texts: {block_text_dir}")
    
    # Delete page texts
    page_text_dir = PAGE_TEXT_DIR_ROOT / newspaper_name
    if page_text_dir.exists():
        shutil.rmtree(page_text_dir)
        logger.info(f"🧹 Deleted page texts: {page_text_dir}")

# ───── Context Manager for Locks ─────
@contextmanager
def locked(lock):
    lock.acquire()
    try:
        yield
    finally:
        lock.release()

# ───── Global pointers for signal handler ─────
CURRENT_CPU_QUEUE = None          # will be set by GPUDominantOCR.process_folder while running
ACTIVE_CPU_PROCESS_PIDS = []      # list of started CPU pids for quick termination
_SIGNAL_RECEIVED = False          # ✅ FIX 1: Global flag for signal detection

# ───── CUDA Optimization Settings ─────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Enable Tensor Cores and optimizations
torch.backends.cudnn.benchmark = True
# ✅ FIX 4: Guard TF32 settings for older PyTorch versions
if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
    torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(1)

# ───── Global State ─────
reader_gpu_global = None

# ───── Runtime counters ─────
COUNTERS = {
    "total_newspapers": 0,
    "total_pages": 0,
    "total_blocks": 0,
    "gpu_blocks": 0,
    "cpu_blocks": 0,
    "gpu_success": 0,
    "cpu_success": 0,
    "gpu_fail": 0,
    "cpu_fail": 0,
    "gpu_oom_recovered": 0,
    "load_balance_moves": 0,
}

# ───── Logging Setup ─────
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_name = f"easyocr_gpu_dominant_{timestamp}.log"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

fh = logging.FileHandler(log_dir / log_name, mode="a", encoding="utf-8")
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# ✅ FIX 3: Better logging handler check - look for specific handler types
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(fh)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(ch)

# ───── NVIDIA ML Initialization (AFTER LOGGER) ─────
GPU_HANDLE = None
if HAS_NVIDIA and nvidia_smi is not None:
    try:
        nvidia_smi.nvmlInit()
        GPU_HANDLE = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        logger.info("🔎 NVML initialized and GPU handle acquired")
    except Exception as e:
        logger.warning(f"⚠️ NVML init failed: {e} — GPU monitoring disabled")
        HAS_NVIDIA = False
        GPU_HANDLE = None

# ───── Cache Path Helper ─────
def _cache_for_path(img_path: Path):
    """Generate unique cache filename using hash of full path to avoid collisions"""
    h = hashlib.blake2b(str(img_path).encode('utf-8'), digest_size=8).hexdigest()
    return CACHE_DIR / f"{img_path.stem}_{h}.npy"

# ───── GPU Memory Management ─────
def clear_gpu_cache():
    """Clear GPU memory to prevent fragmentation"""
    torch.cuda.empty_cache()
    gc.collect()

def get_gpu_memory_used():
    """Get current GPU memory usage in MB"""
    if not HAS_NVIDIA or GPU_HANDLE is None:
        return 0
    try:
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
        return mem.used // (1024 * 1024)
    except Exception as e:
        logger.warning(f"Failed to get GPU memory: {e}")
        return 0

def get_gpu_utilization():
    """Get GPU utilization percentage"""
    if not HAS_NVIDIA or GPU_HANDLE is None:
        return 0
    try:
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(GPU_HANDLE)
        return util.gpu
    except Exception as e:
        logger.warning(f"Failed to get GPU utilization: {e}")
        return 0

def get_cpu_utilization():
    """Get CPU utilization percentage"""
    return psutil.cpu_percent(interval=0.1)

# ───── Image Preprocessing ─────
def preprocess_image(img_path: Path, use_cache: bool = True) -> np.ndarray:
    """Load + resize image with cache support"""
    try:
        cache_path = _cache_for_path(img_path)

        if use_cache and cache_path.exists():
            return np.load(cache_path)

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        target_h = 768
        scale = target_h / h if h > target_h else 1.0
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if use_cache:
            # ✅ FIX 3: Safer temporary .npy cache write using filename
            with tempfile.NamedTemporaryFile(dir=str(CACHE_DIR), suffix='.npy', delete=False) as tf:
                tmp_name = tf.name
            # write with filename string (more portable)
            np.save(tmp_name, image)
            tmp_path = Path(tmp_name)
            
            try:
                os.replace(str(tmp_path), cache_path)
            except FileNotFoundError:
                if cache_path.exists():
                    return np.load(cache_path)
                else:
                    raise
        
        return image

    except Exception as e:
        logger.error(f"Error preprocessing {img_path.name}: {e}")
        raise

def get_image_pixel_count(img_path: Path) -> int:
    """Get image pixel count for intelligent queue assignment"""
    try:
        # Try to read image dimensions without loading full image
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            h, w = image.shape[:2]
            return h * w
        return 0
    except Exception:
        return 0

# ───── Queue-based Result Writer ─────
def result_writer_from_queue(newspaper_name: str, all_img_paths: List[Path], results_queue, stop_event: threading.Event):
    """
    Queue-based result writer - consumes from results_queue and writes files.
    Writes per-block results to disk but only emits INFO when a full page is complete.
    """
    page_map = defaultdict(list)
    for p in all_img_paths:
        m = re.search(r"_p(\d+)", p.name)
        pg = m.group(1) if m else "0"
        page_map[pg].append(p.name)

    prog = load_progress(newspaper_name)
    processed = set()
    stream_logger = logging.getLogger(f"writer_{newspaper_name}")
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stream_logger.addHandler(sh)
    stream_logger.setLevel(logging.INFO)

    def page_complete_check(pg):
        txt_dir = BLOCK_TEXT_DIR_ROOT / newspaper_name
        for fname in page_map[pg]:
            block_file = txt_dir / (fname + ".txt")
            # Only consider blocks that should have produced output (not failed/empty)
            if block_file.exists():
                # File exists and has content
                if block_file.stat().st_size > 0:
                    continue
            # Check if it's marked as failed in progress
            block_key = str(next(p for p in all_img_paths if p.name == fname).resolve())
            if block_key in prog and prog[block_key].get("status") == "failed":
                continue
            # Otherwise, this block is not complete
            return False
        return True

    stream_logger.info(f"📝 Queue-based writer started for {newspaper_name}")

    while True:
        try:
            item = results_queue.get(timeout=0.5)
        except queue.Empty:
            if stop_event.is_set():
                break
            continue
        except Exception as e:
            stream_logger.warning(f"Unexpected queue exception while reading results: {e}")
            if stop_event.is_set():
                break
            continue

        if item is None:  # Poison pill
            break

        try:
            path_str, success, fail, text = item
        except Exception:
            stream_logger.error(f"Writer received malformed item: {item}")
            continue

        # Meta messages for worker init/errors
        if isinstance(path_str, str) and path_str.startswith("_meta_"):
            stream_logger.error(f"Worker error ({path_str}):\n{text}")
            prog[path_str] = {"status": "worker_error", "error": text, "updated_at": datetime.utcnow().isoformat()}
            save_progress(newspaper_name, prog)
            continue

        # Normal block result — write file and update progress (no INFO per-block)
        try:
            pth = Path(path_str)
            outp = BLOCK_TEXT_DIR_ROOT / newspaper_name / (pth.name + ".txt")
            clean_text = (text or "").strip()
            if clean_text:
                # only create a .txt if we have useful content
                atomic_write_text(outp, clean_text)
                status = "ok"
                prog[str(pth.resolve())] = {"status": status, "text_file": str(outp), "updated_at": datetime.utcnow().isoformat()}
            else:
                # failed or empty: don't create an empty file (prevents false resume)
                status = "failed"
                # record failure in progress but no text_file path
                prog[str(pth.resolve())] = {"status": status, "text_file": "", "updated_at": datetime.utcnow().isoformat()}
            save_progress(newspaper_name, prog)
            processed.add(path_str)

            # Per-block log is now DEBUG (won't appear in normal INFO output)
            stream_logger.debug(f"[{pth.name}] Status: {status} - written")

            # Check if page is complete; when it is, emit a single INFO with counts
            m = re.search(r"_p(\d+)", pth.name)
            pg = m.group(1) if m else "0"
            if page_complete_check(pg):
                page_key = f"_page_{pg}_done"
                if not prog.get(page_key):
                    # compute success/total counts for the page
                    txt_dir = BLOCK_TEXT_DIR_ROOT / newspaper_name
                    total_blocks = len(page_map[pg])
                    success_count = 0
                    for fname in page_map[pg]:
                        tf = txt_dir / (fname + ".txt")
                        if tf.exists() and tf.stat().st_size > 0:
                            success_count += 1
                    failed_count = total_blocks - success_count

                    # mark progress and save
                    prog[page_key] = {
                        "done": True,
                        "when": datetime.utcnow().isoformat(),
                        "blocks_total": total_blocks,
                        "blocks_success": success_count,
                        "blocks_failed": failed_count,
                    }
                    save_progress(newspaper_name, prog)

                    # Emit the single page-level INFO the user wants
                    stream_logger.info(f"✅ {newspaper_name} p{pg} done ({success_count}/{total_blocks} blocks)")

        except Exception as e:
            stream_logger.error(f"Writer error for {item}: {e}")

    stream_logger.info("Result writer stopping")
    stream_logger.removeHandler(sh)

# ───── Simplified CPU worker for multiprocessing ─────
def cpu_worker_main(cpu_task_queue, results_queue, shared_lock, shared_counters, langs, worker_id, shutdown_event=None, use_cache=True):
    """
    Simplified CPU worker - just processes tasks from shared queue until empty.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # hide GPU
    # Remove monkeypatching - only use environment variable
    torch.set_num_threads(1)  # Reduce thread oversubscription

    # Reconfigure logging for child process to ensure proper log handling
    worker_logger = logging.getLogger()
    # Clear existing handlers to avoid duplication
    for handler in worker_logger.handlers[:]:
        worker_logger.removeHandler(handler)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - CPU%(processName)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    worker_logger.addHandler(handler)
    worker_logger.setLevel(logging.INFO)

    try:
        cpu_reader = easyocr.Reader(langs, gpu=False)
    except Exception as e:
        tb = traceback.format_exc()
        worker_logger.error(f"❌ CPU{worker_id} failed to init easyocr.Reader: {e}\n{tb}")
        # Send error as a result
        try:
            results_queue.put(("_meta_" + str(os.getpid()), 0, 1, tb))
        except Exception:
            pass
        return

    worker_logger.info(f"🔧 CPU worker {worker_id} started (pid={os.getpid()})")

    task_counter = 0
    try:
        while not (shutdown_event and shutdown_event.is_set()):
            try:
                img_path = cpu_task_queue.get(timeout=0.5)
                if img_path is None:  # Poison pill
                    break
            except (queue.Empty, EOFError, OSError):
                if shutdown_event and shutdown_event.is_set():
                    break
                continue

            try:
                image = preprocess_image(Path(img_path), use_cache=use_cache)
                result = cpu_reader.readtext(image, detail=0, paragraph=True)
                text = "\n".join([ln.strip() for ln in result if ln and str(ln).strip()])
                success = (1, 0) if text else (0, 1)
                
                # ✅ FIX 1: Atomic counter updates with lock
                with locked(shared_lock):
                    shared_counters["total_blocks"] += 1
                    if success[0]:
                        shared_counters["cpu_success"] += 1
                    else:
                        shared_counters["cpu_fail"] += 1
                    shared_counters["cpu_blocks"] += 1
                
                # ✅ FIX 2: Use queue for results
                results_queue.put((str(img_path), success[0], success[1], text))
                
                task_counter += 1
                
                # Lightweight real-time progress
                if task_counter % 20 == 0:
                    with locked(shared_lock):
                        done = shared_counters["total_blocks"]
                    worker_logger.info(f"[progress] {done} blocks done...")
                    
            except Exception as e:
                tb = traceback.format_exc()
                worker_logger.error(f"❌ CPU{worker_id} error on {img_path}: {e}\n{tb}")
                success, text = (0, 1), ""
                
                # ✅ FIX 1: Atomic counter updates with lock
                with locked(shared_lock):
                    shared_counters["total_blocks"] += 1
                    shared_counters["cpu_fail"] += 1
                    shared_counters["cpu_blocks"] += 1
                
                # ✅ FIX 2: Use queue for results
                results_queue.put((str(img_path), 0, 1, ""))

    except Exception as e:
        tb = traceback.format_exc()
        worker_logger.error(f"❌ CPU{worker_id} fatal error: {e}\n{tb}")

    worker_logger.info(f"🔚 CPU worker {worker_id} exiting (pid={os.getpid()})")

# ───── GPU-Dominant OCR System ─────
class GPUDominantOCR:
    def __init__(self, gpu_reader, langs, shutdown_event=None, use_cache=True):
        # 🔧 FIX: Separate queues - local for GPU, multiprocessing for CPU
        self.gpu_task_queue = queue.Queue(maxsize=1024)  # ✅ Added backpressure for large inputs
        # create cpu_task_queue later with a safe context (spawn). Avoid creating
        # multiprocessing queues before threading/fork interactions.
        self.cpu_task_queue = None
        self.results_queue = None
        self.shared_lock = None     
        self.shared_counters = None  
        self.shutdown_event = shutdown_event or threading.Event()
        self.writer_stop_event = threading.Event()
        self.gpu_reader = gpu_reader
        self.langs = langs
        self.gpu_spilled_once = set()  # ✅ Track spilled images to prevent infinite loops
        self.cpu_ok = True  # Assume CPU is OK by default
        self.writer_thread = None
        self.use_cache = use_cache
        
        # ✅ FIX 5: Check shutdown_event type
        if hasattr(shutdown_event, 'is_set') and not hasattr(shutdown_event, 'wait'):
            # quick heuristic — mp.Event has different semantics; warn but continue
            logger.debug("GPUDominantOCR: got shutdown_event (ensure it is a multiprocessing.Event when used with CPU workers)")
        
    def gpu_worker(self, worker_id):
        """GPU worker with dedicated local queue and OOM spillover"""
        logger.info(f"🔧 Starting GPU worker {worker_id}")
        task_counter = 0
        
        while not self.shutdown_event.is_set():
            try:
                # ✅ FIX: Use local queue with timeout for clean shutdown
                img_path = self.gpu_task_queue.get(timeout=1)
                
                # 🔥 CRITICAL FIX: Check for sentinel None to avoid deadlock
                if img_path is None:
                    logger.info(f"🔚 GPU{worker_id} received sentinel — exiting")
                    break
                    
            except queue.Empty:
                continue
                
            try:
                # Periodic GPU memory refresh every 50 tasks
                if task_counter > 0 and task_counter % 50 == 0:
                    logger.debug(f"🧹 GPU{worker_id}: Periodic memory refresh after {task_counter} tasks")
                    clear_gpu_cache()
                
                # Check memory before processing
                current_mem = get_gpu_memory_used()
                if current_mem > GPU_MEMORY_LIMIT_MB:
                    # ✅ Canonicalize keys to prevent Path vs str mismatches
                    key = str(img_path)
                    if key in self.gpu_spilled_once:
                        # already tried spilling once, mark failed to avoid loop
                        logger.warning(f"⚠️ GPU{worker_id} persistent OOM on {Path(img_path).name}, marking as failed")
                        # ✅ FIX 1: Atomic counter updates with lock
                        with locked(self.shared_lock):
                            self.shared_counters["gpu_fail"] += 1
                            self.shared_counters["total_blocks"] += 1
                        # ✅ FIX 2: Use queue for results
                        self.results_queue.put((str(img_path), 0, 1, ""))
                        continue
                    else:
                        logger.warning(f"⚠️ GPU {worker_id} near OOM ({current_mem}MB), attempting spill to CPU")
                        self.gpu_spilled_once.add(key)
                        
                        # PATCH B: Check if CPU is healthy before attempting spill
                        spilled_to_cpu = False
                        if getattr(self, "cpu_ok", True):
                            try:
                                self.cpu_task_queue.put_nowait(img_path)
                                spilled_to_cpu = True
                            except Exception as e:
                                spilled_to_cpu = False
                        else:
                            spilled_to_cpu = False

                        if not spilled_to_cpu:
                            # immediate GPU downscale retry (avoid waiting for CPU)
                            logger.debug("CPU not available for spill — attempting downscaled GPU retry")
                            try:
                                # downscale image more aggressively and retry on GPU once
                                img = preprocess_image(Path(img_path), use_cache=self.use_cache)
                                h, w = img.shape[:2]
                                small = cv2.resize(img, (max(1, w//2), max(1, h//2)), interpolation=cv2.INTER_AREA)
                                result = self.gpu_reader.readtext(small, detail=1, paragraph=True)
                                
                                # Extract text
                                if result and isinstance(result, list) and len(result) > 0:
                                    if isinstance(result[0], tuple) and len(result[0]) >= 2:
                                        text_lines = [item[1] for item in result if item[1].strip()]
                                    else:
                                        text_lines = [str(item).strip() for item in result if str(item).strip()]
                                else:
                                    text_lines = []
                                
                                text = "\n".join(text_lines).strip()
                                success = (1, 0) if text else (0, 1)
                                
                                # ✅ FIX 1: Atomic counter updates with lock
                                with locked(self.shared_lock):
                                    self.shared_counters["total_blocks"] += 1
                                    if success[0]:
                                        self.shared_counters["gpu_success"] += 1
                                    else:
                                        self.shared_counters["gpu_fail"] += 1
                                    self.shared_counters["gpu_blocks"] += 1
                                # ✅ FIX 2: Use queue for results
                                self.results_queue.put((str(img_path), success[0], success[1], text))
                                
                                logger.debug(f"✅ GPU{worker_id}: Processed downscaled {Path(img_path).name}")
                                
                            except Exception as retry_e:
                                # last resort: mark as failed immediately
                                logger.error(f"❌ GPU{worker_id} failed on downscaled {Path(img_path).name}: {retry_e}")
                                # ✅ FIX 1: Atomic counter updates with lock
                                with locked(self.shared_lock):
                                    self.shared_counters["gpu_fail"] += 1
                                    self.shared_counters["total_blocks"] += 1
                                # ✅ FIX 2: Use queue for results
                                self.results_queue.put((str(img_path), 0, 1, ""))
                        
                        # ✅ FIX 1: Atomic counter updates with lock
                        with locked(self.shared_lock):
                            self.shared_counters["gpu_oom_recovered"] += 1
                        clear_gpu_cache()
                        continue
                
                # Process image
                image = preprocess_image(Path(img_path), use_cache=self.use_cache)
                result = self.gpu_reader.readtext(image, detail=1, paragraph=True)
                
                # Extract text
                if result and isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], tuple) and len(result[0]) >= 2:
                        text_lines = [item[1] for item in result if item[1].strip()]
                    else:
                        text_lines = [str(item).strip() for item in result if str(item).strip()]
                else:
                    text_lines = []
                
                text = "\n".join(text_lines).strip()
                success = (1, 0) if text else (0, 1)
                
                # ✅ FIX 1: Atomic counter updates with lock
                with locked(self.shared_lock):
                    self.shared_counters["total_blocks"] += 1
                    if success[0]:
                        self.shared_counters["gpu_success"] += 1
                    else:
                        self.shared_counters["gpu_fail"] += 1
                    self.shared_counters["gpu_blocks"] += 1
                # ✅ FIX 2: Use queue for results
                self.results_queue.put((str(img_path), success[0], success[1], text))
                
                task_counter += 1
                
                # Lightweight real-time progress
                if task_counter % 20 == 0:
                    with locked(self.shared_lock):
                        done = self.shared_counters["total_blocks"]
                    logger.info(f"[progress] {done} blocks done...")
                
                logger.debug(f"✅ GPU{worker_id}: Processed {Path(img_path).name} (task #{task_counter})")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "OOM" in str(e):
                    # ✅ Canonicalize keys to prevent Path vs str mismatches
                    key = str(img_path)
                    if key in self.gpu_spilled_once:
                        # already tried spilling once, mark failed to avoid loop
                        logger.warning(f"⚠️ GPU{worker_id} persistent OOM on {Path(img_path).name}, marking as failed")
                        # ✅ FIX 1: Atomic counter updates with lock
                        with locked(self.shared_lock):
                            self.shared_counters["gpu_fail"] += 1
                            self.shared_counters["total_blocks"] += 1
                        # ✅ FIX 2: Use queue for results
                        self.results_queue.put((str(img_path), 0, 1, ""))
                        continue
                    else:
                        logger.warning(f"⚠️ GPU{worker_id} OOM on {Path(img_path).name}, attempting spill to CPU")
                        self.gpu_spilled_once.add(key)
                        
                        # PATCH B: Check if CPU is healthy before attempting spill
                        spilled_to_cpu = False
                        if getattr(self, "cpu_ok", True):
                            try:
                                self.cpu_task_queue.put_nowait(img_path)
                                spilled_to_cpu = True
                            except Exception as spill_e:
                                spilled_to_cpu = False
                        else:
                            spilled_to_cpu = False

                        if not spilled_to_cpu:
                            # immediate GPU downscale retry (avoid waiting for CPU)
                            logger.debug("CPU not available for spill — attempting downscaled GPU retry")
                            try:
                                # downscale image more aggressively and retry on GPU once
                                img = preprocess_image(Path(img_path), use_cache=self.use_cache)
                                h, w = img.shape[:2]
                                small = cv2.resize(img, (max(1, w//2), max(1, h//2)), interpolation=cv2.INTER_AREA)
                                result = self.gpu_reader.readtext(small, detail=1, paragraph=True)
                                
                                # Extract text
                                if result and isinstance(result, list) and len(result) > 0:
                                    if isinstance(result[0], tuple) and len(result[0]) >= 2:
                                        text_lines = [item[1] for item in result if item[1].strip()]
                                    else:
                                        text_lines = [str(item).strip() for item in result if str(item).strip()]
                                else:
                                    text_lines = []
                                
                                text = "\n".join(text_lines).strip()
                                success = (1, 0) if text else (0, 1)
                                
                                # ✅ FIX 1: Atomic counter updates with lock
                                with locked(self.shared_lock):
                                    self.shared_counters["total_blocks"] += 1
                                    if success[0]:
                                        self.shared_counters["gpu_success"] += 1
                                    else:
                                        self.shared_counters["gpu_fail"] += 1
                                    self.shared_counters["gpu_blocks"] += 1
                                # ✅ FIX 2: Use queue for results
                                self.results_queue.put((str(img_path), success[0], success[1], text))
                                
                                logger.debug(f"✅ GPU{worker_id}: Processed downscaled {Path(img_path).name}")
                                
                            except Exception as retry_e:
                                # last resort: mark as failed immediately
                                logger.error(f"❌ GPU{worker_id} failed on downscaled {Path(img_path).name}: {retry_e}")
                                # ✅ FIX 1: Atomic counter updates with lock
                                with locked(self.shared_lock):
                                    self.shared_counters["gpu_fail"] += 1
                                    self.shared_counters["total_blocks"] += 1
                                # ✅ FIX 2: Use queue for results
                                self.results_queue.put((str(img_path), 0, 1, ""))
                        
                        # ✅ FIX 1: Atomic counter updates with lock
                        with locked(self.shared_lock):
                            self.shared_counters["gpu_oom_recovered"] += 1
                        clear_gpu_cache()
                else:
                    logger.error(f"❌ GPU{worker_id} error on {Path(img_path).name}: {e}")
                    # ✅ FIX 1: Atomic counter updates with lock
                    with locked(self.shared_lock):
                        self.shared_counters["total_blocks"] += 1
                        self.shared_counters["gpu_fail"] += 1
                        self.shared_counters["gpu_blocks"] += 1
                    # ✅ FIX 2: Use queue for results
                    self.results_queue.put((str(img_path), 0, 1, ""))
            except Exception as e:
                logger.error(f"❌ GPU{worker_id} error on {Path(img_path).name}: {e}")
                # ✅ FIX 1: Atomic counter updates with lock
                with locked(self.shared_lock):
                    self.shared_counters["total_blocks"] += 1
                    self.shared_counters["gpu_fail"] += 1
                    self.shared_counters["gpu_blocks"] += 1
                # ✅ FIX 2: Use queue for results
                self.results_queue.put((str(img_path), 0, 1, ""))
        
        logger.info(f"🔚 GPU worker {worker_id} exiting")
    
    def process_folder(self, img_paths: List[Path], newspaper_name: str) -> Dict[Path, Tuple[int, int, str]]:
        """Process all images using GPU-dominant architecture"""
        logger.info(f"[DEBUG] CPU_WORKERS={CPU_WORKERS}, GPU_WORKERS={GPU_WORKERS}")
        logger.info(f"[DEBUG] Total images: {len(img_paths)}")
        logger.info(f"🔄 Starting GPU-dominant processing for {len(img_paths)} images")

        # Reset state
        self.gpu_spilled_once.clear()
        
        # Clear any leftover tasks
        while not self.gpu_task_queue.empty():
            try: 
                self.gpu_task_queue.get_nowait()
            except queue.Empty: 
                break

        # ───── Resume Logic (STRICT) ─────
        prog = load_progress(newspaper_name)

        # If a progress file exists, validate that every expected per-block output exists and is non-empty.
        # If any expected output is missing/empty -> reset everything for this newspaper and start from scratch.
        strict_resume_ok = False
        if prog:
            strict_resume_ok = True
            for img in img_paths:
                key = str(img.resolve())
                entry = prog.get(key)
                # must have an entry and status must be ok/text and a valid non-empty text_file path
                if not entry or entry.get("status") not in ("ok", "text"):
                    strict_resume_ok = False
                    break
                text_path = entry.get("text_file", "")
                if not text_path:
                    strict_resume_ok = False
                    break
                tf = Path(text_path)
                if not tf.exists() or tf.stat().st_size == 0:
                    strict_resume_ok = False
                    break

        if strict_resume_ok:
            # everything already present and valid — nothing to do
            logger.info(f"✅ Strict resume: all expected outputs present for {newspaper_name} — skipping processing")
            return {}

        # Not strict-resume-able: wipe old progress/outputs to force a clean run
        if prog:
            logger.warning(f"⚠️ Incomplete or invalid progress detected for {newspaper_name} — resetting progress & outputs before re-run")
            # this will delete progress json + block/page text dirs for this newspaper
            try:
                reset_progress_and_outputs(newspaper_name)
            except Exception as e:
                logger.error(f"Failed to reset outputs for {newspaper_name}: {e}")
            prog = {}

        # Build fresh work list
        images_to_process = []
        for img in img_paths:
            key = str(img.resolve())
            # if progress says done & file exists -> skip (shouldn't usually happen after reset)
            if key in prog and prog[key].get("status") in ("ok", "text"):
                tf_path = prog[key].get("text_file", "")
                if tf_path and Path(tf_path).exists() and Path(tf_path).stat().st_size > 0:
                    logger.debug(f"[{img.name}] Status: {prog[key]['status']} - skipped")
                    continue
                else:
                    logger.warning(f"[{img.name}] Progress claims done but file missing/corrupt - will requeue")
            # if progress says failed -> skip (do not retry failed blocks automatically)
            if key in prog and prog[key].get("status") == "failed":
                logger.debug(f"[{img.name}] Status: failed (previous run) - skipped")
                continue
            # if full page-level output exists -> skip and mark
            m = re.search(r"_p(\d+)", img.name)
            pg = m.group(1) if m else "0"
            page_out = PAGE_TEXT_DIR_ROOT / newspaper_name / f"{newspaper_name}_p{pg}_text.txt"
            if page_out.exists() and page_out.stat().st_size > 0:
                logger.debug(f"[{img.name}] Page-level output exists - skipping block")
                prog.setdefault(str(img.resolve()), {"status": "ok", "text_file": str(page_out), "updated_at": datetime.utcnow().isoformat()})
                continue
            images_to_process.append(img)
            # quiet per-block line to DEBUG to avoid noisy terminal output
            logger.debug(f"[{img.name}] Status: queued")
        save_progress(newspaper_name, prog)

        if not images_to_process:
            logger.info(f"✅ All images already processed for {newspaper_name}")
            return {}

        # 🧩 Static Boot Balance - 2:1 GPU:CPU ratio
        # Sort by pixel count (larger images first for GPU)
        images_to_process.sort(key=get_image_pixel_count, reverse=True)
        
        split_index = int(len(images_to_process) * (2 / 3))  # 2:1 ratio
        gpu_paths = images_to_process[:split_index]
        cpu_paths = images_to_process[split_index:]

        # Optional: Pre-warm cache for first GPU images to avoid I/O stalls
        if self.use_cache:
            logger.info("🔥 Pre-warming image cache for GPU startup...")
            for p in gpu_paths[:20]:
                try: 
                    preprocess_image(Path(p), use_cache=self.use_cache)
                except Exception:
                    pass

        # Use spawn context for CPU processes to avoid fork-after-thread hazards
        ctx = mp.get_context('spawn')

        # ✅ FIX: Initialize to safe defaults before try block
        manager = None
        local_results = {}
        local_counters = {}

        # ✅ FIX 1: Safe defaults so cleanup never fails if an early exception occurs
        cpu_processes = []
        gpu_threads = []

        try:
            # Use spawn Manager so proxies/locks are created in the same context
            manager = ctx.Manager()
            
            # ✅ FIX 1: Create shared lock for atomic counter updates
            self.shared_lock = manager.Lock()
            
            # ✅ CRITICAL FIX: Make COUNTERS shared across all processes
            global COUNTERS
            self.shared_counters = manager.dict(COUNTERS)

            # ✅ FIX 2: Create queue for results instead of manager.dict()
            self.results_queue = ctx.Queue(maxsize=CPU_WORKERS * 8)

            # Create a multiprocessing queue using the spawn context with backpressure
            self.cpu_task_queue = ctx.Queue(maxsize=CPU_WORKERS * 4)

            # 1) START CPU WORKERS FIRST (so they can consume while we enqueue)
            logger.info("🚀 Starting CPU workers (spawn)...")
            for i in range(CPU_WORKERS):
                p = ctx.Process(
                    target=cpu_worker_main,
                    args=(self.cpu_task_queue, self.results_queue, self.shared_lock, self.shared_counters, self.langs, i+1, self.shutdown_event, self.use_cache),
                )
                p.start()
                cpu_processes.append(p)

            # PATCH: expose queue & pids for global signal handler (best-effort)
            try:
                globals()['CURRENT_CPU_QUEUE'] = self.cpu_task_queue
                globals()['ACTIVE_CPU_PROCESS_PIDS'] = [p.pid for p in cpu_processes]
            except Exception:
                pass

            # ✅ FIX 2: Start writer thread AFTER CPU workers to avoid race with _meta_ messages
            self.writer_stop_event.clear()
            self.writer_thread = threading.Thread(
                target=result_writer_from_queue,
                args=(newspaper_name, images_to_process, self.results_queue, self.writer_stop_event),
                daemon=True
            )
            self.writer_thread.start()
            logger.info("📝 Started queue-based result writer thread")

            # PATCH A: CPU health-check after starting workers
            # wait briefly for workers to attempt init (easyocr may crash fast)
            time.sleep(0.5)

            # detect any dead CPU processes
            dead_pids = [p.pid for p in cpu_processes if not p.is_alive()]
            
            # ✅ FIX 2: Quick drain for _meta_ init errors (give workers a short window)
            meta_found = False
            try:
                # try to pull any immediate meta messages without blocking writer semantics
                start_poll = time.time()
                while time.time() - start_poll < 0.6:  # poll up to 600ms
                    try:
                        item = self.results_queue.get_nowait()
                    except queue.Empty:
                        break
                    # if it's a real result re-enqueue it for the writer
                    if isinstance(item, tuple) and isinstance(item[0], str) and item[0].startswith("_meta_"):
                        logger.error(f"❌ CPU init error detected: {item[3]}")
                        meta_found = True
                        # store meta for progress (writer would also do this if it saw it)
                        prog = load_progress(newspaper_name)
                        prog[item[0]] = {"status": "worker_error", "error": item[3], "updated_at": datetime.utcnow().isoformat()}
                        save_progress(newspaper_name, prog)
                    else:
                        # requeue normal results so writer receives them in order
                        try:
                            self.results_queue.put_nowait(item)
                        except Exception:
                            # if requeue fails, just put it normally (blocking a tiny bit)
                            self.results_queue.put(item, timeout=0.5)
            except Exception:
                pass

            if dead_pids or meta_found:
                logger.error(f"❌ CPU workers failed to start or crashed: dead_pids={dead_pids}, meta_found={meta_found}")
                self.cpu_ok = False
            else:
                self.cpu_ok = True
                logger.info("✅ CPU workers healthy and ready")

            # Optional improvement: spawn a short monitor thread
            def _monitor_cpus(procs, check_interval=0.5):
                while not self.shutdown_event.is_set():
                    alive = any(p.is_alive() for p in procs)
                    if not alive:
                        logger.error("🔴 All CPU workers died during run — disabling cpu_ok")
                        self.cpu_ok = False
                        break
                    time.sleep(check_interval)

            monitor_t = threading.Thread(target=_monitor_cpus, args=(cpu_processes,), daemon=True)
            monitor_t.start()

            # FIX A: Non-blocking enqueue to local GPU queue, collect overflow
            overflow = []
            logger.info("📦 Enqueuing GPU tasks (non-blocking)")
            for path in gpu_paths:
                try:
                    self.gpu_task_queue.put_nowait(str(path))
                except queue.Full:
                    overflow.append(str(path))

            # FIX A: Try to flush overflow now that CPU workers are running and we have some headroom
            if overflow:
                logger.info(f"🔄 Flushing {len(overflow)} GPU overflow tasks...")
                remaining = []
                for pth in overflow:
                    try:
                        self.gpu_task_queue.put(pth, timeout=0.2)
                    except Exception:
                        remaining.append(pth)
                overflow = remaining

            # PATCH D: Improved overflow handling with retries and persistence
            if overflow:
                logger.warning(f"⚠️ {len(overflow)} GPU overflow tasks remain — attempting retries then falling back to CPU")
                # retry a few times quickly
                for attempt in range(4):
                    if not overflow:
                        break
                    new_remaining = []
                    for pth in overflow:
                        try:
                            self.gpu_task_queue.put(pth, timeout=0.2)
                        except Exception:
                            new_remaining.append(pth)
                    overflow = new_remaining
                    if overflow:
                        time.sleep(0.05)

                # try CPU fallback next
                if overflow:
                    for pth in list(overflow):
                        try:
                            self.cpu_task_queue.put_nowait(pth)
                            overflow.remove(pth)
                        except Exception:
                            continue

                # if any still remain, persist to disk so they aren't lost
                if overflow:
                    backlog_file = CACHE_DIR / "task_backlog.txt"
                    try:
                        with open(backlog_file, "a", encoding="utf-8") as bf:
                            for pth in overflow:
                                bf.write(pth + "\n")
                        logger.warning(f"🔁 Persisted {len(overflow)} overflow tasks to {backlog_file}")
                    except Exception as e:
                        logger.error(f"❌ Failed to persist overflow tasks: {e}")

            # FIX C: Add GPU sentinel only AFTER overflow flush
            for _ in range(GPU_WORKERS):
                try:
                    self.gpu_task_queue.put_nowait(None)
                except queue.Full:
                    # guaranteed to succeed now — wait tiny bit then put
                    try:
                        self.gpu_task_queue.put(None, timeout=0.5)
                    except queue.Full:
                        logger.warning("Failed to add GPU sentinel - queue persistently full")

            # Now enqueue CPU tasks non-blocking, collect overflow
            cpu_overflow = []
            for path in cpu_paths:
                try:
                    self.cpu_task_queue.put_nowait(str(path))
                except Exception:
                    # queue might be temporarily full — push to disk/local overflow to flush later
                    cpu_overflow.append(str(path))

            logger.info(f"⚖️ Initial 2:1 load balance: GPU={len(gpu_paths)} | CPU={len(cpu_paths)}")
            logger.info(f"📦 Queue overflow: GPU={len(overflow)}, CPU={len(cpu_overflow)}")

            # ✅ FIX: Start GPU worker IMMEDIATELY after queues are filled
            logger.info("🚀 Starting GPU worker...")
            for i in range(min(GPU_WORKERS, 1)):
                t = threading.Thread(target=self.gpu_worker, args=(i+1,))
                t.daemon = True
                t.start()
                gpu_threads.append(t)

            # FIX D: Improved CPU overflow flushing with retries
            def flush_cpu_overflow(max_retries=5, delay=0.05):
                for retry in range(max_retries):
                    if not cpu_overflow:
                        break
                    for pth in cpu_overflow[:]:
                        try:
                            self.cpu_task_queue.put(pth, timeout=0.2)
                            cpu_overflow.remove(pth)
                        except Exception:
                            continue
                    if cpu_overflow:
                        time.sleep(delay)

            # Attempt to flush CPU overflow now that workers are running
            if cpu_overflow:
                logger.info(f"🔄 Flushing {len(cpu_overflow)} CPU overflow tasks...")
                flush_cpu_overflow()
                if cpu_overflow:
                    logger.warning(f"⚠️ Still {len(cpu_overflow)} CPU overflow tasks remaining")

            logger.info("✅ All workers started - GPU should be processing immediately")

            try:
                # Wait for GPU thread to finish (they will exit when they get sentinel None)
                logger.info("⏳ Waiting for GPU to complete...")
                for t in gpu_threads:
                    t.join()
                
                # ✅ FIX: Signal completion to CPU workers
                logger.info("🛑 GPU finished - signaling CPU workers to complete...")
                self.shutdown_event.set()
                
                # Send poison pills to CPU workers
                for _ in range(CPU_WORKERS):
                    try:
                        self.cpu_task_queue.put(None, timeout=1)
                    except Exception:
                        pass
                
                # Wait for CPU processes to finish
                logger.info("⏳ Waiting for CPU workers to complete...")
                for p in cpu_processes:
                    p.join(timeout=10)
                    if p.is_alive():
                        logger.warning(f"🔪 CPU worker {p.pid} still alive after join - terminating")
                        p.terminate()
                        p.join(timeout=1)
                
            except KeyboardInterrupt:
                logger.warning("🛑 Interrupt received, shutting down workers...")
                self.shutdown_event.set()
                # Send poison pills
                for _ in range(CPU_WORKERS):
                    try:
                        self.cpu_task_queue.put_nowait(None)
                    except:
                        pass
            
            finally:
                # Signal stop to threads
                self.shutdown_event.set()

                # Stop writer thread after workers to ensure it writes remaining results
                logger.info("🛑 Stopping writer thread...")
                self.writer_stop_event.set()
                # Send poison pill to writer queue
                try:
                    self.results_queue.put(None)
                except:
                    pass
                if self.writer_thread and self.writer_thread.is_alive():
                    self.writer_thread.join(timeout=5)

                # Send poison pills to CPU workers (safe guard)
                if self.cpu_task_queue is not None:
                    for _ in range(CPU_WORKERS):
                        try:
                            self.cpu_task_queue.put_nowait(None)
                        except Exception:
                            pass

                # Terminate any alive CPU processes with short timeouts for quick shutdown
                for p in cpu_processes:
                    if p.is_alive():
                        logger.warning(f"🔪 Terminating CPU worker pid={p.pid}")
                        try:
                            p.terminate()
                            p.join(timeout=2)
                        except Exception as e:
                            logger.error(f"Failed to terminate/join CPU worker {p.pid}: {e}")

                # Join CPU processes cleanly
                for p in cpu_processes:
                    if p.exitcode is None:
                        try:
                            p.join(timeout=2)
                        except Exception:
                            pass

                # Stop GPU threads
                for t in gpu_threads:
                    if t.is_alive():
                        # they will exit when shutdown_event is set
                        t.join(timeout=2)

                # ✅ Clean shutdown of NVML
                try:
                    if HAS_NVIDIA and nvidia_smi is not None:
                        nvidia_smi.nvmlShutdown()
                        logger.debug("NVML shutdown complete")
                except Exception as e:
                    logger.debug(f"NVML shutdown error: {e}")
                
                # Force GPU cleanup
                logger.info("🧹 Final GPU memory cleanup...")
                try:
                    clear_gpu_cache()
                except Exception as e:
                    logger.error(f"Error clearing GPU cache: {e}")
                gc.collect()

        finally:
            # ✅ CRITICAL FIX: Copy counters BEFORE manager shutdown
            try:
                local_counters = dict(self.shared_counters) if self.shared_counters is not None else {}
            except Exception:
                local_counters = {}

            # PATCH: clean global registry
            try:
                globals()['CURRENT_CPU_QUEUE'] = None
                globals()['ACTIVE_CPU_PROCESS_PIDS'] = []
            except Exception:
                pass

            # ✅ Safe manager shutdown with guard
            try:
                if manager is not None:
                    manager.shutdown()
            except Exception:
                pass

        # Since we're using queue-based results, we return empty dict (files are written by writer)
        self.results = {}
        
        # ✅ CRITICAL FIX: Update global COUNTERS from local copy (manager is already shutdown)
        for key, val in local_counters.items():
            COUNTERS[key] = val
        
        logger.info(f"✅ GPU-dominant processing complete - Processed: {len(images_to_process)} images")
        return self.results

# ───── Helpers ─────
def get_lang_for_newspaper(newspaper_name: str):
    n = newspaper_name.lower()
    hindi_keys = ("db", "dainik", "navbharat", "amar", "hindi", "amarujala", "bhaskar", "jantak")
    english_keys = ("the", "indianexpress", "express", "times", "hindu", "ie", "th", "english")
    if any(k in n for k in hindi_keys):
        return ["hi", "en"]
    if any(k in n for k in english_keys):
        return ["en"]
    return DEFAULT_LANGS

# ───── Terminal Summary Display ─────
def print_terminal_summary(elapsed: float):
    """Print a comprehensive summary to the terminal"""
    total_success = COUNTERS['gpu_success'] + COUNTERS['cpu_success']
    total_fail = COUNTERS['gpu_fail'] + COUNTERS['cpu_fail']
    total_blocks = COUNTERS['total_blocks']
    success_rate = (total_success / total_blocks * 100) if total_blocks > 0 else 0.0

    gpu_blocks = COUNTERS.get('gpu_blocks', 0)
    cpu_blocks = COUNTERS.get('cpu_blocks', 0)
    gpu_success_rate = (COUNTERS['gpu_success'] / gpu_blocks * 100) if gpu_blocks > 0 else 0.0
    cpu_success_rate = (COUNTERS['cpu_success'] / cpu_blocks * 100) if cpu_blocks > 0 else 0.0

    blocks_per_second = total_blocks / elapsed if elapsed > 0 else 0.0
    pages_per_minute = (COUNTERS['total_pages'] / elapsed * 60) if elapsed > 0 else 0.0

    gpu_pct = (gpu_blocks / total_blocks * 100) if total_blocks > 0 else 0.0
    cpu_pct = (cpu_blocks / total_blocks * 100) if total_blocks > 0 else 0.0

    print("\n" + "=" * 70)
    print("🎉 GPU-DOMINANT OCR COMPLETE - FINAL SUMMARY")
    print("=" * 70)

    print(f"\n📊 PROCESSING OVERVIEW")
    print(f"   ├── Newspapers: {COUNTERS['total_newspapers']:>6}")
    print(f"   ├── Pages:      {COUNTERS['total_pages']:>6}")
    print(f"   └── Blocks:     {total_blocks:>6}")

    print(f"\n⚡ PERFORMANCE METRICS")
    print(f"   ├── Total Time:    {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   ├── Blocks/sec:    {blocks_per_second:.1f}")
    print(f"   └── Pages/min:     {pages_per_minute:.1f}")

    print(f"\n🎯 SUCCESS RATES")
    print(f"   ├── Overall:       {success_rate:>6.1f}% ({total_success}/{total_blocks if total_blocks>0 else 0})")
    print(f"   ├── GPU:           {gpu_success_rate:>6.1f}% ({COUNTERS['gpu_success']}/{gpu_blocks})")
    print(f"   └── CPU:           {cpu_success_rate:>6.1f}% ({COUNTERS['cpu_success']}/{cpu_blocks})")

    print(f"\n🔧 WORKLOAD DISTRIBUTION")
    print(f"   ├── GPU Blocks:    {gpu_blocks:>6} ({gpu_pct:.1f}%)")
    print(f"   ├── CPU Blocks:    {cpu_blocks:>6} ({cpu_pct:.1f}%)")
    print(f"   ├── OOM Recoveries:{COUNTERS.get('gpu_oom_recovered', 0):>6}")

    print(f"\n🚀 GPU-DOMINANT ARCHITECTURE")
    print("   ✅ Dedicated local queue for GPU (no IPC)")
    print("   ✅ GPU starts immediately after queue fill")
    print("   ✅ Simplified CPU workers (no rebalancing)")
    print("   ✅ OOM spillover to CPU")
    print("   ✅ Clean shutdown with poison pills")

    print("=" * 70)
    print("💡 Detailed logs available in: logs/" + log_name)
    print("=" * 70)

# ───── OCR Folder Processing ─────
def extract_text_by_page_folder(refined_folder: Path, shutdown_event=None, use_cache=True):
    newspaper_name = refined_folder.name
    logger.info(f"📰 Processing newspaper folder: {newspaper_name}")

    output_dir = PAGE_TEXT_DIR_ROOT / newspaper_name
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = [p for p in refined_folder.rglob("*") if p.suffix.lower() in EXTS]
    
    # Early verification of found images
    logger.info(f"[scan] Found {len(exts)} images in {refined_folder.name}")
    if exts:
        sample_names = ", ".join([p.name for p in exts[:6]])
        logger.info(f"[scan] sample files: {sample_names}")
    
    if not exts:
        logger.warning(f"[🚫] No images found in {newspaper_name}")
        return (0, 0, 0, 0)

    # Group by page number
    page_groups = defaultdict(list)
    for img_path in exts:
        match = re.search(r"_p(\d+)", img_path.name, re.IGNORECASE)
        if match:
            page_number = match.group(1)
            page_groups[page_number].append(img_path)
        else:
            page_groups["0"].append(img_path)

    total_pages = total_blocks = success_blocks = failed_blocks = 0

    try:
        # PROCESS ALL PAGES TOGETHER for maximum throughput
        all_images = []
        for page, imgs in page_groups.items():
            all_images.extend(imgs)

        total_pages = len(page_groups)
        logger.info(f"📃 {newspaper_name}: {len(all_images)} total blocks across {len(page_groups)} pages")

        # Process all images together with GPU-dominant architecture
        ocr_processor = GPUDominantOCR(reader_gpu_global, DEFAULT_LANGS, shutdown_event, use_cache=use_cache)
        results = ocr_processor.process_folder(all_images, newspaper_name)

        # ───── Assemble pages from block text files ─────
        logger.info(f"📄 Assembling pages from block text files for {newspaper_name}...")
        for page, imgs in sorted(page_groups.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
            page_texts = []
            page_success = 0
            page_failed = 0
            
            for img_path in imgs:
                txt_file = BLOCK_TEXT_DIR_ROOT / newspaper_name / (img_path.name + ".txt")
                if txt_file.exists() and txt_file.stat().st_size > 0:
                    txt = txt_file.read_text(encoding="utf-8").strip()
                    if txt:
                        page_texts.append(txt)
                        page_success += 1
                        success_blocks += 1
                    else:
                        page_failed += 1
                        failed_blocks += 1
                else:
                    page_failed += 1
                    failed_blocks += 1
                
                total_blocks += 1
            
            # Save page text
            if page_texts:
                page_text = "\n\n".join(page_texts)
                out_file = output_dir / f"{newspaper_name}_p{page}_text.txt"
                atomic_write_text(out_file, page_text)
                logger.info(f"✅ {newspaper_name} p{page} done ({len(page_texts)} blocks)")
                logger.info(f"[💾] Saved page {page} to {out_file.name}")
            else:
                logger.warning(f"[❌] No text extracted for page {page}")
                logger.info(f"❌ {newspaper_name} p{page} failed ({page_failed} blocks)")
    
    except KeyboardInterrupt:
        logger.warning("⚠️ User interrupted mid-page. Partial OCR results will be saved.")
        return (total_pages, total_blocks, success_blocks, failed_blocks)

    return (total_pages, total_blocks, success_blocks, failed_blocks)

# ───── Entry Point ─────
def extract_all_folders(base_dir="data/job_blocks_refined", shutdown_event=None, use_cache=True):
    global reader_gpu_global
    
    start_time = datetime.now()

    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"[🚫] Base path does not exist: {base_dir}")
        return

    folders = [f for f in base_path.iterdir() if f.is_dir()]
    if not folders:
        logger.warning("[🚫] No folders found in job_blocks_refined")
        return

    COUNTERS["total_newspapers"] = len(folders)
    
    print("\n" + "=" * 60)
    print(f"🚀 GPU-DOMINANT OCR RUN STARTED | Folders: {len(folders)}")
    print("=" * 60)
    print("📘 GPU-Dominant Architecture Features:")
    print(f"   • {GPU_WORKERS} GPU worker (dedicated local queue)")
    print(f"   • {CPU_WORKERS} CPU workers (simplified, no rebalancing)") 
    print("   • 2:1 initial GPU:CPU split")
    print("   • GPU starts immediately after queue fill")
    print("   • OOM spillover to CPU")
    print("   • Clean shutdown with poison pills\n")
    print("=" * 60 + "\n")

    # ───── Initialize EasyOCR Readers (Single GPU Instance) ─────
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available — will attempt CPU-only fallback")

    # Set mixed precision
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
        logger.info("Enabled mixed precision matrix multiplication")

    logger.info("Initializing EasyOCR readers...")
    
    # Single GPU reader only - EasyOCR uses single CUDA context
    logger.info("🔄 Initializing single GPU reader...")
    try:
        reader_gpu_global = easyocr.Reader(DEFAULT_LANGS, gpu=True)
        
        # Enable FP16 precision (faster)
        try:
            if hasattr(reader_gpu_global, 'recog_network') and reader_gpu_global.recog_network is not None:
                reader_gpu_global.recog_network = reader_gpu_global.recog_network.half()
                logger.info("Enabled FP16 precision for GPU reader")
        except Exception as e:
            logger.warning(f"Could not enable FP16 precision: {e}")

        # Warm up GPU
        logger.info("🔋 Warming up GPU reader...")
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        _ = reader_gpu_global.readtext(dummy, detail=0)
        logger.info("✅ GPU reader warmed up")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize GPU reader: {e}")
        logger.warning("🔄 Falling back to CPU-only mode...")
        reader_gpu_global = easyocr.Reader(DEFAULT_LANGS, gpu=False)
    
    logger.info("✅ All systems initialized and ready for GPU-dominant processing")

    total_pages = total_blocks = success_blocks = failed_blocks = 0
    
    # Process folders sequentially
    logger.info(f"⚙️ Processing {len(folders)} folders with GPU-dominant pipeline...")
    
    try:
        for folder in sorted(folders, key=lambda p: p.name):
            if shutdown_event and shutdown_event.is_set():
                logger.warning("🛑 Shutdown signal received - stopping processing")
                break
                
            p, b, s, f = extract_text_by_page_folder(folder, shutdown_event, use_cache=use_cache)
            total_pages += p
            total_blocks += b
            success_blocks += s
            failed_blocks += f
            logger.info(f"📦 Completed folder: {folder.name}")
            
    except KeyboardInterrupt:
        logger.warning("⚠️ Interrupted by user — saving progress...")
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"⏱️ Partial run completed in {elapsed:.2f} seconds")
        logger.info(f"📊 Progress: {total_pages} pages, {total_blocks} blocks processed")
        raise SystemExit(0)

    # Update final counters
    COUNTERS["total_pages"] = total_pages

    elapsed = (datetime.now() - start_time).total_seconds()

    # Print comprehensive terminal summary
    print_terminal_summary(elapsed)

    # Final logging
    logger.info("\n========== GPU-DOMINANT OCR SUMMARY ==========")
    logger.info(f"🗞️ Total newspapers processed : {COUNTERS['total_newspapers']}")
    logger.info(f"📄 Total pages processed      : {COUNTERS['total_pages']}")
    logger.info(f"🧩 Total blocks processed     : {COUNTERS['total_blocks']}")
    logger.info(f"⏱️ Total time taken           : {elapsed:.2f} seconds ({elapsed/60:.2f} min)")

    total_success = COUNTERS['gpu_success'] + COUNTERS['cpu_success']
    total_fail = COUNTERS['gpu_fail'] + COUNTERS['cpu_fail']
    success_rate = (total_success / (total_success + total_fail) * 100) if (total_success + total_fail) > 0 else 0
    logger.info(f"🎯 Overall success rate       : {success_rate:.2f}%")

    logger.info(f"🔄 OOM Recoveries            : {COUNTERS['gpu_oom_recovered']}")
    
    logger.info("=====================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="data/job_blocks_refined")
    parser.add_argument("--force-reset", action="store_true", help="Delete progress & outputs before starting")
    parser.add_argument("--no-cache", action="store_true", help="Ignore/clear preprocessed cache")
    args = parser.parse_args()

    if args.force_reset:
        # reset every folder present so run is deterministic
        base_path = Path(args.base_dir)
        if base_path.exists():
            for d in base_path.iterdir():
                if d.is_dir():
                    reset_progress_and_outputs(d.name)
                    logger.info(f"🧹 Reset progress and outputs for: {d.name}")

    if args.no_cache:
        try:
            if CACHE_DIR.exists():
                shutil.rmtree(CACHE_DIR)
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                logger.info("🧹 Cleared preprocess cache (cache_blocks/)")
        except Exception as e:
            logger.warning(f"Could not clear cache dir: {e}")

    # Create spawn context for multiprocessing
    ctx = mp.get_context('spawn')
    shutdown_event = ctx.Event()   # multiprocessing-safe

    # --- Replace custom swallowing handler with one that raises KeyboardInterrupt ---
    def _handle_sigint(sig, frame):
        """
        Minimal handler that raises KeyboardInterrupt in main thread so the
        normal exception-handling / cleanup path runs.
        """
        # Re-raise KeyboardInterrupt into the main thread — this is the standard
        # behavior Python would have produced if we didn't install a custom handler.
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    try:
        extract_all_folders(base_dir=args.base_dir, shutdown_event=shutdown_event, use_cache=not args.no_cache)
    except KeyboardInterrupt:
        logger.warning("🛑 KeyboardInterrupt received in main - initiating cleanup")
        
        # Poison CPU queue eagerly so CPU workers see None and exit asap
        q = globals().get('CURRENT_CPU_QUEUE')
        if q is not None:
            for _ in range(CPU_WORKERS):
                try:
                    q.put_nowait(None)
                except Exception:
                    try:
                        q.put(None, timeout=0.1)
                    except Exception:
                        pass
        
        # Force-terminate stubborn child processes after a short grace period
        time.sleep(0.2)
        for p in mp.active_children():
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(0.2)
        for p in mp.active_children():
            try:
                p.kill()
            except Exception:
                pass
                
    except SystemExit:
        pass
    finally:
        # ✅ FIX 1: Perform queue poisoning and cleanup in main thread context (safe)
        if globals().get('_SIGNAL_RECEIVED'):
            q = globals().get('CURRENT_CPU_QUEUE')
            if q is not None:
                for _ in range(CPU_WORKERS):
                    try:
                        q.put_nowait(None)
                    except Exception:
                        try:
                            q.put(None, timeout=0.1)
                        except Exception:
                            pass
            # terminate any active children politely
            for p in mp.active_children():
                try:
                    p.terminate()
                except Exception:
                    pass