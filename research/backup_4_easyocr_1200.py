#!/usr/bin/env python3
"""
gpu_multilang_easyocr_optimized.py

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
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# ───── Global pointers for signal handler ─────
CURRENT_CPU_QUEUE = None          # will be set by GPUDominantOCR.process_folder while running
ACTIVE_CPU_PROCESS_PIDS = []      # list of started CPU pids for quick termination

# ───── CUDA Optimization Settings ─────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Enable Tensor Cores and optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
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

logger.addHandler(fh)
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
def preprocess_image(img_path: Path) -> np.ndarray:
    """Load + resize image with cache support"""
    try:
        cache_path = _cache_for_path(img_path)

        if cache_path.exists():
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

        # ✅ CORRECTED: Atomic cache write to prevent corruption
        with tempfile.NamedTemporaryFile(dir=str(CACHE_DIR), suffix='.npy', delete=False) as tf:
            # tf is an open file; np.save accepts a file-like object
            np.save(tf, image)
            tmp_path = Path(tf.name)

        # atomically replace target cache file with temp file
        try:
            os.replace(tmp_path, cache_path)
        except FileNotFoundError:
            # maybe another process already created cache_path; try to load cache if present
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

# ───── Simplified CPU worker for multiprocessing ─────
def cpu_worker_main(cpu_task_queue, results_proxy, results_lock, shared_counters, langs, worker_id, shutdown_event=None):
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
        # mark that this worker failed (so parent won't wait forever)
        try:
            results_lock.acquire()
            # Store using per-pid _meta keys (simpler & safe)
            meta_key = f"_meta_{os.getpid()}"
            results_proxy[meta_key] = tb  # tb is the traceback string you already built
            results_lock.release()
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
                image = preprocess_image(Path(img_path))
                result = cpu_reader.readtext(image, detail=0, paragraph=True)
                text = "\n".join([ln.strip() for ln in result if ln and str(ln).strip()])
                success = (1, 0) if text else (0, 1)
                
                # ✅ Use explicit lock acquire/release for compatibility
                results_lock.acquire()
                try:
                    shared_counters["total_blocks"] += 1
                    if success[0]:
                        shared_counters["cpu_success"] += 1
                    else:
                        shared_counters["cpu_fail"] += 1
                    shared_counters["cpu_blocks"] += 1
                finally:
                    results_lock.release()
                
                task_counter += 1
                
                # Lightweight real-time progress
                if task_counter % 20 == 0:
                    results_lock.acquire()
                    try:
                        done = shared_counters["total_blocks"]
                    finally:
                        results_lock.release()
                    worker_logger.info(f"[progress] {done} blocks done...")
                    
            except Exception as e:
                tb = traceback.format_exc()
                worker_logger.error(f"❌ CPU{worker_id} error on {img_path}: {e}\n{tb}")
                success, text = (0, 1), ""
                
                # ✅ Use explicit lock acquire/release for compatibility
                results_lock.acquire()
                try:
                    shared_counters["total_blocks"] += 1
                    shared_counters["cpu_fail"] += 1
                    shared_counters["cpu_blocks"] += 1
                finally:
                    results_lock.release()

            # ✅ Store results with string keys for consistency
            results_lock.acquire()
            try:
                results_proxy[str(img_path)] = (*success, text)
            finally:
                results_lock.release()
    except Exception as e:
        tb = traceback.format_exc()
        worker_logger.error(f"❌ CPU{worker_id} fatal error: {e}\n{tb}")

    worker_logger.info(f"🔚 CPU worker {worker_id} exiting (pid={os.getpid()})")

# ───── GPU-Dominant OCR System ─────
class GPUDominantOCR:
    def __init__(self, gpu_reader, langs, shutdown_event=None):
        # 🔧 FIX: Separate queues - local for GPU, multiprocessing for CPU
        self.gpu_task_queue = queue.Queue(maxsize=1024)  # ✅ Added backpressure for large inputs
        # create cpu_task_queue later with a safe context (spawn). Avoid creating
        # multiprocessing queues before threading/fork interactions.
        self.cpu_task_queue = None
        self.results = None          
        self.results_lock = None     
        self.shared_counters = None  
        self.shutdown_event = shutdown_event or threading.Event()
        self.gpu_reader = gpu_reader
        self.langs = langs
        self.gpu_spilled_once = set()  # ✅ Track spilled images to prevent infinite loops
        self.cpu_ok = True  # Assume CPU is OK by default
        
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
                        self.results_lock.acquire()
                        try:
                            self.results[key] = (0, 1, "")
                            self.shared_counters["gpu_fail"] += 1
                            self.shared_counters["total_blocks"] += 1
                        finally:
                            self.results_lock.release()
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
                                img = preprocess_image(Path(img_path))
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
                                
                                self.results_lock.acquire()
                                try:
                                    self.results[str(img_path)] = (*success, text)
                                    self.shared_counters["total_blocks"] += 1
                                    if success[0]:
                                        self.shared_counters["gpu_success"] += 1
                                    else:
                                        self.shared_counters["gpu_fail"] += 1
                                    self.shared_counters["gpu_blocks"] += 1
                                finally:
                                    self.results_lock.release()
                                
                                logger.debug(f"✅ GPU{worker_id}: Processed downscaled {Path(img_path).name}")
                                
                            except Exception as retry_e:
                                # last resort: mark as failed immediately
                                logger.error(f"❌ GPU{worker_id} failed on downscaled {Path(img_path).name}: {retry_e}")
                                self.results_lock.acquire()
                                try:
                                    self.results[str(img_path)] = (0, 1, "")
                                    self.shared_counters["gpu_fail"] += 1
                                    self.shared_counters["total_blocks"] += 1
                                finally:
                                    self.results_lock.release()
                        
                        self.results_lock.acquire()
                        try:
                            self.shared_counters["gpu_oom_recovered"] += 1
                        finally:
                            self.results_lock.release()
                        clear_gpu_cache()
                        continue
                
                # Process image
                image = preprocess_image(Path(img_path))
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
                
                # ✅ Update shared counters safely with string keys
                self.results_lock.acquire()
                try:
                    self.results[str(img_path)] = (*success, text)
                    self.shared_counters["total_blocks"] += 1
                    if success[0]:
                        self.shared_counters["gpu_success"] += 1
                    else:
                        self.shared_counters["gpu_fail"] += 1
                    self.shared_counters["gpu_blocks"] += 1
                finally:
                    self.results_lock.release()
                
                task_counter += 1
                
                # Lightweight real-time progress
                if task_counter % 20 == 0:
                    self.results_lock.acquire()
                    try:
                        done = self.shared_counters["total_blocks"]
                    finally:
                        self.results_lock.release()
                    logger.info(f"[progress] {done} blocks done...")
                
                logger.debug(f"✅ GPU{worker_id}: Processed {Path(img_path).name} (task #{task_counter})")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "OOM" in str(e):
                    # ✅ Canonicalize keys to prevent Path vs str mismatches
                    key = str(img_path)
                    if key in self.gpu_spilled_once:
                        # already tried spilling once, mark failed to avoid loop
                        logger.warning(f"⚠️ GPU{worker_id} persistent OOM on {Path(img_path).name}, marking as failed")
                        self.results_lock.acquire()
                        try:
                            self.results[key] = (0, 1, "")
                            self.shared_counters["gpu_fail"] += 1
                            self.shared_counters["total_blocks"] += 1
                        finally:
                            self.results_lock.release()
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
                                img = preprocess_image(Path(img_path))
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
                                
                                self.results_lock.acquire()
                                try:
                                    self.results[str(img_path)] = (*success, text)
                                    self.shared_counters["total_blocks"] += 1
                                    if success[0]:
                                        self.shared_counters["gpu_success"] += 1
                                    else:
                                        self.shared_counters["gpu_fail"] += 1
                                    self.shared_counters["gpu_blocks"] += 1
                                finally:
                                    self.results_lock.release()
                                
                                logger.debug(f"✅ GPU{worker_id}: Processed downscaled {Path(img_path).name}")
                                
                            except Exception as retry_e:
                                # last resort: mark as failed immediately
                                logger.error(f"❌ GPU{worker_id} failed on downscaled {Path(img_path).name}: {retry_e}")
                                self.results_lock.acquire()
                                try:
                                    self.results[str(img_path)] = (0, 1, "")
                                    self.shared_counters["gpu_fail"] += 1
                                    self.shared_counters["total_blocks"] += 1
                                finally:
                                    self.results_lock.release()
                        
                        self.results_lock.acquire()
                        try:
                            self.shared_counters["gpu_oom_recovered"] += 1
                        finally:
                            self.results_lock.release()
                        clear_gpu_cache()
                else:
                    logger.error(f"❌ GPU{worker_id} error on {Path(img_path).name}: {e}")
                    self.results_lock.acquire()
                    try:
                        self.results[str(img_path)] = (0, 1, "")
                        self.shared_counters["total_blocks"] += 1
                        self.shared_counters["gpu_fail"] += 1
                        self.shared_counters["gpu_blocks"] += 1
                    finally:
                        self.results_lock.release()
            except Exception as e:
                logger.error(f"❌ GPU{worker_id} error on {Path(img_path).name}: {e}")
                self.results_lock.acquire()
                try:
                    self.results[str(img_path)] = (0, 1, "")
                    self.shared_counters["total_blocks"] += 1
                    self.shared_counters["gpu_fail"] += 1
                    self.shared_counters["gpu_blocks"] += 1
                finally:
                    self.results_lock.release()
        
        logger.info(f"🔚 GPU worker {worker_id} exiting")
    
    def process_folder(self, img_paths: List[Path]) -> Dict[Path, Tuple[int, int, str]]:
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

        # 🧩 Static Boot Balance - 2:1 GPU:CPU ratio
        # Sort by pixel count (larger images first for GPU)
        img_paths.sort(key=get_image_pixel_count, reverse=True)
        
        split_index = int(len(img_paths) * (2 / 3))  # 2:1 ratio
        gpu_paths = img_paths[:split_index]
        cpu_paths = img_paths[split_index:]

        # Optional: Pre-warm cache for first GPU images to avoid I/O stalls
        logger.info("🔥 Pre-warming image cache for GPU startup...")
        for p in gpu_paths[:20]:
            try: 
                preprocess_image(Path(p))
            except Exception:
                pass

        # Use spawn context for CPU processes to avoid fork-after-thread hazards
        ctx = mp.get_context('spawn')

        # ✅ FIX: Initialize to safe defaults before try block
        manager = None
        results_proxy = {}
        results_lock = None
        local_results = {}
        local_counters = {}

        try:
            # Use spawn Manager so proxies/locks are created in the same context
            manager = ctx.Manager()
            results_proxy = manager.dict()
            results_lock = manager.Lock()
            
            # ✅ CRITICAL FIX: Make COUNTERS shared across all processes
            global COUNTERS
            self.shared_counters = manager.dict(COUNTERS)
            
            self.results = results_proxy
            self.results_lock = results_lock

            # Create a multiprocessing queue using the spawn context with backpressure
            self.cpu_task_queue = ctx.Queue(maxsize=CPU_WORKERS * 4)

            # 1) START CPU WORKERS FIRST (so they can consume while we enqueue)
            logger.info("🚀 Starting CPU workers (spawn)...")
            cpu_processes = []
            for i in range(CPU_WORKERS):
                p = ctx.Process(
                    target=cpu_worker_main,
                    args=(self.cpu_task_queue, results_proxy, results_lock, self.shared_counters, self.langs, i+1, self.shutdown_event),
                )
                p.start()
                cpu_processes.append(p)

            # PATCH: expose queue & pids for global signal handler (best-effort)
            try:
                globals()['CURRENT_CPU_QUEUE'] = self.cpu_task_queue
                globals()['ACTIVE_CPU_PROCESS_PIDS'] = [p.pid for p in cpu_processes]
            except Exception:
                pass

            # PATCH A: CPU health-check after starting workers
            # wait briefly for workers to attempt init (easyocr may crash fast)
            time.sleep(0.5)

            # detect any dead CPU processes
            dead_pids = [p.pid for p in cpu_processes if not p.is_alive()]
            # PATCH: Read meta entries using the new per-pid keys
            cpu_init_failures = {}
            try:
                # Look for any _meta_* keys
                for key, value in results_proxy.items():
                    if isinstance(key, str) and key.startswith("_meta_"):
                        cpu_init_failures[key] = value
            except Exception:
                pass

            if dead_pids or cpu_init_failures:
                logger.error(f"❌ CPU workers failed to start or crashed immediately: dead_pids={dead_pids}, init_errors={list(cpu_init_failures.keys())}")
                # disable CPU fallback to avoid GPU long waits
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
            gpu_threads = []
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
            # ✅ CRITICAL FIX: Copy results and counters BEFORE manager shutdown
            try:
                local_results = dict(results_proxy) if results_proxy is not None else {}
            except Exception:
                local_results = {}

            # PATCH: gather meta entries BEFORE filtering (any keys starting with "_meta")
            meta_entries = {k: v for k, v in local_results.items() if isinstance(k, str) and k.startswith("_meta")}
            if meta_entries:
                logger.error(f"❌ Worker meta errors found: {meta_entries}")

            # Now filter out internal keys so they won't be treated as page results
            local_results = {k: v for k, v in local_results.items() 
                            if not (isinstance(k, str) and (k.startswith("_meta") or k.startswith("__cpu_worker_")))}

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

        # Convert results back to dict[Path]
        self.results = {Path(k): v for k, v in local_results.items()}
        
        # ✅ CRITICAL FIX: Update global COUNTERS from local copy (manager is already shutdown)
        for key, val in local_counters.items():
            COUNTERS[key] = val
        
        logger.info(f"✅ GPU-dominant processing complete - Processed: {len(self.results)}/{len(img_paths)} images")
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
def extract_text_by_page_folder(refined_folder: Path, shutdown_event=None):
    newspaper_name = refined_folder.name
    logger.info(f"📰 Processing newspaper folder: {newspaper_name}")

    output_dir = Path("data/page_texts") / newspaper_name
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
        ocr_processor = GPUDominantOCR(reader_gpu_global, DEFAULT_LANGS, shutdown_event)
        results = ocr_processor.process_folder(all_images)

        # Group results back by page for saving and summary
        for page, img_paths in sorted(page_groups.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
            page_texts = []
            page_success = 0
            page_failed = 0
            
            for img_path in img_paths:
                if img_path in results:
                    s, f, text = results[img_path]
                    total_blocks += 1
                    success_blocks += s
                    failed_blocks += f
                    page_success += s
                    page_failed += f
                    
                    if text:
                        page_texts.append(text)
                else:
                    # Handle unprocessed images
                    total_blocks += 1
                    failed_blocks += 1
                    page_failed += 1
            
            # Save page text
            if page_texts:
                page_text = "\n\n".join(page_texts)
                out_file = output_dir / f"{newspaper_name}_p{page}_text.txt"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(page_text)
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
def extract_all_folders(base_dir="data/job_blocks_refined", shutdown_event=None):
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
                
            p, b, s, f = extract_text_by_page_folder(folder, shutdown_event)
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
    # Create spawn context for multiprocessing
    ctx = mp.get_context('spawn')
    shutdown_event = ctx.Event()   # multiprocessing-safe

    # PATCH C: Stronger signal handler that wakes queues and terminates children
    def _handle_sigint(sig, frame):
        logger.warning("🛑 SIGINT/SIGTERM received: setting shutdown_event and waking workers")
        try:
            shutdown_event.set()
        except Exception:
            pass

        # Best-effort: push poison pills to known active CPU queue
        try:
            q = globals().get('CURRENT_CPU_QUEUE', None)
            if q is not None:
                for _ in range(CPU_WORKERS):
                    try:
                        q.put_nowait(None)
                    except Exception:
                        try:
                            q.put(None, timeout=0.1)
                        except Exception:
                            pass
        except Exception:
            pass

        # Best-effort: terminate any active child processes started earlier
        try:
            for p in mp.active_children():
                try:
                    p.terminate()
                except Exception:
                    pass
            # Also kill pids we tracked (if any)
            for pid in globals().get('ACTIVE_CPU_PROCESS_PIDS', []):
                try:
                    # double-check it's still running and terminate politely
                    # On unix you could os.kill(pid, signal.SIGTERM) but keep it simple:
                    pass
                except Exception:
                    pass
        except Exception:
            pass

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    try:
        extract_all_folders(base_dir="data/job_blocks_refined", shutdown_event=shutdown_event)
    except SystemExit:
        pass