
import os
import sys
import time
import json
import psutil
import subprocess
import threading
import shutil
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / "4_env" / "bin" / "python"
RAW_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
BENCH_13_DIR = PROJECT_ROOT / "data" / "benchmark_13_raw"
STATE_DIR = PROJECT_ROOT / "run_state"

class ResourceMonitor(threading.Thread):
    def __init__(self, interval=2):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = True
        self.history = []

    def run(self):
        while self.running:
            try:
                gpu_out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used", "--format=csv,noheader,nounits"],
                    encoding="utf-8"
                ).strip()
                util, power, vram = gpu_out.split(",")
                cpu_util = psutil.cpu_percent()
                ram_util = psutil.virtual_memory().percent
                self.history.append({
                    "time": time.time(),
                    "gpu_util": float(util),
                    "gpu_power": float(power),
                    "gpu_vram": float(vram),
                    "cpu_util": cpu_util,
                    "ram_util": ram_util
                })
            except: pass
            time.sleep(self.interval)

    def stop(self):
        self.running = False

    def get_averages(self):
        if not self.history: return {}
        return {
            "avg_gpu_util": sum(x["gpu_util"] for x in self.history) / len(self.history),
            "avg_cpu_util": sum(x["cpu_util"] for x in self.history) / len(self.history),
            "peak_vram": max(x["gpu_vram"] for x in self.history),
            "avg_ram_util": sum(x["ram_util"] for x in self.history) / len(self.history),
        }

def run_bench(label: str, pdf_count: int):
    print(f"\n>>> BENCHMARK: {label} ({pdf_count} PDFs)")
    
    # MANDATORY CLEAN SETUP
    for d in ["data/pdf2img", "data/crops", "run_state/detections", "data/job_blocks_smart"]:
        p = PROJECT_ROOT / d
        if p.exists(): shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
    for f in STATE_DIR.glob("*.done"): f.unlink()
    for f in STATE_DIR.glob("*.jsonl"): f.unlink()
    
    # 1. Setup PDFs
    for f in RAW_DIR.glob("*"): f.unlink()
    pdfs = sorted(list(BENCH_13_DIR.glob("*.pdf")))[:pdf_count]
    for f in pdfs: shutil.copy(f, RAW_DIR / f.name)
    
    monitor = ResourceMonitor()
    monitor.start()
    
    start_total = time.time()
    timings = {}
    
    # Stage 1: Render
    s_start = time.time()
    subprocess.run([str(PYTHON_BIN), "src/pipeline/stage01_pdf_to_images.py", "--move-processed", "false"], check=True)
    timings["S1"] = time.time() - s_start
    
    # Stage 2: Detection
    s_start = time.time()
    subprocess.run([str(PYTHON_BIN), "src/pipeline/stage02_block_detection_parallel.py"], check=True)
    timings["S2"] = time.time() - s_start
    
    # Stage 3: OCR
    s_start = time.time()
    subprocess.run([str(PYTHON_BIN), "src/pipeline/stage03_ocr.py"], check=True)
    timings["S3"] = time.time() - s_start
    
    end_total = time.time()
    monitor.stop()
    res = monitor.get_averages()
    
    crop_count = 0
    if (STATE_DIR / "ocr_manifest.jsonl").exists():
        with open(STATE_DIR / "ocr_manifest.jsonl", "r") as f:
            crop_count = sum(1 for _ in f)
        
    print(f"\n--- RESULTS: {label} ---")
    print(f"Total Time: {end_total - start_total:.2f}s")
    print(f"S1: {timings['S1']:.1f}s | S2: {timings['S2']:.1f}s | S3: {timings['S3']:.1f}s")
    print(f"Crops: {crop_count} | Throughput: {crop_count / max(1, timings['S3']):.2f} crops/sec")
    
    return {
        "label": label,
        "total_time": end_total - start_total,
        "timings": timings,
        "crop_count": crop_count,
        "metrics": res
    }

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "1pdf"
    all_results = []
    if mode in ["1pdf", "all"]:
        all_results.append(run_bench("1-PDF", 1))
    if mode in ["13pdf", "all"]:
        all_results.append(run_bench("13-PDF", 13))
    with open("benchmark_results_current.json", "w") as f:
        json.dump(all_results, f, indent=2)
