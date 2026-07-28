
import os
import sys
import time
import subprocess
import threading
import json
import psutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / "4_env" / "bin" / "python"
RAW_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
BENCH_13_DIR = PROJECT_ROOT / "data" / "benchmark_13_raw"
STATE_DIR = PROJECT_ROOT / "run_state"

STAGES = [
    "src/pipeline/stage01_pdf_to_images.py",
    "src/pipeline/stage02_block_detection.py",
    "src/pipeline/stage03_block_refiner.py",
    "src/pipeline/stage03_ocr.py",
]

class ResourceMonitor(threading.Thread):
    def __init__(self, interval=2):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = True
        self.stats = []

    def run(self):
        while self.running:
            try:
                # GPU stats
                gpu_out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used", "--format=csv,noheader,nounits"],
                    encoding="utf-8"
                ).strip()
                util, power, vram = gpu_out.split(",")
                
                # CPU stats
                cpu_util = psutil.cpu_percent()
                ram_util = psutil.virtual_memory().percent
                
                stat = {
                    "time": time.time(),
                    "gpu_util": float(util),
                    "gpu_power": float(power),
                    "gpu_vram": float(vram),
                    "cpu_util": cpu_util,
                    "ram_util": ram_util
                }
                self.stats.append(stat)
                
                # print(f"  [Monitor] GPU: {util}% ({power}W) | CPU: {cpu_util}% | RAM: {ram_util}%")
            except:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.running = False

def run_stages(label):
    print(f"\n>>> STARTING RUN: {label}")
    start_run = time.time()
    
    # Cleanup done markers
    for f in STATE_DIR.glob("*.done"):
        f.unlink()

    stage_timings = {}
    
    monitor = ResourceMonitor()
    monitor.start()

    for stage in STAGES:
        print(f"  Executing {stage}...")
        s_start = time.time()
        res = subprocess.run([str(PYTHON_BIN), stage], cwd=str(PROJECT_ROOT))
        s_end = time.time()
        stage_timings[stage] = s_end - s_start
        if res.returncode != 0:
            print(f"  ERROR: Stage {stage} failed with code {res.returncode}")
            break
    
    monitor.stop()
    end_run = time.time()
    
    total = end_run - start_run
    print(f"\n>>> COMPLETED RUN: {label}")
    print(f"    Total Time: {total:.2f}s")
    for s, t in stage_timings.items():
        print(f"    - {s}: {t:.2f}s")
    
    if monitor.stats:
        avg_gpu = sum(s["gpu_util"] for s in monitor.stats) / len(monitor.stats)
        avg_cpu = sum(s["cpu_util"] for s in monitor.stats) / len(monitor.stats)
        max_vram = max(s["gpu_vram"] for s in monitor.stats)
        print(f"    - Avg GPU Util: {avg_gpu:.1f}%")
        print(f"    - Avg CPU Util: {avg_cpu:.1f}%")
        print(f"    - Peak VRAM: {max_vram:.1f} MB")

    return {
        "label": label,
        "total": total,
        "stages": stage_timings,
        "avg_gpu": avg_gpu if monitor.stats else 0,
        "avg_cpu": avg_cpu if monitor.stats else 0,
        "peak_vram": max_vram if monitor.stats else 0
    }

def main():
    # Setup 1 PDF
    if not RAW_DIR.exists(): RAW_DIR.mkdir(parents=True)
    for f in RAW_DIR.glob("*"): f.unlink()
    
    # Take first PDF
    sample_pdf = next(BENCH_13_DIR.glob("*.pdf"))
    shutil.copy(sample_pdf, RAW_DIR / sample_pdf.name)
    
    results = []
    
    # Run 1 PDF
    results.append(run_stages("1 PDF"))
    
    # Setup 13 PDFs
    for f in RAW_DIR.glob("*"): f.unlink()
    for f in BENCH_13_DIR.glob("*.pdf"):
        shutil.copy(f, RAW_DIR / f.name)
    
    # Run 13 PDFs
    results.append(run_stages("13 PDFs"))
    
    print("\n\n" + "="*40)
    print("FINAL BENCHMARK REPORT")
    print("="*40)
    for r in results:
        print(f"\n{r['label']}:")
        print(f"  Wall Time: {r['total']:.2f}s")
        print(f"  Avg GPU:   {r['avg_gpu']:.1f}%")
        print(f"  Avg CPU:   {r['avg_cpu']:.1f}%")
        print(f"  Peak VRAM: {r['peak_vram']:.1f} MB")
    
if __name__ == "__main__":
    import shutil
    main()
