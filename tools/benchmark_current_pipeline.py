
import os
import sys
import time
import subprocess
import json
import shutil
import threading
import psutil
from pathlib import Path
from datetime import datetime

# Path Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / "4_env" / "bin" / "python"
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "validation-data" / "new-data" / "raw_pdfs"
RUN_STATE_DIR = PROJECT_ROOT / "run_state"
PDF2IMG_DIR = DATA_DIR / "pdf2img"
CROPS_DIR = DATA_DIR / "crops"
BLOCKS_DIR = DATA_DIR / "job_blocks_smart"
DETECTIONS_DIR = RUN_STATE_DIR / "detections"

STAGE1 = "src/pipeline/stage01_pdf_to_images.py"
STAGE2 = "src/pipeline/stage02_block_detection.py"
STAGE3 = "src/pipeline/stage03_ocr.py"

ALL_PDFS = [
    "BS English-Delhi 07-04.pdf",
    "ET-Delhi 07-04.pdf",
    "FE-Delhi 07-04.pdf",
    "Free Press-Mumbai 07-04.pdf",
    "IE-Delhi 07-04.pdf",
    "Mint Delhi 07-04.pdf",
    "NIE Chennai 07-04.pdf",
    "Orissa-Post-07-04.pdf",
    "Statesman-Delhi 07-04.pdf",
    "TH- Delhi 07-04.pdf",
    "The Tribune Delhi 07-04.pdf",
    "TOI-Delhi 07-04.pdf",
    "UHT Delhi 07-04.pdf",
]

def get_vram_usage():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        return int(output.strip())
    except:
        return 0

class ResourceMonitor(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.history = []

    def run(self):
        while not self.stop_event.is_set():
            ram = psutil.virtual_memory().used / (1024 * 1024)
            vram = get_vram_usage()
            self.history.append({
                "time": time.time(),
                "ram": ram,
                "vram": vram
            })
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

def run_stage(cmd_list, name):
    print(f"--- Running {name} ---")
    start = time.time()
    proc = subprocess.Popen([str(PYTHON_BIN)] + cmd_list, cwd=str(PROJECT_ROOT))
    proc.wait()
    end = time.time()
    if proc.returncode != 0:
        print(f"Error: {name} failed with exit code {proc.returncode}")
        # sys.exit(1)
    return end - start

def clean_dirs():
    dirs_to_clean = [
        RUN_STATE_DIR, 
        PDF2IMG_DIR, 
        CROPS_DIR, 
        BLOCKS_DIR,
        DETECTIONS_DIR
    ]
    for d in dirs_to_clean:
        if d.exists():
            print(f"Cleaning {d}...")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

def setup_pdfs(pdfs, target_dir):
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for p in pdfs:
        src = RAW_PDF_DIR / p
        dst = target_dir / p
        shutil.copy2(src, dst)

def count_jsonl(path):
    if not path.exists():
        return 0
    with open(path, "r") as f:
        return sum(1 for _ in f)

def get_ocr_stats():
    # We'll check the logs for tiled/downscaled if possible
    # Or check if any ocr_manifest records have specific flags
    ocr_manifest = RUN_STATE_DIR / "ocr_manifest.jsonl"
    downscaled = 0
    tiled = 0
    if ocr_manifest.exists():
        with open(ocr_manifest, "r") as f:
            for line in f:
                data = json.loads(line)
                # Check for tiled or downscaled markers if they exist
                # If they don't, we can check stage03_ocr.py logs
                pass
    
    # Check logs for "downscaling and retrying"
    # Stage 3 logs are in logs/step03_ocr_*.log
    downscaled = 0
    log_dir = PROJECT_ROOT / "logs"
    for log_file in log_dir.glob("step03_ocr_*.log"):
        with open(log_file, "r") as f:
            for line in f:
                if "downscaling and retrying" in line:
                    downscaled += 1
    
    return {"downscaled": downscaled, "tiled": tiled}

def run_benchmark(pdf_count):
    print(f"\n===== BENCHMARK: {pdf_count} PDFs =====")
    pdfs = ALL_PDFS[:pdf_count]
    benchmark_raw_dir = DATA_DIR / f"benchmark_{pdf_count}_raw"
    setup_pdfs(pdfs, benchmark_raw_dir)
    
    clean_dirs()
    
    monitor = ResourceMonitor()
    monitor.start()
    
    t_start = time.time()
    
    # Stage 1
    s1_time = run_stage([
        STAGE1, 
        "--pdf-input", str(benchmark_raw_dir),
        "--move-processed", "false"
    ], "Stage 1")
    
    # Stage 2
    s2_time = run_stage([STAGE2, "--force"], "Stage 2")
    
    # Stage 3
    s3_time = run_stage([STAGE3], "Stage 3")
    
    t_end = time.time()
    monitor.stop()
    monitor.join()
    
    total_time = t_end - t_start
    
    # Stats
    pages = count_jsonl(RUN_STATE_DIR / "page_manifest.jsonl")
    crops = count_jsonl(RUN_STATE_DIR / "crop_manifest.jsonl")
    ocr_stats = get_ocr_stats()
    
    results = {
        "pdf_count": pdf_count,
        "total_time": total_time,
        "s1_time": s1_time,
        "s2_time": s2_time,
        "s3_time": s3_time,
        "total_pages": pages,
        "total_crops": crops,
        "pages_per_sec": pages / total_time if total_time > 0 else 0,
        "crops_per_sec": crops / total_time if total_time > 0 else 0,
        "peak_ram": max(h["ram"] for h in monitor.history) if monitor.history else 0,
        "peak_vram": max(h["vram"] for h in monitor.history) if monitor.history else 0,
        "downscaled_count": ocr_stats["downscaled"],
        "tiled_count": ocr_stats["tiled"],
        "history": monitor.history
    }
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", nargs="+", type=int, default=[1, 13])
    args = parser.parse_args()

    final_data = {}
    if os.path.exists("benchmark_results.json"):
        try:
            with open("benchmark_results.json", "r") as f:
                final_data = json.load(f)
        except:
            pass

    for count in args.counts:
        res = run_benchmark(count)
        final_data[f"benchmark_{count}"] = res
        
        # Save intermediate results
        with open("benchmark_results.json", "w") as f:
            json.dump(final_data, f, indent=2)
    
    print("\nBenchmark Complete. Results saved to benchmark_results.json")

if __name__ == "__main__":
    main()
