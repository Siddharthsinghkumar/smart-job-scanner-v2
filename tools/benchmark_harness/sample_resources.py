#!/usr/bin/env python3
import time
import json
import psutil
import subprocess
import argparse
from pathlib import Path

def get_gpu_stats():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        util, mem = map(int, output.strip().split(","))
        return util, mem
    except:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_file = run_dir / "resources.jsonl"
    
    with open(out_file, "w") as f:
        while True:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory()
                gpu_util, vram = get_gpu_stats()
                disk_io = psutil.disk_io_counters()
                
                record = {
                    "ts": time.time(),
                    "cpu_percent": cpu_percent,
                    "ram_used_mb": ram.used / (1024 * 1024),
                    "gpu_util": gpu_util,
                    "vram_used_mb": vram,
                    "disk_read_bytes": disk_io.read_bytes if disk_io else 0,
                    "disk_write_bytes": disk_io.write_bytes if disk_io else 0,
                    "disk_read_count": disk_io.read_count if disk_io else 0,
                    "disk_write_count": disk_io.write_count if disk_io else 0
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                time.sleep(0.3)
            except KeyboardInterrupt:
                break
            except Exception:
                time.sleep(0.3)

if __name__ == "__main__":
    main()
