
import os
import time
import subprocess
import json
from pathlib import Path

# Config
CROP_MANIFEST = "run_state/crop_manifest_subset.jsonl"
OUTPUT_DIR = Path("logs/tuning_subset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Search space
PARAM_GRID = {
    "OCR_BATCH_SIZE": [1, 4, 8],
    "OCR_CORE_LIMIT": [6, 12]
}

RESULTS = []

def run_benchmark(params, run_idx):
    env = os.environ.copy()
    for k, v in params.items():
        env[k] = str(v)
    env["OCR_CPU_WORKERS"] = "0"
    
    print(f"Testing Subset: {params}")
    start_time = time.time()
    
    try:
        cmd = ["./4_env/bin/python", "src/pipeline/stage03_ocr.py", "--crop-manifest", CROP_MANIFEST]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time
        
        reported_elapsed = 0
        full_output = proc.stdout + proc.stderr
        for line in full_output.splitlines():
            if "Total elapsed:" in line:
                try: reported_elapsed = float(line.split("Total elapsed:")[1].split("s")[0].strip())
                except: pass
        
        return {
            "params": params,
            "reported_elapsed": reported_elapsed,
            "wall_elapsed": elapsed,
            "success": proc.returncode == 0 and reported_elapsed > 0
        }
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {"params": params, "error": "Timeout", "success": False}

def main():
    import itertools
    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for i, combo in enumerate(combinations):
        res = run_benchmark(combo, i+1)
        RESULTS.append(res)
        print(f"  Result: {res.get('reported_elapsed')}s")

    print("\nSUMMARY:")
    for r in RESULTS:
        print(f"{r['params']} -> {r.get('reported_elapsed')}s")

if __name__ == "__main__":
    main()
