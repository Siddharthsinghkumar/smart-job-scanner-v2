
import os
import time
import subprocess
import json
from pathlib import Path

# Config
CROP_MANIFEST = "run_state/crop_manifest.jsonl"
OUTPUT_DIR = Path("logs/tuning_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Search space
PARAM_GRID = {
    "OCR_BATCH_SIZE": [1, 2, 4],
    "OCR_CORE_LIMIT": [6, 8],
    "OCR_TASK_QUEUE_SIZE": [64]
}

RESULTS = []

def run_benchmark(params, manifest_path):
    env = os.environ.copy()
    for k, v in params.items():
        env[k] = str(v)
    env["OCR_CPU_WORKERS"] = "0"
    
    print(f"Testing {manifest_path}: {params}")
    start_time = time.time()
    
    try:
        cmd = ["./4_env/bin/python", "src/pipeline/stage03_ocr.py", "--crop-manifest", manifest_path]
        # Use a timeout of 10 mins for 1 PDF
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - start_time
        
        if proc.returncode != 0:
            return {"error": proc.stderr, "params": params, "success": False}
        
        reported_elapsed = 0
        for line in proc.stdout.splitlines():
            if "Total elapsed:" in line:
                try: reported_elapsed = float(line.split(":")[1].split("s")[0].strip())
                except: pass
        
        return {
            "params": params,
            "manifest": str(manifest_path),
            "reported_elapsed": reported_elapsed,
            "success": True
        }
    except subprocess.TimeoutExpired:
        # Kill the process group if it hangs
        print("TIMEOUT - killing process")
        return {"error": "Timeout", "params": params, "success": False}

def main():
    import itertools
    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for i, combo in enumerate(combinations):
        print(f"\nRun {i+1}/{len(combinations)}")
        res = run_benchmark(combo, CROP_MANIFEST)
        RESULTS.append(res)
        with open(OUTPUT_DIR / "tuning_results.json", "w") as f:
            json.dump(RESULTS, f, indent=2)

    best = min([r for r in RESULTS if r.get("success")], key=lambda x: x["reported_elapsed"], default=None)
    print("\nBEST CONFIG:")
    print(json.dumps(best, indent=2))

if __name__ == "__main__":
    main()
