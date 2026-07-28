
import os
import time
import subprocess
import json
from pathlib import Path

# Config
CROP_MANIFEST = "run_state/crop_manifest.jsonl"
OUTPUT_DIR = Path("logs/tuning_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Search space
PARAM_GRID = {
    "OCR_BATCH_SIZE": [1, 4, 8],
    "OCR_CORE_LIMIT": [6, 8]
}

RESULTS = []

def run_benchmark(params, manifest_path, run_idx):
    env = os.environ.copy()
    for k, v in params.items():
        env[k] = str(v)
    env["OCR_CPU_WORKERS"] = "0"
    
    print(f"Testing {manifest_path}: {params}")
    start_time = time.time()
    
    try:
        cmd = ["./4_env/bin/python", "src/pipeline/stage03_ocr.py", "--crop-manifest", manifest_path]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
        elapsed = time.time() - start_time
        
        # Save logs
        with open(OUTPUT_DIR / f"run_{run_idx}.stdout.log", "w") as f: f.write(proc.stdout)
        with open(OUTPUT_DIR / f"run_{run_idx}.stderr.log", "w") as f: f.write(proc.stderr)
        
        reported_elapsed = 0
        # Combine stdout and stderr for parsing
        full_output = proc.stdout + proc.stderr
        for line in full_output.splitlines():
            if "Total elapsed:" in line:
                try: 
                    # Handle "Total elapsed: 249.7s (4.2 min)"
                    reported_elapsed = float(line.split("Total elapsed:")[1].split("s")[0].strip())
                except: pass
        
        return {
            "params": params,
            "manifest": str(manifest_path),
            "reported_elapsed": reported_elapsed,
            "wall_elapsed": elapsed,
            "success": proc.returncode == 0,
            "returncode": proc.returncode
        }
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {"error": "Timeout", "params": params, "success": False}

def main():
    import itertools
    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for i, combo in enumerate(combinations):
        print(f"\nRun {i+1}/{len(combinations)}")
        res = run_benchmark(combo, CROP_MANIFEST, i+1)
        RESULTS.append(res)
        with open(OUTPUT_DIR / "tuning_results.json", "w") as f:
            json.dump(RESULTS, f, indent=2)

    best = min([r for r in RESULTS if r.get("success") and r.get("reported_elapsed", 0) > 0], 
               key=lambda x: x["reported_elapsed"], default=None)
    print("\nBEST CONFIG:")
    print(json.dumps(best, indent=2))

if __name__ == "__main__":
    main()
