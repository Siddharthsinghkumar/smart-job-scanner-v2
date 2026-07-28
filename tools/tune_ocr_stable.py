
import os
import time
import subprocess
import json
from pathlib import Path

# Config
OUTPUT_DIR = Path("logs/tuning_stable")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Search space - Reduced for stability and time
PARAM_GRID = {
    "OCR_BATCH_SIZE": [1, 4],
    "OCR_CORE_LIMIT": [6]
}

MANIFESTS = {
    "1_pdf": "run_state/crop_manifest.jsonl",
    "13_pdf": "data/bench13_workspace/run_state/crop_manifest.jsonl"
}

RESULTS = []

def run_benchmark(params, manifest_name, manifest_path, run_idx):
    env = os.environ.copy()
    for k, v in params.items():
        env[k] = str(v)
    env["OCR_CPU_WORKERS"] = "0"
    
    print(f"\n[RUN {run_idx}] Testing {manifest_name}: {params}")
    start_time = time.time()
    
    try:
        timeout = 7200 if "13" in manifest_name else 1200
        cmd = ["./4_env/bin/python", "src/pipeline/stage03_ocr.py", "--crop-manifest", manifest_path]
        if "13" in manifest_name:
            cmd += ["--crops-dir", "data/bench13_workspace/crops"]
            
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start_time
        
        log_prefix = f"{manifest_name}_run{run_idx}"
        with open(OUTPUT_DIR / f"{log_prefix}.stdout.log", "w") as f: f.write(proc.stdout)
        with open(OUTPUT_DIR / f"{log_prefix}.stderr.log", "w") as f: f.write(proc.stderr)
        
        reported_elapsed = 0
        full_output = proc.stdout + proc.stderr
        for line in full_output.splitlines():
            if "Total elapsed:" in line:
                try: reported_elapsed = float(line.split("Total elapsed:")[1].split("s")[0].strip())
                except: pass
        
        peak_rss = 0
        for line in full_output.splitlines():
            if "ProcRSS=" in line:
                try: peak_rss = max(peak_rss, float(line.split("ProcRSS=")[1].split("MB")[0]))
                except: pass

        return {
            "manifest": manifest_name,
            "params": params,
            "reported_elapsed": reported_elapsed,
            "wall_elapsed": elapsed,
            "peak_rss_mb": peak_rss,
            "success": proc.returncode == 0 and reported_elapsed > 0
        }
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {"manifest": manifest_name, "params": params, "error": "Timeout", "success": False}

def main():
    import itertools
    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    run_counter = 1
    for m_name, m_path in MANIFESTS.items():
        if not Path(m_path).exists():
            continue
        for combo in combinations:
            res = run_benchmark(combo, m_name, m_path, run_counter)
            RESULTS.append(res)
            run_counter += 1
            with open(OUTPUT_DIR / "tuning_results.json", "w") as f:
                json.dump(RESULTS, f, indent=2)

    print("\nDONE")

if __name__ == "__main__":
    main()
