
import os
import time
import subprocess
import json
from pathlib import Path

# Config
OUTPUT_DIR = Path("logs/tuning_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Search space
PARAM_GRID = {
    "OCR_BATCH_SIZE": [1, 4, 8],
    "OCR_CORE_LIMIT": [6, 12] # Test 6 and 12 cores
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
    
    print(f"\n[RUN {run_idx}] Testing {manifest_name} ({manifest_path}): {params}")
    start_time = time.time()
    
    try:
        # Increase timeout for 13 PDFs
        timeout = 7200 if "13" in manifest_name else 1200
        
        cmd = ["./4_env/bin/python", "src/pipeline/stage03_ocr.py", "--crop-manifest", manifest_path]
        # For 13 PDFs, we need to point to the right crops dir too
        if "13" in manifest_name:
            cmd += ["--crops-dir", "data/bench13_workspace/crops"]
            
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start_time
        
        # Save logs
        log_prefix = f"{manifest_name}_run{run_idx}"
        with open(OUTPUT_DIR / f"{log_prefix}.stdout.log", "w") as f: f.write(proc.stdout)
        with open(OUTPUT_DIR / f"{log_prefix}.stderr.log", "w") as f: f.write(proc.stderr)
        
        reported_elapsed = 0
        full_output = proc.stdout + proc.stderr
        for line in full_output.splitlines():
            if "Total elapsed:" in line:
                try: 
                    reported_elapsed = float(line.split("Total elapsed:")[1].split("s")[0].strip())
                except: pass
        
        # Peak RAM from logs
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
            print(f"Skipping {m_name}, manifest not found: {m_path}")
            continue
            
        for combo in combinations:
            res = run_benchmark(combo, m_name, m_path, run_counter)
            RESULTS.append(res)
            run_counter += 1
            with open(OUTPUT_DIR / "tuning_results.json", "w") as f:
                json.dump(RESULTS, f, indent=2)

    # Final Report
    print("\n" + "="*60)
    print("FINAL PERFORMANCE REPORT")
    print("="*60)
    
    for m_name in MANIFESTS.keys():
        m_results = [r for r in RESULTS if r["manifest"] == m_name and r["success"]]
        if not m_results:
            print(f"\nNo successful runs for {m_name}")
            continue
            
        best = min(m_results, key=lambda x: x["reported_elapsed"])
        print(f"\nBEST FOR {m_name}:")
        print(f"  Params: {best['params']}")
        print(f"  Time:   {best['reported_elapsed']:.1f}s")
        print(f"  RAM:    {best['peak_rss_mb']:.1f}MB")
        
        # Throughput
        crops = 161 if m_name == "1_pdf" else 3841
        print(f"  Throughput: {crops / best['reported_elapsed']:.2f} crops/sec")

if __name__ == "__main__":
    main()
