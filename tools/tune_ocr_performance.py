
import os
import time
import subprocess
import json
from pathlib import Path

# Config
CROP_MANIFEST = "run_state/crop_manifest.jsonl"
OUTPUT_DIR = Path("logs/tuning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Grid Search Parameters
PARAM_GRID = {
    "OCR_BATCH_SIZE": [1, 4, 8],
    "OCR_TASK_QUEUE_SIZE": [64, 128],
    "OCR_CORE_LIMIT": [4, 6, 8],
    "OCR_CPU_WORKERS": [0]
}

RESULTS = []

def run_benchmark(params, manifest_path):
    env = os.environ.copy()
    for k, v in params.items():
        env[k] = str(v)
    
    print(f"Testing {manifest_path}: {params}")
    start_time = time.time()
    
    try:
        cmd = ["./4_env/bin/python", "src/pipeline/stage03_ocr.py", "--crop-manifest", manifest_path]
        # Use a timeout appropriate for the manifest size
        # 1 PDF (161 crops) ~ 5 mins
        # 13 PDFs (~2000 crops) ~ 60 mins
        timeout = 1800 if "161" in str(manifest_path) or "1_raw" in str(manifest_path) else 7200
        
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start_time
        
        if proc.returncode != 0:
            print(f"FAILED: {proc.stderr}")
            return {"error": proc.stderr, "params": params, "manifest": str(manifest_path)}
        
        reported_elapsed = 0
        lines = proc.stdout.splitlines()
        for line in lines:
            if "Total elapsed:" in line:
                try:
                    reported_elapsed = float(line.split(":")[1].split("s")[0].strip())
                except:
                    pass
        
        # Extract RAM/VRAM info from log
        log_file = None
        for line in lines:
            if "log              =" in line:
                log_file = line.split("=")[1].strip()
        
        peak_rss = 0
        peak_vram = 0
        if log_file and Path(log_file).exists():
            # Check for the subprocess log which has the ProcRSS entries
            # Log file is step03_ocr_parent_...
            # We need the sub log: step03_ocr_sub_...
            sub_logs = list(Path("logs").glob("step03_ocr_sub_*.log"))
            sub_logs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            if sub_logs:
                with open(sub_logs[0], "r") as f:
                    for l in f:
                        if "ProcRSS=" in l:
                            try:
                                rss = float(l.split("ProcRSS=")[1].split("MB")[0])
                                peak_rss = max(peak_rss, rss)
                            except: pass
                        if "VRAM" in l and "/" in l:
                            try:
                                vram = float(l.split(":")[1].split("/")[0].strip())
                                peak_vram = max(peak_vram, vram)
                            except: pass

        return {
            "params": params,
            "manifest": str(manifest_path),
            "elapsed": elapsed,
            "reported_elapsed": reported_elapsed,
            "peak_rss_mb": peak_rss,
            "peak_vram_mb": peak_vram,
            "success": True
        }
    except subprocess.TimeoutExpired:
        return {"error": "Timeout", "params": params, "manifest": str(manifest_path)}

def main():
    import itertools
    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    manifests = ["run_state/crop_manifest.jsonl"] # Start with 1 PDF
    
    print(f"Starting Grid Search: {len(combinations)} combinations across {len(manifests)} manifests")
    
    for manifest in manifests:
        for i, combo in enumerate(combinations):
            print(f"\nRun {i+1}/{len(combinations)} for {manifest}")
            res = run_benchmark(combo, manifest)
            RESULTS.append(res)
            
            with open(OUTPUT_DIR / "tuning_results.json", "w") as f:
                json.dump(RESULTS, f, indent=2)

    # Report for 1 PDF
    results_1pdf = [r for r in RESULTS if r.get("success") and "crop_manifest.jsonl" in r["manifest"]]
    if results_1pdf:
        best_1pdf = min(results_1pdf, key=lambda x: x["reported_elapsed"])
        print("\nBEST CONFIG FOR 1 PDF:")
        print(json.dumps(best_1pdf, indent=2))

if __name__ == "__main__":
    main()
