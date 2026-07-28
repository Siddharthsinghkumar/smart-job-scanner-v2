import subprocess
import time
import json
import sys
import os
from pathlib import Path

def run_bench(version, script_path, pdf_path):
    print(f"[{version}] Running on {pdf_path.name}...")
    start = time.time()
    cmd = [sys.executable, script_path, "--pdf", str(pdf_path)]
    # v2 uses --pdf but also might need other flags, let's assume standard
    res = subprocess.run(cmd, capture_output=True, text=True)
    end = time.time()
    return end - start

def audit(pdf_stem):
    # Uses our already created generic_audit.py
    cmd = [sys.executable, "tools/generic_audit.py", pdf_stem]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip()

corpus = [
    "UHT Delhi 07-04",
    "Mint Delhi 07-04",
    "The Tribune Delhi 07-04"
]

results = []

for name in corpus:
    pdf = Path(f"data/benchmark_13_raw/{name}.pdf")
    
    # Run v2
    dur_v2 = run_bench("v2", "src/pipeline/stage02_block_detection.py", pdf)
    aud_v2 = audit(name)
    
    # Run v11.1
    dur_v11 = run_bench("v11.1", "src/pipeline/stage02_block_detection_v6.py", pdf)
    aud_v11 = audit(name)
    
    results.append({
        "name": name,
        "v2": {"dur": dur_v2, "audit": aud_v2},
        "v11.1": {"dur": dur_v11, "audit": aud_v11}
    })

print(json.dumps(results, indent=2))
