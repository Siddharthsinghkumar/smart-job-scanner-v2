import subprocess
import time
import json
import sys
import os
from pathlib import Path

def audit(pdf_stem):
    cmd = [sys.executable, "tools/generic_audit.py", pdf_stem]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip()

corpus = [
    "UHT Delhi 07-04",
    "Statesman-Delhi 07-04",
    "The Tribune Delhi 07-04",
    "TOI-Delhi 07-04",
    "TH- Delhi 07-04"
]

results = []

for name in corpus:
    pdf = Path(f"data/benchmark_13_raw/{name}.pdf")
    if not pdf.exists(): continue
    
    # --- Benchmark v2 (Baseline) ---
    os.system("rm -rf run_state/* data/pdf2img/* data/crops/*")
    subprocess.run([sys.executable, "src/pipeline/stage01_pdf_to_images.py", "--pdf", str(pdf)], capture_output=True)
    start_v2 = time.time()
    subprocess.run([sys.executable, "src/pipeline/stage02_block_detection.py"], capture_output=True)
    dur_v2 = time.time() - start_v2
    aud_v2 = audit(name)
    
    # --- Benchmark v12.0 (Hybrid) ---
    os.system("rm -rf run_state/* data/pdf2img/* data/crops/*")
    start_hyb = time.time()
    subprocess.run([sys.executable, "src/pipeline/stage02_block_detection_hybrid.py", "--pdf", str(pdf)], capture_output=True)
    dur_hyb = time.time() - start_hyb
    aud_hyb = audit(name)
    
    results.append({
        "name": name,
        "v2": {"dur": dur_v2, "audit": aud_v2},
        "hybrid": {"dur": dur_hyb, "audit": aud_hyb}
    })

print(json.dumps(results, indent=2))
