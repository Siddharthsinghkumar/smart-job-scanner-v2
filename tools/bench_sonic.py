import subprocess
import time
import os
import shutil
from pathlib import Path

corpus = [
    "UHT Delhi 07-04",
    "Mint Delhi 07-04",
    "TH- Delhi 07-04",
    "TOI-Delhi 07-04",
    "The Tribune Delhi 07-04",
    "BS English-Delhi 07-04",
    "ET-Delhi 07-04",
    "FE-Delhi 07-04",
    "NIE Chennai 07-04",
    "Orissa-Post-07-04"
]

print(f"{'PDF Name':<25} | {'S2 TP':<5} | {'S3 TP':<5} | {'Time'}")
print("-" * 55)

for name in corpus:
    pdf = f"data/benchmark_13_raw/{name}.pdf"
    if not os.path.exists(pdf): continue
    
    # ATOMIC CLEANUP: remove and recreate dirs to avoid "argument list too long"
    for d in ["run_state", "data/crops", "data/pdf2img"]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    start = time.time()
    # Stage 2
    subprocess.run(["./4_env/bin/python", "src/pipeline/stage02_block_detection_v6.py", "--pdf", pdf], capture_output=True)
    # Stage 3
    subprocess.run(["./4_env/bin/python", "src/pipeline/stage03_ocr.py"], capture_output=True)
    dur = time.time() - start
    
    # Audit
    def get_tp(manifest):
        res = subprocess.run(["./4_env/bin/python", "tools/generic_audit.py", manifest, name], capture_output=True, text=True).stdout.strip()
        parts = res.split("|")
        return parts[2] if parts[0] == "OK" else "0"

    tp2 = get_tp("run_state/crop_manifest.jsonl")
    tp3 = get_tp("run_state/ocr_manifest.jsonl")
    
    print(f"{name[:25]:<25} | {tp2:<5} | {tp3:<5} | {dur:.1f}s")
