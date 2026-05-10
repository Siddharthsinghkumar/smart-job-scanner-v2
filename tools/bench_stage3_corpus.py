import subprocess
import time
import json
import sys
import os
import shutil
from pathlib import Path

def audit(manifest_path, pdf_stem):
    cmd = [sys.executable, "tools/generic_audit.py", manifest_path, pdf_stem]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip()

corpus = [
    "UHT Delhi 07-04",
    "BS English-Delhi 07-04",
    "ET-Delhi 07-04",
    "FE-Delhi 07-04",
    "IE-Delhi 07-04",
    "Mint Delhi 07-04",
    "NIE Chennai 07-04",
    "Orissa-Post-07-04",
    "Statesman-Delhi 07-04",
    "TH- Delhi 07-04"
]

results = []
print(f"{'PDF Name':<25} | {'S2 TP':<5} | {'S3 TP':<5} | {'S3 Time':<8} | {'Total'}")
print("-" * 75)

for name in corpus:
    pdf = Path(f"data/benchmark_13_raw/{name}.pdf")
    if not pdf.exists(): continue
    
    # 1. Faster Cleanup
    for d in ["run_state", "data/pdf2img", "data/crops"]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    # 2. Stage 2
    t2_start = time.time()
    subprocess.run([sys.executable, "src/pipeline/stage02_block_detection_v6.py", "--pdf", str(pdf)], capture_output=True)
    t2_dur = time.time() - t2_start
    aud_s2 = audit("run_state/crop_manifest.jsonl", name)
    _, gt, tp_s2, _ = aud_s2.split("|")
    
    # 3. Stage 3
    t3_start = time.time()
    subprocess.run([sys.executable, "src/pipeline/stage03_ocr.py"], capture_output=True)
    t3_dur = time.time() - t3_start
    aud_s3 = audit("run_state/ocr_manifest.jsonl", name)
    _, _, tp_s3, _ = aud_s3.split("|")
    
    total_dur = t2_dur + t3_dur
    print(f"{name[:25]:<25} | {tp_s2:<5} | {tp_s3:<5} | {t3_dur:>6.1f}s | {total_dur:>6.1f}s")
    
    results.append({
        "name": name, "tp_s2": int(tp_s2), "tp_s3": int(tp_s3),
        "gt": int(gt), "dur_s2": t2_dur, "dur_s3": t3_dur, "total": total_dur
    })

if results:
    avg_s3 = sum(r['dur_s3'] for r in results) / len(results)
    avg_total = sum(r['total'] for r in results) / len(results)
    print("-" * 75)
    print(f"BEST TIME    : {min(r['total'] for r in results):.2f}s")
    print(f"AVG TOTAL    : {avg_total:.2f}s (OCR Avg: {avg_s3:.2f}s)")
    print(f"WORST TIME   : {max(r['total'] for r in results):.2f}s")
    print(f"CORPUS TOTAL : {sum(r['total'] for r in results):.2f}s")
