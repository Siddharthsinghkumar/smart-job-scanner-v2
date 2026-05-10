import subprocess
import time
import json
import sys
import os
from pathlib import Path

def audit(pdf_stem):
    cmd = [sys.executable, "tools/generic_audit.py", "run_state/crop_manifest.jsonl", pdf_stem]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip()

corpus = [
    "BS English-Delhi 07-04",
    "ET-Delhi 07-04",
    "FE-Delhi 07-04",
    "IE-Delhi 07-04",
    "Mint Delhi 07-04",
    "NIE Chennai 07-04",
    "Orissa-Post-07-04",
    "Statesman-Delhi 07-04",
    "TH- Delhi 07-04",
    "TOI-Delhi 07-04"
]

results = []
print(f"{'PDF Name':<30} | {'TP/GT':<8} | {'Recall':<8} | {'FP':<8} | {'Time':<6}")
print("-" * 75)

for name in corpus:
    pdf = Path(f"data/benchmark_13_raw/{name}.pdf")
    if not pdf.exists(): continue
    
    os.system("rm -rf run_state/* data/crops/*")
    start = time.time()
    # Run v11.1 logic (stage02_block_detection_v6.py)
    subprocess.run([sys.executable, "src/pipeline/stage02_block_detection_v6.py", "--pdf", str(pdf)], capture_output=True)
    dur = time.time() - start
    
    aud = audit(name)
    parts = aud.split("|")
    if parts[0] == "OK":
        gt, tp, det = parts[1], parts[2], parts[3]
        rec = f"{float(tp)*100/float(gt):.1f}%" if float(gt)>0 else "N/A"
        fp = int(det) - int(tp)
        print(f"{name[:30]:<30} | {tp}/{gt:<5} | {rec:<8} | {fp:<8} | {dur:.1f}s")
        results.append({"dur": dur, "tp": int(tp), "gt": int(gt)})
    else:
        print(f"{name[:30]:<30} | ERROR    | N/A      | N/A      | {dur:.1f}s")

if results:
    avg_dur = sum(r['dur'] for r in results) / len(results)
    avg_rec = sum(r['tp'] for r in results) * 100 / sum(r['gt'] for r in results)
    print("-" * 75)
    print(f"AVG TIME PER PDF: {avg_dur:.2f}s")
    print(f"AVG CORPUS RECALL: {avg_rec:.2f}%")
