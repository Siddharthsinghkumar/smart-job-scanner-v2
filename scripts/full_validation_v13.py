import subprocess
import time
import json
import sys
import os
from pathlib import Path

def run_cmd(name, cmd):
    print(f"\n>>> Starting {name}...")
    start = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    end = time.time()
    if res.returncode != 0:
        print(f"Error in {name}: {res.stderr}")
    return end - start

pdf = "data/benchmark_13_raw/UHT Delhi 07-04.pdf"
pdf_stem = "UHT Delhi 07-04"

# Cleanup
os.system("rm -rf run_state/* data/pdf2img/* data/crops/*")

# Resource Watcher (1s interval)
watcher = subprocess.Popen(["./tools/hardware_watcher.sh"], stdout=open("hw_full_log.txt", "w"))

print(f"🚀 Full-Spectrum Validation v13: {pdf_stem}")
t1 = run_cmd("Stage 1 (Render)", ["./4_env/bin/python", "src/pipeline/stage01_pdf_to_images.py", "--pdf", pdf])
t2 = run_cmd("Stage 2 (Detection)", ["./4_env/bin/python", "src/pipeline/stage02_block_detection_v6.py", "--pdf", pdf])

# Audit S2
res_s2 = subprocess.run(["./4_env/bin/python", "tools/generic_audit.py", "run_state/crop_manifest.jsonl", pdf_stem], capture_output=True, text=True).stdout.strip()

t3 = run_cmd("Stage 3 (OCR)", ["./4_env/bin/python", "src/pipeline/stage03_ocr.py"])

# Audit S3
res_s3 = subprocess.run(["./4_env/bin/python", "tools/generic_audit.py", "run_state/ocr_manifest.jsonl", pdf_stem], capture_output=True, text=True).stdout.strip()

watcher.terminate()

def parse_audit(res):
    parts = res.split("|")
    if parts[0] != "OK": return 0, 0, 0
    return int(parts[1]), int(parts[2]), int(parts[3])

gt, tp_s2, det_s2 = parse_audit(res_s2)
_, tp_s3, det_s3 = parse_audit(res_s3)

print("\n" + "="*50)
print(f"FINAL REPORT: {pdf_stem}")
print("="*50)
print(f"GT Total Ads : {gt}")
print(f"S2 TP / Det  : {tp_s2} / {det_s2} ({float(tp_s2)*100/gt:.1f}% recall)")
print(f"S3 TP / Det  : {tp_s3} / {det_s3} ({float(tp_s3)*100/gt:.1f}% recall)")
print("-"*50)
print(f"S1 Time      : {t1:.2f}s")
print(f"S2 Time      : {t2:.2f}s")
print(f"S3 Time      : {t3:.2f}s")
print(f"TOTAL TIME   : {t1+t2+t3:.2f}s")
print("="*50)
