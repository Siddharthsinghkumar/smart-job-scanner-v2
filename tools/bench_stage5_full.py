import subprocess
import time
import os
import json
from pathlib import Path

def audit(manifest_path, pdf_stem):
    cmd = ["./4_env/bin/python", "tools/generic_audit.py", str(manifest_path), pdf_stem]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip()

def run_challenge():
    pdf_dir = Path("data/continuum_test")
    pdfs = sorted(list(pdf_dir.glob("*.pdf")))
    
    print(f"{'PDF Name':<25} | {'S4 TP':<5} | {'S5 TP':<5} | {'Time Total'}")
    print("-" * 65)
    
    total_start = time.time()
    results = []

    for pdf in pdfs:
        # 1. Pipeline S1-3
        subprocess.run(["./4_env/bin/python", "scripts/run_supersonic_v16_6.py", "--dir", str(pdf_dir), "--only", pdf.name], capture_output=True)
        # 2. Stage 4
        subprocess.run(["./4_env/bin/python", "src/pipeline/stage04_candidate_scorer.py"], capture_output=True)
        # 3. Stage 5 (Vector)
        s5_start = time.time()
        subprocess.run(["./4_env/bin/python", "src/pipeline/stage05_vector_surgeon.py"], capture_output=True)
        
        # Audit
        aud_s4 = audit("run_state/stage4_final_ads.jsonl", pdf.stem)
        aud_s5 = audit("run_state/stage5_vector_ads.jsonl", pdf.stem)
        
        _, gt, tp4, _ = aud_s4.split("|")
        _, _, tp5, det5 = aud_s5.split("|")
        
        total_pdf_time = time.time() - total_start # Cumulative
        print(f"{pdf.stem[:25]:<25} | {tp4:<5} | {tp5:<5} | {det5:<7} | {total_pdf_time:>7.1f}s")
        
        results.append({
            "name": pdf.name, "tp": int(tp5), "gt": int(gt), "time": total_pdf_time
        })

    total_dur = time.time() - total_start
    print("-" * 65)
    print(f"CHALLENGE TOTAL: {total_dur:.2f}s (Avg: {total_dur/len(pdfs):.2f}s/PDF)")

if __name__ == "__main__":
    run_challenge()
