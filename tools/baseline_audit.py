import subprocess
import time
import os
import json
from pathlib import Path

def audit(manifest_path, pdf_stem):
    cmd = ["./4_env/bin/python", "tools/generic_audit.py", str(manifest_path), pdf_stem]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip()

def run_baseline():
    pdf_dir = Path("data/continuum_test")
    pdfs = sorted(list(pdf_dir.glob("*.pdf")))
    
    print(f"{'PDF Name':<25} | {'TP/GT':<8} | {'Time S1-3':<10} | {'Time S4':<8} | {'Total'}")
    print("-" * 65)
    
    total_start = time.time()
    results = []

    for pdf in pdfs:
        # 1. Stage 1-3
        s13_start = time.time()
        # We call the script but only for this PDF to keep it simple in the wrapper
        subprocess.run(["./4_env/bin/python", "scripts/run_supersonic_v16_6.py", "--dir", str(pdf_dir), "--only", pdf.name], capture_output=True)
        s13_time = time.time() - s13_start
        
        # 2. Stage 4
        s4_start = time.time()
        subprocess.run(["./4_env/bin/python", "src/pipeline/stage04_candidate_scorer.py"], capture_output=True)
        s4_time = time.time() - s4_start
        
        # 3. Audit
        audit_res = audit("run_state/stage4_final_ads.jsonl", pdf.stem)
        status, gt, tp, det = audit_res.split("|")
        
        total_pdf_time = s13_time + s4_time
        print(f"{pdf.stem[:25]:<25} | {tp}/{gt:<5} | {s13_time:>8.1f}s | {s4_time:>7.2f}s | {total_pdf_time:>7.1f}s")
        
        results.append({
            "name": pdf.name,
            "tp": int(tp),
            "gt": int(gt),
            "time": total_pdf_time
        })

    total_dur = time.time() - total_start
    print("-" * 65)
    print(f"BASELINE TOTAL: {total_dur:.2f}s (Avg: {total_dur/len(pdfs):.2f}s/PDF)")

if __name__ == "__main__":
    # Small patch to run_supersonic_v16_6.py to allow --only flag
    with open("scripts/run_supersonic_v16_6.py", "r") as f:
        code = f.read()
    if 'parser.add_argument("--only",' not in code:
        code = code.replace('parser.add_argument("--dir", required=True)', 
                            'parser.add_argument("--dir", required=True)\n    parser.add_argument("--only", help="Process only this filename")')
        code = code.replace('pdfs = sorted(list(Path(pdf_dir).glob("*.pdf")))',
                            'pdfs = sorted(list(Path(pdf_dir).glob("*.pdf")))\n    if args.only: pdfs = [p for p in pdfs if p.name == args.only]')
        with open("scripts/run_supersonic_v16_6.py", "w") as f:
            f.write(code)
            
    run_baseline()
