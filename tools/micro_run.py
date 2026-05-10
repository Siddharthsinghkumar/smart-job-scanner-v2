import fitz
import os
import sys
from pathlib import Path
import subprocess

# Simple script to run 2 pages for schema extraction
def micro_run():
    pdf = "data/benchmark_13_raw/UHT Delhi 07-04.pdf"
    doc = fitz.open(pdf)
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=13, to_page=14) # The "Hot Pages"
    new_pdf = "data/micro_uht.pdf"
    new_doc.save(new_pdf)
    new_doc.close()
    doc.close()
    
    # Run the fast v15.6 engine on this 2-page PDF
    os.system("mkdir -p data/micro_test")
    os.system(f"cp {new_pdf} data/micro_test/")
    subprocess.run(["./4_env/bin/python", "scripts/run_supersonic_v15_4.py", "--dir", "data/micro_test"])

if __name__ == "__main__":
    micro_run()
