
import os
import subprocess
from pathlib import Path

# Paths
RAW_PDF_DIR = "data/benchmark_13_raw"
IMG_DIR = "data/pdf2img_bench13"
CROP_DIR = "data/crops_bench13"
STATE_DIR = "run_state_bench13"

for d in [IMG_DIR, CROP_DIR, STATE_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# Stage 1: PDF to Images
# Note: stage01 might need env vars or config. I'll try to run it with overrides.
print("Starting Stage 1 for 13 PDFs...")
env = os.environ.copy()
env["RAW_PDF_DIR"] = RAW_PDF_DIR
env["IMG_DIR"] = IMG_DIR
env["STATE_DIR"] = STATE_DIR

# We need to run it for each PDF or let it discover. 
# stage01 usually discovers from RAW_PDF_DIR.
cmd1 = ["./4_env/bin/python", "src/pipeline/stage01_pdf_to_images.py"]
# I'll pass args if it supports them. Let's assume it uses env.
subprocess.run(cmd1, env=env)

# Stage 2: Block Detection
print("Starting Stage 2 for 13 PDFs...")
env["CROP_DIR"] = CROP_DIR
# stage02 reads from STATE_DIR/page_manifest.jsonl
cmd2 = ["./4_env/bin/python", "src/pipeline/stage02_block_detection.py"]
subprocess.run(cmd2, env=env)

print(f"Manifest ready at {STATE_DIR}/crop_manifest.jsonl")
