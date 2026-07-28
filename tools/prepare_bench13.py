
import os
import sys
import shutil
from pathlib import Path

# Paths
RAW_PDF_DIR = Path("data/benchmark_13_raw")
BENCH_ROOT = Path("data/bench13_workspace")
IMG_DIR = BENCH_ROOT / "pdf2img"
CROP_DIR = BENCH_ROOT / "crops"
STATE_DIR = BENCH_ROOT / "run_state"

for d in [IMG_DIR, CROP_DIR, STATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("--- STAGE 1: PDF TO IMAGES ---")
cmd1 = [
    "./4_env/bin/python", "src/pipeline/stage01_pdf_to_images.py",
    "--pdf-input", str(RAW_PDF_DIR),
    "--images-output", str(IMG_DIR),
    "--manifest-output", str(STATE_DIR / "page_manifest.jsonl"),
    "--workers", "8"
]
os.system(" ".join(cmd1))

print("--- STAGE 2: BLOCK DETECTION ---")
# We need to tell stage02 where to look for images and blocks
# Since stage02 is stubborn about config, we'll patch it or use env vars if it supports them.
# Actually, stage02 uses get_path("images_output", config).
# We'll create a temp config.
config_path = BENCH_ROOT / "pipeline_paths.json"
config_content = {
    "paths": {
        "pdf_input": str(RAW_PDF_DIR),
        "images_output": str(IMG_DIR),
        "blocks_output": str(BENCH_ROOT / "job_blocks_smart"),
        "detections_output": str(BENCH_ROOT / "detections"),
        "crops_output": str(CROP_DIR)
    }
}
import json
with open(config_path, "w") as f:
    json.dump(config_content, f)

# We also need to hack DEFAULT_CROP_MANIFEST_JSONL in stage02 or move the file.
# I'll just move the file after.
cmd2 = [
    "./4_env/bin/python", "src/pipeline/stage02_block_detection.py",
    "--config", str(config_path),
    "--page-manifest", str(STATE_DIR / "page_manifest.jsonl"),
    "--crops-output", str(CROP_DIR),
    "--force"
]
# We must temporarily backup run_state/crop_manifest.jsonl
orig_manifest = Path("run_state/crop_manifest.jsonl")
backup_manifest = Path("run_state/crop_manifest.jsonl.bak")
if orig_manifest.exists():
    shutil.copy(orig_manifest, backup_manifest)

os.system(" ".join(cmd2))

# Move the result to our bench workspace
if orig_manifest.exists():
    shutil.move(orig_manifest, STATE_DIR / "crop_manifest.jsonl")
    
# Restore backup
if backup_manifest.exists():
    shutil.move(backup_manifest, orig_manifest)

print(f"DONE. 13-PDF Crop Manifest ready at {STATE_DIR / 'crop_manifest.jsonl'}")
