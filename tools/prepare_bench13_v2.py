
import os
import sys
import shutil
import json
from pathlib import Path

# Paths
RAW_PDF_DIR = Path("data/benchmark_13_raw")
BENCH_ROOT = Path("data/bench13_workspace")
IMG_DIR = BENCH_ROOT / "pdf2img"
CROP_DIR = BENCH_ROOT / "crops"
STATE_DIR = BENCH_ROOT / "run_state"

# Ensure dirs exist
for d in [IMG_DIR, CROP_DIR, STATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 1. Create a dedicated config for this run
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
with open(config_path, "w") as f:
    json.dump(config_content, f, indent=2)

print(f"Created config at {config_path}")

# 2. Run Stage 2
# We must temporarily backup run_state/crop_manifest.jsonl
orig_manifest = Path("run_state/crop_manifest.jsonl")
backup_manifest = Path("run_state/crop_manifest.jsonl.bak")
if orig_manifest.exists():
    shutil.copy(orig_manifest, backup_manifest)
    print("Backed up original manifest")

cmd2 = [
    "./4_env/bin/python", "src/pipeline/stage02_block_detection.py",
    "--config", str(config_path),
    "--page-manifest", str(STATE_DIR / "page_manifest.jsonl"),
    "--crops-output", str(CROP_DIR),
    "--force"
]
print(f"Running: {' '.join(cmd2)}")
os.system(" ".join(cmd2))

# 3. Move the result to our bench workspace
if orig_manifest.exists():
    shutil.move(orig_manifest, STATE_DIR / "crop_manifest.jsonl")
    print(f"Moved result to {STATE_DIR / 'crop_manifest.jsonl'}")
    
# 4. Restore backup
if backup_manifest.exists():
    shutil.move(backup_manifest, orig_manifest)
    print("Restored original manifest")

# Final Check
if (STATE_DIR / "crop_manifest.jsonl").exists():
    with open(STATE_DIR / "crop_manifest.jsonl") as f:
        count = sum(1 for _ in f)
    print(f"SUCCESS: 13-PDF Crop Manifest has {count} crops")
