
import os
import shutil
import json
from pathlib import Path

# Config
PAGE_MANIFEST = Path("run_state/page_manifest.jsonl")
CROP_MANIFEST = Path("run_state/crop_manifest.jsonl")
IMG_DIR = Path("data/bench13_workspace/pdf2img")
CROP_DIR = Path("data/bench13_workspace/crops")
BENCH_ROOT = Path("data/bench13_workspace")

# 1. Create a dedicated config for this run
config_path = BENCH_ROOT / "pipeline_paths.json"
config_content = {
    "paths": {
        "pdf_input": "data/benchmark_13_raw",
        "images_output": str(IMG_DIR),
        "blocks_output": str(BENCH_ROOT / "job_blocks_smart"),
        "detections_output": str(BENCH_ROOT / "detections"),
        "crops_output": str(CROP_DIR)
    }
}
with open(config_path, "w") as f:
    json.dump(config_content, f, indent=2)

# 2. Run Stage 2
# We must temporarily backup run_state/crop_manifest.jsonl
if CROP_MANIFEST.exists():
    shutil.copy(CROP_MANIFEST, CROP_MANIFEST.with_suffix(".bak"))

cmd = [
    "./4_env/bin/python", "src/pipeline/stage02_block_detection.py",
    "--config", str(config_path),
    "--page-manifest", str(PAGE_MANIFEST),
    "--crops-output", str(CROP_DIR),
    "--force"
]
print(f"Running: {' '.join(cmd)}")
os.system(" ".join(cmd))

# 3. Save the result
if CROP_MANIFEST.exists():
    shutil.move(CROP_MANIFEST, BENCH_ROOT / "run_state/crop_manifest.jsonl")
    print(f"SUCCESS: 13-PDF Crop Manifest saved to {BENCH_ROOT / 'run_state/crop_manifest.jsonl'}")

# 4. Restore original manifest if it existed
if Path("run_state/crop_manifest.jsonl.bak").exists():
    shutil.move(Path("run_state/crop_manifest.jsonl.bak"), CROP_MANIFEST)
