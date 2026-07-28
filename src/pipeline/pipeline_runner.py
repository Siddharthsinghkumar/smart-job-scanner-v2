#!/usr/bin/env python3
"""Optional Python pipeline controller that reuses existing stage scripts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any
from prometheus_client import start_http_server, Counter, Gauge, Histogram


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
STATE_DIR = PROJECT_ROOT / "run_state"
METRICS_FILE = PROJECT_ROOT / "data" / "pipeline_metrics.json"
STRUCTURED_LOG = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d')}.jsonl"
DEFAULT_VENV_PYTHON = PROJECT_ROOT / "4_env" / "bin" / "python"

DEFAULT_COMMANDS = [
    "src/pipeline/stage01_pdf_to_images.py",
    "src/pipeline/stage02_block_detection.py",
    "src/pipeline/stage03_block_refiner.py",
    "src/pipeline/stage04_ocr.py",
    "src/pipeline/stage05_translation.py",
    "src/pipeline/stage06_batch_builder.py",
    "src/pipeline/stage07_llm_extraction.py --no-hybrid",
    "src/pipeline/stage08_post_processing.py",
    "src/pipeline/stage09_dynamic_resumes.py",
    "src/pipeline/stage09_local_filter.py",
    "src/pipeline/stage09_llm_filter.py",
    "src/pipeline/stage09_shortlist.py",
    "src/pipeline/stage10_notification.py",
    "src/pipeline/stage11_cleanup.py",
]

# --- Prometheus metrics ---
STAGE_DURATION = Histogram(
    'jobv2_stage_duration_seconds',
    'Duration of each pipeline stage',
    ['stage_name', 'status']
)
STAGE_RUNS = Counter(
    'jobv2_stage_runs_total',
    'Total stage executions',
    ['stage_name', 'status']
)
PIPELINE_STATUS = Gauge(
    'jobv2_pipeline_running',
    'Whether the pipeline is currently running (1=yes, 0=no)'
)
JOBS_PROCESSED = Counter(
    'jobv2_jobs_processed_total',
    'Total jobs processed across all runs'
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional Python pipeline controller")
    parser.add_argument("--force", action="store_true", help="Ignore .done markers")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan only")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per stage (0 = unlimited)")
    parser.add_argument(
        "--python-bin",
        default=str(DEFAULT_VENV_PYTHON),
        help="Python executable to run stage scripts",
    )
    return parser.parse_args()


def _read_stage_commands() -> list[str]:
    pipeline_config = PROJECT_ROOT / "configs" / "pipeline_config.json"
    if pipeline_config.exists():
        try:
            payload = json.loads(pipeline_config.read_text(encoding="utf-8"))
            stages = payload.get("pipeline", {}).get("stages")
            if isinstance(stages, list) and stages:
                return [str(item) for item in stages]
        except Exception:  # noqa: BLE001
            pass
    return DEFAULT_COMMANDS


def _safe_name(command: str) -> str:
    cleaned = command.strip().replace("/", "_").replace(" ", "_")
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in cleaned)


def _append_structured_log(stage_name: str, status: str, duration_seconds: float, processed_items: int = 0) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage_name": stage_name,
        "status": status,
        "duration_seconds": round(float(duration_seconds), 3),
        "processed_items": int(processed_items),
    }
    with STRUCTURED_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _update_metrics(stage_name: str, status: str, duration_seconds: float, processed_items: int = 0) -> None:
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    defaults: dict[str, Any] = {
        "pages_processed": 0,
        "jobs_detected": 0,
        "jobs_filtered": 0,
        "jobs_sent_to_telegram": 0,
        "total_pipeline_runtime": 0.0,
        "stages_completed": 0,
        "last_stage": None,
        "last_status": None,
        "updated_at": None,
    }
    if METRICS_FILE.exists():
        try:
            payload = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {}
        except Exception:  # noqa: BLE001
            payload = {}
    else:
        payload = {}

    for k, v in defaults.items():
        payload.setdefault(k, v)

    payload["total_pipeline_runtime"] = round(
        float(payload.get("total_pipeline_runtime", 0.0)) + max(float(duration_seconds), 0.0),
        3,
    )
    payload["last_stage"] = stage_name
    payload["last_status"] = status
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    if status == "success":
        payload["stages_completed"] = int(payload.get("stages_completed", 0)) + 1

    if stage_name.startswith("src/pipeline/stage01"):
        payload["pages_processed"] = int(payload.get("pages_processed", 0)) + int(processed_items)
    if stage_name.startswith("src/pipeline/stage09"):
        payload["jobs_filtered"] = int(payload.get("jobs_filtered", 0)) + int(processed_items)
    if stage_name.startswith("src/pipeline/stage10"):
        payload["jobs_sent_to_telegram"] = int(payload.get("jobs_sent_to_telegram", 0)) + int(processed_items)
        
    if stage_name != "pipeline":
        STAGE_DURATION.labels(stage_name=stage_name, status=status).observe(max(float(duration_seconds), 0.0))
        STAGE_RUNS.labels(stage_name=stage_name, status=status).inc()
    if stage_name.startswith("src/pipeline/stage01") and status == "success":
        JOBS_PROCESSED.inc(int(processed_items))

    METRICS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _validate_prereqs(commands: list[str]) -> None:
    required_files = [
        PROJECT_ROOT / "configs" / "gemini_config.json",
        PROJECT_ROOT / "configs" / "newspaper_config.json",
        PROJECT_ROOT / "configs" / "pipeline_config.json",
    ]
    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing config: {file_path}")
        json.loads(file_path.read_text(encoding="utf-8"))

    for command in commands:
        script_rel = shlex.split(command)[0]
        script_path = PROJECT_ROOT / script_rel
        if not script_path.is_file():
            raise FileNotFoundError(f"Missing stage script: {script_rel}")


_active_process: subprocess.Popen | None = None
_shutdown_requested = False

def _signal_handler(sig, frame):
    global _shutdown_requested
    _shutdown_requested = True
    sig_name = signal.Signals(sig).name
    print(f"\n[!] Received {sig_name}, initiating graceful shutdown...")
    if _active_process and _active_process.poll() is None:
        _active_process.terminate()
        try:
            _active_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _active_process.kill()
    _append_structured_log("pipeline", "interrupted", 0.0, 0)
    _update_metrics("pipeline", "interrupted", 0.0, 0)
    sys.exit(128 + sig)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _run_stage(python_bin: str, command: str) -> int:
    global _active_process
    parts = shlex.split(command)
    script_rel, args = parts[0], parts[1:]
    script_abs = PROJECT_ROOT / script_rel
    cmd = [python_bin, "-u", str(script_abs), *args]
    if script_rel.endswith("stage11_cleanup.py"):
        _active_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
        _active_process.communicate(input="y\n")
        rc = _active_process.returncode
    else:
        _active_process = subprocess.Popen(cmd)
        rc = _active_process.wait()
    _active_process = None
    return rc


def main() -> int:
    start_http_server(8765)
    PIPELINE_STATUS.set(1)
    
    args = parse_args()
    if args.max_retries < 0:
        PIPELINE_STATUS.set(0)
        raise SystemExit("--max-retries must be >= 0")

    commands = _read_stage_commands()
    _validate_prereqs(commands)

    print("Pipeline stage plan:")
    for idx, command in enumerate(commands, start=1):
        print(f"  {idx:02d}. {command}")

    if args.dry_run:
        print("Dry-run completed successfully (no stages executed).")
        PIPELINE_STATUS.set(0)
        return 0

    python_bin = args.python_bin
    if not Path(python_bin).exists():
        PIPELINE_STATUS.set(0)
        raise SystemExit(f"Python executable not found: {python_bin}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _append_structured_log("pipeline", "start", 0.0, 0)
    _update_metrics("pipeline", "start", 0.0, 0)

    for command in commands:
        done_file = STATE_DIR / f"{_safe_name(command)}.done"
        if done_file.exists() and not args.force:
            print(f"Skipping '{command}' (already completed).")
            _append_structured_log(command, "skipped", 0.0, 0)
            _update_metrics(command, "skipped", 0.0, 0)
            continue
        if done_file.exists() and args.force:
            done_file.unlink()

        attempt = 0
        while True:
            attempt += 1
            stage_start = time.time()
            _append_structured_log(command, "start", 0.0, 0)
            print(f"Starting attempt #{attempt}: {command}")
            rc = _run_stage(python_bin, command)
            duration = time.time() - stage_start
            if rc == 0:
                done_file.write_text(
                    f"Completed on {datetime.now().isoformat(timespec='seconds')}\n",
                    encoding="utf-8",
                )
                _append_structured_log(command, "success", duration, 0)
                _update_metrics(command, "success", duration, 0)
                break

            _append_structured_log(command, "failure", duration, 0)
            _update_metrics(command, "failure", duration, 0)
            print(f"Stage failed with code {rc}: {command}")
            if args.max_retries > 0 and attempt >= args.max_retries:
                _append_structured_log("pipeline", "failed", duration, 0)
                _update_metrics("pipeline", "failed", duration, 0)
                PIPELINE_STATUS.set(0)
                return 3

    _append_structured_log("pipeline", "success", 0.0, 0)
    _update_metrics("pipeline", "success", 0.0, 0)
    print("All commands finished successfully.")
    PIPELINE_STATUS.set(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
