#!/usr/bin/env python3
"""Primary Python pipeline orchestrator for Smart Job Scanner v2.

Keeps Bash compatibility scripts in place while providing a Python-first engine.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = PROJECT_ROOT / "4_env" / "bin" / "python"
LOG_DIR = PROJECT_ROOT / "logs"
STATE_DIR = PROJECT_ROOT / "run_state"
METRICS_FILE = PROJECT_ROOT / "data" / "pipeline_metrics.json"
STRUCTURED_LOG_FILE = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d')}.jsonl"
PIPELINE_CONFIG = PROJECT_ROOT / "configs" / "pipeline_config.json"
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test_data"
TEST_OUTPUT_ROOT = PROJECT_ROOT / "data" / "test_output"

INITIAL_SLEEP = 10
MAX_SLEEP = 300
BACKOFF_FACTOR = 2

DEFAULT_STAGES = [
    "src/pipeline/stage01_pdf_to_images.py",
    "src/pipeline/stage02_block_detection.py",
    "src/pipeline/stage03_block_refiner.py",
    "src/pipeline/stage03_ocr.py",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR + LLM pipeline stages in sequence")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/config and print stage plan")
    parser.add_argument("--debug", action="store_true", help="Enable verbose runner output")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run in sandbox mode with test data, reduced limits, and no Telegram stage",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit test-run pages (auto-enables --test-run)")
    parser.add_argument("--force", action="store_true", help="Ignore existing run_state/*.done markers")
    parser.add_argument(
        "--retry",
        type=int,
        default=5,
        help="Retries per stage (0 means unlimited retries)",
    )
    parser.add_argument(
        "--python-bin",
        default=str(DEFAULT_PYTHON),
        help="Python executable used to run stage scripts",
    )
    return parser.parse_args()


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_name(command: str) -> str:
    trimmed = command.strip()
    replaced = trimmed.replace(" ", "_").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", replaced)


def load_stage_commands() -> list[str]:
    if PIPELINE_CONFIG.is_file():
        try:
            payload = json.loads(PIPELINE_CONFIG.read_text(encoding="utf-8"))
            stages = payload.get("pipeline", {}).get("stages")
            if isinstance(stages, list) and stages:
                normalized = [str(item).strip() for item in stages if str(item).strip()]
                if normalized:
                    return normalized
        except Exception:
            pass
    return list(DEFAULT_STAGES)


def apply_test_run_stage_policy(commands: list[str]) -> list[str]:
    # Test mode focuses on detector/refiner evaluation path.
    allowed = (
        "src/pipeline/stage01_pdf_to_images.py",
        "src/pipeline/stage02_block_detection.py",
        "src/pipeline/stage03_block_refiner.py",
    )
    return [cmd for cmd in commands if cmd.startswith(allowed)]


def validate_configs(check_python: str) -> None:
    cfgs = [
        PROJECT_ROOT / "configs" / "gemini_config.json",
        PROJECT_ROOT / "configs" / "newspaper_config.json",
        PROJECT_ROOT / "configs" / "pipeline_config.json",
    ]
    for cfg in cfgs:
        if not cfg.is_file():
            raise FileNotFoundError(f"Missing config: {cfg}")

    # Mirror bash behavior: ensure valid JSON object.
    for cfg in cfgs:
        with cfg.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid config {cfg}: top-level JSON must be an object")

    # Keep interface parity with shell script using python validator.
    if not shutil_which(check_python):
        raise RuntimeError("python3 not found for config validation")


def validate_required_dirs() -> None:
    required = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "src" / "pipeline",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "run_state",
    ]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing required directories: {', '.join(missing)}")


def validate_stage_scripts(commands: list[str]) -> None:
    missing: list[str] = []
    for command in commands:
        parts = shlex.split(command)
        if not parts:
            missing.append(command)
            continue
        script_rel = parts[0]
        if not (PROJECT_ROOT / script_rel).is_file():
            missing.append(script_rel)
    if missing:
        raise FileNotFoundError(f"Missing stage script(s): {', '.join(missing)}")


def print_stage_plan(commands: list[str]) -> None:
    print("Pipeline stage plan:")
    for idx, command in enumerate(commands, start=1):
        print(f"  {idx:02d}. {command}")


def append_structured_log(stage: str, status: str, duration: float = 0.0, processed_items: int = 0) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        "duration": round(float(duration), 3),
        "processed_items": int(processed_items),
    }
    with STRUCTURED_LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_metrics(stage: str, status: str, duration: float = 0.0, processed_items: int = 0) -> None:
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    defaults: dict[str, Any] = {
        "pages_processed": 0,
        "jobs_detected": 0,
        "jobs_filtered": 0,
        "jobs_sent_to_telegram": 0,
        "pipeline_runtime_seconds": 0.0,
        # Backward-compatible alias for older tooling.
        "total_pipeline_runtime": 0.0,
        "stages_completed": 0,
        "last_stage": None,
        "last_status": None,
        "updated_at": None,
    }

    payload: dict[str, Any]
    if METRICS_FILE.is_file():
        try:
            raw = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
            payload = raw if isinstance(raw, dict) else {}
        except Exception:
            payload = {}
    else:
        payload = {}

    for key, value in defaults.items():
        payload.setdefault(key, value)

    duration_v = max(float(duration), 0.0)
    runtime_total = round(float(payload.get("pipeline_runtime_seconds", 0.0)) + duration_v, 3)
    payload["pipeline_runtime_seconds"] = runtime_total
    payload["total_pipeline_runtime"] = runtime_total
    payload["last_stage"] = stage
    payload["last_status"] = status
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()

    if status == "success":
        payload["stages_completed"] = int(payload.get("stages_completed", 0)) + 1

    processed_v = int(processed_items)
    if stage.startswith("src/pipeline/stage01") and processed_v > 0:
        payload["pages_processed"] = int(payload.get("pages_processed", 0)) + processed_v
    if stage.startswith("src/pipeline/stage07") and processed_v > 0:
        payload["jobs_detected"] = int(payload.get("jobs_detected", 0)) + processed_v
    if stage.startswith("src/pipeline/stage09") and processed_v > 0:
        payload["jobs_filtered"] = int(payload.get("jobs_filtered", 0)) + processed_v
    if stage.startswith("src/pipeline/stage10") and processed_v > 0:
        payload["jobs_sent_to_telegram"] = int(payload.get("jobs_sent_to_telegram", 0)) + processed_v

    METRICS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def shutil_which(binary: str) -> str | None:
    return shutil.which(binary) or (binary if Path(binary).exists() else None)


def cleanup_old_logs(days: int = 7) -> int:
    archive_dir = LOG_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - (days * 24 * 60 * 60)
    moved = 0

    for path in LOG_DIR.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                shutil.move(str(path), str(archive_dir / path.name))
                moved += 1
        except Exception:
            continue
    return moved


def stream_to_logs(pipe: Any, target: Any, combined: Any, is_stderr: bool = False) -> None:
    for line in iter(pipe.readline, ""):
        target.write(line)
        target.flush()
        combined.write(line)
        combined.flush()
        if is_stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
        else:
            sys.stdout.write(line)
            sys.stdout.flush()
    pipe.close()


def prepare_test_run_sandbox() -> Path:
    sandbox_root = TEST_OUTPUT_ROOT / "runtime"
    sandbox_data = sandbox_root / "data"
    sandbox_logs = sandbox_root / "logs"
    sandbox_state = sandbox_root / "run_state"
    sandbox_cfg = sandbox_root / "configs"

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if sandbox_root.exists():
        shutil.rmtree(sandbox_root)
    sandbox_data.mkdir(parents=True, exist_ok=True)
    sandbox_logs.mkdir(parents=True, exist_ok=True)
    sandbox_state.mkdir(parents=True, exist_ok=True)

    sandbox_cfg.symlink_to(PROJECT_ROOT / "configs", target_is_directory=True)
    (sandbox_data / "test_data").symlink_to(TEST_DATA_DIR, target_is_directory=True)
    (sandbox_data / "raw_pdfs").symlink_to(TEST_DATA_DIR, target_is_directory=True)
    return sandbox_root


def apply_test_run_stage_limits(command: str, sandbox_root: Path, max_pages: int, max_llm_calls: int) -> None:
    if command.startswith("src/pipeline/stage01_pdf_to_images.py"):
        img_root = sandbox_root / "data" / "pdf2img"
        if img_root.exists():
            for paper_dir in sorted([p for p in img_root.iterdir() if p.is_dir()]):
                pages = sorted(paper_dir.glob("*.png"))
                for extra in pages[max_pages:]:
                    extra.unlink(missing_ok=True)
    if command.startswith("src/pipeline/stage06_batch_builder.py"):
        batch_dir = sandbox_root / "data" / "batch_inputs"
        if batch_dir.exists():
            txts = sorted(batch_dir.glob("*.txt"))
            for extra in txts[max_llm_calls:]:
                extra.unlink(missing_ok=True)


def run_stage(
    python_bin: str,
    command: str,
    stdout_log: Path,
    stderr_log: Path,
    combined_log: Path,
    workdir: Path,
    env: dict[str, str] | None = None,
) -> int:
    parts = shlex.split(command)
    script_rel, script_args = parts[0], parts[1:]
    script_path = PROJECT_ROOT / script_rel

    cmd = [python_bin, "-u", str(script_path), *script_args]
    stage_input: str | None = "y\n" if script_rel == "src/pipeline/stage11_cleanup.py" else None

    with stdout_log.open("a", encoding="utf-8") as out_fh, stderr_log.open("a", encoding="utf-8") as err_fh, combined_log.open("a", encoding="utf-8") as comb_fh:
        process = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=str(workdir),
            env=env,
            stdin=subprocess.PIPE if stage_input is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        if stage_input is not None and process.stdin is not None:
            process.stdin.write(stage_input)
            process.stdin.flush()
            process.stdin.close()

        t_out = threading.Thread(target=stream_to_logs, args=(process.stdout, out_fh, comb_fh, False), daemon=True)
        t_err = threading.Thread(target=stream_to_logs, args=(process.stderr, err_fh, comb_fh, True), daemon=True)
        t_out.start()
        t_err.start()
        t_out.join()
        t_err.join()
        return process.wait()


def run_dry_run(commands: list[str], check_python: str) -> int:
    print(f"{ts()} - Dry-run mode enabled: no stages will be executed.")
    print_stage_plan(commands)

    print(f"{ts()} - Validating required directories...")
    validate_required_dirs()

    print(f"{ts()} - Validating stage script paths...")
    validate_stage_scripts(commands)

    print(f"{ts()} - Validating config files...")
    validate_configs(check_python)

    print("✅ Config files are valid JSON")
    print(f"{ts()} - Dry-run validation completed successfully.")
    return 0


def main() -> int:
    global LOG_DIR, STATE_DIR, METRICS_FILE, STRUCTURED_LOG_FILE

    args = parse_args()

    if args.retry < 0:
        print("ERROR: --retry must be a non-negative integer", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit <= 0:
        print("ERROR: --limit must be a positive integer", file=sys.stderr)
        return 2
    if args.limit is not None and not args.test_run:
        # Safe default: limit mode implies sandbox/test inputs.
        args.test_run = True

    os.chdir(PROJECT_ROOT)

    python_bin = args.python_bin
    check_python = python_bin if Path(python_bin).exists() else "python3"

    commands = load_stage_commands()
    stage_env: dict[str, str] | None = None
    workdir = PROJECT_ROOT
    test_max_pages = max(1, int(args.limit)) if args.limit is not None else 2
    test_max_llm_calls = 5

    if args.test_run:
        commands = apply_test_run_stage_policy(commands)
        sandbox_root = prepare_test_run_sandbox()
        workdir = sandbox_root
        LOG_DIR = sandbox_root / "logs"
        STATE_DIR = sandbox_root / "run_state"
        METRICS_FILE = sandbox_root / "data" / "pipeline_metrics.json"
        STRUCTURED_LOG_FILE = LOG_DIR / f"pipeline_test_{datetime.now().strftime('%Y%m%d')}.jsonl"

        stage_env = os.environ.copy()
        stage_env.update(
            {
                "PIPELINE_TEST_RUN": "1",
                "PIPELINE_MAX_PAGES": str(test_max_pages),
                "PIPELINE_MAX_LLM_CALLS": str(test_max_llm_calls),
                "PIPELINE_SKIP_TELEGRAM": "1",
                "PIPELINE_OUTPUT_ROOT": str(sandbox_root / "data"),
                "RAW_PDF_DIR": str(TEST_DATA_DIR),
                "PYTHONPATH": str(PROJECT_ROOT),
                "TELEGRAM_BOT_TOKEN": stage_env.get("TELEGRAM_BOT_TOKEN", "test_token"),
                "TELEGRAM_CHAT_ID": stage_env.get("TELEGRAM_CHAT_ID", "0"),
                "TELEGRAM_API_ID": stage_env.get("TELEGRAM_API_ID", "0"),
                "TELEGRAM_API_HASH": stage_env.get("TELEGRAM_API_HASH", "test_hash"),
                "GOOGLE_API_KEYS": stage_env.get("GOOGLE_API_KEYS", "test_key"),
            }
        )
        if args.debug:
            print(f"{ts()} - Test-run mode enabled. Sandbox root: {sandbox_root}")
            print(f"{ts()} - Test-run limits: pages={test_max_pages}, llm_calls={test_max_llm_calls}")

    if args.dry_run:
        try:
            return run_dry_run(commands, check_python)
        except Exception as exc:
            print(f"{ts()} - Dry-run failed: {exc}", file=sys.stderr)
            return 2

    if not Path(python_bin).exists():
        print(f"ERROR: virtualenv python not found at: {python_bin}", file=sys.stderr)
        return 2

    if args.force:
        print("Force mode enabled: will ignore existing .done files and run all commands.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    moved_logs = cleanup_old_logs(days=7)
    if moved_logs > 0:
        print(f"{ts()} - Archived {moved_logs} log file(s) older than 7 days to logs/archive/.")

    try:
        validate_stage_scripts(commands)
        validate_configs(check_python)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    update_metrics("pipeline", "start", 0.0, 0)
    append_structured_log("pipeline", "start", 0.0, 0)

    def handle_interrupt(signum: int, _frame: Any) -> None:
        print(f"\n{ts()} - Received signal {signum}. Exiting runner.")
        append_structured_log("pipeline", "interrupted", 0.0, 0)
        update_metrics("pipeline", "interrupted", 0.0, 0)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    print(f"{ts()} - Retry policy: RETRY={args.retry} (0 means unlimited)")
    if args.debug:
        print(f"{ts()} - Stage count: {len(commands)}")

    if args.test_run and not any(TEST_DATA_DIR.glob("*.pdf")):
        print(f"{ts()} - Test-run mode: no PDFs found in data/test_data; validation-only run completed.")
        append_structured_log("pipeline_test", "success", 0.0, 0)
        update_metrics("pipeline_test", "success", 0.0, 0)
        return 0

    for command in commands:
        marker_name = safe_name(command)
        done_file = STATE_DIR / f"{marker_name}.done"
        stdout_log = LOG_DIR / f"{marker_name}.out.log"
        stderr_log = LOG_DIR / f"{marker_name}.err.log"
        combined_log = LOG_DIR / f"{marker_name}.combined.log"

        if done_file.exists() and not args.force:
            print(f"{ts()} - Skipping '{command}' (already completed).")
            append_structured_log(command, "skipped", 0.0, 0)
            update_metrics(command, "skipped", 0.0, 0)
            continue

        if done_file.exists() and args.force:
            print(f"{ts()} - Force mode: removing existing done file for '{command}'")
            done_file.unlink(missing_ok=True)

        attempt = 0
        sleep_time = INITIAL_SLEEP

        while True:
            attempt += 1
            start = time.time()
            print("=" * 60)
            print(f"{ts()} - START attempt #{attempt} for: {command}")

            append_structured_log(command, "start", 0.0, 0)

            with combined_log.open("a", encoding="utf-8") as comb_fh:
                comb_fh.write(f"{ts()} - Running: {python_bin} {command}\n")

            rc = run_stage(
                python_bin,
                command,
                stdout_log,
                stderr_log,
                combined_log,
                workdir=workdir,
                env=stage_env,
            )
            duration = time.time() - start

            if rc == 0:
                print(f"{ts()} - SUCCESS: {command} completed (exit 0).")
                done_file.write_text(f"Completed on {ts()}\n", encoding="utf-8")
                append_structured_log(command, "success", duration, 0)
                update_metrics(command, "success", duration, 0)
                if args.test_run:
                    apply_test_run_stage_limits(command, workdir, test_max_pages, test_max_llm_calls)
                break

            print(f"{ts()} - FAILURE: {command} exited with code {rc}.")
            append_structured_log(command, "failure", duration, 0)
            update_metrics(command, "failure", duration, 0)

            if args.retry > 0 and attempt >= args.retry:
                print(f"{ts()} - Reached retry limit ({args.retry}) for {command}. Exiting pipeline.")
                append_structured_log("pipeline", "failed", duration, 0)
                update_metrics("pipeline", "failed", duration, 0)
                return 3

            print(f"{ts()} - Will retry '{command}' after {sleep_time} seconds (attempt #{attempt + 1}).")
            append_structured_log(command, "retrying", 0.0, 0)
            time.sleep(sleep_time)
            sleep_time = min(sleep_time * BACKOFF_FACTOR, MAX_SLEEP)

    print(f"{ts()} - All commands finished successfully.")
    append_structured_log("pipeline", "success", 0.0, 0)
    update_metrics("pipeline", "success", 0.0, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
