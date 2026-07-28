#!/usr/bin/env bash
# run_pipeline.sh
# Usage: ./run_pipeline.sh [--force] [--dry-run]
# It runs each stage in order using the virtualenv at ./4_env.
# Use --force to ignore existing .done files.
# Use --dry-run to validate structure/config/inputs and print plan without executing stages.

set -o pipefail

FORCE_RUN=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE_RUN=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      echo "Usage: ./scripts/run_pipeline.sh [--force] [--dry-run]"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      echo "Usage: ./scripts/run_pipeline.sh [--force] [--dry-run]"
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Path to venv python
VENV_PYTHON="$PROJECT_ROOT/4_env/bin/python"
CHECK_PYTHON="$VENV_PYTHON"
if [ ! -x "$CHECK_PYTHON" ]; then
  CHECK_PYTHON="$(command -v python3 || true)"
fi

# Directories for logs/state/metrics
LOGDIR="$PROJECT_ROOT/logs"
STATEDIR="$PROJECT_ROOT/run_state"
METRICS_FILE="$PROJECT_ROOT/data/pipeline_metrics.json"
STRUCTURED_LOG_FILE="$LOGDIR/pipeline_$(date +%Y%m%d).jsonl"

# Configure retry/backoff behavior
INITIAL_SLEEP=10
MAX_SLEEP=300
BACKOFF_FACTOR=2
MAX_RETRIES="${MAX_RETRIES:-5}"

commands=(
  "src/pipeline/stage01_pdf_to_images.py"
  "src/pipeline/stage02_block_detection.py"
  "src/pipeline/stage03_block_refiner.py"
  "src/pipeline/stage04_ocr.py"
  "src/pipeline/stage05_translation.py"
  "src/pipeline/stage06_batch_builder.py"
  "src/pipeline/stage07_llm_extraction.py --no-hybrid"
  "src/pipeline/stage08_post_processing.py"
  "src/pipeline/stage09_dynamic_resumes.py"
  "src/pipeline/stage09_local_filter.py"
  "src/pipeline/stage09_llm_filter.py"
  "src/pipeline/stage09_shortlist.py"
  "src/pipeline/stage10_notification.py"
  "src/pipeline/stage11_cleanup.py"
)

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

die() {
  echo "ERROR: $*"
  exit 2
}

safe_name() {
  local cmd="$1"
  echo "$cmd" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/[ \/]/_/g' -e 's/[^A-Za-z0-9_.-]/_/g'
}

validate_stage_scripts() {
  local missing=0
  for cmd in "${commands[@]}"; do
    local script_file
    script_file="$(echo "$cmd" | awk '{print $1}')"
    local script_path="$PROJECT_ROOT/$script_file"
    if [ ! -f "$script_path" ]; then
      echo "❌ Missing stage script: $script_file"
      missing=1
    fi
  done
  return "$missing"
}

validate_configs() {
  local cfgs=(
    "$PROJECT_ROOT/configs/gemini_config.json"
    "$PROJECT_ROOT/configs/newspaper_config.json"
    "$PROJECT_ROOT/configs/pipeline_config.json"
  )

  for cfg in "${cfgs[@]}"; do
    if [ ! -f "$cfg" ]; then
      echo "❌ Missing config: $cfg"
      return 1
    fi
  done

  if [ -z "$CHECK_PYTHON" ]; then
    echo "❌ python3 not found for config validation"
    return 1
  fi

  "$CHECK_PYTHON" - "${cfgs[@]}" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError("top-level JSON must be an object")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Invalid config {path}: {exc}")
        raise SystemExit(1)
print("✅ Config files are valid JSON")
PY
}

validate_required_dirs() {
  local dirs=(
    "$PROJECT_ROOT/src"
    "$PROJECT_ROOT/src/pipeline"
    "$PROJECT_ROOT/scripts"
    "$PROJECT_ROOT/configs"
    "$PROJECT_ROOT/data"
    "$PROJECT_ROOT/logs"
    "$PROJECT_ROOT/run_state"
  )

  local missing=0
  for d in "${dirs[@]}"; do
    if [ ! -d "$d" ]; then
      echo "❌ Missing required directory: $d"
      missing=1
    fi
  done
  return "$missing"
}

print_stage_plan() {
  echo "Pipeline stage plan:"
  local idx=1
  for cmd in "${commands[@]}"; do
    printf '  %02d. %s\n' "$idx" "$cmd"
    idx=$((idx+1))
  done
}

append_structured_log() {
  local stage_name="$1"
  local status="$2"
  local duration_seconds="$3"
  local processed_items="$4"

  "$VENV_PYTHON" - "$STRUCTURED_LOG_FILE" "$stage_name" "$status" "$duration_seconds" "$processed_items" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import json
import sys

log_path, stage_name, status, duration, processed = sys.argv[1:]

try:
    duration_v = float(duration)
except ValueError:
    duration_v = 0.0

try:
    processed_v = int(processed)
except ValueError:
    processed_v = 0

entry = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage_name": stage_name,
    "status": status,
    "duration_seconds": round(duration_v, 3),
    "processed_items": processed_v,
}

with open(log_path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
PY
}

update_metrics() {
  local stage_name="$1"
  local status="$2"
  local duration_seconds="$3"
  local processed_items="$4"

  "$VENV_PYTHON" - "$METRICS_FILE" "$stage_name" "$status" "$duration_seconds" "$processed_items" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import json
import sys
from pathlib import Path

metrics_path, stage_name, status, duration, processed = sys.argv[1:]
path = Path(metrics_path)
path.parent.mkdir(parents=True, exist_ok=True)

defaults = {
    "pages_processed": 0,
    "jobs_detected": 0,
    "jobs_filtered": 0,
    "jobs_sent_to_telegram": 0,
    "pipeline_runtime_seconds": 0.0,
    # Backward-compatible legacy key retained for existing dashboards/scripts.
    "total_pipeline_runtime": 0.0,
    "stages_completed": 0,
    "last_stage": None,
    "last_status": None,
    "updated_at": None,
}

if path.exists():
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except Exception:  # noqa: BLE001
        payload = {}
else:
    payload = {}

for k, v in defaults.items():
    payload.setdefault(k, v)

try:
    duration_v = float(duration)
except ValueError:
    duration_v = 0.0

try:
    processed_v = int(processed)
except ValueError:
    processed_v = 0

runtime_total = round(float(payload.get("pipeline_runtime_seconds", 0.0)) + max(duration_v, 0.0), 3)
payload["pipeline_runtime_seconds"] = runtime_total
payload["total_pipeline_runtime"] = runtime_total
payload["last_stage"] = stage_name
payload["last_status"] = status
payload["updated_at"] = datetime.now(timezone.utc).isoformat()

if status == "success":
    payload["stages_completed"] = int(payload.get("stages_completed", 0)) + 1

# Lightweight, non-invasive counters by stage category.
if stage_name.startswith("src/pipeline/stage01") and processed_v > 0:
    payload["pages_processed"] = int(payload.get("pages_processed", 0)) + processed_v
if stage_name.startswith("src/pipeline/stage07") and processed_v > 0:
    payload["jobs_detected"] = int(payload.get("jobs_detected", 0)) + processed_v
if stage_name.startswith("src/pipeline/stage09") and processed_v > 0:
    payload["jobs_filtered"] = int(payload.get("jobs_filtered", 0)) + processed_v
if stage_name.startswith("src/pipeline/stage10") and processed_v > 0:
    payload["jobs_sent_to_telegram"] = int(payload.get("jobs_sent_to_telegram", 0)) + processed_v

path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
}

run_dry_run() {
  echo "$(timestamp) - Dry-run mode enabled: no stages will be executed."

  print_stage_plan

  echo "$(timestamp) - Validating required directories..."
  validate_required_dirs || {
    echo "$(timestamp) - Dry-run failed: missing required directories."
    exit 2
  }

  echo "$(timestamp) - Validating stage script paths..."
  validate_stage_scripts || {
    echo "$(timestamp) - Dry-run failed: missing stage script(s)."
    exit 2
  }

  echo "$(timestamp) - Validating config files..."
  validate_configs || {
    echo "$(timestamp) - Dry-run failed: invalid configuration."
    exit 2
  }

  echo "$(timestamp) - Dry-run validation completed successfully."
  exit 0
}

if [ "$DRY_RUN" = true ]; then
  run_dry_run
fi

if [ "$FORCE_RUN" = true ]; then
  echo "Force mode enabled: will ignore existing .done files and run all commands."
fi

if [ ! -x "$VENV_PYTHON" ]; then
  die "virtualenv python not found at: $VENV_PYTHON"
fi

if ! [[ "$MAX_RETRIES" =~ ^[0-9]+$ ]]; then
  die "MAX_RETRIES must be a non-negative integer (got '$MAX_RETRIES')"
fi

mkdir -p "$LOGDIR" "$STATEDIR"

# Initialize metrics file lazily without altering stage behavior
update_metrics "pipeline" "start" "0" "0"

echo "$(timestamp) - Retry policy: MAX_RETRIES=$MAX_RETRIES (0 means unlimited)"

trap 'echo; echo "$(timestamp) - Received SIGINT/SIGTERM. Exiting runner."; append_structured_log "pipeline" "interrupted" "0" "0"; update_metrics "pipeline" "interrupted" "0" "0"; exit 130' SIGINT SIGTERM

validate_stage_scripts || die "One or more stage scripts are missing"
validate_configs || die "Configuration validation failed"

for cmd in "${commands[@]}"; do
  name="$(safe_name "$cmd")"
  done_file="$STATEDIR/$name.done"
  stdout_log="$LOGDIR/$name.out.log"
  stderr_log="$LOGDIR/$name.err.log"
  combined_log="$LOGDIR/$name.combined.log"

  if [ -f "$done_file" ] && [ "$FORCE_RUN" = false ]; then
    echo "$(timestamp) - Skipping '$cmd' (already completed)."
    append_structured_log "$cmd" "skipped" "0" "0"
    update_metrics "$cmd" "skipped" "0" "0"
    continue
  fi

  if [ "$FORCE_RUN" = true ] && [ -f "$done_file" ]; then
    echo "$(timestamp) - Force mode: removing existing done file for '$cmd'"
    rm "$done_file"
  fi

  attempt=0
  sleep_time=$INITIAL_SLEEP
  while true; do
    attempt=$((attempt+1))
    stage_start_epoch=$(date +%s)
    echo "============================================================" | tee -a "$combined_log"
    echo "$(timestamp) - START attempt #$attempt for: $cmd" | tee -a "$combined_log"

    script_file="$(echo "$cmd" | awk '{print $1}')"
    script_path="$PROJECT_ROOT/$script_file"
    script_args="$(echo "$cmd" | cut -s -d' ' -f2-)"

    echo "$(timestamp) - Running: $VENV_PYTHON $script_path $script_args" >> "$combined_log"
    append_structured_log "$cmd" "start" "0" "0"

    if [ "$script_file" = "src/pipeline/stage11_cleanup.py" ]; then
      if command -v stdbuf >/dev/null 2>&1; then
        printf 'y\n' | stdbuf -oL -eL "$VENV_PYTHON" -u "$script_path" $script_args \
          > >(tee -a "$stdout_log" >>"$combined_log") \
          2> >(tee -a "$stderr_log" >>"$combined_log" >&2)
      else
        printf 'y\n' | "$VENV_PYTHON" -u "$script_path" $script_args \
          > >(tee -a "$stdout_log" >>"$combined_log") \
          2> >(tee -a "$stderr_log" >>"$combined_log" >&2)
      fi
    else
      if command -v stdbuf >/dev/null 2>&1; then
        if [ -n "$script_args" ]; then
          stdbuf -oL -eL "$VENV_PYTHON" -u "$script_path" $script_args \
            > >(tee -a "$stdout_log" >>"$combined_log") \
            2> >(tee -a "$stderr_log" >>"$combined_log" >&2)
        else
          stdbuf -oL -eL "$VENV_PYTHON" -u "$script_path" \
            > >(tee -a "$stdout_log" >>"$combined_log") \
            2> >(tee -a "$stderr_log" >>"$combined_log" >&2)
        fi
      else
        if [ -n "$script_args" ]; then
          "$VENV_PYTHON" -u "$script_path" $script_args \
            > >(tee -a "$stdout_log" >>"$combined_log") \
            2> >(tee -a "$stderr_log" >>"$combined_log" >&2)
        else
          "$VENV_PYTHON" -u "$script_path" \
            > >(tee -a "$stdout_log" >>"$combined_log") \
            2> >(tee -a "$stderr_log" >>"$combined_log" >&2)
        fi
      fi
    fi

    rc=$?
    stage_end_epoch=$(date +%s)
    stage_duration=$((stage_end_epoch - stage_start_epoch))

    if [ $rc -eq 0 ]; then
      echo "$(timestamp) - SUCCESS: $cmd completed (exit 0)." | tee -a "$combined_log"
      echo "$(timestamp) - Completed on $(timestamp)" > "$done_file"
      append_structured_log "$cmd" "success" "$stage_duration" "0"
      update_metrics "$cmd" "success" "$stage_duration" "0"
      break
    else
      echo "$(timestamp) - FAILURE: $cmd exited with code $rc." | tee -a "$combined_log"
      append_structured_log "$cmd" "failure" "$stage_duration" "0"
      update_metrics "$cmd" "failure" "$stage_duration" "0"

      if [ "$MAX_RETRIES" -gt 0 ] && [ "$attempt" -ge "$MAX_RETRIES" ]; then
        echo "$(timestamp) - Reached MAX_RETRIES ($MAX_RETRIES) for $cmd. Exiting pipeline." | tee -a "$combined_log"
        append_structured_log "pipeline" "failed" "$stage_duration" "0"
        update_metrics "pipeline" "failed" "$stage_duration" "0"
        exit 3
      fi

      echo "$(timestamp) - Will retry '$cmd' after $sleep_time seconds (attempt #$((attempt+1)))." | tee -a "$combined_log"
      append_structured_log "$cmd" "retrying" "0" "0"
      sleep "$sleep_time"
      sleep_time=$((sleep_time * BACKOFF_FACTOR))
      if [ "$sleep_time" -gt "$MAX_SLEEP" ]; then
        sleep_time=$MAX_SLEEP
      fi
    fi
  done

done

echo "$(timestamp) - All commands finished successfully."
append_structured_log "pipeline" "success" "0" "0"
update_metrics "pipeline" "success" "0" "0"
exit 0
