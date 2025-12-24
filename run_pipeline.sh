#!/usr/bin/env bash
# run_pipeline.sh
# Usage: ./run_pipeline.sh [--force]
# Place this file in your project root: ~/project/smart-job-scanner-v2/
# It will run each Python script in order using the virtualenv at ./4_env.
# If a script exits with a non-zero code (or is interrupted), it will be retried until it completes successfully.
# Logs are stored in ./logs/, state files in ./run_state/
# Use --force to ignore existing .done files and run all commands from scratch.

set -o pipefail

# Parse command line arguments
FORCE_RUN=false
if [[ "$1" == "--force" ]]; then
  FORCE_RUN=true
  echo "Force mode enabled: will ignore existing .done files and run all commands."
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Path to venv python (adjust if your venv is somewhere else)
VENV_PYTHON="$PROJECT_ROOT/4_env/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "ERROR: virtualenv python not found at: $VENV_PYTHON"
  echo "If your virtualenv is elsewhere, edit VENV_PYTHON in this script."
  exit 2
fi

# Directories for logs and state
LOGDIR="$PROJECT_ROOT/logs"
STATEDIR="$PROJECT_ROOT/run_state"
mkdir -p "$LOGDIR" "$STATEDIR"

# Configure retry/backoff behavior
INITIAL_SLEEP=10       # seconds before first retry
MAX_SLEEP=300          # maximum backoff (5 minutes)
BACKOFF_FACTOR=2       # multiply sleep by this each retry
# If you want unlimited retries set MAX_RETRIES=0
MAX_RETRIES=0          # 0 = unlimited retries, otherwise number of attempts per script

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

# Trap so ctrl-c prints status
trap 'echo; echo "$(timestamp) - Received SIGINT/SIGTERM. Exiting runner."; exit 130' SIGINT SIGTERM

# List of commands (script + args) in the order to run
# IMPORTANT: these are relative to project root
commands=(
  "src/1_pdf_to_images.py"
  "src/2_run_smart_detector_batch.py"
  "src/3_block_refiner.py"
  "src/4_gpu_multilang_easyocr_working_slow_but_accurate.py"
  "src/5_argos_translate_batch.py"
  "src/6_create_batches_for_ollama.py"
  "src/7_final_ollama_pipeline.py --no-hybrid"
  "src/8_post_processing.py"
  "src/9-1_dynamic_resumes_full.py"
  "src/9-2_local_filter.py"
  "src/9-4_llm_search.py"
  "src/9-5_generate_shortlist_latest.py"
  "src/10_notify_shortlist_telegram.py"
  "src/11_cleanup_data.py"
)

# Helper: make a safe filename for logs/state from the command
safe_name() {
  local cmd="$1"
  # remove leading/trailing whitespace and replace spaces/slashes with underscores
  echo "$cmd" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/[ \/]/_/g' -e 's/[^A-Za-z0-9_.-]/_/g'
}

# Run each command sequentially, retrying until success
for cmd in "${commands[@]}"; do
  name="$(safe_name "$cmd")"
  done_file="$STATEDIR/$name.done"
  stdout_log="$LOGDIR/$name.out.log"
  stderr_log="$LOGDIR/$name.err.log"
  combined_log="$LOGDIR/$name.combined.log"

  if [ -f "$done_file" ] && [ "$FORCE_RUN" = false ]; then
    echo "$(timestamp) - Skipping '$cmd' (already completed)."
    continue
  fi

  # If force mode is enabled and done file exists, we'll remove it to ensure clean run
  if [ "$FORCE_RUN" = true ] && [ -f "$done_file" ]; then
    echo "$(timestamp) - Force mode: removing existing done file for '$cmd'"
    rm "$done_file"
  fi

  attempt=0
  sleep_time=$INITIAL_SLEEP
  while true; do
    attempt=$((attempt+1))
    echo "============================================================" | tee -a "$combined_log"
    echo "$(timestamp) - START attempt #$attempt for: $cmd" | tee -a "$combined_log"

    # ---- begin run block with optional auto-confirm for step 11 ----
    # Determine script path and args separately
    script_file="$(echo "$cmd" | awk '{print $1}')"
    script_path="$PROJECT_ROOT/$script_file"
    # remainder (args) if any
    script_args="$(echo "$cmd" | cut -s -d' ' -f2-)"  # empty if no args

    echo "$(timestamp) - Running: $VENV_PYTHON $script_path $script_args" >> "$combined_log"

    # if this is the cleanup script that requires interactive 'y', send it one 'y\n'
    if [ "$script_file" = "src/11_cleanup_data.py" ]; then
      # send single 'y' followed by newline to confirm
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
      # normal execution for other scripts
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
    # ---- end run block ----

    rc=$?
    if [ $rc -eq 0 ]; then
      echo "$(timestamp) - SUCCESS: $cmd completed (exit 0)." | tee -a "$combined_log"
      # mark done
      echo "$(timestamp) - Completed on $(timestamp)" > "$done_file"
      break
    else
      echo "$(timestamp) - FAILURE: $cmd exited with code $rc." | tee -a "$combined_log"
      # If max retries is set and reached, exit entire pipeline with non-zero
      if [ "$MAX_RETRIES" -gt 0 ] && [ "$attempt" -ge "$MAX_RETRIES" ]; then
        echo "$(timestamp) - Reached MAX_RETRIES ($MAX_RETRIES) for $cmd. Exiting pipeline." | tee -a "$combined_log"
        exit 3
      fi

      echo "$(timestamp) - Will retry '$cmd' after $sleep_time seconds (attempt #$((attempt+1)))." | tee -a "$combined_log"
      sleep $sleep_time
      # exponential backoff (cap at MAX_SLEEP)
      sleep_time=$((sleep_time * BACKOFF_FACTOR))
      if [ $sleep_time -gt $MAX_SLEEP ]; then
        sleep_time=$MAX_SLEEP
      fi
      # loop will retry
    fi
  done

done

echo "$(timestamp) - All commands finished successfully."
exit 0