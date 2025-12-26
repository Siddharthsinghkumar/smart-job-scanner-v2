#!/usr/bin/env bash
set -eo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="$BASE_DIR/jobs.conf"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*"; }
now() { date +%s; }

log "Scheduler booting..."
log "Base dir: $BASE_DIR"
log "Config: $CONFIG_FILE"
log "Logs: $LOG_DIR"

if [[ ! -f "$CONFIG_FILE" ]]; then
  log "ERROR: jobs.conf not found"
  exit 1
fi

# Arrays for jobs
declare -a IDS TIMES CWDS CMDS NEXT ENABLED

# Load config (id|type|time|cwd|command|enabled)
i=0
while IFS='|' read -r id type time cwd cmd enabled || [[ -n "${id:-}" ]]; do
  # skip empty / comment
  [[ -z "$id" || "$id" =~ ^# ]] && continue
  IDS[i]="$id"
  TIMES[i]="$time"
  CWDS[i]="$cwd"
  CMDS[i]="$cmd"
  ENABLED[i]="${enabled:-yes}"
  NEXT[i]=0
  log "Loaded job: $id at ${time}"
  ((i++))
done < "$CONFIG_FILE" || true

# helper to compute next daily run
compute_next() {
  local idx=$1
  local t="${TIMES[idx]}"
  local now_ts=$(now)
  local next
  next=$(date -d "today ${t}" +%s 2>/dev/null) || return 1
  (( next <= now_ts )) && next=$(date -d "tomorrow ${t}" +%s)
  NEXT[idx]=$next
}

# compute next runs
for idx in "${!IDS[@]}"; do
  [[ "${ENABLED[idx]}" != "yes" ]] && continue
  compute_next "$idx"
done

# helper: run job by index (background) and rotate its NEXT to next day
run_job_by_idx() {
  local idx=$1
  local id="${IDS[idx]}"
  local cwd="${CWDS[idx]}"
  local cmd="${CMDS[idx]}"
  local logfile="$LOG_DIR/${id}_$(date +%Y%m%d_%H%M%S).log"

  log "Immediate run -> $id (cwd=$cwd) : $cmd"
  (
    cd "$cwd" 2>/dev/null || true
    echo "===== START $(date) =====" >> "$logfile"
    bash -lc "$cmd" >> "$logfile" 2>&1 || echo "Command exited non-zero" >> "$logfile"
    echo "===== END $(date) =====" >> "$logfile"
  ) &

  # schedule next day's run at same time
  if [[ -n "${NEXT[idx]:-}" && "${NEXT[idx]}" -ne 0 ]]; then
    epoch="${NEXT[idx]}"
    # convert epoch to human time, then add 1 day
    base="$(date -d "@$epoch" '+%F %T' 2>/dev/null || true)"
    if [[ -n "$base" ]]; then
      NEXT[idx]="$(date -d "$base +1 day" +%s 2>/dev/null || true)"
    else
      NEXT[idx]=0
    fi
  else
    compute_next "$idx" || true
    if [[ -n "${NEXT[idx]:-}" && "${NEXT[idx]}" -ne 0 ]]; then
      epoch="${NEXT[idx]}"
      base="$(date -d "@$epoch" '+%F %T' 2>/dev/null || true)"
      NEXT[idx]="$(date -d "$base +1 day" +%s 2>/dev/null || true)"
    fi
  fi
}

# find job indexes
find_job_idx() {
  local search_id="$1"
  for idx in "${!IDS[@]}"; do
    if [[ "${IDS[idx]}" == "$search_id" ]]; then
      echo "$idx"
      return 0
    fi
  done
  return 1
}

# --- TIME-AWARE STARTUP CONDITIONAL LOGIC ---

now_ts=$(now)

today_date="$(date +%F)"
today_1150_ts=$(date -d "${today_date} 11:50" +%s)
today_2330_ts=$(date -d "${today_date} 23:30" +%s)
yesterday_2300_ts=$(date -d "yesterday 23:00" +%s)

download_idx="$(find_job_idx auto_download_pdfs || true)"
pipeline_idx="$(find_job_idx force_pipeline || true)"

downloader_ran_today_after_1150() {
  local since_ts
  since_ts=$(date -d "$(date +%F) 11:50" +%s)
  local since_str
  since_str="$(date -d "@$since_ts" '+%F %T')"

  find "$LOG_DIR" -type f -name 'auto_download_pdfs_*.log' \
    -newermt "$since_str" -print -quit | grep -q .
}

pipeline_ran_recently() {
  # any pipeline log newer than yesterday 23:00
  local since_ts="$yesterday_2300_ts"
  local since_str
  since_str="$(date -d "@$since_ts" '+%F %T')"
  find "$LOG_DIR" -type f -name 'force_pipeline_*.log' -newermt "$since_str" -print -quit | grep -q .
}

# -------- Downloader logic (CORRECT) --------
if (( now_ts > today_1150_ts )); then
  if ! downloader_ran_today_after_1150; then
    log "Startup check: After 11:50 and downloader NOT run today -> triggering downloader."
    [[ -n "$download_idx" ]] && run_job_by_idx "$download_idx"
  else
    log "Startup check: Downloader already ran today after 11:50; skipping."
  fi
else
  log "Startup check: Before 11:50; downloader not triggered."
fi

# -------- Pipeline logic --------
if (( now_ts > today_2330_ts )); then
  if ! pipeline_ran_recently; then
    log "Startup check: After 23:30 and pipeline NOT run since yesterday 23:00 -> triggering pipeline."
    [[ -n "$pipeline_idx" ]] && run_job_by_idx "$pipeline_idx"
  else
    log "Startup check: Pipeline already ran recently; skipping."
  fi
else
  log "Startup check: Before 23:30; pipeline not triggered."
fi

trap 'log "Scheduler stopped"; exit 0' SIGINT SIGTERM
log "Scheduler started successfully"

# Main loop: sleep until next job(s)
while true; do
  sleep 1
  ts=$(now)

  # find nearest next
  min_next=0; min_set=0
  for idx in "${!IDS[@]}"; do
    nr=${NEXT[idx]:-0}
    (( nr == 0 )) && continue
    if (( ! min_set )) || (( nr < min_next )); then
      min_next=$nr
      min_set=1
    fi
  done

  if (( min_set == 0 )); then
    # nothing scheduled, sleep and recompute daily
    sleep 60
    for idx in "${!IDS[@]}"; do
      [[ "${ENABLED[idx]}" != "yes" ]] && continue
      compute_next "$idx" || true
    done
    continue
  fi

  sleep_sec=$(( min_next - ts ))
  if (( sleep_sec > 0 )); then
    # cap long sleeps
    if (( sleep_sec > 3600 )); then
      sleep 3600
    else
      sleep "$sleep_sec"
    fi
  fi

  ts=$(now)
  for idx in "${!IDS[@]}"; do
    nr=${NEXT[idx]:-0}
    if (( nr != 0 && nr <= ts )); then
      id="${IDS[idx]}"
      cwd="${CWDS[idx]}"
      cmd="${CMDS[idx]}"
      logfile="$LOG_DIR/${id}_$(date +%Y%m%d_%H%M%S).log"

      log "Running job: $id (scheduled)"
      (
        cd "$cwd" 2>/dev/null || true
        echo "===== START $(date) =====" >> "$logfile"
        bash -lc "$cmd" >> "$logfile" 2>&1 || echo "Command exited non-zero" >> "$logfile"
        echo "===== END $(date) =====" >> "$logfile"
      ) &

      # reschedule for next day
      epoch="$nr"
      base="$(date -d "@$epoch" '+%F %T' 2>/dev/null || true)"
      if [[ -n "$base" ]]; then
        NEXT[idx]="$(date -d "$base +1 day" +%s 2>/dev/null || true)"
      else
        NEXT[idx]=0
      fi
    fi
  done
done