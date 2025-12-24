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

declare -a IDS TIMES CWDS CMDS NEXT ENABLED

i=0
while IFS='|' read -r id type time cwd cmd enabled; do
  [[ -z "$id" || "$id" =~ ^# ]] && continue

  IDS[i]="$id"
  TIMES[i]="$time"
  CWDS[i]="$cwd"
  CMDS[i]="$cmd"
  ENABLED[i]="${enabled:-yes}"
  NEXT[i]=0

  log "Loaded job: $id at $time"
  ((i++))
done < "$CONFIG_FILE" || true

compute_next() {
  local idx=$1
  local now_ts=$(now)
  local next

  next=$(date -d "today ${TIMES[idx]}" +%s)
  (( next <= now_ts )) && next=$(date -d "tomorrow ${TIMES[idx]}" +%s)
  NEXT[idx]=$next
}

for i in "${!IDS[@]}"; do
  [[ "${ENABLED[i]}" != "yes" ]] && continue
  compute_next "$i"
done

trap 'log "Scheduler stopped"; exit 0' SIGINT SIGTERM

log "Scheduler started successfully"

while true; do
  sleep 1
  ts=$(now)

  for i in "${!IDS[@]}"; do
    [[ "${ENABLED[i]}" != "yes" ]] && continue

    if (( ts >= NEXT[i] )); then
      id="${IDS[i]}"
      cwd="${CWDS[i]}"
      cmd="${CMDS[i]}"
      logfile="$LOG_DIR/${id}_$(date +%Y%m%d_%H%M%S).log"

      log "Running job: $id"
      (
        cd "$cwd"
        echo "===== START $(date) =====" >> "$logfile"
        bash -lc "$cmd" >> "$logfile" 2>&1
        echo "===== END $(date) =====" >> "$logfile"
      ) &

      NEXT[i]=$(date -d "@${NEXT[i]} +1 day" +%s)
    fi
  done
done
