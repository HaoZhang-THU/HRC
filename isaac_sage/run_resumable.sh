#!/usr/bin/env bash
# Run train_sage.py and restart it from <log_dir>/models/latest.pt whenever the
# process exits before the run is finished (crash, or the stall watchdog firing).
#
#   isaac_sage/run_resumable.sh <log_dir> [train_sage args ...]
#
# Uses $PYTHON (default: python) and appends all output to <log_dir>/train.log.
# The run is finished when train() writes <log_dir>/DONE.
# Stop everything with:  pkill -f run_resumable.sh; pkill -f isaac_sage.train_sage
set -u
LOG_DIR="$1"; shift
PYTHON="${PYTHON:-python}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
cd "$(dirname "$0")/.."
mkdir -p "$LOG_DIR"

for attempt in $(seq 0 "$MAX_RESTARTS"); do
    RESUME=()
    if [ -f "$LOG_DIR/models/latest.pt" ]; then
        RESUME=(--resume "$LOG_DIR/models/latest.pt" --resume_continue)
    fi
    echo "[supervisor] $(date '+%F %T') attempt $attempt ${RESUME[*]:-fresh}" >> "$LOG_DIR/train.log"
    "$PYTHON" -u -m isaac_sage.train_sage --log_dir "$LOG_DIR" "$@" "${RESUME[@]}" >> "$LOG_DIR/train.log" 2>&1
    code=$?
    if [ -f "$LOG_DIR/DONE" ]; then
        echo "[supervisor] $(date '+%F %T') finished" >> "$LOG_DIR/train.log"
        exit 0
    fi
    echo "[supervisor] $(date '+%F %T') exit code $code before DONE; restarting in 20 s" >> "$LOG_DIR/train.log"
    sleep 20
done
echo "[supervisor] giving up after $MAX_RESTARTS restarts" >> "$LOG_DIR/train.log"
exit 1
