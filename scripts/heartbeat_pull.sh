#!/usr/bin/env bash
# Heartbeat: append data-pull progress every 5 min to /tmp/overnight_pull/heartbeat.log
#
# Usage:
#   nohup bash scripts/heartbeat_pull.sh > /tmp/overnight_pull/heartbeat_run.log 2>&1 &

set -uo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
LOG_DIR="/tmp/overnight_pull"
mkdir -p "$LOG_DIR"

while true; do
    n_spec_5=$(ls data/spectrograms 2>/dev/null | wc -l | tr -d ' ')
    n_spec_60=$(ls data/spectrograms_60s 2>/dev/null | wc -l | tr -d ' ')
    n_v7bulk=$(find data/training/v7_bulk -name "*.jsonl" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    n_mbari_proc=$(pgrep -f bootstrap_mbari_diverse 2>/dev/null | wc -l | tr -d ' ')
    n_orca_proc=$(pgrep -f bootstrap_orcasound 2>/dev/null | wc -l | tr -d ' ')
    n_sanct_proc=$(pgrep -f bootstrap_sanctsound 2>/dev/null | wc -l | tr -d ' ')
    echo "$(date '+%Y-%m-%d %H:%M:%S')  spec5=$n_spec_5  spec60=$n_spec_60  v7bulk_rows=$n_v7bulk  procs[mbari=$n_mbari_proc orca=$n_orca_proc sanct=$n_sanct_proc]" \
        >> "$LOG_DIR/heartbeat.log"
    sleep 300
done
