#!/usr/bin/env bash
# Serial Orcasound pull — runs each hydrophone one at a time to avoid
# hitting the GFW API rate limit (1 concurrent report/token).
#
# Launched separately from MBARI parallel pulls so they don't fight
# over the same token. MBARI uses ~1 AIS call/day so 6× parallel is OK;
# Orcasound calls AIS per window so must serialize.
#
# Usage:
#   nohup bash scripts/overnight_orca_serial.sh > /tmp/overnight_pull/orca_serial.log 2>&1 &

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/venv/bin/python"
LOG_DIR="/tmp/overnight_pull"
mkdir -p "$LOG_DIR"

echo "[$(date '+%H:%M:%S')] orca_serial starting"

NODES=(
    bush-point
    sunset-bay
    orcasound-lab
    north-sjc
    point-robinson
    port-townsend
    andrews-bay
    mast-center
)

for node in "${NODES[@]}"; do
    echo ""
    echo "[$(date '+%H:%M:%S')] === orca $node start ==="
    PYTHONPATH=src "$VENV_PY" scripts/bootstrap_orcasound_v2.py \
        --hydrophone "$node" --step 120 --radius-km 10 \
        > "$LOG_DIR/orca_${node}.log" 2>&1
    rc=$?
    echo "[$(date '+%H:%M:%S')] === orca $node done exit=$rc ==="
done

echo ""
echo "[$(date '+%H:%M:%S')] === sanctsound_more start ==="
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_sanctsound_more_60s.py \
    > "$LOG_DIR/sanctsound_more.log" 2>&1
rc=$?
echo "[$(date '+%H:%M:%S')] === sanctsound_more done exit=$rc ==="

echo ""
echo "[$(date '+%H:%M:%S')] orca_serial finished"
