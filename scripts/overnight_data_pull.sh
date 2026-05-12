#!/usr/bin/env bash
# Overnight DATA PULL ONLY — no training, no eval.
# Goal: scale training pool from ~72.5k → 200-500k samples via parallel
# bootstrap across MBARI, Orcasound, AIS-correlated streams, and SanctSound.
#
# Designed to be safely re-runnable: every bootstrap_* script dedupes via
# event_id / capture_start so re-launching skips already-pulled chunks.
#
# Usage:
#   nohup bash scripts/overnight_data_pull.sh > /tmp/overnight_pull/main.log 2>&1 &
#
# Progress:
#   ls /tmp/overnight_pull/         — per-process logs
#   cat /tmp/overnight_pull/manifest.json    — written when done
#   tail -f /tmp/overnight_pull/heartbeat.log — appended every 10 min

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/venv/bin/python"
LOG_DIR="/tmp/overnight_pull"
mkdir -p "$LOG_DIR"

START_TS=$(date '+%s')
echo "[$(date '+%H:%M:%S')] overnight_data_pull starting"

if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i -w $$ &
    echo "[$(date '+%H:%M:%S')] caffeinate active (PID $!)"
fi

step() {
    echo ""
    echo "===================================================="
    echo "  $(date '+%Y-%m-%d %H:%M:%S')  $1"
    echo "===================================================="
}

# ============================================================
# Heartbeat: append a counts snapshot every 10 min so the
# hourly check-in can see whether new data is actually landing.
# ============================================================
heartbeat() {
    while true; do
        sleep 600
        local n_spec_5
        local n_spec_60
        local n_jsonl
        n_spec_5=$(ls data/spectrograms 2>/dev/null | wc -l | tr -d ' ')
        n_spec_60=$(ls data/spectrograms_60s 2>/dev/null | wc -l | tr -d ' ')
        n_jsonl=$(find data/training/v7_bulk -name "*.jsonl" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
        echo "$(date '+%Y-%m-%d %H:%M:%S') spec5=$n_spec_5 spec60=$n_spec_60 v7_bulk_rows=$n_jsonl" \
            >> "$LOG_DIR/heartbeat.log"
    done
}
heartbeat &
HB_PID=$!
echo "[$(date '+%H:%M:%S')] heartbeat pid=$HB_PID"

# ============================================================ PHASE 1
# MBARI diverse — aggressive multi-month, multi-day, parallel.
# Cap-per-day-per-label=500 (was 200/400), chunks-per-hour=20 (was 8/15).
# Each run handles 5-6 days; 6 runs in parallel × ~6 days = 36 day-runs.
# Yield est: 6 × 6 days × 24h × 20 chunks × 2 labels capped @ 500/day
#         ≈ 36k chunks worst-case, more like 20-30k after caps.
# ============================================================
step "PHASE 1: MBARI diverse parallel (6 branches)"

start_mbari() {
    local label="$1"; shift
    local out="$1"; shift
    local days="$1"; shift
    echo "[$(date '+%H:%M:%S')] mbari $label days=$days -> $out"
    PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
        --days "$days" \
        --chunks-per-hour 20 --cap-per-day-per-label 500 --radius-km 10 \
        --out "$out" \
        > "$LOG_DIR/mbari_${label}.log" 2>&1 &
}

start_mbari "2023_h1" "data/training/v7_bulk/mbari_2023h1.jsonl" \
    "2023-01-15,2023-02-15,2023-03-15,2023-04-15,2023-05-15,2023-06-15"
M1=$!

start_mbari "2023_h2" "data/training/v7_bulk/mbari_2023h2.jsonl" \
    "2023-07-15,2023-08-15,2023-09-15,2023-10-15,2023-11-15,2023-12-15"
M2=$!

start_mbari "2024_extra1" "data/training/v7_bulk/mbari_2024x1.jsonl" \
    "2024-01-08,2024-02-12,2024-03-19,2024-04-23,2024-05-27,2024-07-01"
M3=$!

start_mbari "2024_extra2" "data/training/v7_bulk/mbari_2024x2.jsonl" \
    "2024-07-29,2024-08-26,2024-09-23,2024-10-21,2024-11-18,2024-12-23"
M4=$!

start_mbari "2025_h1" "data/training/v7_bulk/mbari_2025h1.jsonl" \
    "2025-01-15,2025-02-15,2025-03-15,2025-04-15,2025-05-15,2025-06-15"
M5=$!

start_mbari "2025_h2" "data/training/v7_bulk/mbari_2025h2.jsonl" \
    "2025-07-15,2025-08-15,2025-09-15,2025-10-15,2025-11-15,2025-12-15"
M6=$!

# ============================================================ PHASE 2
# Orcasound — 8 hydrophone nodes, finer step (every 2 min vs 5 min).
# Each node pull is independent (different IO endpoint). max-windows
# unset = pull whatever is available.
# Yield est: ~720 windows/day per node, but depends on stream availability.
# ============================================================
step "PHASE 2: Orcasound 8 nodes parallel"

start_orca() {
    local node="$1"
    local step="$2"
    echo "[$(date '+%H:%M:%S')] orca $node step=${step}s"
    PYTHONPATH=src "$VENV_PY" scripts/bootstrap_orcasound_v2.py \
        --hydrophone "$node" --step "$step" --radius-km 10 \
        > "$LOG_DIR/orca_${node}.log" 2>&1 &
}

start_orca bush-point     120; O1=$!
start_orca sunset-bay     120; O2=$!
start_orca orcasound-lab  120; O3=$!
start_orca north-sjc      120; O4=$!
start_orca point-robinson 120; O5=$!
start_orca port-townsend  120; O6=$!
start_orca andrews-bay    120; O7=$!
start_orca mast-center    120; O8=$!

# ============================================================ PHASE 3
# SanctSound more — already pulled 16 deployments at default settings;
# re-launch resumes and pulls any new FLAC files that have appeared.
# Lightweight, non-fatal if it skips everything.
# ============================================================
step "PHASE 3: SanctSound more 60s (resume)"
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_sanctsound_more_60s.py \
    > "$LOG_DIR/sanctsound_more.log" 2>&1 &
S1=$!

# ============================================================ WAIT
# Block until every parallel branch returns. None of these are fatal
# to the overall pull — we collect exit codes for the manifest.
# ============================================================
step "WAITING on 15 parallel pulls"

wait_pid() {
    local label="$1"; local pid="$2"
    wait "$pid"; local rc=$?
    echo "[$(date '+%H:%M:%S')] $label (pid=$pid) exit=$rc"
    echo "$rc"
}

RC_M1=$(wait_pid "mbari_2023h1"   $M1 | tail -1)
RC_M2=$(wait_pid "mbari_2023h2"   $M2 | tail -1)
RC_M3=$(wait_pid "mbari_2024x1"   $M3 | tail -1)
RC_M4=$(wait_pid "mbari_2024x2"   $M4 | tail -1)
RC_M5=$(wait_pid "mbari_2025h1"   $M5 | tail -1)
RC_M6=$(wait_pid "mbari_2025h2"   $M6 | tail -1)
RC_O1=$(wait_pid "orca_bush"      $O1 | tail -1)
RC_O2=$(wait_pid "orca_sunset"    $O2 | tail -1)
RC_O3=$(wait_pid "orca_lab"       $O3 | tail -1)
RC_O4=$(wait_pid "orca_north"     $O4 | tail -1)
RC_O5=$(wait_pid "orca_robinson"  $O5 | tail -1)
RC_O6=$(wait_pid "orca_townsend"  $O6 | tail -1)
RC_O7=$(wait_pid "orca_andrews"   $O7 | tail -1)
RC_O8=$(wait_pid "orca_mast"      $O8 | tail -1)
RC_S1=$(wait_pid "sanctsound_more" $S1 | tail -1)

# Stop heartbeat now that work is done
kill $HB_PID 2>/dev/null || true

# ============================================================ MANIFEST
step "DONE — manifest"

END_TS=$(date '+%s')
ELAPSED_MIN=$(( (END_TS - START_TS) / 60 ))

N_SPEC_5=$(ls data/spectrograms 2>/dev/null | wc -l | tr -d ' ')
N_SPEC_60=$(ls data/spectrograms_60s 2>/dev/null | wc -l | tr -d ' ')
TOTAL_ROWS=$(PYTHONPATH=src "$VENV_PY" scripts/v7_data_inventory.py 2>/dev/null \
    | grep "Total trainable rows" | awk '{print $4}' | tr -d ',')

cat > "$LOG_DIR/manifest.json" <<EOF
{
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "elapsed_minutes": $ELAPSED_MIN,
  "final_counts": {
    "spectrograms_5s":   $N_SPEC_5,
    "spectrograms_60s":  $N_SPEC_60,
    "trainable_rows":    "${TOTAL_ROWS:-unknown}"
  },
  "exit_codes": {
    "mbari_2023h1":   $RC_M1, "mbari_2023h2":  $RC_M2,
    "mbari_2024x1":   $RC_M3, "mbari_2024x2":  $RC_M4,
    "mbari_2025h1":   $RC_M5, "mbari_2025h2":  $RC_M6,
    "orca_bush":      $RC_O1, "orca_sunset":   $RC_O2,
    "orca_lab":       $RC_O3, "orca_north":    $RC_O4,
    "orca_robinson":  $RC_O5, "orca_townsend": $RC_O6,
    "orca_andrews":   $RC_O7, "orca_mast":     $RC_O8,
    "sanctsound_more": $RC_S1
  },
  "logs": "$LOG_DIR"
}
EOF
cat "$LOG_DIR/manifest.json"

echo ""
echo "[$(date '+%H:%M:%S')] overnight_data_pull DONE in ${ELAPSED_MIN} min"
echo "Spectrograms: 5s=$N_SPEC_5  60s=$N_SPEC_60   total rows=${TOTAL_ROWS:-?}"
