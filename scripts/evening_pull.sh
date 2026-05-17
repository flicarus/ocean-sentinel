#!/usr/bin/env bash
# Evening pull — 5 parallel MBARI workers on 2023 dates (fresh year).
# Runs alongside v7.4 training. ~5h target, finishes ~21:30.
#
# Why 2023: every 2024 day we've already used would dedup-skip; 2023 is
# entirely fresh data the model has never seen. Adds temporal diversity
# (different annual cycle, different vessel patterns, different seasonal
# biology).
#
# Params bumped to chunks-per-hour=22, cap=500 — that puts the cap right
# at where it starts binding (24*22=528 candidates/day, cap=500), so we
# get ~500 samples per day per worker. 5 workers × 8 days × 500 = 20k max.
#
# Usage:
#   mkdir -p /tmp/eve && nohup bash scripts/evening_pull.sh > /tmp/eve/main.log 2>&1 &

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/venv/bin/python"
LOG_DIR="/tmp/eve"
mkdir -p "$LOG_DIR"

if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i -w $$ &
fi

step() {
    echo ""
    echo "===================================================="
    echo "  $(date '+%Y-%m-%d %H:%M:%S')  $1"
    echo "===================================================="
}

START_TS=$(date '+%s')

CHUNKS_PER_HOUR=22
CAP=500

step "5-way parallel MBARI on FRESH 2023 dates"

# Worker A: 2023 Jan-Feb
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2023-01-04,2023-01-15,2023-01-25,2023-02-02,2023-02-12,2023-02-19,2023-02-25,2023-01-09 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_eve_a.jsonl \
    > "$LOG_DIR/A_2023_janfeb.log" 2>&1 &
PID_A=$!

# Worker B: 2023 Mar-Apr
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2023-03-05,2023-03-15,2023-03-22,2023-03-28,2023-04-02,2023-04-10,2023-04-18,2023-04-25 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_eve_b.jsonl \
    > "$LOG_DIR/B_2023_marapr.log" 2>&1 &
PID_B=$!

# Worker C: 2023 May-Jun
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2023-05-03,2023-05-13,2023-05-22,2023-05-29,2023-06-04,2023-06-12,2023-06-20,2023-06-27 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_eve_c.jsonl \
    > "$LOG_DIR/C_2023_mayjun.log" 2>&1 &
PID_C=$!

# Worker D: 2023 Jul-Aug
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2023-07-04,2023-07-13,2023-07-21,2023-07-29,2023-08-05,2023-08-13,2023-08-21,2023-08-28 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_eve_d.jsonl \
    > "$LOG_DIR/D_2023_julaug.log" 2>&1 &
PID_D=$!

# Worker E: 2023 Sep-Oct (cooler waters, autumn migration)
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2023-09-05,2023-09-13,2023-09-21,2023-09-28,2023-10-04,2023-10-12,2023-10-20,2023-10-28 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_eve_e.jsonl \
    > "$LOG_DIR/E_2023_sepoct.log" 2>&1 &
PID_E=$!

echo "[$(date '+%H:%M:%S')] waiting on 5 parallel pulls..."
wait $PID_A; RC_A=$?
wait $PID_B; RC_B=$?
wait $PID_C; RC_C=$?
wait $PID_D; RC_D=$?
wait $PID_E; RC_E=$?

count_lines() {
    [[ -f "$1" ]] && wc -l < "$1" | tr -d ' ' || echo 0
}

A_N=$(count_lines data/training/v7_bulk/mbari_diverse_eve_a.jsonl)
B_N=$(count_lines data/training/v7_bulk/mbari_diverse_eve_b.jsonl)
C_N=$(count_lines data/training/v7_bulk/mbari_diverse_eve_c.jsonl)
D_N=$(count_lines data/training/v7_bulk/mbari_diverse_eve_d.jsonl)
E_N=$(count_lines data/training/v7_bulk/mbari_diverse_eve_e.jsonl)
TOTAL=$(( A_N + B_N + C_N + D_N + E_N ))

END_TS=$(date '+%s')
ELAPSED_MIN=$(( (END_TS - START_TS) / 60 ))

cat > "$LOG_DIR/manifest.json" <<EOF
{
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "elapsed_minutes": $ELAPSED_MIN,
  "exit_codes": {"A": $RC_A, "B": $RC_B, "C": $RC_C, "D": $RC_D, "E": $RC_E},
  "rows_per_worker": {"A_2023_janfeb": $A_N, "B_2023_marapr": $B_N, "C_2023_mayjun": $C_N, "D_2023_julaug": $D_N, "E_2023_sepoct": $E_N},
  "total_new": $TOTAL
}
EOF
cat "$LOG_DIR/manifest.json"
echo ""
echo "[$(date '+%H:%M:%S')] Evening pull DONE in ${ELAPSED_MIN} min — $TOTAL new rows."
