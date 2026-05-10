#!/usr/bin/env bash
# Afternoon pull — runs alongside v7.3 retrain.
# Uses FRESH MBARI days from Jan-Aug 2024 (every day below is unpulled).
# 5 parallel MBARI workers × 7 days = 35 fresh days.
#
# Usage:
#   mkdir -p /tmp/pm && nohup bash scripts/afternoon_pull.sh > /tmp/pm/main.log 2>&1 &
#
# Coexistence with training: train uses MPS GPU + small CPU dataloader,
# pull is HTTP-bound + light spec compute. M5 Pro 12-core handles both.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/venv/bin/python"
LOG_DIR="/tmp/pm"
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

CHUNKS_PER_HOUR=16
CAP=500

step "5-way parallel MBARI on FRESH days (35 days, all Jan-Aug 2024)"

# Worker A: scattered Jan
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-01-01,2024-01-03,2024-01-06,2024-01-09,2024-01-13,2024-01-17,2024-01-21 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_pm_a.jsonl \
    > "$LOG_DIR/A_jan.log" 2>&1 &
PID_A=$!

# Worker B: scattered Feb
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-02-03,2024-02-07,2024-02-11,2024-02-14,2024-02-18,2024-02-22,2024-02-25 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_pm_b.jsonl \
    > "$LOG_DIR/B_feb.log" 2>&1 &
PID_B=$!

# Worker C: Mar+Apr fresh
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-03-06,2024-03-11,2024-03-15,2024-03-20,2024-03-26,2024-03-30,2024-04-04 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_pm_c.jsonl \
    > "$LOG_DIR/C_marapr.log" 2>&1 &
PID_C=$!

# Worker D: May+Jun fresh
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-05-04,2024-05-11,2024-05-19,2024-05-25,2024-06-01,2024-06-08,2024-06-14 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_pm_d.jsonl \
    > "$LOG_DIR/D_mayjun.log" 2>&1 &
PID_D=$!

# Worker E: Jul+Aug fresh
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-07-04,2024-07-11,2024-07-18,2024-07-24,2024-07-30,2024-08-04,2024-08-12 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_pm_e.jsonl \
    > "$LOG_DIR/E_julaug.log" 2>&1 &
PID_E=$!

echo "[$(date '+%H:%M:%S')] waiting on 5 parallel MBARI pulls..."
wait $PID_A; RC_A=$?
wait $PID_B; RC_B=$?
wait $PID_C; RC_C=$?
wait $PID_D; RC_D=$?
wait $PID_E; RC_E=$?

count_lines() {
    [[ -f "$1" ]] && wc -l < "$1" | tr -d ' ' || echo 0
}

A_N=$(count_lines data/training/v7_bulk/mbari_diverse_pm_a.jsonl)
B_N=$(count_lines data/training/v7_bulk/mbari_diverse_pm_b.jsonl)
C_N=$(count_lines data/training/v7_bulk/mbari_diverse_pm_c.jsonl)
D_N=$(count_lines data/training/v7_bulk/mbari_diverse_pm_d.jsonl)
E_N=$(count_lines data/training/v7_bulk/mbari_diverse_pm_e.jsonl)
TOTAL=$(( A_N + B_N + C_N + D_N + E_N ))

END_TS=$(date '+%s')
ELAPSED_MIN=$(( (END_TS - START_TS) / 60 ))

cat > "$LOG_DIR/manifest.json" <<EOF
{
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "elapsed_minutes": $ELAPSED_MIN,
  "exit_codes": {"A": $RC_A, "B": $RC_B, "C": $RC_C, "D": $RC_D, "E": $RC_E},
  "rows_per_worker": {"A_jan": $A_N, "B_feb": $B_N, "C_marapr": $C_N, "D_mayjun": $D_N, "E_julaug": $E_N},
  "total_new": $TOTAL
}
EOF
cat "$LOG_DIR/manifest.json"
echo ""
echo "[$(date '+%H:%M:%S')] PM pull DONE in ${ELAPSED_MIN} min — $TOTAL new rows."
