#!/usr/bin/env bash
# Concurrent data pull — runs alongside training, uses fresh MBARI dates
# from confirmed-working months (Jan-Aug 2024). Sep-Dec 2024 returns
# 404 from the Pacific Sound bucket (not yet published), so we avoid them.
#
# Why this is safe to run alongside training:
#   - Training uses MPS GPU + small CPU for dataloader. Pulls are HTTP/S3
#     bound, mostly idle. Different resources, no contention.
#   - Disk: training reads ~50KB specs, pulls write ~50KB specs. M5 Pro
#     NVMe handles this trivially.
#   - GFW key: MBARI does 1 AIS lookup PER DAY (not per chunk), so 5
#     extra workers add ~5 AIS calls per minute — well under rate limit.
#
# Days picked to be DISJOINT from existing q1a/b/c, q3a, and original
# mbari_diverse pulls — every day below is fresh.
#
# Output: data/training/v7_bulk/mbari_diverse_extra_{a-e}.jsonl
# Trainer globs v7_bulk/*.jsonl so v7.3 (next retrain) picks them up.
#
# Usage:
#   mkdir -p /tmp/extra && nohup bash scripts/concurrent_day_pull.sh > /tmp/extra/main.log 2>&1 &

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/venv/bin/python"
LOG_DIR="/tmp/extra"
mkdir -p "$LOG_DIR"

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

START_TS=$(date '+%s')

# Common params: chunks-per-hour=16 → 24×16=384 candidates/day.
# cap=500 means cap won't actually bite unless a day is unusually busy;
# we want most chunks landed.
CHUNKS_PER_HOUR=16
CAP=500

step "5-way parallel MBARI on FRESH days (Jan-Aug 2024 only)"

# Worker A: rest of January
echo "[$(date '+%H:%M:%S')] launching A: Jan-extras ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-01-05,2024-01-12,2024-01-15,2024-01-26,2024-01-29 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_extra_a.jsonl \
    > "$LOG_DIR/A_jan.log" 2>&1 &
PID_A=$!

# Worker B: rest of February
echo "[$(date '+%H:%M:%S')] launching B: Feb-extras ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-02-01,2024-02-09,2024-02-16,2024-02-23,2024-02-26 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_extra_b.jsonl \
    > "$LOG_DIR/B_feb.log" 2>&1 &
PID_B=$!

# Worker C: Mar+Apr extras
echo "[$(date '+%H:%M:%S')] launching C: Mar+Apr-extras ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-03-08,2024-03-22,2024-03-25,2024-04-08,2024-04-22 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_extra_c.jsonl \
    > "$LOG_DIR/C_mar_apr.log" 2>&1 &
PID_C=$!

# Worker D: May+Jun extras
echo "[$(date '+%H:%M:%S')] launching D: May+Jun-extras ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-05-01,2024-05-13,2024-05-24,2024-06-07,2024-06-21 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_extra_d.jsonl \
    > "$LOG_DIR/D_may_jun.log" 2>&1 &
PID_D=$!

# Worker E: Jul+Aug extras (avoid 08, 22, 05, 19 already pulled)
echo "[$(date '+%H:%M:%S')] launching E: Jul+Aug-extras ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-07-01,2024-07-15,2024-07-26,2024-08-08,2024-08-22 \
    --chunks-per-hour $CHUNKS_PER_HOUR --cap-per-day-per-label $CAP --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_extra_e.jsonl \
    > "$LOG_DIR/E_jul_aug.log" 2>&1 &
PID_E=$!

# 2 Orcasound re-pulls at finer step on nodes not touched today
echo "[$(date '+%H:%M:%S')] launching F: sunset-bay step=100 ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_orcasound_v2.py \
    --hydrophone sunset-bay --step 100 --radius-km 10 \
    > "$LOG_DIR/F_orca_sunset_bay.log" 2>&1 &
PID_F=$!

echo "[$(date '+%H:%M:%S')] launching G: orcasound-lab step=100 ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_orcasound_v2.py \
    --hydrophone orcasound-lab --step 100 --radius-km 10 \
    > "$LOG_DIR/G_orca_lab.log" 2>&1 &
PID_G=$!

echo "[$(date '+%H:%M:%S')] waiting on 7 parallel pulls..."
wait $PID_A; RC_A=$?
wait $PID_B; RC_B=$?
wait $PID_C; RC_C=$?
wait $PID_D; RC_D=$?
wait $PID_E; RC_E=$?
wait $PID_F; RC_F=$?
wait $PID_G; RC_G=$?
echo "[$(date '+%H:%M:%S')] all done — RCs: A=$RC_A B=$RC_B C=$RC_C D=$RC_D E=$RC_E F=$RC_F G=$RC_G"

# Manifest
count_lines() {
    local f="$1"
    [[ -f "$f" ]] && wc -l < "$f" | tr -d ' ' || echo 0
}

A_N=$(count_lines data/training/v7_bulk/mbari_diverse_extra_a.jsonl)
B_N=$(count_lines data/training/v7_bulk/mbari_diverse_extra_b.jsonl)
C_N=$(count_lines data/training/v7_bulk/mbari_diverse_extra_c.jsonl)
D_N=$(count_lines data/training/v7_bulk/mbari_diverse_extra_d.jsonl)
E_N=$(count_lines data/training/v7_bulk/mbari_diverse_extra_e.jsonl)
SUNSET_N=$(count_lines data/training/v7_bulk/orcasound_sunset-bay.jsonl)
LAB_N=$(count_lines data/training/v7_bulk/orcasound_orcasound-lab.jsonl)

MBARI_TOTAL=$(( A_N + B_N + C_N + D_N + E_N ))

END_TS=$(date '+%s')
ELAPSED_MIN=$(( (END_TS - START_TS) / 60 ))

cat > "$LOG_DIR/manifest.json" <<EOF
{
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "elapsed_minutes": $ELAPSED_MIN,
  "exit_codes": {"A": $RC_A, "B": $RC_B, "C": $RC_C, "D": $RC_D, "E": $RC_E, "F_sunset": $RC_F, "G_lab": $RC_G},
  "new_rows": {
    "mbari_extra_a_jan":   $A_N,
    "mbari_extra_b_feb":   $B_N,
    "mbari_extra_c_marapr": $C_N,
    "mbari_extra_d_mayjun": $D_N,
    "mbari_extra_e_julaug": $E_N,
    "mbari_extra_total":   $MBARI_TOTAL,
    "orca_sunset_bay_total": $SUNSET_N,
    "orca_lab_total":      $LAB_N
  }
}
EOF
cat "$LOG_DIR/manifest.json"
echo ""
echo "[$(date '+%H:%M:%S')] Concurrent pull DONE in ${ELAPSED_MIN} min."
echo "MBARI extra total: $MBARI_TOTAL rows."
