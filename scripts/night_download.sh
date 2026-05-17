#!/usr/bin/env bash
# Night data pull — multi-process but single-source-light (in-bed safe).
#
# Usage (before sleep):
#   nohup bash scripts/night_download.sh > /tmp/night/main.log 2>&1 &
#
# Why this is fast: bootstrap_mbari_diverse.py was capped at 30 ship + 30
# ambient PER DAY, single-threaded. Two changes:
#   1. cap-per-day-per-label bumped to 200 (still balanced, just more
#      data per day).
#   2. Three parallel workers writing to distinct --out files
#      (mbari_diverse_q1a/_q1b/_q1c.jsonl). Trainer globs v7_bulk/*.jsonl
#      so all three feed into v7.2.
#
# In-bed safety:
#   - 4 parallel workers max, each is HTTP/S3-bound (low CPU).
#   - Plug in the charger.
#   - Use a hard surface, not blanket, for vents.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/venv/bin/python"
LOG_DIR="/tmp/night"
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

# ============================================================ Phase A
step "PHASE A: AIS-relabel oc01 + sb01 (sequential, light)"
PYTHONPATH=src "$VENV_PY" scripts/sanctsound_ais_relabel.py \
    > "$LOG_DIR/A_relabel.log" 2>&1 \
    || echo "[WARN] relabel failed — see A_relabel.log; continuing"

# ============================================================ Phase B
step "PHASE B: 3-way parallel MBARI Q1+Q2 + 1 Orcasound"

# B1: MBARI Jan-Feb (cargo-heavy winter shipping)
echo "[$(date '+%H:%M:%S')] launching B1: MBARI Jan-Feb ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-01-08,2024-01-22,2024-02-05,2024-02-19 \
    --chunks-per-hour 15 --cap-per-day-per-label 400 --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_q1a.jsonl \
    > "$LOG_DIR/B1_mbari_q1a.log" 2>&1 &
PID_B1=$!

# B2: MBARI Mar-Apr (spring fishery openings)
echo "[$(date '+%H:%M:%S')] launching B2: MBARI Mar-Apr ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-03-04,2024-03-18,2024-04-01,2024-04-15 \
    --chunks-per-hour 15 --cap-per-day-per-label 400 --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_q1b.jsonl \
    > "$LOG_DIR/B2_mbari_q1b.log" 2>&1 &
PID_B2=$!

# B3: MBARI May-Jun (calm sea, deep canyon ambient bias)
echo "[$(date '+%H:%M:%S')] launching B3: MBARI May-Jun ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-05-06,2024-05-20,2024-06-03,2024-06-17 \
    --chunks-per-hour 15 --cap-per-day-per-label 400 --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_q1c.jsonl \
    > "$LOG_DIR/B3_mbari_q1c.log" 2>&1 &
PID_B3=$!

# B4: Orcasound port-townsend (new node, parallel — different endpoint)
echo "[$(date '+%H:%M:%S')] launching B4: Orcasound port-townsend ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_orcasound_v2.py \
    --hydrophone port-townsend --step 150 --radius-km 10 \
    > "$LOG_DIR/B4_orca_port_townsend.log" 2>&1 &
PID_B4=$!

echo "[$(date '+%H:%M:%S')] waiting on 4 parallel pulls..."
wait $PID_B1; RC_B1=$?
wait $PID_B2; RC_B2=$?
wait $PID_B3; RC_B3=$?
wait $PID_B4; RC_B4=$?
echo "[$(date '+%H:%M:%S')] phase B done — exit codes: B1=$RC_B1 B2=$RC_B2 B3=$RC_B3 B4=$RC_B4"

# ============================================================ Phase C
step "PHASE C: Orcasound bush-point re-pull, finer step=150"
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_orcasound_v2.py \
    --hydrophone bush-point --step 150 --radius-km 10 \
    > "$LOG_DIR/C_orca_bush_point.log" 2>&1 \
    || echo "[WARN] orca bush-point re-pull failed — continuing"

# ============================================================ Manifest
step "DONE — writing manifest"

count_lines() {
    local f="$1"
    [[ -f "$f" ]] && wc -l < "$f" | tr -d ' ' || echo 0
}

END_TS=$(date '+%s')
ELAPSED_MIN=$(( (END_TS - START_TS) / 60 ))

NEW_MBARI=$(( $(count_lines data/training/v7_bulk/mbari_diverse_q1a.jsonl) +
              $(count_lines data/training/v7_bulk/mbari_diverse_q1b.jsonl) +
              $(count_lines data/training/v7_bulk/mbari_diverse_q1c.jsonl) ))
NEW_ORCA=$(( $(count_lines data/training/v7_bulk/orcasound_port-townsend.jsonl) +
             $(count_lines data/training/v7_bulk/orcasound_bush-point.jsonl) ))

cat > "$LOG_DIR/manifest.json" <<EOF
{
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "elapsed_minutes": $ELAPSED_MIN,
  "phase_b_exit_codes": {"B1_q1a": $RC_B1, "B2_q1b": $RC_B2, "B3_q1c": $RC_B3, "B4_orca": $RC_B4},
  "new_rows": {
    "mbari_q1_total":           $NEW_MBARI,
    "orcasound_port_townsend":  $(count_lines data/training/v7_bulk/orcasound_port-townsend.jsonl),
    "orcasound_bush_point":     $(count_lines data/training/v7_bulk/orcasound_bush-point.jsonl),
    "sanctsound_corrected":     $(count_lines data/training/sanctsound_corrected.jsonl)
  },
  "logs": "$LOG_DIR"
}
EOF
cat "$LOG_DIR/manifest.json"
echo ""
echo "[$(date '+%H:%M:%S')] Night pull DONE in ${ELAPSED_MIN} min."
echo "MBARI Q1 total: $NEW_MBARI rows; Orcasound total: $NEW_ORCA rows."
echo "Run morning_bootstrap.sh next."
