#!/usr/bin/env bash
# Morning bootstrap — second data batch (parallel) + hard-neg + train v7.2.
#
# Assumes night_download.sh ran overnight and produced:
#   data/training/sanctsound_corrected.jsonl  (relabeled)
#   data/training/v7_bulk/mbari_diverse.jsonl       (Q1 days)
#   data/training/v7_bulk/orcasound_port-townsend.jsonl
#   data/training/v7_bulk/orcasound_bush-point.jsonl  (re-pulled finer)
#
# This script pulls a DIFFERENT set (Q3 days, different Orcasound nodes)
# in parallel — different network endpoints + adapters mean parallelism
# actually saves wall time, not just CPU time.
#
# Usage (before leaving for work):
#   nohup bash scripts/morning_bootstrap.sh > /tmp/morning/main.log 2>&1 &
#
# Check on return:
#   cat /tmp/morning/comparison.txt   # v7 vs v7.1 vs v7.2 side-by-side
#   cat /tmp/morning/manifest.json    # artifact paths
#
# Timing budget (~7-8h):
#   Phase 1: PARALLEL data pull (3 branches concurrently)   ~1.5 h
#       1A: MBARI diverse Q3 days
#       1B: Orcasound point-robinson (new node)
#       1C: Orcasound andrews-bay re-pull, finer step=150
#   Phase 2: Hard-negative mining on v7.1                   ~15 min
#   Phase 3: Train v7.2 (40 epochs, hard negs ×5)           ~3-4 h
#   Phase 4: Calibrate conformal v7.2                       ~10 min
#   Phase 5: Per-site raw eval v7.2                         ~30 min
#   Phase 6: End-to-end eval v7.2 (oracle + none)           ~30 min
#   Phase 7: Manifest + comparison report                   ~5 min
#
# Failure policy: data pulls non-fatal (continue with whatever pulled).
# Hard-neg mining and training are fatal — no point training on bad data.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/venv/bin/python"
LOG_DIR="/tmp/morning"
mkdir -p "$LOG_DIR"

if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i -w $$ &
    echo "[$(date '+%H:%M:%S')] caffeinate active (PID $!)"
fi

step() {
    local name="$1"
    echo ""
    echo "===================================================="
    echo "  $(date '+%Y-%m-%d %H:%M:%S')  $name"
    echo "===================================================="
}

START_TS=$(date '+%s')

# ============================================================ PHASE 1
step "PHASE 1: PARALLEL data pull (6 branches — empty home, full throttle)"

# 1A: MBARI Jul-Aug (summer, fishery + cruise)
echo "[$(date '+%H:%M:%S')] launching 1A: MBARI Jul-Aug ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-07-08,2024-07-22,2024-08-05,2024-08-19 \
    --chunks-per-hour 15 --cap-per-day-per-label 400 --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_q3a.jsonl \
    > "$LOG_DIR/1A_mbari_q3a.log" 2>&1 &
PID_1A=$!

# 1B: MBARI Sep-Oct (autumn equinox migration)
echo "[$(date '+%H:%M:%S')] launching 1B: MBARI Sep-Oct ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-09-02,2024-09-16,2024-09-30,2024-10-14 \
    --chunks-per-hour 15 --cap-per-day-per-label 400 --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_q3b.jsonl \
    > "$LOG_DIR/1B_mbari_q3b.log" 2>&1 &
PID_1B=$!

# 1C: MBARI Nov-Dec (winter storms, dense traffic)
echo "[$(date '+%H:%M:%S')] launching 1C: MBARI Nov-Dec ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-11-04,2024-11-18,2024-12-02,2024-12-16 \
    --chunks-per-hour 15 --cap-per-day-per-label 400 --radius-km 10 \
    --out data/training/v7_bulk/mbari_diverse_q3c.jsonl \
    > "$LOG_DIR/1C_mbari_q3c.log" 2>&1 &
PID_1C=$!

# 1D: Orcasound point-robinson (new node)
echo "[$(date '+%H:%M:%S')] launching 1D: Orcasound point-robinson ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_orcasound_v2.py \
    --hydrophone point-robinson --step 150 --radius-km 10 \
    > "$LOG_DIR/1D_orca_point_robinson.log" 2>&1 &
PID_1D=$!

# 1E: Orcasound andrews-bay re-pull, finer step
echo "[$(date '+%H:%M:%S')] launching 1E: Orcasound andrews-bay re-pull ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_orcasound_v2.py \
    --hydrophone andrews-bay --step 120 --radius-km 10 \
    > "$LOG_DIR/1E_orca_andrews_bay.log" 2>&1 &
PID_1E=$!

# 1F: Orcasound mast-center re-pull, finer step
echo "[$(date '+%H:%M:%S')] launching 1F: Orcasound mast-center re-pull ..."
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_orcasound_v2.py \
    --hydrophone mast-center --step 150 --radius-km 10 \
    > "$LOG_DIR/1F_orca_mast_center.log" 2>&1 &
PID_1F=$!

echo "[$(date '+%H:%M:%S')] waiting on 6 parallel pulls..."
wait $PID_1A; RC_1A=$?
wait $PID_1B; RC_1B=$?
wait $PID_1C; RC_1C=$?
wait $PID_1D; RC_1D=$?
wait $PID_1E; RC_1E=$?
wait $PID_1F; RC_1F=$?
echo "[$(date '+%H:%M:%S')] phase 1 done — exit codes: 1A=$RC_1A 1B=$RC_1B 1C=$RC_1C 1D=$RC_1D 1E=$RC_1E 1F=$RC_1F"

# ============================================================ PHASE 2
step "PHASE 2: hard-negative mining on v7.1"
PYTHONPATH=src "$VENV_PY" scripts/extract_hard_negatives.py \
    --model data/models/cnn_v7_1.pt \
    --out data/training/v7_bulk/hard_negatives.jsonl \
    --max-per-site 200 \
    > "$LOG_DIR/2_hardneg.log" 2>&1
RC=$?
if [[ $RC -ne 0 ]]; then
    echo "[FATAL] hard-neg mining failed — see 2_hardneg.log"
    exit 1
fi
HARD_NEG_COUNT=$(wc -l < data/training/v7_bulk/hard_negatives.jsonl 2>/dev/null | tr -d ' ' || echo 0)
echo "[$(date '+%H:%M:%S')] hard negatives: $HARD_NEG_COUNT"

# ============================================================ PHASE 3
step "PHASE 3: train v7.2 (40 epochs, hard-neg oversample 5)"
PYTHONPATH=src "$VENV_PY" scripts/train_cnn_v7_2.py \
    --epochs 40 --batch-size 32 --lr 3e-4 \
    --hard-negatives data/training/v7_bulk/hard_negatives.jsonl \
    --hard-neg-oversample 5 \
    --out data/models/cnn_v7_2.pt \
    > "$LOG_DIR/3_train.log" 2>&1
RC=$?
if [[ $RC -ne 0 ]]; then
    echo "[FATAL] v7.2 training failed — see 3_train.log"
    exit 1
fi

# ============================================================ PHASE 4
step "PHASE 4: calibrate conformal v7.2"
PYTHONPATH=src "$VENV_PY" scripts/calibrate_conformal.py \
    --model data/models/cnn_v7_2.pt \
    --out data/calibration/conformal_v7_2.json \
    > "$LOG_DIR/4_conformal.log" 2>&1 \
    || echo "[WARN] conformal calibration failed; eval will use uncalibrated"

# ============================================================ PHASE 5
step "PHASE 5: per-site raw eval v7.2"
PYTHONPATH=src "$VENV_PY" scripts/eval_per_site.py \
    --model data/models/cnn_v7_2.pt \
    --out data/eval/per_site_v7_2.json \
    > "$LOG_DIR/5_persite.log" 2>&1 \
    || echo "[WARN] per-site eval failed"

# ============================================================ PHASE 6
step "PHASE 6: end-to-end eval v7.2 (oracle + none, parallel)"
CONFORMAL_PATH="data/calibration/conformal_v7_2.json"
[[ -f "$CONFORMAL_PATH" ]] || CONFORMAL_PATH="data/calibration/conformal.json"

PYTHONPATH=src "$VENV_PY" scripts/eval_end_to_end.py \
    --model data/models/cnn_v7_2.pt --conformal "$CONFORMAL_PATH" \
    --ais-mode oracle \
    --out data/eval/end_to_end_v7_2_oracle.json \
    > "$LOG_DIR/6a_e2e_oracle.log" 2>&1 &
PID_6A=$!

PYTHONPATH=src "$VENV_PY" scripts/eval_end_to_end.py \
    --model data/models/cnn_v7_2.pt --conformal "$CONFORMAL_PATH" \
    --ais-mode none \
    --out data/eval/end_to_end_v7_2_no_ais.json \
    > "$LOG_DIR/6b_e2e_no_ais.log" 2>&1 &
PID_6B=$!

wait $PID_6A; RC_6A=$?
wait $PID_6B; RC_6B=$?
[[ $RC_6A -ne 0 ]] && echo "[WARN] e2e oracle failed"
[[ $RC_6B -ne 0 ]] && echo "[WARN] e2e no_ais failed"

# ============================================================ PHASE 7
step "PHASE 7: manifest + comparison report"

COMPARISON_TXT="$LOG_DIR/comparison.txt"
{
    echo "=================================================="
    echo "Morning bootstrap finished $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================="
    "$VENV_PY" scripts/compare_models.py \
        --raw \
            data/eval/per_site.json:v7 \
            data/eval/per_site_v7_1.json:v7.1 \
            data/eval/per_site_v7_2.json:v7.2 \
        --e2e \
            data/eval/end_to_end_v7_1_oracle.json:v7.1_oracle \
            data/eval/end_to_end_v7_2_oracle.json:v7.2_oracle \
        2>/dev/null
} > "$COMPARISON_TXT"
cat "$COMPARISON_TXT"

END_TS=$(date '+%s')
ELAPSED_MIN=$(( (END_TS - START_TS) / 60 ))

cat > "$LOG_DIR/manifest.json" <<EOF
{
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "elapsed_minutes": $ELAPSED_MIN,
  "hard_negatives_count": $HARD_NEG_COUNT,
  "phase1_exit_codes": {
    "1A_mbari_q3a": $RC_1A, "1B_mbari_q3b": $RC_1B, "1C_mbari_q3c": $RC_1C,
    "1D_point_robinson": $RC_1D, "1E_andrews_bay": $RC_1E, "1F_mast_center": $RC_1F
  },
  "artifacts": {
    "model_v7_2":         "data/models/cnn_v7_2.pt",
    "conformal_v7_2":     "data/calibration/conformal_v7_2.json",
    "hard_negatives":     "data/training/v7_bulk/hard_negatives.jsonl",
    "per_site_v7_2":      "data/eval/per_site_v7_2.json",
    "e2e_v7_2_oracle":    "data/eval/end_to_end_v7_2_oracle.json",
    "e2e_v7_2_no_ais":    "data/eval/end_to_end_v7_2_no_ais.json",
    "comparison_report":  "$COMPARISON_TXT"
  },
  "logs": "$LOG_DIR"
}
EOF
cat "$LOG_DIR/manifest.json"
echo ""
echo "[$(date '+%H:%M:%S')] Morning bootstrap DONE in ${ELAPSED_MIN} min."
echo "Read: $COMPARISON_TXT"
