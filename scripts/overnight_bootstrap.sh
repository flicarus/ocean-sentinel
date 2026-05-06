#!/usr/bin/env bash
# Overnight: pull diverse data, retrain v7.1, calibrate conformal, eval.
#
# Usage (run before sleep, then check /tmp/overnight/manifest.json in the morning):
#   nohup bash scripts/overnight_bootstrap.sh > /tmp/overnight/main.log 2>&1 &
#
# What it does, in order:
#   1. Pull 5 missing Orcasound nodes (port_townsend, sunset_bay,
#      mast_center, andrews_bay, north_sjc) — adds OOD coverage.
#   2. Pull diverse MBARI hours across 6 days of 2024 — broadens the
#      MBARI distribution from "quiet canyon" to "what live MBARI
#      actually sounds like".
#   3. Smoke-test the v7.1 trainer on 1 epoch — fast sanity check
#      that loss isn't NaN and the new recipe runs end-to-end.
#   4. Full retrain v7.1 — 30 epochs, focal loss + low KL + cross-site
#      mix + session split.
#   5. Calibrate conformal on the freshly trained v7.1.
#   6. Per-site eval to compare v7 baseline vs v7.1.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/venv/bin/python"
LOG_DIR="/tmp/overnight"
mkdir -p "$LOG_DIR"

# Keep macOS awake while this script runs (idle sleep would freeze the
# retraining mid-epoch). caffeinate -w $$ exits when our PID dies, so we
# never leave a stray no-sleep daemon behind. Display CAN still sleep.
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i -w $$ &
    echo "[$(date '+%H:%M:%S')] caffeinate active (PID $!) — Mac will not idle-sleep"
fi

step() {
    local name="$1"
    echo ""
    echo "===================================================="
    echo "  $(date '+%Y-%m-%d %H:%M:%S')  $name"
    echo "===================================================="
}

# Phase 1: pull 5 missing Orcasound nodes ----------------------------
step "PHASE 1: pull 5 unused Orcasound nodes"
NODES=(port-townsend sunset-bay mast-center andrews-bay north-sjc)
for node in "${NODES[@]}"; do
    echo "[$(date '+%H:%M:%S')] node=$node ..."
    PYTHONPATH=src "$VENV_PY" scripts/bootstrap_orcasound_v2.py \
        --hydrophone "$node" --step 300 --radius-km 10 \
        > "$LOG_DIR/orca_${node}.log" 2>&1 \
        || { echo "node $node failed — check $LOG_DIR/orca_${node}.log"; }
done

# Phase 2: diverse MBARI ---------------------------------------------
step "PHASE 2: diverse MBARI hours across 2024"
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_mbari_diverse.py \
    --days 2024-02-01,2024-04-01,2024-06-01,2024-08-01,2024-10-01,2024-12-01 \
    --chunks-per-hour 2 --radius-km 10 \
    > "$LOG_DIR/mbari_diverse.log" 2>&1 \
    || echo "mbari_diverse failed — check log"

# Phase 3: smoke-test v7.1 trainer (1 epoch) -------------------------
step "PHASE 3: smoke-test v7.1 trainer (1 epoch)"
PYTHONPATH=src "$VENV_PY" scripts/train_cnn_v7_1.py \
    --epochs 1 --batch-size 32 \
    --out "/tmp/cnn_v7_1_smoke.pt" \
    > "$LOG_DIR/v71_smoke.log" 2>&1 \
    || { echo "smoke test failed — STOPPING"; exit 1; }

# Phase 4: full v7.1 retrain -----------------------------------------
step "PHASE 4: full v7.1 retrain (30 epochs)"
PYTHONPATH=src "$VENV_PY" scripts/train_cnn_v7_1.py \
    --epochs 30 --batch-size 32 \
    --out data/models/cnn_v7_1.pt \
    > "$LOG_DIR/v71_train.log" 2>&1

# Phase 5: calibrate conformal on v7.1 -------------------------------
step "PHASE 5: calibrate conformal on v7.1"
PYTHONPATH=src "$VENV_PY" scripts/calibrate_conformal.py \
    --model data/models/cnn_v7_1.pt \
    --out data/calibration/conformal.json \
    > "$LOG_DIR/conformal.log" 2>&1

# Phase 6: per-site eval ---------------------------------------------
step "PHASE 6: per-site eval v7.1"
PYTHONPATH=src "$VENV_PY" scripts/eval_per_site.py \
    --model data/models/cnn_v7_1.pt \
    --out data/eval/per_site_v7_1.json \
    > "$LOG_DIR/eval.log" 2>&1

# Manifest -----------------------------------------------------------
step "DONE — writing manifest"
cat > "$LOG_DIR/manifest.json" <<EOF
{
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "artifacts": {
    "model_v7_1":         "data/models/cnn_v7_1.pt",
    "conformal":          "data/calibration/conformal.json",
    "eval_v7_baseline":   "data/eval/per_site.json",
    "eval_v7_1":          "data/eval/per_site_v7_1.json",
    "diverse_mbari":      "data/training/v7_bulk/mbari_diverse.jsonl",
    "orcasound_pulls": [
$(for node in "${NODES[@]}"; do
    echo "      \"data/training/v7_bulk/orcasound_${node}.jsonl\","
done | sed '$ s/,$//')
    ]
  },
  "logs": "$LOG_DIR/"
}
EOF
cat "$LOG_DIR/manifest.json"
echo ""
echo "All phases finished. Compare v7 vs v7.1:"
echo "  diff <(jq -S .per_site data/eval/per_site.json) \\"
echo "       <(jq -S .per_site data/eval/per_site_v7_1.json)"
