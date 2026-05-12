#!/usr/bin/env bash
# OVERNIGHT v7.6 — aggressive non-MBARI rebalance + balanced training.
#
# Phase 1 (~5h): pull many AIS-correlated dates per node (8 parallel,
#                serialized through GFW 1-concurrent on AIS calls only)
#                + expand SanctSound deployments
# Phase 2 (~3-4h): train v7.6 with WeightedRandomSampler on full pool
# Phase 3 (~30min): per_site eval + comparison report
#
# Total budget: ~8-9h. Launch before sleeping, results by morning.
#
# Usage:
#   nohup bash scripts/overnight_v76_balanced.sh > /tmp/v76/main.log 2>&1 &
#   disown $!

set -uo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/venv/bin/python"
LOG_DIR="/tmp/v76"
mkdir -p "$LOG_DIR" "$LOG_DIR/ais" "$LOG_DIR/sanct"

# Renew caffeinate for the whole night — die if parent dies
if command -v caffeinate >/dev/null 2>&1; then
    nohup caffeinate -i -t 36000 > "$LOG_DIR/caffeinate.log" 2>&1 &
    CAFF_PID=$!
    disown $CAFF_PID 2>/dev/null
    echo "[$(date '+%H:%M:%S')] caffeinate PID=$CAFF_PID (10h)"
fi

# Heartbeat: append progress every 5 min
(
  while true; do
    sleep 300
    n_specs=$(ls data/spectrograms 2>/dev/null | wc -l | tr -d ' ')
    n_specs60=$(ls data/spectrograms_60s 2>/dev/null | wc -l | tr -d ' ')
    n_procs=$(pgrep -f "bootstrap_(ais|orcasound|sanctsound)" 2>/dev/null | wc -l | tr -d ' ')
    rows=$(find data/training/v7_bulk -name "*.jsonl" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    echo "$(date '+%H:%M:%S') spec5=$n_specs spec60=$n_specs60 bulk_rows=$rows procs=$n_procs" \
        >> "$LOG_DIR/heartbeat.log"
  done
) &
HB_PID=$!
disown $HB_PID 2>/dev/null

step() {
    echo ""
    echo "===================================================="
    echo "  $(date '+%Y-%m-%d %H:%M:%S')  $1"
    echo "===================================================="
}

START_TS=$(date +%s)

# ============================================================ PHASE 1A
# AIS-correlated across many dates per node.
# 8 parallel batches (one per node), each iterates through 40 dates serially.
# Each (node, date) yields ~100-300 chunks. Total target: ~50-80k.
# ============================================================
step "PHASE 1A: AIS-correlated × 40 dates × 8 nodes (~50-80k chunks)"

HYDROS=(bush-point sunset-bay orcasound-lab north-sjc point-robinson port-townsend mast-center andrews-bay)

# 40 spread dates across 2024 (different from already-pulled today)
DATES=(
  2024-01-05 2024-01-19 2024-02-02 2024-02-16 2024-03-01
  2024-03-22 2024-04-05 2024-04-26 2024-05-10 2024-05-31
  2024-06-07 2024-06-28 2024-07-12 2024-07-26 2024-08-09
  2024-08-30 2024-09-06 2024-09-20 2024-10-04 2024-10-25
  2024-11-08 2024-11-29 2024-12-06 2024-12-27 2023-04-18
  2023-05-09 2023-06-13 2023-07-04 2023-08-15 2023-09-12
  2023-10-10 2023-11-07 2023-12-12 2024-04-19 2024-07-19
  2024-10-18 2024-11-15 2024-12-13 2023-03-21 2023-04-25
)

pull_node_serial() {
    local node="$1"
    for d in "${DATES[@]}"; do
        PYTHONPATH=src "$VENV_PY" scripts/bootstrap_ais_correlated.py \
            --hydrophone "$node" --date "$d" --step 120 --radius-km 10 \
            > "$LOG_DIR/ais/${node}_${d}.log" 2>&1
    done
}

for node in "${HYDROS[@]}"; do
    pull_node_serial "$node" &
done

# ============================================================ PHASE 1B
# SanctSound — re-run with expanded deployment list (background, low priority)
# ============================================================
step "PHASE 1B: SanctSound expanded (background)"

# Patch SITES_TO_PULL in place — backup first
cp scripts/bootstrap_sanctsound_more_60s.py "$LOG_DIR/bootstrap_sanctsound_more_60s.py.bak"

# Add more deployments to the script's hardcoded list using a simple Python patch.
# Each SITE has multiple deployment numbers — try 01-04 for each.
"$VENV_PY" - <<'PYTHON_PATCH'
from pathlib import Path
p = Path("scripts/bootstrap_sanctsound_more_60s.py")
src = p.read_text()
old = '''SITES_TO_PULL: list[tuple[str, str]] = [
    ("hi03", "02"),
    ("hi03", "03"),
    ("hi04", "01"),
    ("hi04", "02"),
    ("hi06", "01"),
    ("hi07", "01"),
    ("oc02", "01"),
    ("oc02", "02"),
    ("oc03", "01"),
    ("oc04", "01"),
    ("ci02", "01"),
    ("ci03", "01"),
    ("ci04", "01"),
    ("ci05", "01"),
    ("fk02", "01"),
    ("fk03", "01"),
    ("sb03", "01"),
]'''
# Expanded: try multiple deployments per site
expanded_entries = []
for site in ["hi03", "hi04", "hi06", "hi07", "oc02", "oc03", "oc04",
             "ci02", "ci03", "ci04", "ci05", "fk02", "fk03", "sb03"]:
    for dep in ["01", "02", "03", "04"]:
        expanded_entries.append(f'    ("{site}", "{dep}"),')
new = "SITES_TO_PULL: list[tuple[str, str]] = [\n" + "\n".join(expanded_entries) + "\n]"
if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print(f"Patched: {len(expanded_entries)} deployments")
else:
    print("WARN: old block not found, skipping patch")
PYTHON_PATCH

PYTHONPATH=src "$VENV_PY" scripts/bootstrap_sanctsound_more_60s.py \
    > "$LOG_DIR/sanctsound_expanded.log" 2>&1 &
SANCT_PID=$!

# ============================================================ WAIT FOR PHASE 1
# Block until all 8 AIS pulls finish AND sanctsound finishes
# ============================================================
step "WAITING for Phase 1 to complete (target ~5h)"

wait
P1_END=$(date +%s)
P1_MIN=$(( (P1_END - START_TS) / 60 ))
echo "[$(date '+%H:%M:%S')] Phase 1 done in ${P1_MIN} min"

# Restore SanctSound script
cp "$LOG_DIR/bootstrap_sanctsound_more_60s.py.bak" scripts/bootstrap_sanctsound_more_60s.py

# ============================================================ PHASE 2
# Train v7.6 with WeightedRandomSampler — balanced site sampling.
# ============================================================
step "PHASE 2: train v7.6 with balanced sampler"

PYTHONPATH=src "$VENV_PY" scripts/train_cnn_v7_6.py \
    --epochs 30 --batch-size 32 --lr 3e-4 \
    --hard-negatives data/training/v7_bulk/hard_negatives_v4.jsonl \
    --hard-neg-oversample 5 \
    --balanced-sampler \
    --out data/models/cnn_v7_6.pt \
    > "$LOG_DIR/train.log" 2>&1

RC=$?
if [[ $RC -ne 0 ]]; then
    echo "[FATAL] v7.6 training failed (exit $RC) — see $LOG_DIR/train.log"
    kill $HB_PID 2>/dev/null
    exit 1
fi

# ============================================================ PHASE 3
# Conformal calibrate + per-site eval
# ============================================================
step "PHASE 3: calibrate + eval"

PYTHONPATH=src "$VENV_PY" scripts/calibrate_conformal.py \
    --model data/models/cnn_v7_6.pt \
    --out data/calibration/conformal_v7_6.json \
    > "$LOG_DIR/calibrate.log" 2>&1

PYTHONPATH=src "$VENV_PY" scripts/eval_per_site.py \
    --model data/models/cnn_v7_6.pt \
    --out data/eval/per_site_v7_6.json \
    --limit-per-site 300 \
    > "$LOG_DIR/eval.log" 2>&1

# Stop heartbeat
kill $HB_PID 2>/dev/null || true

END_TS=$(date +%s)
TOTAL_MIN=$(( (END_TS - START_TS) / 60 ))

step "DONE in ${TOTAL_MIN} min"

# Comparison report
"$VENV_PY" - <<'PYTHON_REPORT'
import json
v5 = json.load(open("data/eval/per_site_v7_5.json"))
v6 = json.load(open("data/eval/per_site_v7_6.json"))
print(f"v7.5 overall: {v5['overall']['accuracy']:.4f}")
print(f"v7.6 overall: {v6['overall']['accuracy']:.4f}   Δ {(v6['overall']['accuracy']-v5['overall']['accuracy'])*100:+.1f}pp")
print()
print(f"{'site':<32s} {'v7.5':>6} {'v7.6':>6} {'Δpp':>7}")
sites = sorted(set(v5['per_site']) | set(v6['per_site']))
for s in sites:
    a5 = v5['per_site'].get(s,{}).get('accuracy', 0)
    a6 = v6['per_site'].get(s,{}).get('accuracy', 0)
    delta = (a6-a5)*100
    print(f"{s:<32s} {a5:.3f}  {a6:.3f}  {delta:+6.1f}")
PYTHON_REPORT
