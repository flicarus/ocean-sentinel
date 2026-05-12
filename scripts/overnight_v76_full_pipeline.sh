#!/usr/bin/env bash
# LAST-CHANCE OVERNIGHT + WORKDAY PIPELINE.
#
# Goal: fill every passive hour user has (23:00 today → 14:00 tomorrow).
# Train 2 candidate models (v7.6, v7.7) so user can pick the better one
# evening of 2026-05-12. No idle time during their work hours.
#
# Phases:
#   1. Non-MBARI pull       (5-6h)
#   2. Train v7.6           (4h)   — balanced sampler + keep cross_site_mix
#   3. Eval v7.6            (30 min)
#   4. Train v7.7           (4h)   — balanced sampler + NO cross_site_mix
#   5. Eval v7.7            (30 min)
#   6. Final comparison     (15 min)
#
# Usage:
#   nohup bash scripts/overnight_v76_full_pipeline.sh > /tmp/v76_data/main.log 2>&1 &
#   disown $!

set -uo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/venv/bin/python"
LOG_DIR="/tmp/v76_data"
mkdir -p "$LOG_DIR" "$LOG_DIR/ais" "$LOG_DIR/sanct"
> "$LOG_DIR/pipeline.log"

# Renew caffeinate — 16h covers full pipeline + buffer
nohup caffeinate -i -t 57600 > "$LOG_DIR/caffeinate.log" 2>&1 &
disown $! 2>/dev/null

phase() {
    echo "" | tee -a "$LOG_DIR/pipeline.log"
    echo "====================================================" | tee -a "$LOG_DIR/pipeline.log"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a "$LOG_DIR/pipeline.log"
    echo "====================================================" | tee -a "$LOG_DIR/pipeline.log"
}

START_TS=$(date +%s)

# ============================================================ PHASE 1
# 8 AIS-correlated chains with offset start dates + SanctSound expanded.
# Adaptive controller scales workers ±2 every 15 min based on 429 rate.
# Hard limit: 6h elapsed regardless of completion status.
# ============================================================
phase "PHASE 1: non-MBARI pull (adaptive 12 workers, 6h max)"

HYDROS=(bush-point sunset-bay orcasound-lab north-sjc point-robinson port-townsend mast-center andrews-bay)
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

pull_ais_chain() {
    local node="$1"; local offset="$2"
    local n=${#DATES[@]}
    for ((i=0; i<n; i++)); do
        local idx=$(( (i + offset) % n ))
        local d="${DATES[$idx]}"
        PYTHONPATH=src "$VENV_PY" scripts/bootstrap_ais_correlated.py \
            --hydrophone "$node" --date "$d" --step 120 --radius-km 10 \
            > "$LOG_DIR/ais/${node}_${d}.log" 2>&1
        sleep 3
    done
}

# Patch SanctSound to expand deployment list (14 sites × 4 deps = 56)
cp scripts/bootstrap_sanctsound_more_60s.py "$LOG_DIR/bootstrap_sanctsound_more_60s.py.bak"
"$VENV_PY" - <<'PYTHON_PATCH'
from pathlib import Path
import re
p = Path("scripts/bootstrap_sanctsound_more_60s.py")
src = p.read_text()
m = re.search(r"SITES_TO_PULL: list\[tuple\[str, str\]\] = \[(.+?)\n\]", src, re.S)
if m:
    expanded = []
    for site in ["hi03", "hi04", "hi06", "hi07", "oc02", "oc03", "oc04",
                 "ci02", "ci03", "ci04", "ci05", "fk02", "fk03", "sb03"]:
        for dep in ["01", "02", "03", "04"]:
            expanded.append(f'    ("{site}", "{dep}"),')
    new_block = "SITES_TO_PULL: list[tuple[str, str]] = [\n" + "\n".join(expanded) + "\n]"
    src = src.replace(m.group(0), new_block)
    p.write_text(src)
    print(f"Patched: {len(expanded)} deployments")
PYTHON_PATCH

# Launch sanctsound in background
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_sanctsound_more_60s.py \
    > "$LOG_DIR/sanctsound_expanded.log" 2>&1 &
disown $! 2>/dev/null

# Launch 8 AIS chains with staggered offset dates
for i in 0 1 2 3 4 5 6 7; do
    node="${HYDROS[$i]}"
    offset=$(( i * 5 ))
    pull_ais_chain "$node" "$offset" &
    disown $! 2>/dev/null
    sleep 2
done

echo "[$(date '+%H:%M:%S')] Phase 1 workers launched (8 AIS + 1 SanctSound)" \
    | tee -a "$LOG_DIR/pipeline.log"

# Adaptive controller — runs for 6h max
PHASE1_DEADLINE=$(( $(date +%s) + 21600 ))
EXTRA_DATES=(
  2024-02-20 2024-03-15 2024-05-18 2024-07-09 2024-09-13
  2024-11-06 2023-02-11 2023-06-21 2023-09-19 2023-11-22
)
EXTRA_IDX=0
EXTRA_NODE_IDX=0

while [[ $(date +%s) -lt $PHASE1_DEADLINE ]]; do
    sleep 900
    # 429 count in logs touched in last 5 min
    N_429=$(find "$LOG_DIR/ais" -name "*.log" -mmin -5 -exec grep -l "429" {} + 2>/dev/null | wc -l | tr -d ' ')
    N_PROCS=$(pgrep -f "bootstrap_(ais|sanctsound)" 2>/dev/null | wc -l | tr -d ' ')
    ROWS=$(find data/training/v7_bulk -name "*.jsonl" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    REMAINING_MIN=$(( (PHASE1_DEADLINE - $(date +%s)) / 60 ))

    DECISION=""
    if [[ $N_429 -gt 30 && $N_PROCS -gt 6 ]]; then
        VICTIMS=$(pgrep -f "bootstrap_ais_correlated" | tail -2)
        if [[ -n "$VICTIMS" ]]; then
            echo "$VICTIMS" | xargs kill 2>/dev/null
            DECISION="SCALE_DOWN -2 (429=$N_429)"
        fi
    elif [[ $N_429 -lt 5 && $N_PROCS -lt 14 ]]; then
        for _ in 1 2; do
            node="${HYDROS[$((EXTRA_NODE_IDX % 8))]}"
            d="${EXTRA_DATES[$((EXTRA_IDX % 10))]}"
            PYTHONPATH=src "$VENV_PY" scripts/bootstrap_ais_correlated.py \
                --hydrophone "$node" --date "$d" --step 120 --radius-km 10 \
                > "$LOG_DIR/ais/${node}_${d}_extra${EXTRA_IDX}.log" 2>&1 &
            disown $! 2>/dev/null
            EXTRA_NODE_IDX=$(( EXTRA_NODE_IDX + 1 ))
            EXTRA_IDX=$(( EXTRA_IDX + 1 ))
        done
        DECISION="SCALE_UP +2 (429=$N_429)"
    else
        DECISION="STEADY (429=$N_429, procs=$N_PROCS)"
    fi
    echo "$(date '+%H:%M:%S')  procs=$N_PROCS  429=$N_429  rows=$ROWS  remaining=${REMAINING_MIN}m  $DECISION" \
        >> "$LOG_DIR/adaptive.log"
done

# Phase 1 done — kill remaining AIS chains, keep sanctsound finishing
phase "PHASE 1 deadline reached — killing remaining pull workers"
pkill -f bootstrap_ais_correlated 2>/dev/null || true
# Let sanctsound finish naturally if still running

# Restore sanctsound script
cp "$LOG_DIR/bootstrap_sanctsound_more_60s.py.bak" scripts/bootstrap_sanctsound_more_60s.py

# Snapshot source distribution post-pull
PYTHONPATH=src "$VENV_PY" -c "
import json, glob
from collections import Counter
files = glob.glob('data/training/v7_bulk/*.jsonl') + ['data/training/gemma_labels.v7.jsonl', 'data/training/sanctsound_corrected.jsonl', 'data/training/sanctsound_diverse.jsonl', 'data/training/sanctsound_more_60s.jsonl']
c = Counter()
for f in files:
    try:
        for line in open(f):
            r=json.loads(line); src=(r.get('provenance') or {}).get('source_id','?')
            if 'mbari' in src: cat='mbari'
            elif 'ais-correlated' in src: cat='ais/orca'
            elif 'sanctsound-more' in src: cat='sanct_more'
            elif src.startswith('sanctsound'): cat='sanct_other'
            else: cat=src
            c[cat]+=1
    except: pass
total=sum(c.values())
for k,v in c.most_common():
    print(f'  {k:<20s} {v:>8d}  {100*v/total:>5.1f}%')
print(f'  TOTAL: {total}')
" | tee -a "$LOG_DIR/pipeline.log"

# ============================================================ PHASE 2
# Train v7.6 — balanced sampler ON, cross_site_mix ON (as v7.5)
# ============================================================
phase "PHASE 2: train v7.6 (balanced + gentle mix 0.25 + hardneg x2)"

# Codex review: hard-neg x5 + balanced sampler = point-robinson gets 1000 false
# 'not_ship' replays per epoch which would crush its already-collapsed recall.
# Cross-site mix 0.5 too hot for rare sites — use 0.25 dose.
PYTHONPATH=src "$VENV_PY" scripts/train_cnn_v7_6.py \
    --epochs 30 --batch-size 32 --lr 3e-4 \
    --hard-negatives data/training/v7_bulk/hard_negatives_v4.jsonl \
    --hard-neg-oversample 2 \
    --balanced-sampler \
    --cross-site-mix-prob 0.25 \
    --out data/models/cnn_v7_6.pt \
    > "$LOG_DIR/v76_train.log" 2>&1
RC=$?
[[ $RC -ne 0 ]] && echo "[FATAL] v7.6 training failed exit=$RC" | tee -a "$LOG_DIR/pipeline.log"

# ============================================================ PHASE 3
# Eval v7.6
# ============================================================
phase "PHASE 3: eval v7.6 + conformal calibrate"
PYTHONPATH=src "$VENV_PY" scripts/eval_per_site.py \
    --model data/models/cnn_v7_6.pt --out data/eval/per_site_v7_6.json \
    --limit-per-site 300 > "$LOG_DIR/v76_eval.log" 2>&1
PYTHONPATH=src "$VENV_PY" scripts/calibrate_conformal.py \
    --model data/models/cnn_v7_6.pt --out data/calibration/conformal_v7_6.json \
    > "$LOG_DIR/v76_calibrate.log" 2>&1

# ============================================================ PHASE 4
# Train v7.7 — balanced sampler ON + cross_site_mix OFF
# Hypothesis: cross_site_mix=0.5 was poisoning under-represented sites.
# ============================================================
phase "PHASE 4: train v7.7 (balanced + NO mix + hardneg x2 — clean ablation)"

# Same hardneg reduction as v7.6 — keeps the bracket scientifically valid
# (only cross_site_mix differs between v7.6 and v7.7).
PYTHONPATH=src "$VENV_PY" scripts/train_cnn_v7_6.py \
    --epochs 30 --batch-size 32 --lr 3e-4 \
    --hard-negatives data/training/v7_bulk/hard_negatives_v4.jsonl \
    --hard-neg-oversample 2 \
    --balanced-sampler --no-cross-site-mix \
    --out data/models/cnn_v7_7.pt \
    > "$LOG_DIR/v77_train.log" 2>&1
RC=$?
[[ $RC -ne 0 ]] && echo "[FATAL] v7.7 training failed exit=$RC" | tee -a "$LOG_DIR/pipeline.log"

# ============================================================ PHASE 5
# Eval v7.7
# ============================================================
phase "PHASE 5: eval v7.7 + conformal calibrate"
PYTHONPATH=src "$VENV_PY" scripts/eval_per_site.py \
    --model data/models/cnn_v7_7.pt --out data/eval/per_site_v7_7.json \
    --limit-per-site 300 > "$LOG_DIR/v77_eval.log" 2>&1
PYTHONPATH=src "$VENV_PY" scripts/calibrate_conformal.py \
    --model data/models/cnn_v7_7.pt --out data/calibration/conformal_v7_7.json \
    > "$LOG_DIR/v77_calibrate.log" 2>&1

# ============================================================ PHASE 6
# Comparison report — v7.5 vs v7.6 vs v7.7 per site
# ============================================================
phase "PHASE 6: comparison report"

"$VENV_PY" - <<'PYTHON_REPORT' | tee "$LOG_DIR/comparison.txt" | tee -a "$LOG_DIR/pipeline.log"
import json
from pathlib import Path

evals = {}
for name in ("v7_5", "v7_6", "v7_7"):
    p = Path(f"data/eval/per_site_{name}.json")
    if p.exists():
        evals[name] = json.load(open(p))

if "v7_5" in evals:
    print(f"{'overall':<28s}  v7.5={evals['v7_5']['overall']['accuracy']:.4f}", end="")
    if "v7_6" in evals: print(f"  v7.6={evals['v7_6']['overall']['accuracy']:.4f}", end="")
    if "v7_7" in evals: print(f"  v7.7={evals['v7_7']['overall']['accuracy']:.4f}", end="")
    print()
    print()

# Per-site comparison
sites = set()
for e in evals.values():
    sites.update(e['per_site'])

header = f"{'site':<28s}"
for name in ("v7_5", "v7_6", "v7_7"):
    if name in evals: header += f"  {name}"
print(header)

for s in sorted(sites):
    line = f"{s:<28s}"
    for name in ("v7_5", "v7_6", "v7_7"):
        if name not in evals:
            continue
        a = evals[name]['per_site'].get(s, {}).get('accuracy')
        line += f"  {a:.3f}" if a is not None else "    -  "
    print(line)

# Codex: overall acc can lie under site imbalance — surface recall/precision
# for the sites that regressed in v7.5 so we pick winner on actual fix quality.
print()
print("=== REGRESSION SITES — recall / precision ===")
WATCH = ["point-robinson", "ais-correlated-point-robinson", "sanctsound",
         "sb02", "sb03", "mb02"]
for s in WATCH:
    parts = [f"{s:<32s}"]
    for name in ("v7_5", "v7_6", "v7_7"):
        if name not in evals: continue
        site = evals[name]['per_site'].get(s)
        if site:
            parts.append(f"{name}: rec={site.get('recall',-1):.2f} prec={site.get('precision',-1):.2f}")
        else:
            parts.append(f"{name}: -")
    print("  ".join(parts))
PYTHON_REPORT

END_TS=$(date +%s)
TOTAL_MIN=$(( (END_TS - START_TS) / 60 ))
phase "PIPELINE DONE in ${TOTAL_MIN} min"
echo "Best of v7.5/v7.6/v7.7 is your ship candidate. Check $LOG_DIR/comparison.txt"
