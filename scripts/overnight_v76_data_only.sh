#!/usr/bin/env bash
# OVERNIGHT — DATA ONLY (no training).
#
# Day 14 lesson: 83% of pool was MBARI. Day 15 fix: rebalance pool with
# many more non-MBARI samples (AIS-correlated × 40 dates × 8 nodes +
# expanded SanctSound deployments). Training v7.6 deferred until pool
# is materially balanced.
#
# Usage:
#   nohup bash scripts/overnight_v76_data_only.sh > /tmp/v76_data/main.log 2>&1 &
#   disown $!

set -uo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/venv/bin/python"
LOG_DIR="/tmp/v76_data"
mkdir -p "$LOG_DIR" "$LOG_DIR/ais" "$LOG_DIR/sanct"

# Renew caffeinate — 10h, detached so it survives parent shell
if command -v caffeinate >/dev/null 2>&1; then
    nohup caffeinate -i -t 36000 > "$LOG_DIR/caffeinate.log" 2>&1 &
    CAFF_PID=$!
    disown $CAFF_PID 2>/dev/null
    echo "[$(date '+%H:%M:%S')] caffeinate PID=$CAFF_PID (10h)"
fi

# Heartbeat: source-distribution tracker every 5 min
(
  while true; do
    sleep 300
    rows=$(find data/training/v7_bulk -name "*.jsonl" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    n_procs=$(pgrep -f "bootstrap_(ais|orcasound|sanctsound)" 2>/dev/null | wc -l | tr -d ' ')
    # also count per-source rows so we can see MBARI share drop in real time
    mbari_share=$(PYTHONPATH=src "$VENV_PY" -c "
import json, glob
files = glob.glob('data/training/v7_bulk/*.jsonl') + ['data/training/gemma_labels.v7.jsonl', 'data/training/sanctsound_corrected.jsonl', 'data/training/sanctsound_diverse.jsonl', 'data/training/sanctsound_more_60s.jsonl']
mbari=0; total=0
for f in files:
    try:
        for line in open(f):
            r=json.loads(line); src=(r.get('provenance') or {}).get('source_id','')
            if 'mbari' in src: mbari+=1
            total+=1
    except: pass
print(f'{100*mbari/max(total,1):.1f}')
" 2>/dev/null)
    echo "$(date '+%H:%M:%S')  bulk_rows=$rows  procs=$n_procs  mbari_share=${mbari_share}%" \
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
# AIS-correlated × 40 dates × 8 nodes, serialized PER NODE (parallel
# across nodes). Each (node,date) yields ~100-300 chunks.
# GFW limit is per-token, but burst-then-cache pattern means parallel
# across nodes is fine after initial AIS hits.
# ============================================================
step "PHASE 1A: AIS-correlated × 40 dates × 8 nodes"

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
# SanctSound — expand SITES_TO_PULL to all 14 sites × 4 deployments = 56
# entries. Patch in place, restore after.
# ============================================================
step "PHASE 1B: SanctSound expanded (background)"

cp scripts/bootstrap_sanctsound_more_60s.py "$LOG_DIR/bootstrap_sanctsound_more_60s.py.bak"

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
expanded_entries = []
for site in ["hi03", "hi04", "hi06", "hi07", "oc02", "oc03", "oc04",
             "ci02", "ci03", "ci04", "ci05", "fk02", "fk03", "sb03"]:
    for dep in ["01", "02", "03", "04"]:
        expanded_entries.append(f'    ("{site}", "{dep}"),')
new = "SITES_TO_PULL: list[tuple[str, str]] = [\n" + "\n".join(expanded_entries) + "\n]"
if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print(f"Patched: {len(expanded_entries)} deployments (was 17)")
else:
    print("WARN: SITES_TO_PULL block not found — skipping patch")
PYTHON_PATCH

PYTHONPATH=src "$VENV_PY" scripts/bootstrap_sanctsound_more_60s.py \
    > "$LOG_DIR/sanctsound_expanded.log" 2>&1 &

# ============================================================ WAIT
# Block until all 8 AIS-per-node chains + sanctsound expansion finish
# ============================================================
step "WAITING for Phase 1 to complete (target ~5-6h)"

wait
END_TS=$(date +%s)
ELAPSED_MIN=$(( (END_TS - START_TS) / 60 ))
echo "[$(date '+%H:%M:%S')] Phase 1 done in ${ELAPSED_MIN} min"

# Restore SanctSound script to original
cp "$LOG_DIR/bootstrap_sanctsound_more_60s.py.bak" scripts/bootstrap_sanctsound_more_60s.py

# Stop heartbeat
kill $HB_PID 2>/dev/null || true

# ============================================================ FINAL REPORT
step "DONE — source distribution after rebalance"

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
            elif 'ais-correlated' in src: cat='orcasound/ais-correlated'
            elif 'sanctsound-more' in src: cat='sanctsound_more_60s'
            elif src.startswith('sanctsound-diverse'): cat='sanctsound_diverse'
            elif src.startswith('sanctsound-corrected'): cat='sanctsound_corrected'
            elif src=='sanctsound': cat='sanctsound (raw)'
            elif 'orcasound' in src: cat='orcasound (raw)'
            else: cat=src
            c[cat]+=1
    except: pass
total=sum(c.values())
print(f'{\"source\":<30s} {\"rows\":>8s} {\"share\":>8s}')
print('-'*52)
for k,v in c.most_common():
    print(f'{k:<30s} {v:>8d} {100*v/total:>7.1f}%')
print('-'*52)
print(f'{\"TOTAL\":<30s} {total:>8d}')
"

echo ""
echo "[$(date '+%H:%M:%S')] overnight_v76_data_only DONE in ${ELAPSED_MIN} min"
echo "Next step: review distribution, then manually train v7.6 with --balanced-sampler"
