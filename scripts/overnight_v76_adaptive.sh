#!/usr/bin/env bash
# OVERNIGHT — DATA ONLY, ADAPTIVE SCALING.
#
# Starts with 12 workers (8 AIS-correlated, offset start dates + 4 SanctSound chains).
# Adaptive controller checks 429 rate every 15 min:
#   - 429-rate > 30/5min  → scale down by 2 workers (target 6 min)
#   - 429-rate < 5/5min   → scale up by 2 workers (target 16 max)
#
# Usage:
#   nohup bash scripts/overnight_v76_adaptive.sh > /tmp/v76_data/main.log 2>&1 &
#   disown $!

set -uo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/venv/bin/python"
LOG_DIR="/tmp/v76_data"
mkdir -p "$LOG_DIR" "$LOG_DIR/ais" "$LOG_DIR/sanct"
> "$LOG_DIR/adaptive.log"   # truncate previous adaptive log

# Renew caffeinate (10h)
nohup caffeinate -i -t 36000 > "$LOG_DIR/caffeinate.log" 2>&1 &
disown $! 2>/dev/null
echo "[$(date '+%H:%M:%S')] adaptive overnight started"

# ============================================================ INITIAL WORKERS
# 8 AIS-correlated chains, one per node, OFFSET START DATES so we don't
# hammer GFW with 8 simultaneous AIS calls on day=2024-01-05.
# ============================================================

HYDROS=(bush-point sunset-bay orcasound-lab north-sjc point-robinson port-townsend mast-center andrews-bay)

# 40 dates spread across 2023-2024
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

# Each chain rotates dates with offset = (node_idx * 5) to stagger AIS calls
pull_ais_chain() {
    local node="$1"; local offset="$2"
    local n_dates=${#DATES[@]}
    for ((i=0; i<n_dates; i++)); do
        local idx=$(( (i + offset) % n_dates ))
        local d="${DATES[$idx]}"
        PYTHONPATH=src "$VENV_PY" scripts/bootstrap_ais_correlated.py \
            --hydrophone "$node" --date "$d" --step 120 --radius-km 10 \
            > "$LOG_DIR/ais/${node}_${d}.log" 2>&1
        # Sleep 5s between dates to give GFW token room to breathe
        sleep 5
    done
}

# ============================================================ SANCTSOUND PARALLEL
# Patch sanctsound script to expand deployment list, then run 4 parallel
# chains, each handling 14 deployments serially.
# ============================================================

cp scripts/bootstrap_sanctsound_more_60s.py "$LOG_DIR/bootstrap_sanctsound_more_60s.py.bak"
"$VENV_PY" - <<'PYTHON_PATCH'
from pathlib import Path
p = Path("scripts/bootstrap_sanctsound_more_60s.py")
src = p.read_text()
# Match the SITES_TO_PULL block (any prior content) and replace
import re
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
else:
    print("WARN: SITES_TO_PULL block not found")
PYTHON_PATCH

# Run sanctsound in single proc (script doesn't support per-site CLI) —
# but it processes them serially through the expanded list.
PYTHONPATH=src "$VENV_PY" scripts/bootstrap_sanctsound_more_60s.py \
    > "$LOG_DIR/sanctsound_expanded.log" 2>&1 &
SANCT_PID=$!
disown $SANCT_PID 2>/dev/null

# ============================================================ LAUNCH 8 AIS CHAINS
# Each gets a different starting offset so they hit GFW with different dates first
# ============================================================
# Note: macOS bash 3.2 doesn't support declare -A — track PIDs in plain array.
CHAIN_PIDS=()
for i in 0 1 2 3 4 5 6 7; do
    node="${HYDROS[$i]}"
    offset=$(( i * 5 ))
    pull_ais_chain "$node" "$offset" &
    pid=$!
    CHAIN_PIDS+=("$pid")
    disown $pid 2>/dev/null
    echo "[$(date '+%H:%M:%S')] AIS chain $node (offset $offset) PID=$pid"
    sleep 2  # stagger launches by 2s
done

echo "[$(date '+%H:%M:%S')] 9 workers launched (8 AIS chains + 1 SanctSound serial)"

# ============================================================ ADAPTIVE CONTROLLER
# Runs in main script. Every 15 min:
#   1. Count 429 in last 5 min across all logs
#   2. Count current bootstrap_ procs
#   3. If 429 high → kill extra chains (down to min 6)
#   4. If 429 low and procs < 12 → spawn 2 more chains for fresh dates
#   5. Log to adaptive.log
# Total runtime: 6h. After that, all chains finish naturally.
# ============================================================

DEADLINE=$(( $(date +%s) + 21600 ))  # 6h from now
EXTRA_DATES=(
  2024-02-20 2024-03-15 2024-05-18 2024-07-09 2024-09-13
  2024-11-06 2023-02-11 2023-06-21 2023-09-19 2023-11-22
  2024-04-12 2024-08-23 2024-10-11 2023-05-26 2023-07-30
)
EXTRA_IDX=0
EXTRA_NODE_IDX=0

while [[ $(date +%s) -lt $DEADLINE ]]; do
    sleep 900  # 15 min check
    NOW=$(date '+%H:%M:%S')
    # Count 429s in last 5 min (find files modified within 5 min)
    N_429=$(find "$LOG_DIR/ais" -name "*.log" -mmin -5 -exec grep -l "429" {} + 2>/dev/null | wc -l | tr -d ' ')
    N_PROCS=$(pgrep -f "bootstrap_(ais|sanctsound)" 2>/dev/null | wc -l | tr -d ' ')

    DECISION=""
    if [[ $N_429 -gt 30 ]]; then
        # Scale down — kill 2 newest AIS chains
        VICTIMS=$(pgrep -f "bootstrap_ais_correlated" | tail -2)
        if [[ -n "$VICTIMS" ]]; then
            echo "$VICTIMS" | xargs kill 2>/dev/null
            DECISION="SCALE_DOWN (killed 2, 429=$N_429>30)"
        fi
    elif [[ $N_429 -lt 5 && $N_PROCS -lt 12 ]]; then
        # Scale up — spawn 2 new AIS workers on fresh (node, date) combos
        for _ in 1 2; do
            node="${HYDROS[$((EXTRA_NODE_IDX % ${#HYDROS[@]}))]}"
            d="${EXTRA_DATES[$((EXTRA_IDX % ${#EXTRA_DATES[@]}))]}"
            PYTHONPATH=src "$VENV_PY" scripts/bootstrap_ais_correlated.py \
                --hydrophone "$node" --date "$d" --step 120 --radius-km 10 \
                > "$LOG_DIR/ais/${node}_${d}_extra.log" 2>&1 &
            disown $! 2>/dev/null
            EXTRA_NODE_IDX=$(( EXTRA_NODE_IDX + 1 ))
            EXTRA_IDX=$(( EXTRA_IDX + 1 ))
        done
        DECISION="SCALE_UP (+2, 429=$N_429<5, procs $N_PROCS→$((N_PROCS+2)))"
    else
        DECISION="STEADY (429=$N_429, procs=$N_PROCS)"
    fi

    # snapshot row count + mbari share for monitoring
    ROWS=$(find data/training/v7_bulk -name "*.jsonl" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    echo "$NOW  procs=$N_PROCS  429_5min=$N_429  bulk_rows=$ROWS  $DECISION" \
        >> "$LOG_DIR/adaptive.log"
done

echo "[$(date '+%H:%M:%S')] adaptive controller exited (deadline reached)"

# Wait for any remaining chains
wait

# Restore sanctsound script
cp "$LOG_DIR/bootstrap_sanctsound_more_60s.py.bak" scripts/bootstrap_sanctsound_more_60s.py

echo ""
echo "===================================================="
echo "  $(date '+%Y-%m-%d %H:%M:%S')  DONE — final dist"
echo "===================================================="

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
for k,v in c.most_common():
    print(f'{k:<30s} {v:>8d}  {100*v/total:>5.1f}%')
print('-'*52)
print(f'{\"TOTAL\":<30s} {total:>8d}')
"
