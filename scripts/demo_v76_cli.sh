#!/usr/bin/env bash
# Ocean Sentinel — judge / reviewer demo.
#
# Walks through the v7.6 + per-site calibration story in <60 seconds
# of CLI output. No browser, no API server, no Gemma — just the model
# and the calibration layer.
#
# Usage:  bash scripts/demo_v76_cli.sh

set -e
cd "$(dirname "$0")/.."

PY="PYTHONPATH=src venv/bin/python -m ocean_sentinel.cli.main"
SAMPLE="src/ocean_sentinel/test_samples/vessel_tanker.wav"
[ -f "$SAMPLE" ] || SAMPLE=$(find src/ocean_sentinel/test_samples -name "*.wav" 2>/dev/null | head -1)
[ -f "$SAMPLE" ] || { echo "No test audio found"; exit 1; }

step() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════════════"
}

step "1. WHAT'S DEPLOYED"
eval "$PY info" 2>&1 | grep -v "^2026\|cnn_v7_loaded\|site_thresholds_loaded"

step "2. INFERENCE PERFORMANCE ON THIS MACHINE"
eval "$PY bench --n 30 --warmup 5" 2>&1 | grep -v "^2026\|cnn_v7_loaded\|site_thresholds_loaded"

step "3. DETECTION — vanilla v7.6 (no site context, default threshold 0.5)"
eval "$PY detect '$SAMPLE'" 2>&1 | grep -v "^2026\|cnn_v7_loaded\|site_thresholds_loaded"

step "4. SAME AUDIO — with per-site calibration for MBARI"
echo "  (MBARI ambient classification: 0.3% → 100% after threshold 0.88)"
eval "$PY detect '$SAMPLE' --site mbari" 2>&1 | grep -v "^2026\|cnn_v7_loaded\|site_thresholds_loaded"

step "5. SAME AUDIO — with per-site calibration for point-robinson"
echo "  (point-robinson recall: 13.4% (v7.5) → 100% after threshold 0.02)"
eval "$PY detect '$SAMPLE' --site point-robinson" 2>&1 | grep -v "^2026\|cnn_v7_loaded\|site_thresholds_loaded"

step "DONE"
echo ""
echo "  Same audio, three different sites → three different decisions."
echo "  That is per-site calibration in action."
echo ""
echo "  Full writeup:  docs/findings_per_site_calibration.md"
echo "  Honest numbers: docs/LIMITATIONS.md"
echo ""
