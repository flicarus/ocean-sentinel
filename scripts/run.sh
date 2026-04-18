#!/bin/bash
# Ocean Sentinel — one-command starter.
# Starts Ollama (if not running), API, dashboard, opens browser.
#
# Usage:
#   bash scripts/run.sh                        # just start everything
#   bash scripts/run.sh --scan 2024-02-01      # + trigger a quick scan

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Use venv's python directly (avoids broken activate-script shebangs)
VENV_PY="$PROJECT_DIR/venv/bin/python"
export MPLBACKEND=Agg
mkdir -p data/chromadb data/spectrograms data/training data/models

# --- 1. Ollama -----------------------------------------------------------
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "• Ollama not running — starting it..."
    (ollama serve > /tmp/ollama-serve.log 2>&1) &
    OLLAMA_PID=$!
    for i in $(seq 1 20); do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "• Ollama ready."
            break
        fi
        sleep 1
    done
else
    echo "• Ollama already running."
fi

# --- 2. API --------------------------------------------------------------
echo "• Starting API on http://localhost:8000 ..."
"$VENV_PY" -m uvicorn ocean_sentinel.api.app:app \
    --host 0.0.0.0 --port 8000 \
    --reload \
    --reload-exclude 'dashboard/*' \
    --reload-exclude 'dashboard-react/*' \
    --reload-exclude 'data/*' \
    --app-dir src > /tmp/os-api.log 2>&1 &
API_PID=$!

for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health/ > /dev/null 2>&1; then
        echo "• API ready."
        break
    fi
    sleep 1
done

# --- 3. Dashboard --------------------------------------------------------
echo "• Starting dashboard on http://localhost:5173 ..."
(cd dashboard && node_modules/.bin/vite --host > /tmp/os-dashboard.log 2>&1) &
DASH_PID=$!

# Give vite a moment to bind, then open browser
sleep 3
echo "• Opening browser..."
if command -v open > /dev/null 2>&1; then
    open "http://localhost:5173"
fi

echo ""
echo "════════════════════════════════════════════════"
echo "  Ocean Sentinel is running"
echo "  Dashboard: http://localhost:5173"
echo "  API docs:  http://localhost:8000/docs"
echo "  Logs:      tail -f /tmp/os-api.log  /tmp/os-dashboard.log"
echo "════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop everything."

# Optional one-shot scan via flag
if [ "$1" = "--scan" ] && [ -n "$2" ]; then
    echo ""
    echo "• Triggering pipeline scan for $2 ..."
    sleep 2
    curl -s -X POST http://localhost:8000/pipeline/scan \
        -H "Content-Type: application/json" \
        -d "{\"date\": \"$2\", \"step\": 300, \"sample_every\": 5}" | "$VENV_PY" -m json.tool
fi

# Clean up on Ctrl+C — kill children we started (not Ollama if it was already running)
cleanup() {
    echo ""
    echo "• Stopping services..."
    kill $API_PID $DASH_PID 2>/dev/null || true
    if [ -n "$OLLAMA_PID" ]; then
        kill $OLLAMA_PID 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

wait
