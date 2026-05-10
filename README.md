# Ocean Sentinel

Acoustic-first ocean monitoring — hydrophone audio + vessel tracking (GFW) + oceanographic data, fused by Gemma 4. Built for the **Gemma 4 Good Hackathon** (deadline 2026-05-18).

Three-layer AI pipeline:
1. **CNN** — fast local triage (millisecond classification)
2. **RAG / ChromaDB** — "have I heard this before?" memory
3. **Gemma 4** — multimodal deep analysis on uncertain cases

Detailed architecture: [`hackathon/arch-three-layer-ai`](../../Documents/Obsidian%20Vault/hackathon/arch-three-layer-ai.md) in the Obsidian vault.

---

## Quick start — one command

```bash
cd "path/to/Sofar.ai"
bash scripts/run.sh
```

Starts **everything** at once:
- Ollama (if not already running)
- API server on http://localhost:8000
- Dashboard on http://localhost:5173
- **Opens your browser automatically** after ~3s

Ctrl+C kills all three cleanly.

### Variant: run a scan immediately

```bash
bash scripts/run.sh --scan 2024-02-01
```

Kicks off a `POST /pipeline/scan` for that date right after startup.

### Logs

Background processes log to:
- `/tmp/os-api.log` — FastAPI + pipeline
- `/tmp/ollama-serve.log` — LLM server

The frontend lives in a separate repo (`~/oceansentinelfrontend`, Next.js).
Run it with `pnpm dev` from that directory; it hits this API on `:8000`.

Follow in real time:
```bash
tail -f /tmp/os-api.log
```

---

## Quick start — separate terminals (for debugging)

If you want to watch each service in its own window:

### Terminal 1 — Ollama
```bash
ollama serve
```
(Skip if it's already running — you'll see `address already in use`, that's fine.)

### Terminal 2 — API
```bash
cd "path/to/Sofar.ai"
PYTHONPATH=src venv/bin/python -m uvicorn ocean_sentinel.api.app:app \
    --host 0.0.0.0 --port 8000 \
    --reload \
    --reload-exclude 'data/*' \
    --app-dir src
```
API live at http://localhost:8000 · Docs at http://localhost:8000/docs

### Terminal 3 — Frontend (separate repo)
```bash
cd ~/oceansentinelfrontend
pnpm dev
```
Frontend at http://localhost:3000 (Next.js).

### (Optional) Terminal 4 — bulk scan
Runs many hydrophones × dates × offsets in sequence, writing straight to `data/training/`:
```bash
cd "path/to/Sofar.ai"
PYTHONPATH=src venv/bin/python scripts/bulk_scan.py --step 1800 --max-per-date 8
```

---

## First-time setup

```bash
# clone + enter the repo, then:
python3 -m venv venv
venv/bin/python -m pip install -e .
```

That installs the `ocean_sentinel` package + all declared dependencies (chromadb, librosa, httpx, fastapi, torch, etc.).

Frontend deps (separate repo):
```bash
cd ~/oceansentinelfrontend
pnpm install
```

Environment:
```bash
cp .env.example .env
# edit .env — add GFW / Copernicus / SendGrid tokens
```

Required for full pipeline:
- `OS_GFW_API_TOKEN` — Global Fishing Watch (vessel tracking)
- `OS_COPERNICUS_USERNAME` / `OS_COPERNICUS_PASSWORD` — ocean conditions
- `OS_SENDGRID_API_KEY` — email alerts
- `OS_GEMMA_API_KEY` — Google AI Gemma (optional if using Ollama fallback)

Optional for future week:
- `OS_ONC_TOKEN` — Ocean Networks Canada (register free at https://data.oceannetworks.ca/)

---

## Useful one-liners

### Test the CNN model loads + forward pass works
```bash
PYTHONPATH=src venv/bin/python -m ocean_sentinel.models.cnn
```

### Train the CNN on current data
```bash
PYTHONPATH=src venv/bin/python scripts/train_cnn.py
```

### Predict on one spectrogram
```bash
PYTHONPATH=src venv/bin/python scripts/infer.py data/spectrograms/<file>.npy
```

### Diagnose Orcasound adapter on one node
```bash
PYTHONPATH=src venv/bin/python -c "
import asyncio
from ocean_sentinel.adapters.orcasound import OrcasoundAdapter
from ocean_sentinel.config import Settings

async def go():
    a = OrcasoundAdapter('rpi_bush_point', Settings())
    await a._ensure_stream()
    print(f'{len(a._segments)} segments, bucket={a._bucket}')
    await a.close()

asyncio.run(go())
"
```

---

## Project layout

```
src/ocean_sentinel/
  adapters/       MBARI, Orcasound, GFW, Copernicus, Gemma, ChromaDB, SQLite
  api/            FastAPI routes (scan, events, alerts, memory, logs)
  domain/         pydantic models + protocols
  models/         PyTorch CNN (Backbone + VesselHead + future SpeciesHead)
  services/       AudioAnalyzer, CorrelationService, ThreatClassifierService
scripts/          bulk_scan.py, train_cnn.py, infer.py, run.sh
~/oceansentinelfrontend  Next.js 16 + React 19 + Leaflet + Three.js (separate repo)
data/             chromadb/, spectrograms/, training/, models/
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: ocean_sentinel` | package not installed | `venv/bin/python -m pip install -e .` |
| `ModuleNotFoundError: chromadb` (or any dep) | venv missing deps | same: `pip install -e .` |
| `bind: address already in use` on Ollama | Ollama already running | ignore, it's fine |
| API won't start but no error | Check `/tmp/os-api.log` | often `.env` missing or port 8000 taken |
| `403 Forbidden` on Orcasound | Running OLD adapter code | Restart script — Python cached the old module |
