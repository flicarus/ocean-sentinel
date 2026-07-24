# Ocean Sentinel

**Acoustic AI for dark-vessel detection in Marine Protected Areas.**
Hydrophone audio → CNN classifier → Gemma 4 analyst → live dashboard. Edge-deployable, no cloud key required, **96.4 %** held-out accuracy in **4.56 ms** per inference.

> Built for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/google-gemma-3-good-hackathon) (deadline 2026-05-18) by Jakub Koscielny ([@flicarus](https://github.com/flicarus)) and Maciej Rychlewski.

---

## The problem

Illegal, unreported, unregulated (IUU) fishing steals an estimated **USD 10–23 billion** from coastal communities every year (FAO, Pew). The standard enforcement stack — AIS transponder feeds, satellite radar, patrol vessels — goes blind the moment a fishing boat **turns off its transponder**. That's the exact behaviour Marine Protected Areas need to catch.

Underwater microphones (hydrophones) don't go blind. A 30 m trawler is loud whether it's transmitting or not. Ocean Sentinel turns every hydrophone in the world into a continuous, autonomous, **dark-vessel detector**.

---

## What it does

```
   hydrophone audio          CNN classifier              Gemma 4 analyst
   ───────────────►          ──────────────►             ──────────────►
   60 s @ 16 kHz             ship / not_ship             "DARK_VESSEL  HIGH
   mel-spectrogram           + uncertainty               severity · no AIS
                             + decision tier             contact within 5 km
                                                          → investigate"
                                                                │
                                                                ▼
                                              ┌──────────────────────────────┐
                                              │  vessel_events  (Supabase)   │
                                              │  ─────────────────────────── │
                                              │  realtime dashboard pin      │
                                              │  + audit-log row             │
                                              └──────────────────────────────┘
```

Three-layer pipeline:

| Layer | What | Footprint |
|---|---|---|
| **1. CNN classifier** (PyTorch) | mel-spec → P(ship) + epistemic uncertainty + decision tier (`DARK_VESSEL` / `CONFIRMED_VESSEL` / `AMBIENT` / `UNCERTAIN`) | 9.1 MB model, 18 MB RSS, 4.56 ms inference |
| **2. ChromaDB acoustic memory** | embeds every detection → matches against past confirmed events ("have we heard this signature before?") | local SQLite, no network |
| **3. Gemma 4 analyst** | function-calling agent that composes the analyst paragraph using CNN output + AIS proximity + MPA distance + ocean conditions | runs locally via Ollama, no API key |

Plus a **calibration layer** that's the actual moat: per-site decision thresholds + conformal prediction (Lei 2018) give a provable false-alarm bound. See [the per-site calibration writeup](docs/findings_per_site_calibration.md) for the discovery → fix → out-of-distribution validation story.

---

## Headline numbers

| Metric | Value | n / source |
|---|---|---|
| **Held-out test split accuracy** (PRIMARY) | **96.4 %** | n = 4,044 · `data/calibration/per_site_thresholds_v7_6_honest.json` |
| **Out-of-distribution accuracy** (fresh dates, 8 sites) | **96.0 %** | n = 4,414 · `scripts/eval_ood.py` |
| v7.6 vanilla (default threshold 0.5) | 89.3 % | n = 8,082 · `data/eval/per_site_v7_6.json` |
| Per-site sites at 100 % accuracy | **15 / 29** | same eval |
| Median CNN inference latency | **4.56 ms** | M5 Pro MPS · `data/eval/inference_perf_v7_6.json` |
| End-to-end pipeline latency | **19 ms** | full chain · `data/eval/inference_perf.json` |
| Real-time factor | **3,210 ×** | 60 s of audio / 19 ms |
| Sustained throughput | 52 clips / sec | single process, MPS |
| Cold start | 69 ms | model load on MPS |
| Conformal false-alarm bound | **FA ≤ α** | provable, Lei 2018 finite-sample correction |

The held-out test number (96.4 %) and the fresh-date OOD number (96.0 %) agree within 0.4 pp — calibration generalises; thresholds are not overfit. All numbers are reproducible from `data/eval/*` and trace-back to the scripts in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

---

## The model was right, the label was wrong

While building the calibration layer, the CNN kept disagreeing with "ambient" ground-truth labels at a high-traffic NOAA SanctSound site (OC01). Rather than trust the anomaly, we audited it three ways:

1. **Model self-audit** — invalid by construction: the model had been trained on the labels it was auditing (circular).
2. **Pure acoustic analysis (PSD)** — found a vessel signature at 28–37 Hz (blade-rate harmonics), but cross-recording calibration drift made it inconclusive on its own.
3. **MarineCadastre.gov** — NOAA's own government AIS archive. Decisive: a cargo vessel (JOSCO HUIZHOU) confirmed at 6.81 km from the hydrophone, timestamped to the second (2019-03-09T12:39:55Z), inside the "ambient"-labelled window.

Grad-CAM showed the CNN attends to a completely different acoustic band (cavitation, 500–1000 Hz) than the PSD method (blade-rate, 28–37 Hz) — two independent signal pathways converging on the same vessel at the same timestamp, confirmed by a third (AIS position data).

Extending the audit to the full 800-chunk corpus surfaced something more useful than a list of corrections: at high-AIS-traffic sites, *every* chunk has a vessel within 10 km — "ambient" is a labeling convention, not an acoustic ground truth. We shared this with the NOAA SanctSound team as a methodology note rather than a corrections list.

**Full story:** [Case study, Part I](https://www.sofar-ai.com/case-study) · [Part II: the audit](https://www.sofar-ai.com/case-study/audit) · in-repo: [`docs/findings_per_site_calibration.md`](docs/findings_per_site_calibration.md)

---

## Quick start

### 1. Install

```bash
git clone https://github.com/flicarus/ocean-sentinel.git
cd ocean-sentinel
python3 -m venv venv
venv/bin/python -m pip install -e .
```

That installs the `os` command on your PATH plus all declared dependencies (PyTorch, librosa, chromadb, httpx, fastapi, typer, rich, pyfiglet, ollama, pyyaml).

### 2. Download the model

The CNN checkpoint is a release asset (it doesn't live in git). One command:

```bash
mkdir -p data/models data/calibration
curl -L -o data/models/cnn_v7_6.pt \
    https://github.com/flicarus/ocean-sentinel/releases/download/cnn-v7.6/cnn_v7_6.pt
curl -L -o data/calibration/conformal_v7_6.json \
    https://github.com/flicarus/ocean-sentinel/releases/download/cnn-v7.6/conformal_v7_6.json
curl -L -o data/calibration/per_site_thresholds_v7_6.json \
    https://github.com/flicarus/ocean-sentinel/releases/download/cnn-v7.6/per_site_thresholds_v7_6.json
curl -L -o data/calibration/per_site_thresholds_v7_6_honest.json \
    https://github.com/flicarus/ocean-sentinel/releases/download/cnn-v7.6/per_site_thresholds_v7_6_honest.json
```

### 3. Verify the install

```bash
os doctor
```

Eight green checks across model file, per-site thresholds, conformal calibration, PyTorch device, librosa, end-to-end inference, eval results, and optional API server. Exits non-zero on any failure.

### 4. Run a detection

```bash
# Vanilla — uses default 0.5 threshold
os detect src/ocean_sentinel/test_samples/vessel_tanker.wav

# Site-calibrated — uses MBARI's threshold (0.88) + AIS context
os detect path/to/clip.wav --site mbari --ais 0

# Pipe to jq
os detect clip.wav --json | jq '.decision_tier, .ship_prob'
```

A site-calibrated detection with no AIS contact in radius outputs the **`DARK_VESSEL (HIGH)`** tier — the system's "potentially illegal fishing" signal. Same audio with `--site point-robinson` (threshold 0.02) produces a different decision: that's the per-site calibration moat in action.

---

## The `os` CLI

Twelve commands; each maps to a real user need:

| Command | What it does |
|---|---|
| `os doctor` | Self-check across model / calibration / deps / inference / eval files |
| `os info` | Show deployed model, per-site thresholds (sorted), eval scores, calibrated latency |
| `os bench` | Measure CNN inference latency on this machine |
| `os detect <wav>` | Classify a single clip; `--site` for per-site threshold, `--ais N` for AIS context, `--json` for piping |
| `os list-sites` | Enumerate hydrophone sites the system knows about |
| `os test <site>` | Run bundled known-label samples through the calibrated pipeline |
| `os monitor <site> --watch <dir>` | Operational mode: watch a folder, classify each new clip, push to dashboard |
| `os monitor <site> --replay <dir>` | One-shot replay across an entire folder of clips |
| `os refresh <site>` | Re-fit the per-site adapter + recalibrate the conformal threshold on accumulated ambient |
| `os onboard` | Live Gemma-driven onboarding flow for a new hydrophone site (8 steps, ~5 min) |
| `os onboard --demo` | Same flow with mocked tools — runs offline, no Ollama required |
| `os brief` | Generate an Ocean Intelligence System daily brief for one hydrophone |
| `os alert-demo` | End-to-end mock-stream alert pipeline demo |
| `os identify-vessel` | Gemma 4 multimodal vessel identification from a natural photograph |

All commands respect `OS_LOG_LEVEL` (e.g. `OS_LOG_LEVEL=warning os detect …` for clean output in scripted use).

---

## Live dashboard

The CLI ships detections to a **public Supabase Edge Function** (`ingest-event`) that validates the payload, uploads spectrogram + Grad-CAM PNGs, and inserts one row into `vessel_events`. The dashboard subscribes via Supabase Realtime — new detections appear within ~200 ms, **no page refresh needed**.

End-user installs **do not carry a service-role key**. The Edge Function does the privileged work server-side, so a leaked CLI install can never write directly to the table.

- Frontend repo: separate Next.js 16 / React 19
- Live URL: see [sofar-ai.com](https://sofar-ai.com)
- Self-host: point `OS_INGEST_URL` at your own Edge Function

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         Ocean Sentinel — runtime                          │
└───────────────────────────────────────────────────────────────────────────┘

    hydrophone audio
    (60 s @ 16 kHz mono)
            │
            ▼
    ┌───────────────────┐      ┌──────────────────────┐
    │  preprocess       │      │  per-site adapter    │
    │  high-pass +      │ ───► │  (residual MLP,      │
    │  log-mel 128 ×    │      │   ~33 k params,      │
    │  157 frames       │      │   init-as-identity)  │
    └───────────────────┘      └──────────┬───────────┘
                                          │
                                          ▼
                                ┌─────────────────────┐
                                │  CNN v7.6           │
                                │  (2.3 M params)     │
                                │  → ship_prob        │
                                │  → uncertainty      │
                                └─────────┬───────────┘
                                          │
                                          ▼
                                ┌─────────────────────┐
                                │  decision tier      │
                                │  per-site threshold │
                                │  + AIS context      │
                                │  + conformal bound  │
                                └─────────┬───────────┘
                                          │
                       ┌──────────────────┼──────────────────────┐
                       ▼                  ▼                      ▼
            ┌────────────────┐  ┌─────────────────┐   ┌──────────────────┐
            │ ChromaDB       │  │ local jsonl     │   │ public ingest    │
            │ acoustic       │  │ events log      │   │ gateway          │
            │ memory (RAG)   │  │ (offline source │   │ → Supabase       │
            │                │  │  of truth)      │   │ → dashboard      │
            └────────────────┘  └─────────────────┘   └──────────────────┘

  ┌────────────────────────────────────────────────────────────────────┐
  │  Gemma 4 (function-calling agent, runs locally via Ollama)         │
  │  ───────────────────────────────────────────────────────────────   │
  │  Composes the analyst paragraph from CNN output + AIS proximity    │
  │  + MPA distance + ocean conditions + acoustic memory matches.      │
  │  Called by `os brief`, `os monitor` narration, `os onboard`,       │
  │  and the event-detail panel in the dashboard.                      │
  └────────────────────────────────────────────────────────────────────┘
```

### Per-site calibration — the moat

A generalist CNN can't fit every site's noise floor. Our v7.6 trained with balanced sampling (`WeightedRandomSampler(weight=1/site_count)`) recovered site rarity, but de-weighted the dominant ambient class — MBARI ambient went from 97 % → 0.3 % accuracy. We didn't retrain again. Instead, we applied **per-site decision-threshold calibration**:

| Site | Vanilla threshold | Calibrated | Accuracy lift |
|---|---|---|---|
| MBARI ambient | 0.5 | **0.88** | 0.3 % → 100 % |
| Point Robinson recall | 0.5 | **0.02** | 13.4 % → 100 % |
| **Overall held-out** | 0.5 | **per-site** | 89.3 % → **96.4 %** |

The full discovery → fix → OOD-validation writeup is in [`docs/findings_per_site_calibration.md`](docs/findings_per_site_calibration.md). The empirical-finding companion (cosine similarity does *not* predict accuracy, n=14, r = −0.23 p=0.41) is in [`docs/empirical_findings.md`](docs/empirical_findings.md).

---

## Data sources

| Source | Adapter | What we use it for |
|---|---|---|
| **MBARI MARS** | `adapters/mbari.py` | 24 h continuous hydrophone WAVs on S3 (Monterey Bay) |
| **Orcasound** | `adapters/orcasound.py` | 8 PNW hydrophone nodes, live HLS streams |
| **NOAA SanctSound** | `adapters/sanctsound.py` | Cabled sanctuary deployments — public bucket |
| **Ocean Networks Canada** | `adapters/onc.py` | Cabled observatory, Canadian Pacific |
| **Global Fishing Watch** | `adapters/gfw.py` | AIS gap events (vessels that stopped transmitting) |
| **Copernicus Marine** | `adapters/copernicus.py` | SST + ocean current overlays |
| **ShipsEar + DeepShip** | (training only) | Labeled vessel-class corpora for v7.x training |

The 29-site eval pool spans these sources. Training corpus: ~205 k chunks pre-rebalance; ~150 k post-balanced-sampler. See [`hackathon/data-*`](docs/) Obsidian notes for full data provenance.

---

## Project layout

```
src/ocean_sentinel/
├── adapters/          MBARI · Orcasound · SanctSound · ONC · GFW · Copernicus · Supabase
├── api/               FastAPI routes (scan, events, alerts, memory, logs)
├── cli/               os {doctor, info, bench, detect, monitor, onboard, ...}
├── domain/            pydantic models + decision-tier policy
├── gemma/             Ollama function-calling agent + onboarding state machine
├── models/            PyTorch CNN backbone + VesselHead
├── services/          AudioAnalyzer, EventPersistence, ThreatClassifier, Artifacts
└── test_samples/      bundled .wav fixtures (~4.6 MB)

scripts/               training, calibration, benchmarks, demo tapes (VHS)
docs/                  LIMITATIONS.md, findings_per_site_calibration.md, empirical_findings.md
data/                  models/, calibration/, eval/, sites/* (largely gitignored)
```

The Next.js dashboard lives in a sibling repo (`~/oceansentinelfrontend`). Backend ↔ dashboard contract is documented in `src/ocean_sentinel/services/event_persistence.py`.

---

## What we measured — and what we didn't

[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) is the auditable inventory. The headline:

✅ **Measured**: held-out + OOD accuracy on 29 sites, per-site recall recovery, per-site false-alarm reduction (90.3 % → 3.2 % synthetic reef), conformal bound (Lei 2018), inference latency p50/p95, sustained throughput, cold start, RSS, Wilson 95 % CIs on small-n per-class scores.

⚠️ **Honest scope** — we tested Gemma 4 multimodal on mel spectrograms. It scored **1/4** with a strong "SHIP" bias regardless of ground truth — domain shift (mel-specs aren't natural images Gemma was pre-trained on), not a model-size problem. We documented the failure and **pivoted Gemma's role** from "verifier of CNN decisions" to "analytical synthesiser of structured detection data." The CNN remains the auditable decision-maker; Gemma writes the analyst-grade narrative on top of a number we trust. Native-audio modality would likely solve this, but Ollama doesn't yet expose audio input.

❌ **Out of scope (deferred to pilot)**: long-term temporal drift, adversarial-acoustic robustness, throughput at multi-thousand-stream scale, cross-sensor calibration (different hydrophone hardware models), formal admissibility-of-evidence chain for legal proceedings.

---

## Submission deliverables (hackathon)

Built for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/google-gemma-3-good-hackathon) (Global Resilience track):

- **Open-source repo** — this one
- **Live dashboard** — [sofar-ai.com](https://sofar-ai.com)
- **CNN release** — [cnn-v7.6](https://github.com/flicarus/ocean-sentinel/releases/tag/cnn-v7.6)
- **Documentation** — [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md), [`docs/findings_per_site_calibration.md`](docs/findings_per_site_calibration.md), [`docs/empirical_findings.md`](docs/empirical_findings.md)
- **Demo video** — short-form launch clips rendered with Charm VHS, in `scripts/demo_*.tape`

---

## Configuration

Copy `.env.example` to `.env` and fill in tokens for the data adapters you intend to use:

```bash
cp .env.example .env
# edit .env
```

| Var | Required for | Notes |
|---|---|---|
| `OS_GFW_API_TOKEN` | Live AIS / dark-vessel context | https://globalfishingwatch.org/ |
| `OS_COPERNICUS_USERNAME` / `OS_COPERNICUS_PASSWORD` | Ocean conditions | free Copernicus Marine account |
| `OS_SENDGRID_API_KEY` | Email alerts | optional |
| `OS_GEMMA_API_KEY` | Cloud Gemma (Google AI) | optional — Ollama works offline |
| `OS_ONC_TOKEN` | Ocean Networks Canada streams | free registration |
| `OS_LOG_LEVEL` | Filter log noise from CLI | `warning` / `error` / `info` |
| `OS_INGEST_URL` | Push events to your own gateway | defaults to public sandbox |

The CLI works **completely offline** if you skip all of the above and only run local detection on cached audio.

---

## Running the stack

```bash
# One command — Ollama + API + (optional dashboard hint)
bash scripts/run.sh

# Or pieces separately
PYTHONPATH=src venv/bin/python -m uvicorn ocean_sentinel.api.app:app --host 0.0.0.0 --port 8000 --reload --app-dir src
# Frontend (separate repo)
cd ~/oceansentinelfrontend && pnpm dev
```

API at `http://localhost:8000` · OpenAPI docs at `http://localhost:8000/docs`.

---

## License

MIT — see [`LICENSE`](LICENSE). Training data is sourced from public scientific repositories under their respective licenses (ShipsEar CC-BY-NC-4.0; DeepShip per-author; NOAA SanctSound public domain; MBARI MARS open access).

---

## Credits

- **Jakub Koscielny** ([@flicarus](https://github.com/flicarus)) — CNN, calibration, CLI, Gemma agent
- **Maciej Rychlewski** — data adapters, FastAPI, alerts, frontend integration
- **Datasets** — ShipsEar (UVIGO), DeepShip (Dalhousie), NOAA SanctSound, MBARI MARS, Orcasound
- **Models** — [Gemma 4](https://blog.google/technology/developers/gemma-3/) (Google DeepMind)
- **Tooling** — PyTorch, Rich, Typer, Ollama, Supabase, Charm VHS

> "Every hydrophone in the ocean becomes a witness. Every illegal trawler turns into an audit log entry. Marine Protected Areas stop being lines on a map."
