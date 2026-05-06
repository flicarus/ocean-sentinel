"""End-to-end CRITICAL threat scenario demo.

Synthetic scenario for pipeline demonstration. Real deployment uses live
GFW vessel API + Orcasound/MBARI hydrophone streams + real AIS gap feed;
this script exercises the same classification pipeline with hand-built
inputs so the reasoning chain is observable end-to-end (great for the
submission video).

Scenario:
  - Vessel: PESCA NORTE (MMSI 224157000), Spanish trawler, 3 prior IUU
  - Location: 38.05°N, -123.42°W (Cordell Bank no-take NMS centroid)
  - Audio: 60s of real cargo passage (DeepShip Cargo/103.wav)
  - AIS gap: 90 min, intentional_disabling=true, in_mpa=true
  - Pre-seeded: 3 prior HIGH detections in last 6h within 5km

Expected verdict: CRITICAL.  All 3 tools should fire in one iteration.

Usage:
    PYTHONPATH=src venv/bin/python scripts/demo_critical_scenario.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

sys.path.insert(0, "src")

from ocean_sentinel.adapters.gemma import GemmaAdapter
from ocean_sentinel.adapters.persistence import SQLiteEventStore
from ocean_sentinel.config import Settings
from ocean_sentinel.domain.enums import ThreatLevel
from ocean_sentinel.domain.models import (
    AISGapEvent, AudioSegment, DetectionEvent, GeoPoint, TimeWindow,
)
from ocean_sentinel.services.agent_tools import Toolbox
from ocean_sentinel.services.audio_analyzer import AudioAnalyzer
from ocean_sentinel.services.classifier import ThreatClassifierService
from ocean_sentinel.services.cnn_classifier import CNNClassifier

# ---- Quiet down structlog so the printed trace is the only output -------

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(40),  # ERROR+ only
)

# ---- Scenario parameters -------------------------------------------------

DEMO_DB = "data/demo_critical.db"          # isolated from production DB
CKPT = "data/models/cnn_v6.pt"
AUDIO = Path("data/deepship/Cargo/103.wav")  # 32 kHz mono float, ~3 min cargo

LAT = 38.05                                  # Cordell Bank centroid
LON = -123.42

VESSEL_MMSI = "224157000"                    # PESCA NORTE in mock registry
VESSEL_NAME = "PESCA NORTE"

GAP_HOURS = 1.5
PRIOR_DETECTIONS = 3
PRIOR_WINDOW_HOURS = 6


# ---- Recording wrapper around Toolbox to capture dispatches --------------

class RecordingToolbox(Toolbox):
    """Toolbox that captures each (name, args, result) for later playback."""

    def __init__(self, event_store: SQLiteEventStore | None = None) -> None:
        super().__init__(event_store=event_store)
        self.dispatched: list[dict] = []

    async def dispatch(self, name, args):
        result_json = await super().dispatch(name, args)
        self.dispatched.append({
            "name": name,
            "args": args,
            "result": json.loads(result_json),
        })
        return result_json


# ---- Helpers -------------------------------------------------------------

def hr(title: str = "", char: str = "─") -> None:
    bar = char * 70
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


def load_demo_audio(wav_path: Path, lat: float, lon: float, duration_s: int = 60) -> AudioSegment:
    samples, sr = sf.read(wav_path, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != 16000:
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000).astype(np.float32, copy=False)
        sr = 16000
    samples = samples[: duration_s * sr]

    now = datetime.now(timezone.utc)
    return AudioSegment(
        source_file=f"demo://{wav_path.name}",
        location=GeoPoint(lat=lat, lon=lon),
        time_window=TimeWindow(start=now, end=now + timedelta(seconds=duration_s)),
        sample_rate=sr,
        samples=samples,
        source_id="deepship",   # v6 doesn't have this profile → LOHO path
    )


def build_mock_ais_gap(lat: float, lon: float) -> AISGapEvent:
    now = datetime.now(timezone.utc)
    return AISGapEvent(
        vessel_id=VESSEL_MMSI,
        vessel_name=VESSEL_NAME,
        flag_state="ESP",
        last_known_position=GeoPoint(lat=lat, lon=lon),
        gap_start=now - timedelta(hours=GAP_HOURS),
        gap_end=None,
        gap_duration_hours=GAP_HOURS,
        intentional_disabling=True,
        in_mpa=True,
    )


async def seed_prior_detections(
    store: SQLiteEventStore, lat: float, lon: float, n: int, hours: int,
) -> list[str]:
    now = datetime.now(timezone.utc)
    ids: list[str] = []
    for i in range(n):
        ts = now - timedelta(hours=hours * (i + 1) / (n + 1))
        d_lat = ((1.0 + i * 0.7) / 111.0) * (1 if i % 2 else -1)
        ev = DetectionEvent(
            id=f"demo_prior_{uuid.uuid4().hex[:8]}",
            timestamp=ts,
            location=GeoPoint(lat=lat + d_lat, lon=lon),
            threat_level=ThreatLevel.HIGH,
            confidence=0.91 + i * 0.02,
            classification_reasoning=f"[demo seed] Prior detection #{i+1}",
            ais_gaps=[],
            ocean_conditions=None,
            raw_model_output={"demo": True},
            audio_segment=None,
        )
        await store.save_event(ev)
        ids.append(ev.id)
    return ids


# ---- Main flow -----------------------------------------------------------

async def main() -> None:
    if not AUDIO.exists():
        sys.exit(f"Demo audio not found: {AUDIO}. Run scripts/bootstrap_deepship.py first.")
    if not Path(CKPT).exists():
        sys.exit(f"CNN checkpoint not found: {CKPT}.")

    # Header --------------------------------------------------------------
    hr("OCEAN SENTINEL — END-TO-END CRITICAL SCENARIO DEMO", char="━")
    print(f"  Vessel:         {VESSEL_NAME} (MMSI {VESSEL_MMSI})")
    print(f"  Location:       {LAT:.4f}°N, {LON:.4f}°W (Cordell Bank no-take NMS)")
    print(f"  Audio:          {AUDIO.name} (60s cargo passage)")
    print(f"  AIS gap:        {GAP_HOURS}h, intentional=True, in_mpa=True")
    print(f"  Recent context: {PRIOR_DETECTIONS} HIGH detections, last {PRIOR_WINDOW_HOURS}h")
    print(f"  Production model: cnn_v6 + gemma4:e4b + check_mpa / vessel_registry / recent_detections")
    print(f"  NOTE: synthetic scenario for demonstration. Real deployment uses live GFW + AIS.")

    # Pipeline init -------------------------------------------------------
    settings = Settings()

    # Isolated DB — wipe each run to keep the demo deterministic.
    if Path(DEMO_DB).exists():
        Path(DEMO_DB).unlink()
    store = SQLiteEventStore(DEMO_DB)
    await store.init()
    seeded_ids = await seed_prior_detections(
        store, LAT, LON, PRIOR_DETECTIONS, PRIOR_WINDOW_HOURS,
    )

    analyzer = AudioAnalyzer(settings)
    cnn = CNNClassifier(CKPT)
    toolbox = RecordingToolbox(event_store=store)
    gemma = GemmaAdapter(settings, memory=None, toolbox=toolbox)
    classifier = ThreatClassifierService(analyzer, gemma, cnn=cnn)

    # Build inputs --------------------------------------------------------
    audio = load_demo_audio(AUDIO, LAT, LON)
    ais_gaps = [build_mock_ais_gap(LAT, LON)]

    # Run -----------------------------------------------------------------
    hr("Running pipeline ...")
    t0 = time.perf_counter()
    result = await classifier.classify(audio=audio, ais_gaps=ais_gaps, ocean=None)
    elapsed = time.perf_counter() - t0

    # CNN verdict ---------------------------------------------------------
    cnn_v = result.raw_output.get("cnn", {})
    hr("CNN verdict (tier 1 — binary acoustic classifier)")
    print(f"  label:          {cnn_v.get('label')}")
    print(f"  confidence:     {cnn_v.get('confidence', 0):.3f} (calibrated)")
    print(f"  source_id:      {audio.source_id}")
    print(f"  profile:        {cnn_v.get('profile_applied') or 'none (LOHO path)'}")

    # Tool trace ----------------------------------------------------------
    hr(f"Gemma agent loop — {len(toolbox.dispatched)} tool dispatches")
    for d in toolbox.dispatched:
        print(f"\n  [Tool] {d['name']}({format_args(d['args'])})")
        for k, v in d['result'].items():
            print(f"    → {k}: {v}")

    # Final verdict -------------------------------------------------------
    hr("Final verdict")
    print(f"  threat_level:    {result.threat_level.value}")
    print(f"  confidence:      {result.confidence:.2f}")
    print(f"  vessel_type:     {result.raw_output.get('vessel_type', '—')}")
    print(f"  recommended:     {result.raw_output.get('recommended_action', '—')}")
    print(f"\n  reasoning:")
    for line in chunk_text(result.reasoning, 64):
        print(f"    {line}")

    # Footer --------------------------------------------------------------
    hr(f"Done in {elapsed:.1f}s — {len(toolbox.dispatched)} tools, {ThreatLevel(result.threat_level).value} verdict", char="━")

    await store.close()


def format_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            parts.append(f'{k}="{v}"')
        elif isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def chunk_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines, line = [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line.strip())
            line = w + " "
        else:
            line += w + " "
    if line.strip():
        lines.append(line.strip())
    return lines


if __name__ == "__main__":
    asyncio.run(main())
