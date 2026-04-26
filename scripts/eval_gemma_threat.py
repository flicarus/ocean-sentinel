"""Evaluate Gemma's threat-level reasoning against the rubric.

Builds 9 hand-crafted scenarios spanning the NONE→CRITICAL rubric, calls
the real Ollama Gemma e4b with the production SYSTEM_PROMPT + production
_build_context, prints expected vs actual + reasoning. Lets us see whether
the rubric is being followed or if Gemma is going off-script.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import (
    AISGapEvent, AudioSegment, GeoPoint, OceanConditions, TimeWindow,
)
from ocean_sentinel.adapters.gemma import GemmaAdapter


def _audio(source_id: str = "mbari") -> AudioSegment:
    """Stub AudioSegment — only metadata is rendered into the prompt."""
    now = datetime.now(timezone.utc)
    return AudioSegment(
        source_file=f"{source_id}/synthetic_test_chunk.wav",
        location=GeoPoint(lat=36.7, lon=-122.0),
        time_window=TimeWindow(start=now, end=now + timedelta(seconds=60)),
        sample_rate=16000,
        samples=None,  # type: ignore[arg-type]  # not rendered
        spectrogram=None,
        source_id=source_id,
    )


def _gap(
    vessel: str,
    duration_h: float,
    in_mpa: bool | None,
    intentional: bool | None,
) -> AISGapEvent:
    now = datetime.now(timezone.utc)
    return AISGapEvent(
        vessel_id=f"vessel-{vessel}",
        vessel_name=vessel,
        flag_state="CHN",
        last_known_position=GeoPoint(lat=36.7, lon=-122.0),
        gap_start=now - timedelta(hours=duration_h),
        gap_end=None,
        gap_duration_hours=duration_h,
        intentional_disabling=intentional,
        in_mpa=in_mpa,
    )


def _features(
    engine_band_energy_db: float = -25.0,
    engine_band_ratio: float = 1.10,
    peak_frequency_hz: float = 200.0,
    spectral_flatness: float = 0.15,
    rms_energy: float = 0.04,
    is_engine_band_dominant: bool = True,
) -> dict:
    return {
        "engine_band_energy_db": engine_band_energy_db,
        "engine_band_ratio": engine_band_ratio,
        "peak_frequency_hz": peak_frequency_hz,
        "spectral_flatness": spectral_flatness,
        "rms_energy": rms_energy,
        "is_engine_band_dominant": is_engine_band_dominant,
    }


def _cnn(label: str, confidence: float) -> dict:
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {"not_ship": 1 - confidence if label == "ship" else confidence,
                          "ship":     confidence if label == "ship" else 1 - confidence},
        "embedding": [0.0] * 64,
        "checkpoint": "cnn_v6.pt",
        "profile_applied": "mbari",
    }


@dataclass
class Scenario:
    name: str
    expected: str
    cnn: dict
    features: dict
    gaps: list[AISGapEvent]
    ocean: OceanConditions | None = None


SCENARIOS: list[Scenario] = [
    Scenario(
        name="NONE — quiet ocean, CNN confident not_ship",
        expected="NONE",
        cnn=_cnn("not_ship", 0.98),
        features=_features(engine_band_energy_db=-55.0, is_engine_band_dominant=False,
                           engine_band_ratio=0.6),
        gaps=[],
    ),
    Scenario(
        name="NONE — borderline CNN ship 0.45, no AIS evidence",
        expected="NONE",
        cnn=_cnn("ship", 0.45),
        features=_features(engine_band_energy_db=-30.0),
        gaps=[],
    ),
    Scenario(
        name="LOW — CNN confident ship, no anomalies",
        expected="LOW",
        cnn=_cnn("ship", 0.88),
        features=_features(),
        gaps=[],
    ),
    Scenario(
        name="LOW — CNN ship + brief AIS gap (10 min), no MPA",
        expected="LOW",
        cnn=_cnn("ship", 0.93),
        features=_features(),
        gaps=[_gap("CARGO-A", 0.17, in_mpa=False, intentional=False)],
    ),
    Scenario(
        name="MEDIUM — CNN ship + 30min AIS gap near (not in) MPA",
        expected="MEDIUM",
        cnn=_cnn("ship", 0.91),
        features=_features(),
        gaps=[_gap("FISHER-B", 0.5, in_mpa=False, intentional=None)],
    ),
    Scenario(
        name="HIGH — CNN ship + 2h AIS gap, vessel inside MPA",
        expected="HIGH",
        cnn=_cnn("ship", 0.92),
        features=_features(),
        gaps=[_gap("DARKVESSEL-C", 2.0, in_mpa=True, intentional=False)],
    ),
    Scenario(
        name="HIGH — CNN ship + 30min gap + intentional_disabling=true",
        expected="HIGH",
        cnn=_cnn("ship", 0.90),
        features=_features(),
        gaps=[_gap("EVADER-D", 0.5, in_mpa=False, intentional=True)],
    ),
    Scenario(
        name="CRITICAL — confident ship + intentional + in_mpa + 3h gap",
        expected="CRITICAL",
        cnn=_cnn("ship", 0.96),
        features=_features(engine_band_ratio=1.4, engine_band_energy_db=-15.0),
        gaps=[_gap("IUU-OPERATOR", 3.0, in_mpa=True, intentional=True)],
    ),
    Scenario(
        name="CRITICAL — multiple stacked anomalies",
        expected="CRITICAL",
        cnn=_cnn("ship", 0.95),
        features=_features(engine_band_ratio=1.6, engine_band_energy_db=-12.0),
        gaps=[
            _gap("BAD-ACTOR-1", 4.5, in_mpa=True, intentional=True),
            _gap("BAD-ACTOR-2", 2.0, in_mpa=True, intentional=True),
        ],
    ),
]


async def main() -> None:
    settings = Settings()
    settings.google_ai_api_key = None  # force Ollama path
    gemma = GemmaAdapter(settings, memory=None)

    correct = 0
    print(f"Model: {settings.gemma_model}\n")
    print(f"{'#':>2}  {'expected':>9s}  {'actual':>9s}  {'ok':>3s}  scenario")
    print("-" * 90)

    for i, sc in enumerate(SCENARIOS, 1):
        text = gemma._build_context(
            audio=_audio(sc.cnn.get("profile_applied", "mbari")),
            ais_gaps=sc.gaps,
            ocean=sc.ocean,
            features=sc.features,
            cnn_verdict=sc.cnn,
        )
        try:
            raw = await gemma._call_ollama(text)
        except Exception as e:
            print(f"{i:>2d}  {sc.expected:>9s}  {'ERROR':>9s}  ✗  {sc.name} -- {e}")
            continue
        actual = (raw.get("threat_level") or "").upper()
        ok = "✓" if actual == sc.expected else "✗"
        if actual == sc.expected:
            correct += 1
        reason = (raw.get("reasoning") or "").strip().replace("\n", " ")
        print(f"{i:>2d}  {sc.expected:>9s}  {actual:>9s}  {ok}    {sc.name}")
        print(f"       reasoning: {reason[:140]}")

    print("-" * 90)
    print(f"Score: {correct}/{len(SCENARIOS)} = {correct / len(SCENARIOS):.0%}")

    await gemma.close()


if __name__ == "__main__":
    asyncio.run(main())
