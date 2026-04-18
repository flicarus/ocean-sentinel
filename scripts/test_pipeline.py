"""
End-to-end pipeline test with real MBARI data.

Usage:
    python scripts/test_pipeline.py

Requires:
    - .env with OS_GFW_API_TOKEN set
    - Either OS_GOOGLE_AI_API_KEY in .env  (Google AI)
      or Ollama running locally with gemma model (fallback)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from ocean_sentinel.config import Settings
from ocean_sentinel.adapters.mbari import MBARIAdapter
from ocean_sentinel.adapters.gfw import GFWAdapter
from ocean_sentinel.adapters.copernicus import CopernicusAdapter
from ocean_sentinel.adapters.gemma import GemmaAdapter
from ocean_sentinel.domain.models import GeoPoint, TimeWindow
from ocean_sentinel.services.audio_analyzer import AudioAnalyzer
from ocean_sentinel.services.correlation import CorrelationService
from ocean_sentinel.services.classifier import ThreatClassifierService
from ocean_sentinel.services.pipeline import Pipeline


async def main() -> None:
    settings = Settings()

    # --- Adapters ---
    mbari = MBARIAdapter(settings)
    gfw = GFWAdapter(settings)
    copernicus = CopernicusAdapter(settings)
    gemma = GemmaAdapter(settings)

    # --- Services ---
    analyzer = AudioAnalyzer(settings)
    correlation = CorrelationService(
        vessel_tracker=gfw,
        ocean_source=copernicus,
        settings=settings,
    )
    classifier = ThreatClassifierService(
        analyzer=analyzer,
        gemma=gemma,
    )

    # --- Pipeline ---
    pipeline = Pipeline(
        hydrophone=mbari,
        analyzer=analyzer,
        correlation=correlation,
        classifier=classifier,
    )

    # --- Run ---
    # MBARI MARS hydrophone — Monterey Canyon
    location = GeoPoint(lat=36.7128, lon=-122.186)

    # Pick a date known to have data in the MBARI S3 bucket
    # Pacific Sound archive: 2023 data is reliably available
    time_window = TimeWindow(
        start=datetime(2023, 7, 15, 0, 0, tzinfo=timezone.utc),
        end=datetime(2023, 7, 15, 0, 0, tzinfo=timezone.utc),
    )

    print("=" * 60)
    print("Ocean Sentinel — Pipeline Test")
    print("=" * 60)
    print(f"Location: {location.lat}°N, {location.lon}°W")
    print(f"Time: {time_window.start}")
    print(f"Model: {settings.gemma_model}")
    print(f"Backend: {'Google AI' if settings.google_ai_api_key else 'Ollama'}")
    print("=" * 60)

    print("\n[1/4] Fetching audio from MBARI...")
    segments = await mbari.fetch_audio(location, time_window)
    print(f"       Got {len(segments)} segment(s), "
          f"{len(segments[0].samples)} samples each")

    print("\n[2/4] Analyzing audio...")
    analyzed, features = analyzer.analyze(segments[0])
    print(f"       Engine band energy: {features['engine_band_energy_db']} dB")
    print(f"       Engine band ratio:  {features['engine_band_ratio']}")
    print(f"       Peak frequency:     {features['peak_frequency_hz']} Hz")
    print(f"       Spectral flatness:  {features['spectral_flatness']}")
    print(f"       Engine dominant:    {features['is_engine_band_dominant']}")

    print("\n[3/4] Correlating with AIS + ocean data...")
    ais_gaps, ocean = await correlation.correlate(analyzed, features)
    print(f"       AIS gaps found: {len(ais_gaps)}")
    if ocean:
        print(f"       SST: {ocean.sea_surface_temp_c}°C")
        print(f"       Current: {ocean.current_speed_ms} m/s")
    else:
        print("       Ocean data: unavailable (non-critical)")

    print("\n[4/4] Classifying with Gemma...")
    result = await classifier.classify(
        audio=analyzed,
        ais_gaps=ais_gaps,
        ocean=ocean,
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  Threat level: {result.threat_level.value}")
    print(f"  Confidence:   {result.confidence:.0%}")
    print(f"  Reasoning:    {result.reasoning}")
    print(f"  Raw output:   {result.raw_output}")
    print("=" * 60)

    # Cleanup
    await mbari.close()
    await gemma.close()


if __name__ == "__main__":
    asyncio.run(main())
