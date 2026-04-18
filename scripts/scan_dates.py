"""
Scan multiple MBARI dates to find recordings with ship engine signatures.
Runs audio analysis only (fast) — then full Gemma classification on hits.

Usage:
    python scripts/scan_dates.py
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

# Dates to scan — mix of seasons, times of day, known shipping activity
# Monterey Bay has commercial shipping lanes nearby
# Night hours (UTC 08:00-12:00 = midnight-4am local) = less recreational, more commercial
SCAN_DATES = [
    # Summer 2023 — busy season
    datetime(2023, 6, 1, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 7, 1, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 7, 20, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 8, 5, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 8, 20, 0, 0, tzinfo=timezone.utc),
    # Fall 2023
    datetime(2023, 9, 10, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 10, 1, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 10, 15, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 11, 1, 0, 0, tzinfo=timezone.utc),
    # Winter 2023-2024
    datetime(2023, 12, 1, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 2, 1, 0, 0, tzinfo=timezone.utc),
    # Spring 2024
    datetime(2024, 3, 10, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 4, 1, 0, 0, tzinfo=timezone.utc),
]

LOCATION = GeoPoint(lat=36.7128, lon=-122.186)


async def main() -> None:
    settings = Settings()
    mbari = MBARIAdapter(settings)
    analyzer = AudioAnalyzer(settings)

    # For full classification on hits
    gfw = GFWAdapter(settings)
    copernicus = CopernicusAdapter(settings)
    gemma = GemmaAdapter(settings)
    correlation = CorrelationService(gfw, copernicus, settings)
    classifier = ThreatClassifierService(analyzer, gemma)

    print("=" * 70)
    print("Ocean Sentinel — Multi-Date Scanner")
    print(f"Scanning {len(SCAN_DATES)} dates at Monterey Canyon")
    print("=" * 70)

    hits = []

    # Phase 1: Quick scan — audio analysis only
    print("\n--- PHASE 1: Audio Analysis Scan ---\n")
    print(f"{'Date':<22} {'Engine dB':>10} {'Ratio':>7} {'Peak Hz':>8} {'Flat':>7} {'Engine?':>8}")
    print("-" * 70)

    for dt in SCAN_DATES:
        tw = TimeWindow(start=dt, end=dt)
        try:
            segments = await mbari.fetch_audio(LOCATION, tw)
            analyzed, features = analyzer.analyze(segments[0])

            is_hit = (
                features["is_engine_band_dominant"]
                or features["engine_band_ratio"] > 1.05
            )
            marker = " <<<" if is_hit else ""

            print(
                f"{dt.strftime('%Y-%m-%d %H:%M UTC'):<22}"
                f"{features['engine_band_energy_db']:>10.2f}"
                f"{features['engine_band_ratio']:>7.3f}"
                f"{features['peak_frequency_hz']:>8.1f}"
                f"{features['spectral_flatness']:>7.4f}"
                f"  {'YES' if features['is_engine_band_dominant'] else 'no':>5}"
                f"{marker}"
            )

            if is_hit:
                hits.append((dt, analyzed, features))

        except Exception as e:
            print(f"{dt.strftime('%Y-%m-%d %H:%M UTC'):<22}  FAILED: {e}")

    print("-" * 70)
    print(f"\nHits: {len(hits)} / {len(SCAN_DATES)} dates have engine signatures")

    # Phase 2: Full classification on hits
    if hits:
        print("\n--- PHASE 2: Gemma Classification on Hits ---\n")

        for dt, analyzed, features in hits:
            print(f"\nClassifying {dt.strftime('%Y-%m-%d %H:%M UTC')}...")

            ais_gaps, ocean = await correlation.correlate(analyzed, features)
            result = await classifier.classify(
                audio=analyzed,
                ais_gaps=ais_gaps,
                ocean=ocean,
            )

            print(f"  Threat:     {result.threat_level.value}")
            print(f"  Confidence: {result.confidence:.0%}")
            print(f"  Reasoning:  {result.reasoning}")
            print(f"  AIS gaps:   {len(ais_gaps)}")
            if ocean:
                print(f"  Current:    {ocean.current_speed_ms} m/s")
    else:
        print("\nNo engine signatures found. Try different dates or longer segments.")
        print("Tip: increase OS_MBARI_SEGMENT_SECONDS in .env (default 60s)")

    # Cleanup
    await mbari.close()
    await gemma.close()


if __name__ == "__main__":
    asyncio.run(main())
