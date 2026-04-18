"""
Scan an entire day's MBARI recording chunk by chunk.
Processes 60-second segments sequentially through the full 24h file.

Usage:
    python scripts/scan_full_day.py
    python scripts/scan_full_day.py --date 2024-02-01 --step 300
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from ocean_sentinel.config import Settings
from ocean_sentinel.adapters.mbari import MBARIAdapter
from ocean_sentinel.adapters.gfw import GFWAdapter
from ocean_sentinel.adapters.copernicus import CopernicusAdapter
from ocean_sentinel.adapters.gemma import GemmaAdapter
from ocean_sentinel.domain.models import GeoPoint
from ocean_sentinel.services.audio_analyzer import AudioAnalyzer
from ocean_sentinel.services.correlation import CorrelationService
from ocean_sentinel.services.classifier import ThreatClassifierService

LOCATION = GeoPoint(lat=36.7128, lon=-122.186)
SECONDS_IN_DAY = 86400


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2024-02-01", help="Date to scan (YYYY-MM-DD)")
    parser.add_argument("--step", type=int, default=60, help="Seconds between chunks (60=continuous, 300=every 5min)")
    parser.add_argument("--duration", type=int, default=60, help="Chunk duration in seconds")
    parser.add_argument("--classify", action="store_true", help="Run Gemma classification on hits")
    args = parser.parse_args()

    dt = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    step = args.step
    duration = args.duration
    total_chunks = SECONDS_IN_DAY // step

    settings = Settings()
    mbari = MBARIAdapter(settings)
    analyzer = AudioAnalyzer(settings)

    # For classification (only if --classify)
    gemma = None
    classifier = None
    correlation = None
    if args.classify:
        gfw = GFWAdapter(settings)
        copernicus = CopernicusAdapter(settings)
        gemma = GemmaAdapter(settings)
        correlation = CorrelationService(gfw, copernicus, settings)
        classifier = ThreatClassifierService(analyzer, gemma)

    print("=" * 75)
    print("Ocean Sentinel — Full Day Scanner")
    print("=" * 75)
    print(f"Date:     {args.date}")
    print(f"Step:     {step}s ({'continuous' if step == duration else f'every {step // 60}min'})")
    print(f"Duration: {duration}s per chunk")
    print(f"Chunks:   {total_chunks}")
    print(f"Classify: {'yes (Gemma)' if args.classify else 'no (audio analysis only)'}")
    print("=" * 75)

    print(f"\n{'Time':<10} {'Engine dB':>10} {'Ratio':>7} {'Peak Hz':>8} {'Flat':>7} {'RMS':>10} {'Hit':>5}")
    print("-" * 75)

    hits = []
    scanned = 0
    failed = 0

    for offset in range(0, SECONDS_IN_DAY, step):
        hours = offset // 3600
        minutes = (offset % 3600) // 60
        time_str = f"{hours:02d}:{minutes:02d}"

        try:
            segment = await mbari.fetch_at_offset(dt, offset, duration)
            analyzed, features = analyzer.analyze(segment)
            scanned += 1

            is_hit = (
                features["is_engine_band_dominant"]
                or features["engine_band_ratio"] > 1.05
            )
            marker = " <<<" if is_hit else ""

            print(
                f"{time_str:<10}"
                f"{features['engine_band_energy_db']:>10.2f}"
                f"{features['engine_band_ratio']:>7.3f}"
                f"{features['peak_frequency_hz']:>8.1f}"
                f"{features['spectral_flatness']:>7.4f}"
                f"{features['rms_energy']:>10.6f}"
                f"{'  YES' if is_hit else '   no':>5}"
                f"{marker}"
            )

            if is_hit:
                hits.append((time_str, offset, analyzed, features))

        except Exception as e:
            failed += 1
            print(f"{time_str:<10}  FAILED: {e}")

    print("-" * 75)
    print(f"\nScanned: {scanned}/{total_chunks} chunks | "
          f"Failed: {failed} | "
          f"Hits: {len(hits)}")

    if hits:
        print(f"\nEngine signatures detected at:")
        for time_str, offset, _, features in hits:
            print(f"  {time_str} — "
                  f"ratio: {features['engine_band_ratio']:.3f}, "
                  f"peak: {features['peak_frequency_hz']:.0f} Hz, "
                  f"energy: {features['engine_band_energy_db']:.1f} dB")

        if args.classify and classifier and correlation:
            print(f"\n--- Gemma Classification on {len(hits)} hits ---\n")
            for time_str, offset, analyzed, features in hits:
                print(f"Classifying {time_str}...")
                ais_gaps, ocean = await correlation.correlate(analyzed, features)
                result = await classifier.classify(
                    audio=analyzed, ais_gaps=ais_gaps, ocean=ocean,
                )
                print(f"  Threat:     {result.threat_level.value}")
                print(f"  Confidence: {result.confidence:.0%}")
                print(f"  Reasoning:  {result.reasoning}\n")

    await mbari.close()
    if gemma:
        await gemma.close()


if __name__ == "__main__":
    asyncio.run(main())
