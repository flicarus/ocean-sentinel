"""Pull diverse MBARI hours/days, AIS-labelled, for v7.1 training.

The original MBARI training pull picked specifically quiet deep-canyon
hours so v7 saw "MBARI = no ship". Live MBARI also gets traffic — ferries,
cruise ships, fishing — and that's why v7 mis-flags traffic-laden Monterey
Bay audio as DARK_VESSEL.

This script broadens the MBARI distribution by sampling across:
  - multiple days spread over a year (seasonal coverage)
  - all hours of the day (dawn fish chorus, daytime traffic, night quiet)
  - both ship and ambient labels (AIS-derived)

Output JSONL: data/training/v7_bulk/mbari_diverse.jsonl
Spectrogram dir: data/spectrograms/

Usage:
    PYTHONPATH=src venv/bin/python scripts/bootstrap_mbari_diverse.py \\
        --days 2024-02-01,2024-04-01,2024-06-01,2024-08-01,2024-10-01 \\
        --chunks-per-hour 2 \\
        --radius-km 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import structlog

sys.path.insert(0, "src")

from ocean_sentinel.adapters.gfw import GFWAdapter
from ocean_sentinel.adapters.mbari import MBARIAdapter
from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import GeoPoint, NearbyVessel, TimeWindow
from ocean_sentinel.services.audio_analyzer import AudioAnalyzer

log = structlog.get_logger()


OUT_PATH = Path("data/training/v7_bulk/mbari_diverse.jsonl")
SPEC_DIR = Path("data/spectrograms")


def _severity_rank(v: NearbyVessel) -> int:
    vc = (v.vessel_class or "").lower()
    if vc in ("cargo", "tanker"): return 4
    if v.length_m and v.length_m >= 80: return 3
    if vc == "fishing": return 2
    if v.length_m and v.length_m >= 30: return 2
    if vc == "passenger": return 2
    return 0


def _vessel_type_label(vc: str) -> str:
    return {
        "cargo": "cargo_ship", "tanker": "tanker",
        "fishing": "fishing_vessel", "fishing_vessel": "fishing_vessel",
        "passenger": "passenger_vessel", "passenger_vessel": "passenger_vessel",
    }.get(vc.lower(), vc or "none")


def _load_existing_capture_starts() -> set[str]:
    """Skip already-pulled chunks so re-runs are idempotent."""
    if not OUT_PATH.exists():
        return set()
    seen: set[str] = set()
    with OUT_PATH.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if "audio_capture_start" in r:
                    seen.add(r["audio_capture_start"])
            except json.JSONDecodeError:
                continue
    return seen


def _log_pair(entry: dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


async def main(days: list[str], chunks_per_hour: int, radius_km: float,
               cap_per_day_per_label: int) -> None:
    settings = Settings()
    mbari = MBARIAdapter(settings)
    gfw = GFWAdapter(settings)
    analyzer = AudioAnalyzer(settings)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    seen_starts = _load_existing_capture_starts()
    if seen_starts:
        log.info("dedup_active", existing_chunks=len(seen_starts))

    counters = {"ship": 0, "not_ship": 0, "skipped_seen": 0,
                "skipped_cap": 0, "fetch_failed": 0}

    for day_str in days:
        try:
            day = datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            log.warning("bad_day_format", day=day_str)
            continue

        # Pre-fetch hourly AIS once per day (cheaper than per-chunk).
        try:
            day_window = TimeWindow(start=day, end=day + timedelta(hours=24))
            vessels_today = await gfw.get_vessels_in_radius(
                mbari.location, radius_km, day_window,
            )
        except Exception as e:
            log.warning("ais_fetch_failed", day=day_str, error=str(e))
            vessels_today = []

        # We don't have hourly resolution from get_vessels_in_radius (DAILY),
        # so we treat "any vessel within radius today" as a positive class
        # signal at the day level. For the chunk's actual label we still
        # require AIS presence within the day; absence → ambient.
        has_traffic = len(vessels_today) > 0
        worst = max(vessels_today, key=_severity_rank) if vessels_today else None
        worst_class = (worst.vessel_class if worst else "") or ""

        log.info("day_start", day=day_str,
                 vessels_in_radius=len(vessels_today),
                 worst_class=worst_class)

        per_label_today = {"ship": 0, "not_ship": 0}

        for hour in range(0, 24):
            for sub in range(chunks_per_hour):
                # Distribute chunks within the hour: e.g. 2/hour at minute 15 and 45.
                minute = int(60 * (sub + 0.5) / chunks_per_hour)
                offset_seconds = hour * 3600 + minute * 60

                seg_start = day + timedelta(seconds=offset_seconds)
                key = seg_start.isoformat()
                if key in seen_starts:
                    counters["skipped_seen"] += 1
                    continue

                label = "ship" if has_traffic else "not_ship"
                if per_label_today[label] >= cap_per_day_per_label:
                    counters["skipped_cap"] += 1
                    continue

                try:
                    segment = await mbari.fetch_at_offset(
                        day, offset_seconds=offset_seconds, duration_seconds=60,
                    )
                    analyzed, features = analyzer.analyze(segment)
                except Exception as e:
                    counters["fetch_failed"] += 1
                    log.warning("mbari_chunk_failed",
                                day=day_str, hour=hour, error=str(e))
                    continue

                event_id = f"mbari_diverse_{day.strftime('%Y%m%d')}_{offset_seconds}"
                spec_path = SPEC_DIR / f"{event_id}.npy"
                np.save(spec_path, analyzed.spectrogram)

                vessel_type = _vessel_type_label(worst_class) if has_traffic else "none"
                distance_bucket = "close" if has_traffic else "none"
                threat_level = (
                    "HIGH" if has_traffic and worst_class in ("cargo", "tanker") else
                    "MEDIUM" if has_traffic else "NONE"
                )

                entry = {
                    "event_id": event_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "spectrogram_path": str(spec_path),
                    "audio_capture_start": analyzed.time_window.start.isoformat(),
                    "audio_capture_end": analyzed.time_window.end.isoformat(),
                    "features": {
                        "engine_band_ratio": features["engine_band_ratio"],
                        "peak_frequency_hz": features["peak_frequency_hz"],
                        "spectral_flatness": features["spectral_flatness"],
                        "rms_energy": features["rms_energy"],
                        "engine_band_energy_db": features["engine_band_energy_db"],
                    },
                    "context_text": (
                        f"mbari | {analyzed.time_window.start.isoformat()} | "
                        f"vessel={vessel_type} | radius={radius_km}km"
                    ),
                    "gemma_verdict": {
                        "threat_level": threat_level,
                        "vessel_type": vessel_type,
                        "is_bootstrap": True,
                    },
                    "provenance": {
                        "source_id": "mbari-diverse",
                        "source_file": segment.source_file,
                        "hydrophone": "mbari",
                        "day": day_str,
                        "offset_seconds": offset_seconds,
                    },
                    "label": label,
                    "distance_bucket": distance_bucket,
                    "ground_truth_label": threat_level,
                }
                _log_pair(entry)
                seen_starts.add(key)
                counters[label] += 1
                per_label_today[label] += 1

        log.info("day_done", day=day_str,
                 ship_today=per_label_today["ship"],
                 not_ship_today=per_label_today["not_ship"])

    await mbari.close()
    await gfw.close()
    print(f"\nFinal counters: {counters}")


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--days",
        default="2024-02-01,2024-04-01,2024-06-01,2024-08-01,2024-10-01,2024-12-01",
        help="Comma-separated YYYY-MM-DD list. Default = bi-monthly across 2024.",
    )
    ap.add_argument("--chunks-per-hour", type=int, default=2,
                    help="How many 60s chunks to extract per hour (24 hours covered)")
    ap.add_argument("--radius-km", type=float, default=10.0)
    ap.add_argument("--cap-per-day-per-label", type=int, default=30,
                    help="Max ship + max ambient chunks per day (prevents class skew)")
    args = ap.parse_args()
    days = [d.strip() for d in args.days.split(",") if d.strip()]
    asyncio.run(main(days, args.chunks_per_hour, args.radius_km,
                     args.cap_per_day_per_label))


if __name__ == "__main__":
    cli()
