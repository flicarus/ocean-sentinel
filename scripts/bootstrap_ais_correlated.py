"""Generate AIS-correlated training pairs from Orcasound hydrophones.

For each 60s audio window in a date range:
  1. Ask GFW: what vessels were within radius_km of the hydrophone?
  2. Assign a physics-grounded label based on proximity and vessel size.
  3. Fetch the audio, compute spectrogram + features.
  4. Write a training pair (metadata + .npy file).

Labels are assigned by physics, not AI — a vessel 1km away at time T
objectively means that recording contains ship noise.

Ambiguous windows (vessel just left, partial overlap) are skipped to
keep training data clean.

Usage:
    PYTHONPATH=src venv/bin/python3 scripts/bootstrap_ais_correlated.py \\
        --hydrophone bush-point \\
        --date 2024-06-15 \\
        --radius-km 10 \\
        --step 300
"""

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import structlog

from ocean_sentinel.adapters.gfw import GFWAdapter
from ocean_sentinel.adapters.orcasound import OrcasoundAdapter
from ocean_sentinel.adapters.training_logger import JSONLTrainingLogger
from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import AcousticFeatures, GeoPoint, TimeWindow
from ocean_sentinel.services.audio_analyzer import AudioAnalyzer

log = structlog.get_logger()

# Maps short CLI names to OrcasoundAdapter node names
HYDROPHONE_NODES: dict[str, str] = {
    "bush-point":     "rpi_bush_point",
    "sunset-bay":     "rpi_sunset_bay",
    "orcasound-lab":  "rpi_orcasound_lab",
    "north-sjc":      "rpi_north_sjc",
    "point-robinson": "rpi_point_robinson",
    "port-townsend":  "rpi_port_townsend",
    "andrews-bay":    "rpi_andrews_bay",
    "mast-center":    "rpi_mast_center",
}


def _vessel_type_from_gfw(vessel) -> str:
    """Map GFW vessel class to our taxonomy."""
    mapping = {
        "fishing":   "fishing_vessel",
        "cargo":     "cargo_ship",
        "tanker":    "tanker",
        "passenger": "passenger_vessel",
        "tug":       "tug",
    }
    return mapping.get((vessel.vessel_class or "").lower(), "unknown")


async def label_for_window(
    gfw: GFWAdapter,
    location: GeoPoint,
    window: TimeWindow,
    radius_km: float,
) -> tuple[str, str] | None:
    """Return (threat_level, vessel_type) or None if window is ambiguous.

    None means skip this window — don't pollute training data with
    uncertain labels.
    """
    nearby = await gfw.get_vessels_in_radius(location, radius_km, window)

    if not nearby:
        # No vessels now — check if one was here recently (last 1h)
        extended = TimeWindow(
            start=window.start - timedelta(hours=1),
            end=window.end,
        )
        historical = await gfw.get_vessels_in_radius(location, radius_km, extended)
        if historical:
            return None  # vessel just left, acoustic tail may still be present
        return ("NONE", "none")

    closest = nearby[0]  # already sorted closest-first

    if closest.distance_km <= 2.0:
        # Very close — classify by size
        if closest.length_m and closest.length_m >= 80:
            return ("HIGH", _vessel_type_from_gfw(closest))
        return ("MEDIUM", _vessel_type_from_gfw(closest))

    if closest.distance_km <= radius_km:
        return ("LOW", _vessel_type_from_gfw(closest))

    return None  # shouldn't happen but guard anyway


async def main(
    hydrophone_id: str,
    date: str,
    radius_km: float,
    step: int,
) -> None:
    node_name = HYDROPHONE_NODES[hydrophone_id]
    settings = Settings()

    gfw = GFWAdapter(settings)
    orca = OrcasoundAdapter(node_name=node_name, settings=settings)
    analyzer = AudioAnalyzer(settings)
    tlog = JSONLTrainingLogger(output_dir="data/training")

    spec_dir = Path("data/spectrograms")
    spec_dir.mkdir(parents=True, exist_ok=True)

    base_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    location = orca.location

    counters: dict[str, int] = {
        "NONE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0,
        "SKIPPED": 0, "FAILED": 0,
    }

    total_windows = 86400 // step
    print(f"Hydrophone : {hydrophone_id} ({node_name})")
    print(f"Date       : {date}")
    print(f"Windows    : {total_windows} ({step}s cadence)")
    print(f"Radius     : {radius_km} km")
    print()

    for i, offset in enumerate(range(0, 86400, step)):
        window_start = base_date + timedelta(seconds=offset)
        window = TimeWindow(
            start=window_start,
            end=window_start + timedelta(seconds=60),
        )

        if i % 20 == 0:
            print(f"  [{i}/{total_windows}] {window_start.strftime('%H:%M')} — "
                  f"NONE={counters['NONE']} LOW={counters['LOW']} "
                  f"MEDIUM={counters['MEDIUM']} HIGH={counters['HIGH']} "
                  f"SKIP={counters['SKIPPED']} FAIL={counters['FAILED']}")

        try:
            # 1. Decide label first — skip audio fetch if ambiguous
            label = await label_for_window(gfw, location, window, radius_km)
            if label is None:
                counters["SKIPPED"] += 1
                continue
            threat_level, vessel_type = label

            # 2. Fetch audio from Orcasound
            segment = await orca.fetch_at_offset(base_date, offset, 60)

            # 3. Analyze — spectrogram + features (sync call)
            analyzed, features_dict = analyzer.analyze(segment)

            # 4. Save spectrogram .npy
            event_id = f"ais_{hydrophone_id}_{date}_{offset}"
            spec_path = spec_dir / f"{event_id}.npy"
            np.save(spec_path, analyzed.spectrogram)

            # 5. Write training pair
            await tlog.log(
                event_id=event_id,
                spectrogram_path=str(spec_path),
                features=AcousticFeatures.from_analyzer_dict(features_dict),
                context_text=(
                    f"ais-correlated | hydrophone={hydrophone_id} | "
                    f"window={window_start.isoformat()}"
                ),
                gemma_verdict={
                    "threat_level": threat_level,
                    "confidence": 1.0,
                    "reasoning": "AIS-correlated label — physics, not model prediction",
                    "vessel_type": vessel_type,
                    "recommended_action": "none",
                },
                source_id=f"ais-correlated-{hydrophone_id}",
            )
            counters[threat_level] += 1

        except Exception as e:
            counters["FAILED"] += 1
            log.error(
                "ais_bootstrap_window_failed",
                hydrophone=hydrophone_id,
                offset=offset,
                error=str(e),
            )

    print()
    print("═" * 40)
    print(f"Done: {hydrophone_id} / {date}")
    for k, v in counters.items():
        print(f"  {k:10s}: {v}")
    print("═" * 40)

    await gfw.close()
    await orca.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate AIS-correlated training pairs from Orcasound hydrophones"
    )
    ap.add_argument(
        "--hydrophone", required=True,
        choices=list(HYDROPHONE_NODES.keys()),
        help="Which Orcasound hydrophone to use",
    )
    ap.add_argument("--date", required=True, help="Date to process (YYYY-MM-DD)")
    ap.add_argument("--radius-km", type=float, default=10.0,
                    help="Radius around hydrophone to check for vessels (default 10)")
    ap.add_argument("--step", type=int, default=300,
                    help="Seconds between windows (default 300 = 5 min)")
    args = ap.parse_args()

    asyncio.run(main(args.hydrophone, args.date, args.radius_km, args.step))
