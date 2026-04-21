"""Generate AIS-correlated training pairs from Orcasound hydrophones.

For each 60s audio window in a date range:
  1. Ask GFW (4Wings report): what vessels were in the area this hour?
  2. Assign a physics-grounded label based on vessel class + size.
  3. Fetch the audio, compute spectrogram + features.
  4. Write a training pair (metadata + .npy file).

Label scheme (per Kuba's design):
  HIGH    = cargo/tanker present, or any vessel >= 80m
  MEDIUM  = fishing vessel, or vessel 30-79m, or passenger < 80m
  NONE    = no vessels in current hour AND prior hour
  SKIP    = everything ambiguous (unknown class/size, acoustic tail)
  LOW     = intentionally omitted — comes from ShipsEar dataset instead
  CRITICAL = intentionally omitted — emerges from CNN vs AIS disagreement later

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
from ocean_sentinel.domain.models import AcousticFeatures, GeoPoint, NearbyVessel, TimeWindow
from ocean_sentinel.services.audio_analyzer import AudioAnalyzer

log = structlog.get_logger()

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


def _severity_rank(vessel: NearbyVessel) -> int:
    """Higher number = louder/more threatening vessel. Used to find worst-case."""
    vc = (vessel.vessel_class or "").lower()
    if vc in ("cargo", "tanker"):
        return 4
    if vessel.length_m and vessel.length_m >= 80:
        return 3
    if vc == "fishing":
        return 2
    if vessel.length_m and vessel.length_m >= 30:
        return 2
    if vc == "passenger":
        return 2
    return 0  # unknown/small — not safe to label


def _label_from_vessels(vessels: list[NearbyVessel]) -> tuple[str, str] | None:
    """Return (threat_level, vessel_type) from a list of vessels in the bbox,
    or None if the window should be skipped.

    Does NOT emit LOW (comes from ShipsEar) or CRITICAL (emerges from CNN later).
    """
    if not vessels:
        return None  # caller handles NONE vs SKIP based on prior-hour check

    worst = max(vessels, key=_severity_rank)
    vc = (worst.vessel_class or "").lower()

    # HIGH — large commercial vessels
    if vc in ("cargo", "tanker"):
        return ("HIGH", vc)
    if worst.length_m and worst.length_m >= 80:
        return ("HIGH", vc or "unknown")

    # MEDIUM — fishing, mid-sized, or passenger
    if vc == "fishing":
        return ("MEDIUM", "fishing_vessel")
    if worst.length_m and 30 <= worst.length_m < 80:
        return ("MEDIUM", vc or "unknown")
    if vc == "passenger":
        return ("MEDIUM", "passenger_vessel")

    # Everything else: unknown class + unknown/small length — too ambiguous
    return None  # SKIP


def _vessels_at_time(vessels: list[NearbyVessel], window: TimeWindow) -> list[NearbyVessel]:
    """Filter to vessels whose presence window overlaps with the given time window."""
    result = []
    for v in vessels:
        if v.present_start is None or v.present_end is None:
            result.append(v)
            continue
        if v.present_end >= window.start and v.present_start <= window.end:
            result.append(v)
    return result


async def label_for_window(
    gfw: GFWAdapter,
    location: GeoPoint,
    window: TimeWindow,
    radius_km: float,
) -> tuple[str, str] | None:
    """Return (threat_level, vessel_type) or None (skip this window).

    None is returned when:
    - Only unknown/small vessels present (can't label cleanly)
    - Vessels were present in the prior hour (acoustic tail risk)
    - No vessel class or length available to make a confident decision
    """
    all_day = await gfw.get_vessels_in_radius(location, radius_km, window)
    current = _vessels_at_time(all_day, window)

    if current:
        return _label_from_vessels(current)

    # No vessels this hour — check prior hour for acoustic tail
    prior_window = TimeWindow(
        start=window.start - timedelta(hours=1),
        end=window.start,
    )
    all_prior = await gfw.get_vessels_in_radius(location, radius_km, prior_window)
    prior = _vessels_at_time(all_prior, prior_window)
    if prior:
        return None  # acoustic tail may still be present — skip

    return ("NONE", "none")


def _vessel_type_label(vessel_type: str) -> str:
    mapping = {
        "cargo":            "cargo_ship",
        "tanker":           "tanker",
        "fishing":          "fishing_vessel",
        "fishing_vessel":   "fishing_vessel",
        "passenger":        "passenger_vessel",
        "passenger_vessel": "passenger_vessel",
    }
    return mapping.get(vessel_type.lower(), vessel_type)


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
        "NONE": 0, "MEDIUM": 0, "HIGH": 0, "SKIPPED": 0, "FAILED": 0,
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
            print(
                f"  [{i}/{total_windows}] {window_start.strftime('%H:%M')} — "
                f"NONE={counters['NONE']} MED={counters['MEDIUM']} "
                f"HIGH={counters['HIGH']} SKIP={counters['SKIPPED']} "
                f"FAIL={counters['FAILED']}"
            )

        try:
            label = await label_for_window(gfw, location, window, radius_km)
            if label is None:
                counters["SKIPPED"] += 1
                continue
            threat_level, vessel_type = label

            segment = await orca.fetch_at_offset(base_date, offset, 60)
            analyzed, features_dict = analyzer.analyze(segment)

            event_id = f"ais_{hydrophone_id}_{date}_{offset}"
            spec_path = spec_dir / f"{event_id}.npy"
            np.save(spec_path, analyzed.spectrogram)

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
                    "vessel_type": _vessel_type_label(vessel_type),
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
    print()
    print("Expected Bush Point weekday: NONE=40-60, MEDIUM=80-120, HIGH=20-40, SKIPPED=40-80")

    await gfw.close()
    await orca.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate AIS-correlated training pairs from Orcasound hydrophones"
    )
    ap.add_argument(
        "--hydrophone", required=True,
        choices=list(HYDROPHONE_NODES.keys()),
    )
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--radius-km", type=float, default=10.0)
    ap.add_argument("--step", type=int, default=300)
    args = ap.parse_args()

    asyncio.run(main(args.hydrophone, args.date, args.radius_km, args.step))
