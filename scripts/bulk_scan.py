"""
Bulk scan: run the full pipeline (analyze → classify → persist → log) across
many dates and offsets, accumulating (spectrogram → threat level) training
pairs for the CNN.

Mirrors api/routes/pipeline.py but runs as a standalone script so we can
leave it churning in the background without starting the API server.

Usage:
    python scripts/bulk_scan.py                 # default date list, step=1800
    python scripts/bulk_scan.py --step 3600     # every hour
    python scripts/bulk_scan.py --dates 2023-07-01,2023-12-01
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ocean_sentinel.adapters.chromadb_store import ChromaDBAcousticMemory
from ocean_sentinel.adapters.copernicus import CopernicusAdapter
from ocean_sentinel.adapters.gemma import GemmaAdapter
from ocean_sentinel.adapters.gfw import GFWAdapter
from ocean_sentinel.adapters.mbari import MBARIAdapter
from ocean_sentinel.adapters.orcasound import OrcasoundAdapter
from ocean_sentinel.adapters.persistence import SQLiteEventStore
from ocean_sentinel.adapters.training_logger import JSONLTrainingLogger
from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import AcousticFeatures, DetectionEvent, GeoPoint
from ocean_sentinel.domain.protocols import HydrophoneSource
from ocean_sentinel.logging import configure_logging
from ocean_sentinel.services.audio_analyzer import AudioAnalyzer
from ocean_sentinel.services.classifier import ThreatClassifierService
from ocean_sentinel.services.correlation import CorrelationService

SECONDS_IN_DAY = 86400
LOCATION = GeoPoint(lat=36.7128, lon=-122.186)

DEFAULT_DATES = [
    "2023-06-15",
    "2023-08-05",
    "2023-10-15",
    "2023-12-01",
    "2024-01-15",
    "2024-02-01",
    "2024-03-10",
    "2024-04-01",
]


async def scan_chunk(
    *,
    source: HydrophoneSource,
    date: str,
    offset: int,
    analyzer: AudioAnalyzer,
    correlation: CorrelationService,
    classifier: ThreatClassifierService,
    store: SQLiteEventStore,
    training_logger: JSONLTrainingLogger,
    spec_dir: Path,
) -> str | None:
    dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    segment = await source.fetch_at_offset(dt, offset, 60)
    analyzed, features = analyzer.analyze(segment)

    event_id = f"{source.source_id}_{date}_{offset}"
    spec_path = spec_dir / f"{event_id}.npy"
    np.save(spec_path, analyzed.spectrogram)

    ais_gaps, ocean = await correlation.correlate(analyzed, features)
    result = await classifier.classify(audio=analyzed, ais_gaps=ais_gaps, ocean=ocean)

    await training_logger.log(
        event_id=event_id,
        spectrogram_path=str(spec_path),
        features=AcousticFeatures.from_analyzer_dict(features),
        context_text=f"{source.source_id} | {analyzed.time_window.start.isoformat()}",
        gemma_verdict=result.raw_output,
    )

    event = DetectionEvent(
        id=str(uuid.uuid4()),
        timestamp=analyzed.time_window.start,
        location=analyzed.location,
        threat_level=result.threat_level,
        confidence=result.confidence,
        classification_reasoning=result.reasoning,
        audio_segment=None,
        ais_gaps=ais_gaps,
        ocean_conditions=ocean,
        raw_model_output=result.raw_output,
    )
    await store.save_event(event)
    return result.threat_level.value


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", default=",".join(DEFAULT_DATES),
                        help="Comma-separated YYYY-MM-DD list")
    parser.add_argument("--step", type=int, default=1800,
                        help="Seconds between chunks (default 1800 = 48/day)")
    parser.add_argument("--max-per-date", type=int, default=24,
                        help="Cap chunks per date (default 24)")
    args = parser.parse_args()

    configure_logging()
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]

    settings = Settings()
    hydrophones: list[HydrophoneSource] = [
        MBARIAdapter(settings),
        OrcasoundAdapter("rpi_orcasound_lab", settings),
        OrcasoundAdapter("rpi_port_townsend", settings),
        OrcasoundAdapter("rpi_bush_point", settings),
        OrcasoundAdapter("rpi_sunset_bay", settings),
    ]
    analyzer = AudioAnalyzer(settings)
    gfw = GFWAdapter(settings)
    copernicus = CopernicusAdapter(settings)
    memory = ChromaDBAcousticMemory(persist_dir="data/chromadb")
    gemma = GemmaAdapter(settings, memory=memory)
    correlation = CorrelationService(gfw, copernicus, settings)
    classifier = ThreatClassifierService(analyzer, gemma)

    db_path = settings.database_url.replace("sqlite+aiosqlite:///", "")
    store = SQLiteEventStore(db_path)
    await store.init()

    training_logger = JSONLTrainingLogger(output_dir="data/training")

    spec_dir = Path("data/spectrograms")
    spec_dir.mkdir(parents=True, exist_ok=True)

    offsets = list(range(0, SECONDS_IN_DAY, args.step))[: args.max_per_date]

    totals: dict[str, int] = {}
    ok = fail = 0
    total_chunks = len(hydrophones) * len(dates) * len(offsets)
    done = 0

    print(f"Bulk scan: {len(hydrophones)} sources × {len(dates)} dates × {len(offsets)} offsets = {total_chunks} chunks")
    print(f"Sources: {[h.source_id for h in hydrophones]}")
    print(f"Dates: {dates}")
    print(f"Step: {args.step}s\n", flush=True)

    for source in hydrophones:
        for date in dates:
            for offset in offsets:
                done += 1
                tag = f"[{done}/{total_chunks}] {source.source_id} {date} +{offset}s"
                try:
                    threat = await scan_chunk(
                        source=source,
                        date=date, offset=offset,
                        analyzer=analyzer,
                        correlation=correlation, classifier=classifier,
                        store=store, training_logger=training_logger, spec_dir=spec_dir,
                    )
                    if threat:
                        totals[threat] = totals.get(threat, 0) + 1
                        ok += 1
                        print(f"{tag} → {threat}", flush=True)
                    else:
                        fail += 1
                        print(f"{tag} → SKIP", flush=True)
                except Exception as e:  # noqa: BLE001
                    fail += 1
                    print(f"{tag} → FAIL: {e}", flush=True)
                    traceback.print_exc(file=sys.stdout)

    print("\n=== Summary ===")
    print(f"OK: {ok}  FAIL: {fail}")
    for k, v in sorted(totals.items()):
        print(f"  {k}: {v}")

    for source in hydrophones:
        await source.close()
    await gemma.close()
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
