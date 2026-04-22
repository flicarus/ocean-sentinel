"""Bootstrap not_ship training data from MBARI (+ optional Watkins).

Writes rows via ocean_sentinel.training.TrainingJsonlWriter so the schema
stays identical to the ShipsEar rows we'll migrate in a later step.

Usage:
    PYTHONPATH=src venv/bin/python scripts/bootstrap_not_ship.py --limit 5      # pilot
    PYTHONPATH=src venv/bin/python scripts/bootstrap_not_ship.py                # full (~1200)
"""
from __future__ import annotations

import argparse
import asyncio
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import structlog

from ocean_sentinel.adapters.mbari import MBARIAdapter, MONTEREY_CANYON
from ocean_sentinel.adapters.watkins import WatkinsAdapter, WATKINS_SOURCE_URL
from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import (
    AcousticFeatures, AudioSegment, GeoPoint, TimeWindow,
)
from ocean_sentinel.services.audio_analyzer import AudioAnalyzer
from ocean_sentinel.training.labels import binary_label_for
from ocean_sentinel.training.schema import (
    AudioMeta, Provenance, Taxonomy, TrainingJsonlWriter, TrainingRow,
    utc_now_iso,
)

log = structlog.get_logger()

JSONL_PATH = Path("data/training/gemma_labels.jsonl")
SPEC_DIR = Path("data/spectrograms")
WATKINS_ROOT = Path("data/watkins")

CHUNK_SECONDS = 5
MBARI_CHUNK_DURATION = 60
REPRESENTATION_VERSION = "abs_db_v1"


# ---------------------------------------------------------------------------
# MBARI
# ---------------------------------------------------------------------------

def mbari_sample_points(
    n_chunks: int, seed: int = 42,
) -> list[tuple[datetime, int]]:
    """Deterministic (date, offset) pairs spread across 2022-2023."""
    rng = random.Random(seed)
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end = datetime(2023, 12, 31, tzinfo=timezone.utc)
    span_days = (end - start).days
    return [
        (
            start + timedelta(days=rng.randrange(span_days)),
            rng.randrange(0, 86400 - MBARI_CHUNK_DURATION),
        )
        for _ in range(n_chunks)
    ]


async def collect_mbari(
    n_chunks: int, analyzer: AudioAnalyzer,
    writer: TrainingJsonlWriter, spec_dir: Path,
) -> dict[str, int]:
    """Pull N 60s chunks from MBARI, split each into 5s sub-clips, write rows."""
    settings = Settings()
    adapter = MBARIAdapter(settings)
    stats = {"chunks_fetched": 0, "chunks_failed": 0, "clips_written": 0}
    try:
        for i, (dt, offset) in enumerate(mbari_sample_points(n_chunks)):
            try:
                segment = await adapter.fetch_at_offset(
                    dt, offset_seconds=offset,
                    duration_seconds=MBARI_CHUNK_DURATION,
                )
                stats["chunks_fetched"] += 1
            except Exception as e:
                stats["chunks_failed"] += 1
                log.warning("mbari_fetch_failed",
                            dt=dt.isoformat(), offset=offset, error=str(e))
                continue

            provenance = Provenance(
                source_id="mbari",
                source_file=segment.source_file,
                source_url=segment.source_file,
                original_label="ambient",
                license="public-domain",
                collected_at=utc_now_iso(),
            )
            taxonomy = Taxonomy(
                category="ambient",
                subclass="deep_ocean_ambient",
            )
            n = write_clips_from_samples(
                samples=segment.samples,
                sample_rate=segment.sample_rate,
                location=segment.location,
                capture_time=segment.time_window.start,
                event_id_prefix=f"mbari_{dt.strftime('%Y%m%d')}_{offset}",
                taxonomy=taxonomy,
                provenance=provenance,
                analyzer=analyzer,
                writer=writer,
                spec_dir=spec_dir,
            )
            stats["clips_written"] += n

            if (i + 1) % 10 == 0:
                log.info("mbari_progress", done=i + 1, total=n_chunks, **stats)
    finally:
        await adapter.close()
    return stats


# ---------------------------------------------------------------------------
# Watkins
# ---------------------------------------------------------------------------

def collect_watkins(
    analyzer: AudioAnalyzer,
    writer: TrainingJsonlWriter, spec_dir: Path,
) -> dict[str, int]:
    """Walk data/watkins/ and write rows for every species clip present."""
    adapter = WatkinsAdapter(WATKINS_ROOT)
    stats = {"files_loaded": 0, "clips_written": 0}
    for clip in adapter.enumerate():
        stats["files_loaded"] += 1
        taxonomy = Taxonomy(
            category="biological",
            subclass=clip.subclass,
            species_scientific=clip.species_scientific,
        )
        provenance = Provenance(
            source_id="watkins",
            source_file=str(clip.path.relative_to(WATKINS_ROOT)),
            source_url=WATKINS_SOURCE_URL,
            original_label=clip.species_slug,
            license=WatkinsAdapter.license_name,
            collected_at=utc_now_iso(),
        )
        n = write_clips_from_samples(
            samples=clip.samples,
            sample_rate=clip.sample_rate,
            location=None,
            capture_time=None,
            event_id_prefix=f"watkins_{clip.species_slug}_{clip.path.stem}",
            taxonomy=taxonomy,
            provenance=provenance,
            analyzer=analyzer,
            writer=writer,
            spec_dir=spec_dir,
        )
        stats["clips_written"] += n
    return stats


# ---------------------------------------------------------------------------
# Shared chunking + row writer
# ---------------------------------------------------------------------------

def write_clips_from_samples(
    samples: np.ndarray,
    sample_rate: int,
    location: GeoPoint | None,
    capture_time: datetime | None,
    event_id_prefix: str,
    taxonomy: Taxonomy,
    provenance: Provenance,
    analyzer: AudioAnalyzer,
    writer: TrainingJsonlWriter,
    spec_dir: Path,
) -> int:
    """Split samples into CHUNK_SECONDS pieces, analyze, write. Returns n written."""
    chunk_len = CHUNK_SECONDS * sample_rate
    n_chunks = len(samples) // chunk_len
    if n_chunks == 0:
        return 0

    written = 0
    label = binary_label_for(taxonomy)
    for idx in range(n_chunks):
        event_id = f"{event_id_prefix}_{idx}"
        if writer.already_written(event_id):
            continue

        chunk = samples[idx * chunk_len : (idx + 1) * chunk_len]
        chunk_start = (
            capture_time + timedelta(seconds=idx * CHUNK_SECONDS)
            if capture_time else datetime.now(timezone.utc)
        )
        sub_segment = AudioSegment(
            source_file=f"{provenance.source_file}#chunk={idx}",
            location=location or MONTEREY_CANYON,
            time_window=TimeWindow(
                start=chunk_start,
                end=chunk_start + timedelta(seconds=CHUNK_SECONDS),
            ),
            sample_rate=sample_rate,
            samples=chunk.astype(np.float32, copy=False),
        )

        try:
            analyzed, features_dict = analyzer.analyze(sub_segment)
        except Exception as e:
            log.warning("analyzer_failed", event_id=event_id, error=str(e))
            continue

        spec_path = spec_dir / f"{event_id}.npy"
        np.save(spec_path, analyzed.spectrogram)

        row = TrainingRow(
            event_id=event_id,
            timestamp=utc_now_iso(),
            spectrogram_path=str(spec_path),
            representation_version=REPRESENTATION_VERSION,
            label=label,
            taxonomy=taxonomy,
            provenance=provenance,
            audio=AudioMeta(
                duration_s=float(CHUNK_SECONDS),
                sample_rate=sample_rate,
                location=location,
                capture_time=chunk_start.isoformat() if capture_time else None,
            ),
            features=AcousticFeatures.from_analyzer_dict(features_dict),
        )
        writer.write(row)
        written += 1
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+",
                    default=["mbari", "watkins"],
                    choices=["mbari", "watkins"])
    ap.add_argument("--mbari-chunks", type=int, default=100,
                    help="Number of 60s MBARI chunks (~12 clips each).")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, cap mbari-chunks to this (pilot mode).")
    args = ap.parse_args()

    settings = Settings()
    analyzer = AudioAnalyzer(settings)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    writer = TrainingJsonlWriter(JSONL_PATH)

    log.info("bootstrap_not_ship_started",
             sources=args.sources, already_written=writer.seen_count)

    results: dict[str, dict] = {}
    if "mbari" in args.sources:
        n_chunks = (min(args.mbari_chunks, args.limit)
                    if args.limit else args.mbari_chunks)
        results["mbari"] = asyncio.run(
            collect_mbari(n_chunks, analyzer, writer, SPEC_DIR)
        )
    if "watkins" in args.sources:
        results["watkins"] = collect_watkins(analyzer, writer, SPEC_DIR)

    print("\n=== not_ship bootstrap summary ===")
    for src, stats in results.items():
        print(f"  {src:10s}  {stats}")
    print(f"  total rows in jsonl now: {writer.seen_count}")


if __name__ == "__main__":
    main()
