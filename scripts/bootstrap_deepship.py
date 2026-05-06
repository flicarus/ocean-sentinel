"""Bootstrap DeepShip ship-class training data for v7.

DeepShip provides a NEW hydrophone (IcListen AF) and NEW geography
(Strait of Georgia, BC) — both LOHO-relevant axes that v6 has never seen.
Adding it to the v5 base addresses the 50% ship-class held-out gap on
ShipsEar without needing the full 20.2 GB email-gated dataset.

Usage:
    PYTHONPATH=src venv/bin/python scripts/bootstrap_deepship.py
    PYTHONPATH=src venv/bin/python scripts/bootstrap_deepship.py \
        --max-per-class 125 --jsonl data/training/gemma_labels.v7.jsonl

Default --max-per-class=125 evenly samples ~500 ship chunks across the
4 vessel classes — enough to add diversity without flipping the ship /
not_ship balance past 65/35.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import structlog

from ocean_sentinel.adapters.deepship import (
    CLASS_FOLDERS, DEEPSHIP_SOURCE_URL, DeepShipAdapter,
)
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

DEEPSHIP_ROOT = Path("data/deepship")
SPEC_DIR = Path("data/spectrograms")
DEFAULT_JSONL = Path("data/training/gemma_labels.v7.jsonl")
CHUNK_SECONDS = 5
REPRESENTATION_VERSION = "abs_db_v1"

# Strait of Georgia delta node — Ocean Networks Canada public hydrophone.
SOG_DELTA_NODE = GeoPoint(lat=49.13, lon=-123.32)


def collect_deepship(
    analyzer: AudioAnalyzer,
    writer: TrainingJsonlWriter,
    spec_dir: Path,
    max_per_class: int,
) -> dict[str, dict]:
    """Walk DeepShip class folders, chunk into 5s pieces, write up to
    `max_per_class` chunks per vessel class. Returns per-class stats."""

    adapter = DeepShipAdapter(DEEPSHIP_ROOT)
    stats: dict[str, dict] = {
        c: {"files": 0, "chunks_written": 0}
        for c in CLASS_FOLDERS.values()
    }

    for clip in adapter.enumerate():
        cls = clip.vessel_class
        stats[cls]["files"] += 1
        sr = clip.sample_rate
        chunk_len = CHUNK_SECONDS * sr

        if len(clip.samples) < chunk_len:
            continue

        if stats[cls]["chunks_written"] >= max_per_class:
            log.info("deepship_class_cap_reached", vessel_class=cls)
            continue

        taxonomy = Taxonomy(
            category="vessel",
            subclass=f"deepship_{cls}",
        )
        provenance = Provenance(
            source_id="deepship",
            source_file=str(clip.path.relative_to(DEEPSHIP_ROOT)),
            source_url=DEEPSHIP_SOURCE_URL,
            original_label=cls,
            license=DeepShipAdapter.license_name,
            collected_at=utc_now_iso(),
        )

        n_chunks = len(clip.samples) // chunk_len
        written = 0
        for idx in range(n_chunks):
            if stats[cls]["chunks_written"] >= max_per_class:
                break
            event_id = f"deepship_{cls}_{clip.path.stem}_{idx}"
            if writer.already_written(event_id):
                continue
            chunk = clip.samples[idx * chunk_len : (idx + 1) * chunk_len]

            sub_segment = AudioSegment(
                source_file=f"{provenance.source_file}#chunk={idx}",
                location=SOG_DELTA_NODE,
                time_window=TimeWindow(
                    start=datetime.now(timezone.utc),
                    end=datetime.now(timezone.utc)
                        + timedelta(seconds=CHUNK_SECONDS),
                ),
                sample_rate=sr,
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
                label=binary_label_for(taxonomy),
                taxonomy=taxonomy,
                provenance=provenance,
                audio=AudioMeta(
                    duration_s=float(CHUNK_SECONDS),
                    sample_rate=sr,
                    location=SOG_DELTA_NODE,
                    capture_time=None,
                ),
                features=AcousticFeatures.from_analyzer_dict(features_dict),
            )
            writer.write(row)
            written += 1
            stats[cls]["chunks_written"] += 1

        log.info(
            "deepship_file_done",
            vessel_class=cls,
            file=clip.path.name,
            chunks_written=written,
        )

    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--max-per-class", type=int, default=125,
                    help="Cap chunks per vessel class (default 125 → ~500 total).")
    args = ap.parse_args()

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    analyzer = AudioAnalyzer(settings)
    writer = TrainingJsonlWriter(args.jsonl)

    log.info(
        "bootstrap_deepship_started",
        jsonl=str(args.jsonl),
        already_written=writer.seen_count,
        max_per_class=args.max_per_class,
    )

    stats = collect_deepship(
        analyzer, writer, SPEC_DIR, args.max_per_class,
    )
    total = sum(s["chunks_written"] for s in stats.values())

    log.info("bootstrap_deepship_done", per_class=stats, total_chunks=total)
    print("Per-class:")
    for cls, s in stats.items():
        print(f"  {cls:10s}: {s['files']:3d} files → {s['chunks_written']:4d} chunks")
    print(f"Total chunks written: {total}")
    print(f"Output: {args.jsonl}")


if __name__ == "__main__":
    main()
