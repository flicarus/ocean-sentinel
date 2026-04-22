"""Migrate gemma_labels.jsonl rows to the new rich schema.

Walks an existing JSONL file and writes every row to a fresh output in the
canonical new shape (TrainingRow). Doesn't touch spectrograms on disk —
pure row-shape rewrite. Safe to diff old vs new before swapping.

Row routing:
  - New schema already (top-level "label" + "taxonomy") -> pass through verbatim
  - Old ShipsEar row (gemma_verdict.shipsear_class present)  -> migrate
  - Anything else (legacy Day-5 empty gemma_verdict, etc.)   -> skip, counted

Usage:
    PYTHONPATH=src venv/bin/python scripts/migrate_jsonl.py \\
        --input  data/training/gemma_labels.jsonl \\
        --output data/training/gemma_labels.v2.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import structlog

from ocean_sentinel.domain.models import AcousticFeatures
from ocean_sentinel.training.labels import binary_label_for
from ocean_sentinel.training.schema import (
    AudioMeta, Provenance, Taxonomy, TrainingRow, utc_now_iso,
)

log = structlog.get_logger()

SHIPSEAR_SOURCE_URL = "https://underwaternoise.atlanttic.uvigo.es/"
SHIPSEAR_LICENSE = "CC-BY-NC-4.0"
SHIPSEAR_SAMPLE_RATE = 16000
CHUNK_SECONDS = 5.0


def _parse_source_file(context_text: str) -> str:
    """Extract source_file from 'shipsear-bootstrap | class=A | source_file=...'."""
    for part in context_text.split("|"):
        part = part.strip()
        if part.startswith("source_file="):
            return part[len("source_file="):]
    return ""


def _shipsear_taxonomy(cls: str) -> Taxonomy:
    """Map ShipsEar letter class (A-E) to new taxonomy."""
    if cls in {"A", "B", "C", "D"}:
        return Taxonomy(
            category="vessel",
            subclass=f"shipsear_class_{cls.lower()}",
            shipsear_class=cls,
        )
    if cls == "E":
        return Taxonomy(
            category="ambient",
            subclass="harbor_ambient",
            shipsear_class="E",
        )
    raise ValueError(f"Unknown ShipsEar class: {cls!r}")


def _migrate_shipsear_row(row: dict) -> TrainingRow:
    """Build a new-schema TrainingRow from an old ShipsEar JSONL row."""
    cls = row["gemma_verdict"]["shipsear_class"]
    taxonomy = _shipsear_taxonomy(cls)
    source_file = _parse_source_file(row.get("context_text", ""))
    return TrainingRow(
        event_id=row["event_id"],
        timestamp=row.get("timestamp", utc_now_iso()),
        spectrogram_path=row["spectrogram_path"],
        representation_version=row.get("representation_version", "abs_db_v1"),
        label=binary_label_for(taxonomy),
        taxonomy=taxonomy,
        provenance=Provenance(
            source_id="shipsear",
            source_file=source_file,
            source_url=SHIPSEAR_SOURCE_URL,
            original_label=cls,
            license=SHIPSEAR_LICENSE,
            collected_at=row.get("timestamp", utc_now_iso()),
        ),
        audio=AudioMeta(
            duration_s=CHUNK_SECONDS,
            sample_rate=SHIPSEAR_SAMPLE_RATE,
            location=None,
            capture_time=None,
        ),
        features=AcousticFeatures.from_analyzer_dict(row.get("features", {})),
    )


def _is_new_schema(row: dict) -> bool:
    return "label" in row and "taxonomy" in row


def _is_old_shipsear(row: dict) -> bool:
    return "shipsear_class" in row.get("gemma_verdict", {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("data/training/gemma_labels.jsonl"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/training/gemma_labels.v2.jsonl"))
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    seen: set[str] = set()

    with args.output.open("w") as out:
        for raw_line in args.input.read_text().splitlines():
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                stats["malformed"] += 1
                continue

            event_id = row.get("event_id")
            if event_id and event_id in seen:
                stats["duplicate"] += 1
                continue

            if _is_new_schema(row):
                out.write(raw_line + "\n")
                seen.add(event_id)
                stats[f"passthrough_{row['label']}"] += 1
                continue

            if _is_old_shipsear(row):
                try:
                    new_row = _migrate_shipsear_row(row)
                except Exception as e:
                    stats["migrate_failed"] += 1
                    log.warning("migrate_failed",
                                event_id=event_id, error=str(e))
                    continue
                out.write(json.dumps(new_row.to_jsonl_dict()) + "\n")
                seen.add(new_row.event_id)
                stats[f"migrated_{new_row.label}"] += 1
                continue

            stats["skipped_legacy"] += 1

    print("\n=== JSONL migration summary ===")
    for k, v in sorted(stats.items()):
        print(f"  {k:25s} {v}")
    print(f"  {'total_output_rows':25s} {sum(stats[k] for k in stats if k.startswith(('passthrough_', 'migrated_')))}")
    print(f"\n  input:  {args.input}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
