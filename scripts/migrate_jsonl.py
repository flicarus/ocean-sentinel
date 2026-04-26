"""Migrate one-or-more gemma_labels JSONL files to the new rich schema.

Walks every input JSONL file in order, dedupes on event_id, and writes every
row to a single output file in the canonical new shape (TrainingRow).
Doesn't touch spectrograms on disk — pure row-shape rewrite.

Row routing:
  - New schema already (top-level "label" + "taxonomy") -> pass through verbatim
  - Old ShipsEar row (gemma_verdict.shipsear_class present)  -> migrate
  - Old AIS-correlated row (source_id startswith "ais-correlated-") -> migrate
  - Anything else (legacy Day-5 empty gemma_verdict, etc.)   -> skip, counted

Usage:
    # merge Maciej's AIS drop + our bootstrap, emit v3
    PYTHONPATH=src venv/bin/python scripts/migrate_jsonl.py \\
        --inputs data/training/gemma_labels.jsonl ais_only.jsonl \\
        --output data/training/gemma_labels.v3.jsonl
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

AIS_SOURCE_URL = "https://orcasound.net/"
AIS_LICENSE = "CC-BY-4.0"
AIS_SAMPLE_RATE = 16000
AIS_CHUNK_SECONDS = 60.0


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


def _is_old_ais(row: dict) -> bool:
    return str(row.get("source_id", "")).startswith("ais-correlated-")


# ---------------------------------------------------------------------------
# AIS (Orcasound + GFW) migration
# ---------------------------------------------------------------------------

def _parse_ais_context(context_text: str) -> tuple[str, str]:
    """Extract (hydrophone, window_iso) from
    'ais-correlated | hydrophone=X | window=ISO'."""
    hydrophone = ""
    window_iso = ""
    for part in context_text.split("|"):
        part = part.strip()
        if part.startswith("hydrophone="):
            hydrophone = part[len("hydrophone="):]
        elif part.startswith("window="):
            window_iso = part[len("window="):]
    return hydrophone, window_iso


def _ais_taxonomy(threat_level: str, vessel_type: str) -> Taxonomy:
    """Map AIS verdict to new taxonomy.

    HIGH/MEDIUM -> vessel (ship). NONE -> ambient (not_ship).
    vessel_type (cargo_ship, passenger_vessel, fishing_vessel, tanker, ...)
    becomes the subclass so future heads can read it.
    """
    if threat_level in {"HIGH", "MEDIUM"}:
        return Taxonomy(
            category="vessel",
            subclass=vessel_type or f"ais_{threat_level.lower()}",
            threat_level_legacy=threat_level,
        )
    if threat_level == "NONE":
        return Taxonomy(
            category="ambient",
            subclass="ais_no_vessel",
            threat_level_legacy="NONE",
        )
    raise ValueError(f"Unknown AIS threat_level: {threat_level!r}")


def _ais_session_source_file(hydrophone: str, date: str, offset: int) -> str:
    """Group AIS chunks into hour-long sessions for the session-level split.

    Same-hour windows likely share the same vessel loitering in the radius,
    so keep them on the same side of train/val. 300s stride -> ~12 chunks/hour
    -> ~24 sessions/day per hydrophone: good granularity for stratified split.
    """
    hour = offset // 3600
    return f"ais/{hydrophone}/{date}#hour={hour:02d}"


def _parse_offset_from_event_id(event_id: str) -> int:
    """event_id format: ais_{hydrophone}_{date}_{offset}. Returns offset or 0."""
    try:
        return int(event_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _migrate_ais_row(row: dict) -> TrainingRow:
    """Build a new-schema TrainingRow from an old AIS-correlated JSONL row."""
    verdict = row["gemma_verdict"]
    threat_level = verdict["threat_level"]
    vessel_type = verdict.get("vessel_type", "") or ""
    taxonomy = _ais_taxonomy(threat_level, vessel_type)

    hydrophone, window_iso = _parse_ais_context(row.get("context_text", ""))
    event_id = row["event_id"]
    # event_id = ais_{hydrophone}_{date}_{offset}  -> recover date
    parts = event_id.split("_")
    date = parts[2] if len(parts) >= 4 else ""
    offset = _parse_offset_from_event_id(event_id)
    source_file = _ais_session_source_file(hydrophone, date, offset)

    capture_time = window_iso or row.get("timestamp") or utc_now_iso()

    return TrainingRow(
        event_id=event_id,
        timestamp=row.get("timestamp", utc_now_iso()),
        spectrogram_path=row["spectrogram_path"],
        representation_version=row.get("representation_version", "abs_db_v1"),
        label=binary_label_for(taxonomy),
        taxonomy=taxonomy,
        provenance=Provenance(
            source_id=row["source_id"],  # e.g. ais-correlated-bush-point
            source_file=source_file,
            source_url=AIS_SOURCE_URL,
            original_label=threat_level,
            license=AIS_LICENSE,
            collected_at=row.get("timestamp", utc_now_iso()),
        ),
        audio=AudioMeta(
            duration_s=AIS_CHUNK_SECONDS,
            sample_rate=AIS_SAMPLE_RATE,
            location=None,
            capture_time=capture_time,
        ),
        features=AcousticFeatures.from_analyzer_dict(row.get("features", {})),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+",
                    default=[Path("data/training/gemma_labels.jsonl")],
                    help="One or more JSONL files to merge + migrate.")
    ap.add_argument("--output", type=Path,
                    default=Path("data/training/gemma_labels.v2.jsonl"))
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    seen: set[str] = set()

    with args.output.open("w") as out:
        for input_path in args.inputs:
            if not input_path.exists():
                log.warning("input_missing", path=str(input_path))
                stats[f"input_missing_{input_path.name}"] += 1
                continue
            for raw_line in input_path.read_text().splitlines():
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
                        stats["migrate_failed_shipsear"] += 1
                        log.warning("migrate_failed",
                                    event_id=event_id, error=str(e))
                        continue
                    out.write(json.dumps(new_row.to_jsonl_dict()) + "\n")
                    seen.add(new_row.event_id)
                    stats[f"migrated_shipsear_{new_row.label}"] += 1
                    continue

                if _is_old_ais(row):
                    try:
                        new_row = _migrate_ais_row(row)
                    except Exception as e:
                        stats["migrate_failed_ais"] += 1
                        log.warning("migrate_failed",
                                    event_id=event_id, error=str(e))
                        continue
                    out.write(json.dumps(new_row.to_jsonl_dict()) + "\n")
                    seen.add(new_row.event_id)
                    stats[f"migrated_ais_{new_row.label}"] += 1
                    continue

                stats["skipped_legacy"] += 1

    print("\n=== JSONL migration summary ===")
    for k, v in sorted(stats.items()):
        print(f"  {k:30s} {v}")
    total_out = sum(stats[k] for k in stats
                    if k.startswith(("passthrough_", "migrated_")))
    print(f"  {'total_output_rows':30s} {total_out}")
    print(f"\n  inputs: {[str(p) for p in args.inputs]}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
