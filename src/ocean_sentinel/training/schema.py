"""Training-data schema + idempotent JSONL writer.

Every row written to gemma_labels.jsonl — regardless of source — flows through
TrainingJsonlWriter.write(). This is the single source of truth for the
training format; if two bootstrappers diverge, that's a bug, not a feature.

The schema separates:
  label     — what the hackathon CNN actually trains on (binary)
  taxonomy  — fine-grained truth, reserved for future Sofar heads
              (species, threat-level, vessel-type)
  provenance — citations, licensing, reproducibility
  audio     — clip format + capture context
  features  — AudioAnalyzer numeric features (consumed by Gemma's text prompt)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ocean_sentinel.domain.models import AcousticFeatures, GeoPoint

Label = Literal["ship", "not_ship"]
Category = Literal["vessel", "biological", "ambient", "anthropogenic_other"]


@dataclass(frozen=True, slots=True)
class Taxonomy:
    """Fine-grained labels. CNN ignores this; future heads consume it."""
    category: Category
    subclass: str
    species_scientific: str | None = None
    shipsear_class: str | None = None
    threat_level_legacy: str | None = None


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where this clip came from — citation, licensing, reproducibility."""
    source_id: str
    source_file: str
    source_url: str | None = None
    original_label: str | None = None
    license: str | None = None
    collected_at: str = ""


@dataclass(frozen=True, slots=True)
class AudioMeta:
    """Clip format + capture context."""
    duration_s: float
    sample_rate: int
    location: GeoPoint | None = None
    capture_time: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingRow:
    """One row in gemma_labels.jsonl."""
    event_id: str
    timestamp: str
    spectrogram_path: str
    representation_version: str
    label: Label
    taxonomy: Taxonomy
    provenance: Provenance
    audio: AudioMeta
    features: AcousticFeatures

    def to_jsonl_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "spectrogram_path": self.spectrogram_path,
            "representation_version": self.representation_version,
            "label": self.label,
            "taxonomy": asdict(self.taxonomy),
            "provenance": asdict(self.provenance),
            "audio": {
                "duration_s": self.audio.duration_s,
                "sample_rate": self.audio.sample_rate,
                "location": asdict(self.audio.location) if self.audio.location else None,
                "capture_time": self.audio.capture_time,
            },
            "features": asdict(self.features),
        }


class TrainingJsonlWriter:
    """Append-only writer. Skips rows whose event_id is already present."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = self._load_seen()

    def _load_seen(self) -> set[str]:
        if not self._path.exists():
            return set()
        seen: set[str] = set()
        for line in self._path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                seen.add(json.loads(line)["event_id"])
            except (json.JSONDecodeError, KeyError):
                continue
        return seen

    def already_written(self, event_id: str) -> bool:
        return event_id in self._seen

    def write(self, row: TrainingRow) -> None:
        if row.event_id in self._seen:
            return
        with self._path.open("a") as f:
            f.write(json.dumps(row.to_jsonl_dict()) + "\n")
        self._seen.add(row.event_id)

    @property
    def seen_count(self) -> int:
        return len(self._seen)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
