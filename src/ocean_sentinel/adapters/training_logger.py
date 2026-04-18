from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import structlog

from ocean_sentinel.domain.models import AcousticFeatures

log = structlog.get_logger()


class JSONLTrainingLogger:
    """Append-only JSONL writer that accumulates (input → Gemma verdict) pairs.

    Each line is a self-contained training example. The CNN training script
    reads this file directly — no parsing, no database, just line-by-line JSON.
    """

    def __init__(self, output_dir: str = "data/training") -> None:
        self._path = Path(output_dir)
        self._path.mkdir(parents=True, exist_ok=True)
        self._file = self._path / "gemma_labels.jsonl"

    async def log(
        self,
        event_id: str,
        spectrogram_path: str,
        features: AcousticFeatures,
        context_text: str,
        gemma_verdict: dict,
    ) -> None:
        entry = {
            "event_id": event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "spectrogram_path": spectrogram_path,
            "features": {
                "engine_band_ratio": features.engine_band_ratio,
                "peak_frequency_hz": features.peak_frequency_hz,
                "spectral_flatness": features.spectral_flatness,
                "rms_energy": features.rms_energy,
                "engine_band_energy_db": features.engine_band_energy_db,
            },
            "context_text": context_text,
            "gemma_verdict": gemma_verdict,
        }

        with open(self._file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        log.info(
            "training_example_logged",
            event_id=event_id,
            threat_level=gemma_verdict.get("threat_level"),
            file=str(self._file),
        )
