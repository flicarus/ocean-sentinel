from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from supabase import Client, create_client

from ocean_sentinel.domain.models import AcousticFeatures

log = structlog.get_logger()


class SupabaseTrainingLogger:
    """Saves training pairs to Supabase Postgres + Storage.

    Each classification produces two things:
    - A row in the `training_pairs` table (metadata + Gemma verdict)
    - The .npy spectrogram file uploaded to the `spectrograms` bucket

    The storage key is deterministic ({source_id}/{event_id}.npy) so
    re-running the same event is safe — it overwrites rather than duplicates.
    """

    def __init__(
        self,
        url: str,
        service_role_key: str,
        bucket: str = "spectrograms",
    ) -> None:
        self._client: Client = create_client(url, service_role_key)
        self._bucket = bucket

    async def log(
        self,
        event_id: str,
        spectrogram_path: str,
        features: AcousticFeatures,
        context_text: str,
        gemma_verdict: dict[str, Any],
        source_id: str = "unknown",
        ground_truth_label: str | None = None,
        ground_truth_source: str | None = None,
    ) -> None:
        # 1. Upload the .npy spectrogram file to Supabase Storage
        local_path = Path(spectrogram_path)
        storage_key = f"{source_id}/{local_path.name}"

        if local_path.exists():
            with local_path.open("rb") as f:
                self._client.storage.from_(self._bucket).upload(
                    path=storage_key,
                    file=f.read(),
                    file_options={
                        "contentType": "application/octet-stream",
                        "upsert": "true",
                    },
                )
        else:
            log.warning("spectrogram_file_missing", path=spectrogram_path)

        # 2. Insert a row into the training_pairs table
        payload = {
            "event_id": event_id,
            "source_id": source_id,
            "orig_path": spectrogram_path,
            "spectrogram_bucket": self._bucket,
            "spectrogram_key": storage_key,
            "features": {
                "engine_band_ratio": features.engine_band_ratio,
                "peak_frequency_hz": features.peak_frequency_hz,
                "spectral_flatness": features.spectral_flatness,
                "rms_energy": features.rms_energy,
                "engine_band_energy_db": features.engine_band_energy_db,
            },
            "context_text": context_text,
            "gemma_verdict": gemma_verdict,
            "ground_truth_label": ground_truth_label,
            "ground_truth_source": ground_truth_source,
        }
        self._client.table("training_pairs").insert(payload).execute()

        log.info(
            "training_pair_logged",
            event_id=event_id,
            threat_level=gemma_verdict.get("threat_level"),
            storage_key=storage_key,
        )
