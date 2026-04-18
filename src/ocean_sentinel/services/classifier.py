from __future__ import annotations

import structlog

from ocean_sentinel.domain.models import (
    AISGapEvent,
    AudioSegment,
    ClassificationResult,
    OceanConditions,
)
from ocean_sentinel.services.audio_analyzer import AudioAnalyzer
from ocean_sentinel.adapters.gemma import GemmaAdapter

log = structlog.get_logger()


class ThreatClassifierService:
    """Orchestrates audio analysis → Gemma classification."""

    def __init__(self, analyzer: AudioAnalyzer, gemma: GemmaAdapter) -> None:
        self._analyzer = analyzer
        self._gemma = gemma

    async def classify(
        self,
        audio: AudioSegment,
        ais_gaps: list[AISGapEvent],
        ocean: OceanConditions | None = None,
    ) -> ClassificationResult:
        """Full pipeline: raw audio → spectrogram → features → Gemma → verdict."""

        # Step 1: Generate spectrogram + extract acoustic features
        analyzed_audio, features = self._analyzer.analyze(audio)

        log.info(
            "classification_started",
            source=audio.source_file,
            ais_gaps=len(ais_gaps),
            engine_dominant=features["is_engine_band_dominant"],
        )

        # Step 2: Send everything to Gemma
        result = await self._gemma.classify(
            audio=analyzed_audio,
            ais_gaps=ais_gaps,
            ocean=ocean,
            features=features,
        )

        log.info(
            "classification_complete",
            source=audio.source_file,
            threat_level=result.threat_level.value,
            confidence=result.confidence,
        )

        return result
