from __future__ import annotations

from typing import Awaitable, Callable

import structlog

from ocean_sentinel.domain.enums import ThreatLevel
from ocean_sentinel.domain.models import (
    AISGapEvent,
    AudioSegment,
    ClassificationResult,
    OceanConditions,
)
from ocean_sentinel.services.audio_analyzer import AudioAnalyzer
from ocean_sentinel.services.cnn_classifier import CNNClassifier
from ocean_sentinel.adapters.gemma import GemmaAdapter

CorrelationCallback = Callable[
    [], Awaitable[tuple[list[AISGapEvent], OceanConditions | None]]
]

log = structlog.get_logger()


class ThreatClassifierService:
    """Three-layer classification orchestration:
        Tier 1 — CNN binary triage on spectrogram.
        Tier 2 — RAG (handled inside GemmaAdapter via AcousticMemory).
        Tier 3 — Gemma threat-level reasoning over correlated evidence.
    """

    # If the CNN is at-or-above this confidence on `not_ship`, skip Gemma —
    # not_ship → ThreatLevel.NONE by definition, and a confident CNN call
    # saves an Ollama round-trip per quiet chunk.
    #
    # Threshold picked from scripts/sweep_threshold.py on calibrated v6:
    #   τ=0.98 → 13.4% coverage, 0/701 ships missed on val set.
    # Lower τ trades safety for coverage; never go below ~0.95 without
    # rerunning the sweep (τ=0.85 lost 31/701 ships ≈ 4.4% recall hit).
    SKIP_GEMMA_NOT_SHIP_CONFIDENCE = 0.98

    def __init__(
        self,
        analyzer: AudioAnalyzer,
        gemma: GemmaAdapter,
        cnn: CNNClassifier | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._gemma = gemma
        self._cnn = cnn

    async def classify(
        self,
        audio: AudioSegment,
        ais_gaps: list[AISGapEvent] | None = None,
        ocean: OceanConditions | None = None,
        correlator: CorrelationCallback | None = None,
    ) -> ClassificationResult:
        """Full pipeline: audio → spectrogram → CNN → Gemma → verdict.

        `correlator`, if provided, is invoked lazily AFTER the CNN tier-1
        decides Gemma is needed — so chunks short-circuited by the fast
        path don't pay for an AIS / ocean fetch they won't use.
        """

        analyzed_audio, features = self._analyzer.analyze(audio)

        cnn_verdict: dict | None = None
        if self._cnn is not None:
            cnn_verdict = self._cnn.predict(
                analyzed_audio.spectrogram,
                source_id=audio.source_id,
            )
            log.info(
                "cnn_verdict",
                source=audio.source_file,
                source_id=audio.source_id,
                label=cnn_verdict["label"],
                confidence=cnn_verdict["confidence"],
                profile_applied=cnn_verdict.get("profile_applied"),
            )

            if (
                cnn_verdict["label"] == "not_ship"
                and cnn_verdict["confidence"] >= self.SKIP_GEMMA_NOT_SHIP_CONFIDENCE
            ):
                log.info(
                    "gemma_skipped_cnn_fast_path",
                    source=audio.source_file,
                    confidence=cnn_verdict["confidence"],
                )
                return ClassificationResult(
                    threat_level=ThreatLevel.NONE,
                    confidence=cnn_verdict["confidence"],
                    reasoning=(
                        f"CNN tier-1: not_ship "
                        f"(confidence {cnn_verdict['confidence']:.0%}). "
                        f"Gemma escalation skipped."
                    ),
                    raw_output={"cnn": cnn_verdict, "gemma_skipped": True},
                )

        # Lazy correlation: only fetch AIS / ocean if Gemma is actually
        # going to run. Saves ~13% of GFW calls for not_ship fast-path
        # chunks (measured on val set, τ=0.98).
        if ais_gaps is None and correlator is not None:
            ais_gaps, ocean = await correlator()
        if ais_gaps is None:
            ais_gaps = []

        log.info(
            "classification_started",
            source=audio.source_file,
            ais_gaps=len(ais_gaps),
            engine_dominant=features["is_engine_band_dominant"],
            cnn_label=cnn_verdict["label"] if cnn_verdict else None,
        )

        result = await self._gemma.classify(
            audio=analyzed_audio,
            ais_gaps=ais_gaps,
            ocean=ocean,
            features=features,
            cnn_verdict=cnn_verdict,
        )

        if cnn_verdict is not None:
            # Surface CNN evidence alongside Gemma's verdict for downstream
            # consumers (training logger, dashboard, debugging).
            result.raw_output["cnn"] = cnn_verdict

        log.info(
            "classification_complete",
            source=audio.source_file,
            threat_level=result.threat_level.value,
            confidence=result.confidence,
        )

        return result
