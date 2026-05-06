"""Threat classifier orchestration — wires the decision pipeline together.

Replaces the previous Gemma-as-decision-maker design. Now the verdict is
produced by the deterministic Layer 1-6 stack (sliding window, gates,
conformal, AIS, engine), so every decision has reproducible reasoning
and an immutable provenance record. Gemma is no longer in the critical
path; if we re-introduce it later it will be as a narrative/explanation
layer on top of the engine output, not a verdict source.
"""
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
from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier
from ocean_sentinel.decision import (
    DecisionEngine,
    evaluate_gates,
    predict_windowed,
)

# `correlator` was the legacy Gemma-era hook that lazily fetched AIS gaps
# + ocean conditions. AIS in the new pipeline is queried by DecisionEngine
# itself when (and only when) the decision tree needs it; ocean conditions
# aren't part of the deterministic tier and would belong to a future
# environmental-corroboration gate. We keep the parameter type alias so
# existing callers compile, but it's effectively unused.
CorrelationCallback = Callable[
    [], Awaitable[tuple[list[AISGapEvent], OceanConditions | None]]
]

log = structlog.get_logger()


class ThreatClassifierService:
    """Pipeline orchestration:
        Layer 1: CNN sliding-window inference  (decision/window)
        Layer 2: deterministic acoustic gates  (decision/gates)
        Layer 3: conformal lower bound         (decision/conformal)
        Layer 4: AIS cross-check via GFW       (decision/engine)
        Layer 5: tier decision                 (decision/engine)
        Layer 6: provenance audit record       (decision/engine)
    """

    # Fast-path — skip the rest of the stack for confidently quiet chunks.
    # Uses v7's evidential `mean_uncertainty` because raw confidence is
    # saturated (~0.975 constant). MBARI training ambient sits at 0.044-
    # 0.054; sanctsound corrected (held-out) reaches 0.296. Threshold
    # 0.05 catches the easy ambient majority.
    SKIP_NOT_SHIP_UNCERTAINTY = 0.05

    def __init__(
        self,
        analyzer: AudioAnalyzer,
        cnn: CNNV7Classifier,
        engine: DecisionEngine,
    ) -> None:
        self._analyzer = analyzer
        self._cnn = cnn
        self._engine = engine

    async def classify(
        self,
        audio: AudioSegment,
        ais_gaps: list[AISGapEvent] | None = None,
        ocean: OceanConditions | None = None,
        correlator: CorrelationCallback | None = None,
    ) -> ClassificationResult:
        """Run the full Layer 1-6 pipeline on one AudioSegment.

        `ais_gaps` / `ocean` / `correlator` are accepted for backwards
        compatibility with the previous Gemma-era signature but are not
        consumed — AIS is fetched inside DecisionEngine on demand.
        """
        del ais_gaps, ocean, correlator  # see class docstring

        analyzed_audio, features = self._analyzer.analyze(audio)

        # --- Layer 1: sliding-window inference ----------------------
        windowed = predict_windowed(
            self._cnn, analyzed_audio.spectrogram,
            source_id=audio.source_id,
        )

        # --- Fast-path: confident quiet ------------------------------
        if (
            windowed.label == "not_ship"
            and windowed.mean_uncertainty < self.SKIP_NOT_SHIP_UNCERTAINTY
        ):
            log.info(
                "fast_path_quiet",
                source=audio.source_file,
                mean_uncertainty=windowed.mean_uncertainty,
                ship_fraction=windowed.ship_fraction,
            )
            return ClassificationResult(
                threat_level=ThreatLevel.NONE,
                confidence=windowed.confidence,
                reasoning=(
                    f"CNN sliding-window: not_ship "
                    f"(mean uncertainty {windowed.mean_uncertainty:.3f} "
                    f"< {self.SKIP_NOT_SHIP_UNCERTAINTY}). "
                    f"Decision engine bypassed."
                ),
                raw_output={
                    "windowed": windowed.as_dict(),
                    "fast_path": True,
                },
            )

        # --- Layer 2: gates -----------------------------------------
        gates_report = evaluate_gates(features, windowed)

        # --- Layer 3-6: engine (conformal, AIS, tier, provenance) ---
        decision, provenance = await self._engine.decide(
            spectrogram=analyzed_audio.spectrogram,
            windowed=windowed,
            gates_report=gates_report,
            location=analyzed_audio.location,
            capture_time=analyzed_audio.time_window.start,
            hydrophone_id=audio.source_id,
        )

        log.info(
            "classification_complete",
            source=audio.source_file,
            threat_level=decision.threat_level.value,
            tier=decision.tier_label,
            confidence=decision.confidence,
            requires_review=decision.requires_review,
        )

        return ClassificationResult(
            threat_level=decision.threat_level,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            raw_output={
                "decision": {
                    "tier_label": decision.tier_label,
                    "requires_review": decision.requires_review,
                },
                "windowed": windowed.as_dict(),
                "gates": gates_report.to_dict(),
                "provenance": provenance.to_dict(),
            },
        )
