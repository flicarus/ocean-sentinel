from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog

from ocean_sentinel.config import Settings
from ocean_sentinel.domain.enums import ThreatLevel
from ocean_sentinel.domain.models import (
    AudioSegment,
    DetectionEvent,
    GeoPoint,
    TimeWindow,
)
from ocean_sentinel.domain.protocols import HydrophoneSource
from ocean_sentinel.services.audio_analyzer import AudioAnalyzer
from ocean_sentinel.services.correlation import CorrelationService
from ocean_sentinel.services.classifier import ThreatClassifierService

log = structlog.get_logger()


class Pipeline:
    """End-to-end orchestrator: fetch audio → analyze → correlate → classify → event."""

    def __init__(
        self,
        hydrophone: HydrophoneSource,
        analyzer: AudioAnalyzer,
        correlation: CorrelationService,
        classifier: ThreatClassifierService,
    ) -> None:
        self._hydrophone = hydrophone
        self._analyzer = analyzer
        self._correlation = correlation
        self._classifier = classifier

    async def run(
        self,
        location: GeoPoint,
        time_window: TimeWindow,
    ) -> list[DetectionEvent]:
        """Run full pipeline for a location and time window.

        1. Fetch audio segments from hydrophone
        2. For each segment: analyze → correlate → classify
        3. Return detection events for anything above NONE
        """
        events: list[DetectionEvent] = []

        log.info(
            "pipeline_started",
            lat=location.lat,
            lon=location.lon,
            start=str(time_window.start),
            end=str(time_window.end),
        )

        # Step 1: Fetch audio
        segments = await self._hydrophone.fetch_audio(location, time_window)
        log.info("pipeline_audio_fetched", segments=len(segments))

        # Step 2: Process each segment
        for segment in segments:
            event = await self._process_segment(segment)
            if event:
                events.append(event)

        log.info(
            "pipeline_complete",
            segments_processed=len(segments),
            events_generated=len(events),
        )

        return events

    async def _process_segment(self, segment: AudioSegment) -> DetectionEvent | None:
        """Process a single audio segment through the full pipeline."""

        # Analyze audio → spectrogram + features
        analyzed, features = self._analyzer.analyze(segment)

        # Skip if nothing interesting in the audio
        if not features["is_engine_band_dominant"] and features["engine_band_ratio"] < 0.3:
            log.debug(
                "pipeline_segment_skipped",
                source=segment.source_file,
                reason="no engine signature",
            )
            return None

        # Correlate with AIS + ocean data
        ais_gaps, ocean = await self._correlation.correlate(analyzed, features)

        # Classify threat
        result = await self._classifier.classify(
            audio=analyzed,
            ais_gaps=ais_gaps,
            ocean=ocean,
        )

        # Skip non-threats
        if result.threat_level == ThreatLevel.NONE:
            return None

        # Build detection event
        return DetectionEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            location=segment.location,
            threat_level=result.threat_level,
            confidence=result.confidence,
            classification_reasoning=result.reasoning,
            audio_segment=analyzed,
            ais_gaps=ais_gaps,
            ocean_conditions=ocean,
            raw_model_output=result.raw_output,
        )
