from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from ocean_sentinel.domain.models import (
    AcousticEntry,
    AcousticFeatures,
    AISGapEvent,
    Alert,
    AudioSegment,
    ClassificationResult,
    DetectionEvent,
    GeoPoint,
    OceanConditions,
    SimilarMatch,
    TimeWindow,
    ThreatLevel,
)

class HydrophoneSource(Protocol):
    """Port: fetches hydrophone audio from a public archive.

    Each implementation (MBARI, Orcasound, NOAA PMEL) wraps one network's
    archive format and exposes the same offset-based fetch so the pipeline
    can iterate uniformly across sources.
    """

    location: GeoPoint
    source_id: str

    async def fetch_at_offset(
            self,
            dt: datetime,
            offset_seconds: int,
            duration_seconds: int = 60,
    ) -> AudioSegment: ...

    async def close(self) -> None: ... 

class VesselTracker(Protocol):
    """Port: finds vessels that went dark in a region/time window (GFW)."""

    async def get_ais_gaps(
            self, region: GeoPoint, radius_km: float, time_window: TimeWindow
    ) -> list[AISGapEvent]: ...



class OceanDataSource(Protocol):
    """Port: environmental context for a location/time (Copernicus)."""

    async def get_conditions(
            self, location: GeoPoint, timestamp: datetime
    ) -> OceanConditions: ...


class ThreatClassifier(Protocol):
    """Port: multimodal classification - spectrogram  + AIS + ocean -> verdict."""


    async def classify( 
        self,
        audio: AudioSegment,
        ais_gaps: list[AISGapEvent],
        ocean: OceanConditions | None,
    ) -> ClassificationResult: ...


class AlertSender(Protocol):
    """Port: delivers alert via specific channel (email, SMS, webhook)."""
    
    async def send(self, alert: Alert, event: DetectionEvent) -> Alert: ...


class EventStore(Protocol):
    """Port: persistence for detection events and alerts."""

    async def save_event(self, event: DetectionEvent) -> None: ...

    async def get_event(self, event_id: str) -> DetectionEvent | None: ...

    async def list_events(
            self,
            time_window: TimeWindow | None = None,
            min_threat: ThreatLevel | None = None,
    ) -> list[DetectionEvent]: ...

    async def save_alert(self, alert: Alert) -> None: ...


class AcousticMemory(Protocol):
    """Port: vector store for acoustic signatures — enables RAG retrieval
    of similar past classifications to improve Gemma's context."""

    async def store(self, entry: AcousticEntry) -> None:
        """Persist a classified acoustic entry with its feature embedding."""
        ...

    async def query_similar(
        self, features: AcousticFeatures, n: int = 3,
    ) -> list[SimilarMatch]:
        """Find the N most acoustically similar past classifications."""
        ...

    async def count(self) -> int:
        """Total entries in the memory."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...

class TrainingLogger(Protocol):
    """Port: logs (input, output) pairs from Gemma classifications
    as training data for CNN fine-tuning.
    """

    async def log(
        self,
        event_id: str,
        spectrogram_path: str,
        features: AcousticFeatures,
        context_text: str,
        gemma_verdict: dict[str, Any],
    ) -> None: ...
