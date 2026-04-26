from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np 

@dataclass(frozen=True, slots=True)
class GeoPoint:
    """Latitude/longitude coordinate."""
    lat: float
    lon: float

@dataclass(frozen=True, slots=True)
class TimeWindow:
    """Time range for queries — 'give me data between start and end'."""
    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class AudioSegment:
    """Chunk of hydrophone audio — raw samples + optional spectrogram.

    `source_id` matches the adapter's `source_id` attribute and lets the CNN
    tier-1 classifier look up a per-source frequency profile when one was
    cached at training time.
    """
    source_file: str
    location: GeoPoint
    time_window: TimeWindow
    sample_rate: int
    samples: np.ndarray
    spectrogram: np.ndarray | None = None
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class NearbyVessel:
    """A vessel actively broadcasting AIS near a hydrophone."""
    vessel_id: str
    vessel_name: str | None
    vessel_class: str | None        # "fishing", "cargo", "tanker", "passenger", etc.
    flag_state: str | None
    position: GeoPoint
    distance_km: float
    length_m: float | None          # vessel length from GFW registry
    present_start: datetime | None = None   # entry timestamp from GFW (hourly resolution)
    present_end: datetime | None = None     # exit timestamp from GFW


@dataclass(frozen=True, slots=True)
class AISGapEvent:
    """Vessel that went dark — stopped transmitting AIS."""
    vessel_id: str
    vessel_name: str | None
    flag_state: str | None
    last_known_position: GeoPoint
    gap_start: datetime
    gap_end: datetime | None          # None = still dark
    gap_duration_hours: float
    intentional_disabling: bool | None = None   # GFW flag if available
    in_mpa: bool | None = None                  # was vessel in Marine Protected Area


@dataclass(frozen=True, slots=True)
class OceanConditions:
    """Environmental context from Copernicus — currents, temperature."""
    location: GeoPoint
    timestamp: datetime
    sea_surface_temp_c: float | None
    current_speed_ms: float | None
    current_direction_deg: float | None



from ocean_sentinel.domain.enums import AlertChannel, AlertStatus, ThreatLevel


# ---------------------------------------------------------------------------
# Acoustic memory — domain models for the self-improving RAG loop
# ---------------------------------------------------------------------------

# Normalization bounds — acoustic domain knowledge.
# Defines the meaningful range for each feature so cosine similarity
# treats all dimensions equally. Module-level constant, not on the dataclass.
_ACOUSTIC_BOUNDS: dict[str, tuple[float, float]] = {
    "engine_band_ratio":     (0.5, 3.0),      # ratio rarely exceeds 3×
    "peak_frequency_hz":     (0.0, 1000.0),    # mel spec capped at 1 kHz
    "spectral_flatness":     (0.0, 1.0),       # 0 = pure tone, 1 = white noise
    "rms_energy":            (0.0, 0.1),        # typical hydrophone range
    "engine_band_energy_db": (-80.0, 0.0),      # dB scale
}


@dataclass(frozen=True, slots=True)
class AcousticFeatures:
    """Typed representation of audio features extracted by AudioAnalyzer."""

    engine_band_ratio: float
    peak_frequency_hz: float
    spectral_flatness: float
    rms_energy: float
    engine_band_energy_db: float

    @classmethod
    def from_analyzer_dict(cls, features: dict[str, Any]) -> AcousticFeatures:
        """Construct from the raw dict returned by AudioAnalyzer.extract_features()."""
        return cls(
            engine_band_ratio=features["engine_band_ratio"],
            peak_frequency_hz=features["peak_frequency_hz"],
            spectral_flatness=features["spectral_flatness"],
            rms_energy=features["rms_energy"],
            engine_band_energy_db=features["engine_band_energy_db"],
        )

    def to_vector(self) -> list[float]:
        """Normalize all features to [0, 1] and return as a flat list.

        Order is deterministic — defined by _ACOUSTIC_BOUNDS iteration order
        (dict is insertion-ordered since 3.7).
        """
        vector: list[float] = []
        for field_name, (lo, hi) in _ACOUSTIC_BOUNDS.items():
            raw = getattr(self, field_name)
            normalized = (raw - lo) / (hi - lo) if hi != lo else 0.0
            vector.append(max(0.0, min(1.0, normalized)))
        return vector


@dataclass(frozen=True, slots=True)
class AcousticEntry:
    """Everything stored per classification in the acoustic memory.

    This is the unit of knowledge the system accumulates — one entry
    per Gemma classification, linking the acoustic signature to the
    model's verdict and all contextual evidence.

    `embedding` is the 64-dim shared backbone vector from the CNN. When
    present, it is used as the ChromaDB vector key — far more discriminative
    than the 5-dim hand-crafted `features` vector for retrieval.
    """

    event_id: str
    timestamp: datetime
    location: GeoPoint
    features: AcousticFeatures
    context_text: str          # the text prompt sent to Gemma
    threat_level: ThreatLevel
    confidence: float
    reasoning: str
    vessel_type: str | None
    recommended_action: str | None
    embedding: list[float] | None = None


@dataclass(frozen=True, slots=True)
class SimilarMatch:
    """Result from an acoustic similarity query.

    score: 0.0 = identical acoustic signature, 1.0 = maximally different.
    (ChromaDB cosine distance, not cosine similarity — lower is better.)
    """

    entry: AcousticEntry
    score: float


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """What Gemma returned after analyzing correlated evidence."""
    threat_level: ThreatLevel
    confidence: float                  # 0.0 - 1.0
    reasoning: str                     # Gemma's explanation
    raw_output: dict[str, Any]         # full model response for debugging


@dataclass(frozen=True, slots=True)
class DetectionEvent:
    """Final output — one detected incident with all evidence bundled."""
    id: str                            # UUID
    timestamp: datetime
    location: GeoPoint
    threat_level: ThreatLevel
    confidence: float
    classification_reasoning: str
    audio_segment: AudioSegment | None
    ais_gaps: list[AISGapEvent]
    ocean_conditions: OceanConditions | None
    raw_model_output: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Alert:
    """Notification sent to coast guard / NGO."""
    id: str                            # UUID
    event_id: str                      # links to DetectionEvent
    channel: AlertChannel
    recipient: str
    sent_at: datetime | None
    status: AlertStatus
    failure_reason: str | None = None
