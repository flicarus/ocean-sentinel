"""Layer 1 — sliding-window inference over long mel spectrograms.

Background
----------
v7 was trained on 5-second mel specs (TARGET_FRAMES = 313 @ ~31 fps mel).
The audio pipeline produces 60-second segments (~1876 frames). Feeding a
full 60s spec to v7's predict() center-crops it to 5s, so the model sees
~8% of the audio and the remaining 55s of evidence is silently discarded.

This module slides a 5s window across the full spec, runs predict() on
each, and aggregates the per-window verdicts into one WindowedPrediction.
Downstream layers (gates, conformal, decision tiers, provenance) consume
that aggregate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import structlog

from ocean_sentinel.services.cnn_v7_classifier import TARGET_FRAMES

log = structlog.get_logger()


# 50% overlap between adjacent windows. Standard in audio ML — captures
# events that straddle a window boundary without doubling compute relative
# to no-overlap (~2x windows but each forward pass is cheap).
DEFAULT_HOP_FRAMES = TARGET_FRAMES // 2

# Layer 5 hint: majority-ambient with one very-confident ship window is a
# candidate for human review, not silent dismissal. We don't make the call
# here, just surface a flag the decision engine can read without recomputing.
SINGLE_WINDOW_ANOMALY_MAX_FRACTION = 0.2
SINGLE_WINDOW_ANOMALY_MIN_CONFIDENCE = 0.95


class _SingleWindowClassifier(Protocol):
    """Anything with a v7-shaped predict() signature."""

    def predict(
        self, spectrogram: np.ndarray, source_id: str | None = None,
    ) -> dict: ...


@dataclass(frozen=True)
class WindowPrediction:
    """One slide's verdict — full per-window provenance for the audit log."""

    window_idx: int
    start_frame: int
    end_frame: int
    label: str
    confidence: float           # probability of predicted class
    uncertainty: float          # evidential 2 / S
    total_evidence: float       # alpha.sum() - K
    vessel_type: str
    distance: str


@dataclass(frozen=True)
class WindowedPrediction:
    """Aggregated verdict over a multi-window spectrogram.

    Top-level `label` / `confidence` / `uncertainty` mirror v7 predict()'s
    return shape so this struct is a drop-in replacement for callers that
    don't care about per-window detail (legacy classifier service, RAG
    retrieval, etc).

    Per-window detail lives in `windows`. Aggregate stats (`ship_fraction`,
    `max_ship_confidence`, `single_window_anomaly`) are decision-engine
    inputs computed once here so each downstream layer doesn't recompute.
    """

    n_windows: int
    label: str
    confidence: float
    uncertainty: float
    ship_fraction: float
    mean_uncertainty: float
    max_ship_confidence: float | None
    min_ship_confidence: float | None
    vessel_type: str
    distance: str
    embedding: list[float]
    single_window_anomaly: bool
    windows: tuple[WindowPrediction, ...] = field(default_factory=tuple)
    checkpoint: str | None = None

    def as_dict(self) -> dict:
        """Flatten to a JSON-serializable dict for callers that expect the
        v7 predict() return shape (RAG / training_logger / dashboard).
        Per-window detail is preserved under `windows`.
        """
        return {
            "label": self.label,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "n_windows": self.n_windows,
            "ship_fraction": self.ship_fraction,
            "mean_uncertainty": self.mean_uncertainty,
            "max_ship_confidence": self.max_ship_confidence,
            "min_ship_confidence": self.min_ship_confidence,
            "vessel_type": self.vessel_type,
            "distance": self.distance,
            "embedding": self.embedding,
            "single_window_anomaly": self.single_window_anomaly,
            "checkpoint": self.checkpoint,
            "windows": [
                {
                    "window_idx": w.window_idx,
                    "start_frame": w.start_frame,
                    "end_frame": w.end_frame,
                    "label": w.label,
                    "confidence": w.confidence,
                    "uncertainty": w.uncertainty,
                    "total_evidence": w.total_evidence,
                    "vessel_type": w.vessel_type,
                    "distance": w.distance,
                }
                for w in self.windows
            ],
        }


def _iter_window_ranges(
    n_frames: int, window_frames: int, hop_frames: int,
) -> list[tuple[int, int]]:
    """Returns [(start, end), ...] window ranges over a spectrogram.

    The last window is anchored flush against the tail of the spec so we
    always cover the final samples. That can produce one extra window
    whose step from its predecessor is smaller than `hop_frames` — fine,
    the classifier doesn't care, it just sees a window-wide slice.

    For specs shorter than `window_frames`, returns a single (0, n_frames)
    range; the classifier's own crop/pad handles the size mismatch.
    """
    if n_frames <= window_frames:
        return [(0, n_frames)]
    ranges: list[tuple[int, int]] = []
    start = 0
    while start + window_frames <= n_frames:
        ranges.append((start, start + window_frames))
        start += hop_frames
    last_start = n_frames - window_frames
    if not ranges or ranges[-1][0] != last_start:
        ranges.append((last_start, n_frames))
    return ranges


def predict_windowed(
    classifier: _SingleWindowClassifier,
    spectrogram: np.ndarray,
    source_id: str | None = None,
    window_frames: int = TARGET_FRAMES,
    hop_frames: int = DEFAULT_HOP_FRAMES,
) -> WindowedPrediction:
    """Slide a `window_frames`-wide window across `spectrogram`, classify
    each slice, and aggregate into a single WindowedPrediction.

    For specs at or below `window_frames`, falls back to a single inference
    on the whole spec — same behaviour as calling classifier.predict()
    directly, so it's safe on training-sized 5s specs too.
    """
    if spectrogram.ndim != 2:
        raise ValueError(
            f"Expected 2D spectrogram, got shape {spectrogram.shape}"
        )

    n_frames = spectrogram.shape[1]
    ranges = _iter_window_ranges(n_frames, window_frames, hop_frames)

    windows: list[WindowPrediction] = []
    embeddings: list[list[float]] = []
    checkpoint: str | None = None
    for idx, (start, end) in enumerate(ranges):
        slice_ = spectrogram[:, start:end]
        verdict = classifier.predict(slice_, source_id=source_id)
        embeddings.append(verdict["embedding"])
        checkpoint = verdict.get("checkpoint", checkpoint)
        windows.append(WindowPrediction(
            window_idx=idx,
            start_frame=start,
            end_frame=end,
            label=verdict["label"],
            confidence=float(verdict["confidence"]),
            uncertainty=float(verdict["uncertainty"]),
            total_evidence=float(verdict["total_evidence"]),
            vessel_type=verdict["vessel_type"],
            distance=verdict["distance"],
        ))

    n = len(windows)
    ship_windows = [w for w in windows if w.label == "ship"]
    ship_fraction = len(ship_windows) / n
    label = "ship" if ship_fraction >= 0.5 else "not_ship"

    # Top-level confidence/uncertainty: mean across windows that voted the
    # winning label. Keeps the semantic of "confidence in predicted class"
    # consistent with single-window predict().
    winning_windows = [w for w in windows if w.label == label]
    confidence = float(np.mean([w.confidence for w in winning_windows]))
    uncertainty = float(np.mean([w.uncertainty for w in winning_windows]))
    mean_uncertainty = float(np.mean([w.uncertainty for w in windows]))

    if ship_windows:
        max_ship_conf = float(max(w.confidence for w in ship_windows))
        min_ship_conf = float(min(w.confidence for w in ship_windows))
    else:
        max_ship_conf = None
        min_ship_conf = None

    # vessel_type / distance: most-frequent across ship windows when the
    # aggregate label is ship; otherwise across all windows. Counter ties
    # break by insertion order, which is fine for a tag-style field.
    if label == "ship" and ship_windows:
        vessel_type = Counter(w.vessel_type for w in ship_windows).most_common(1)[0][0]
        distance = Counter(w.distance for w in ship_windows).most_common(1)[0][0]
    else:
        vessel_type = Counter(w.vessel_type for w in windows).most_common(1)[0][0]
        distance = Counter(w.distance for w in windows).most_common(1)[0][0]

    # Mean embedding across all windows for RAG retrieval. The acoustic-
    # memory path in gemma.py wants list[float], so flatten to that.
    mean_embedding = np.mean(np.asarray(embeddings, dtype=np.float32), axis=0)
    embedding_list = mean_embedding.tolist()

    single_anomaly = (
        0 < ship_fraction <= SINGLE_WINDOW_ANOMALY_MAX_FRACTION
        and max_ship_conf is not None
        and max_ship_conf >= SINGLE_WINDOW_ANOMALY_MIN_CONFIDENCE
    )

    log.info(
        "windowed_prediction",
        n_windows=n,
        label=label,
        ship_fraction=round(ship_fraction, 3),
        confidence=round(confidence, 3),
        mean_uncertainty=round(mean_uncertainty, 3),
        single_window_anomaly=single_anomaly,
    )

    return WindowedPrediction(
        n_windows=n,
        label=label,
        confidence=confidence,
        uncertainty=uncertainty,
        ship_fraction=ship_fraction,
        mean_uncertainty=mean_uncertainty,
        max_ship_confidence=max_ship_conf,
        min_ship_confidence=min_ship_conf,
        vessel_type=vessel_type,
        distance=distance,
        embedding=embedding_list,
        single_window_anomaly=single_anomaly,
        windows=tuple(windows),
        checkpoint=checkpoint,
    )
