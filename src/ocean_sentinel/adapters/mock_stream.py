"""MockStreamFeeder — replays a WAV file at near-real-time cadence, emitting
60-second chunks for the CNN classifier. Lets us demonstrate the full
detection → threat-scoring → alert pipeline without needing a live
hydrophone or actual streaming infrastructure.

Usage:
    feeder = MockStreamFeeder("data/deepship/Tug/49.wav", chunk_s=60.0)
    for spec, t_offset in feeder.iter_chunks():
        # spec is the 128-bin mel spectrogram, ready for classifier.predict()
        ...
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import librosa
import numpy as np

_MEL_N = 128
_FMAX = 1000.0
_TARGET_SR = 8000


@dataclass
class StreamChunk:
    """One emitted chunk from the mock stream."""
    spectrogram: np.ndarray  # (128, T) mel-dB
    samples: np.ndarray      # raw mono float32
    sample_rate: int
    offset_s: float          # seconds into the source file
    chunk_index: int


class MockStreamFeeder:
    """Yield audio chunks from a WAV at a configurable cadence.

    Parameters
    ----------
    source : path
        WAV/FLAC file to replay.
    chunk_s : float
        Length of each emitted chunk in seconds (default 60).
    realtime : bool
        If True, sleep between chunks so they emit at real-time pace.
        Set False for fast tests / smoke runs.
    loop : bool
        If True, restart from the top after reaching EOF.
    """

    def __init__(
        self,
        source: str | Path,
        chunk_s: float = 60.0,
        realtime: bool = True,
        loop: bool = False,
        target_sr: int = _TARGET_SR,
    ) -> None:
        self.source = Path(source)
        self.chunk_s = chunk_s
        self.realtime = realtime
        self.loop = loop
        self.target_sr = target_sr

        if not self.source.exists():
            raise FileNotFoundError(f"mock stream source not found: {source}")

        self._samples, self._sr = librosa.load(
            str(self.source), sr=target_sr, mono=True,
        )
        self._chunk_samples = int(self.chunk_s * self._sr)
        self._duration_s = len(self._samples) / self._sr

    @property
    def duration_s(self) -> float:
        return self._duration_s

    @property
    def n_chunks(self) -> int:
        return max(1, int(self._duration_s / self.chunk_s))

    def _make_spec(self, samples: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=samples, sr=self._sr, n_mels=_MEL_N, fmax=_FMAX,
        )
        return librosa.power_to_db(mel, ref=1.0)

    def iter_chunks(self) -> Iterator[StreamChunk]:
        """Yield consecutive non-overlapping chunks of `chunk_s` seconds."""
        idx = 0
        offset = 0
        while True:
            end = offset + self._chunk_samples
            if end > len(self._samples):
                if self.loop:
                    offset = 0
                    end = self._chunk_samples
                else:
                    return
            chunk_samples = self._samples[offset:end].copy()
            spec = self._make_spec(chunk_samples)
            yield StreamChunk(
                spectrogram=spec,
                samples=chunk_samples,
                sample_rate=self._sr,
                offset_s=offset / self._sr,
                chunk_index=idx,
            )
            idx += 1
            offset = end
            if self.realtime:
                time.sleep(self.chunk_s)
