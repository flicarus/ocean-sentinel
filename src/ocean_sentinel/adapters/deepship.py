"""DeepShip — Strait of Georgia commercial vessel acoustic dataset.

Recorded 2016-2018 at the Ocean Networks Canada Strait of Georgia delta
node with an IcListen AF hydrophone (~141-147m depth). Full dataset is
47h 04min / 265 ships / 4 classes; the public partial release on GitHub
(~95 min, 63 WAV files) is enough to add real-world hydrophone and
geographic diversity to the v7 binary ship classifier.

Drop the partial release under data/deepship/<Class>/*.wav. Class folders
are: Cargo, Passengership, Tanker, Tug.

Label strategy: every DeepShip clip is `ship`. The dataset is curated to
single-ship-within-2km windows, so label noise is ~0%. Subclass carries
the vessel category for downstream analysis; the binary CNN only sees
`ship`.

Implementation notes:
- Files are 32 kHz mono FLOAT WAV, ~3 min each. We resample to
  TARGET_SAMPLE_RATE=16 kHz mono on load to match the rest of the corpus.
- Single hydrophone for the whole dataset → all clips share source_id
  'deepship' so per-source freq-profile caching learns one consistent
  spectral baseline. source_id is intentionally NOT class-specific —
  class diversity drives ship-recognition generalization, but spectral
  baseline is hardware-dependent (one IcListen unit).
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

log = structlog.get_logger()

DEEPSHIP_SOURCE_URL = "https://github.com/irfankamboh/DeepShip"
DEEPSHIP_LICENSE = "research-use-cite-eswa.2021.115270"
TARGET_SAMPLE_RATE = 16000

CLASS_FOLDERS: dict[str, str] = {
    "Cargo":         "cargo",
    "Passengership": "passenger",
    "Tanker":        "tanker",
    "Tug":           "tug",
}


@dataclass(frozen=True, slots=True)
class DeepShipClip:
    """One DeepShip WAV loaded as mono float32 at TARGET_SAMPLE_RATE.

    A full clip is 1-5 minutes — modest, but callers should still chunk
    immediately rather than holding many in memory.
    """
    path: Path
    vessel_class: str       # cargo / passenger / tanker / tug
    samples: np.ndarray
    sample_rate: int


class DeepShipAdapter:
    """Walks data/deepship/<Class>/*.wav, yields DeepShipClip per file."""

    source_id = "deepship"
    license_name = DEEPSHIP_LICENSE

    def __init__(self, root: Path) -> None:
        self._root = root

    def exists(self) -> bool:
        return self._root.exists() and any(self._root.iterdir())

    def enumerate(self) -> Iterator[DeepShipClip]:
        if not self.exists():
            log.info("deepship_dir_missing", path=str(self._root))
            return

        for folder_name, vessel_class in CLASS_FOLDERS.items():
            class_dir = self._root / folder_name
            if not class_dir.is_dir():
                continue
            for wav in sorted(class_dir.glob("*.wav")):
                try:
                    samples, sr = sf.read(wav, dtype="float32")
                    if samples.ndim > 1:
                        samples = samples.mean(axis=1)
                    if sr != TARGET_SAMPLE_RATE:
                        import librosa
                        samples = librosa.resample(
                            samples, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE,
                        ).astype(np.float32, copy=False)
                    log.info(
                        "deepship_clip_loaded",
                        path=str(wav.name), vessel_class=vessel_class,
                        samples=len(samples), sr=TARGET_SAMPLE_RATE,
                    )
                    yield DeepShipClip(
                        path=wav,
                        vessel_class=vessel_class,
                        samples=samples,
                        sample_rate=TARGET_SAMPLE_RATE,
                    )
                except Exception as e:
                    log.warning(
                        "deepship_clip_skipped",
                        path=str(wav), error=str(e),
                    )
