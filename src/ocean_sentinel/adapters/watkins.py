"""Watkins Marine Mammal Sound Database — local-directory adapter.

Watkins clips must be pre-downloaded from
  https://cis.whoi.edu/science/B/whalesounds/
and dropped at data/watkins/<species_slug>/*.wav. Matches the on-disk pattern
already used by ShipsEar (shipsear_5s_16k/<class>/*.wav).

If data/watkins/ is missing or empty, .enumerate() returns []. This keeps the
bootstrap orchestrator running on MBARI alone until Watkins is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

log = structlog.get_logger()

WATKINS_SOURCE_URL = "https://cis.whoi.edu/science/B/whalesounds/"

# Species slug -> (subclass, scientific_name). Extend as needed.
# Slugs not in this map still load, with subclass=slug, species_scientific=None.
SPECIES_MAP: dict[str, tuple[str, str | None]] = {
    "orca":               ("orca",              "Orcinus orca"),
    "humpback_whale":     ("humpback_whale",    "Megaptera novaeangliae"),
    "blue_whale":         ("blue_whale",        "Balaenoptera musculus"),
    "sperm_whale":        ("sperm_whale",       "Physeter macrocephalus"),
    "fin_whale":          ("fin_whale",         "Balaenoptera physalus"),
    "gray_whale":         ("gray_whale",        "Eschrichtius robustus"),
    "right_whale":        ("right_whale",       "Eubalaena glacialis"),
    "bottlenose_dolphin": ("dolphin",           "Tursiops truncatus"),
    "common_dolphin":     ("dolphin",           "Delphinus delphis"),
    "pilot_whale":        ("pilot_whale",       "Globicephala melas"),
}


@dataclass(frozen=True, slots=True)
class WatkinsClip:
    path: Path
    species_slug: str
    subclass: str
    species_scientific: str | None
    samples: np.ndarray     # mono float32
    sample_rate: int


class WatkinsAdapter:
    """Walks data/watkins/<species_slug>/*.wav and yields loaded clips."""

    source_id = "watkins"
    license_name = "research-fair-use"

    def __init__(self, root: Path) -> None:
        self._root = root

    def exists(self) -> bool:
        return self._root.exists() and any(self._root.iterdir())

    def enumerate(self) -> list[WatkinsClip]:
        if not self.exists():
            log.info("watkins_dir_missing", path=str(self._root))
            return []
        clips: list[WatkinsClip] = []
        for species_dir in sorted(p for p in self._root.iterdir() if p.is_dir()):
            slug = species_dir.name
            subclass, sci = SPECIES_MAP.get(slug, (slug, None))
            for wav in sorted(species_dir.rglob("*.wav")):
                try:
                    samples, sr = sf.read(wav, dtype="float32")
                    if samples.ndim > 1:
                        samples = samples.mean(axis=1)
                    clips.append(WatkinsClip(
                        path=wav, species_slug=slug, subclass=subclass,
                        species_scientific=sci, samples=samples,
                        sample_rate=int(sr),
                    ))
                except Exception as e:
                    log.warning("watkins_clip_skipped",
                                path=str(wav), error=str(e))
        return clips
