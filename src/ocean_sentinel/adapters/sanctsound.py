"""SanctSound — NOAA Passive Acoustic archive from U.S. National Marine Sanctuaries.

The SanctSound project collected ~300 TB of passive acoustic data 2018-2021
across 30 recording sites in 8 sanctuaries (East/West Coast, Gulf, Hawaii,
Pacific). Public bucket: gs://noaa-passive-bioacoustic/sanctsound/

Drop FLAC files under data/sanctsound/<site>/*.flac. Site name ('sb01',
'oc01', 'hi03', ...) becomes the subclass. If data/sanctsound/ is missing
or empty, .enumerate() yields nothing and the caller skips this source.

Label strategy: every SanctSound chunk is emitted as 'ambient' (not_ship).
Some recordings contain transiting vessels — accepted as label noise.
SanctSound's real value is the geographic diversity of 8 sanctuaries,
which dwarfs the <20% label noise cost.

Implementation notes:
- Files are 4-6h FLAC, ~600 MB each, usually 48 kHz stereo. We resample
  to 16 kHz mono on load so spectrograms match other sources.
- `.enumerate()` is a generator (not list) because a full SanctSound file
  becomes ~350 MB of float32 after resample; loading many at once would
  blow memory.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

log = structlog.get_logger()

SANCTSOUND_SOURCE_URL = "https://sanctsound.ioos.us/"
SANCTSOUND_BUCKET = "gs://noaa-passive-bioacoustic/sanctsound/"
SANCTSOUND_LICENSE = "public-domain-noaa"

# Our downstream analyzer assumes 16 kHz mono. Everything else matches.
TARGET_SAMPLE_RATE = 16000

# site slug -> (subclass, sanctuary_name). Sites not listed load with
# subclass=slug, sanctuary=None.
SITE_MAP: dict[str, tuple[str, str]] = {
    # Channel Islands NMS (CA)
    "ci01": ("channel_islands_ambient", "Channel Islands"),
    "ci02": ("channel_islands_ambient", "Channel Islands"),
    "ci03": ("channel_islands_ambient", "Channel Islands"),
    "ci04": ("channel_islands_ambient", "Channel Islands"),
    "ci05": ("channel_islands_ambient", "Channel Islands"),
    # Florida Keys NMS (FL)
    "fk01": ("florida_keys_ambient", "Florida Keys"),
    "fk02": ("florida_keys_ambient", "Florida Keys"),
    "fk03": ("florida_keys_ambient", "Florida Keys"),
    "fk04": ("florida_keys_ambient", "Florida Keys"),
    # Gray's Reef NMS (GA)
    "gr01": ("grays_reef_ambient", "Gray's Reef"),
    "gr02": ("grays_reef_ambient", "Gray's Reef"),
    "gr03": ("grays_reef_ambient", "Gray's Reef"),
    # Hawaiian Islands Humpback Whale NMS (HI)
    "hi01": ("hawaii_ambient", "Hawaiian Islands"),
    "hi03": ("hawaii_ambient", "Hawaiian Islands"),
    "hi04": ("hawaii_ambient", "Hawaiian Islands"),
    "hi05": ("hawaii_ambient", "Hawaiian Islands"),
    "hi06": ("hawaii_ambient", "Hawaiian Islands"),
    # Monterey Bay NMS (CA) — separate from MBARI deep-canyon hydrophone
    "mb01": ("monterey_bay_ambient", "Monterey Bay"),
    "mb02": ("monterey_bay_ambient", "Monterey Bay"),
    "mb03": ("monterey_bay_ambient", "Monterey Bay"),
    # Olympic Coast NMS (WA)
    "oc01": ("olympic_coast_ambient", "Olympic Coast"),
    "oc02": ("olympic_coast_ambient", "Olympic Coast"),
    "oc03": ("olympic_coast_ambient", "Olympic Coast"),
    "oc04": ("olympic_coast_ambient", "Olympic Coast"),
    # Papahānaumokuākea MNM (HI)
    "pm01": ("papahanaumokuakea_ambient", "Papahānaumokuākea"),
    "pm02": ("papahanaumokuakea_ambient", "Papahānaumokuākea"),
    "pm05": ("papahanaumokuakea_ambient", "Papahānaumokuākea"),
    # Stellwagen Bank NMS (MA)
    "sb01": ("stellwagen_bank_ambient", "Stellwagen Bank"),
    "sb02": ("stellwagen_bank_ambient", "Stellwagen Bank"),
    "sb03": ("stellwagen_bank_ambient", "Stellwagen Bank"),
}


@dataclass(frozen=True, slots=True)
class SanctSoundClip:
    """One SanctSound FLAC loaded as mono float32 at TARGET_SAMPLE_RATE.

    Long (hours). Callers should chunk immediately and not hold many
    instances in memory at once.
    """
    path: Path
    site: str
    subclass: str
    sanctuary: str | None
    samples: np.ndarray
    sample_rate: int  # always TARGET_SAMPLE_RATE after load


class SanctSoundAdapter:
    """Walks data/sanctsound/<site>/*.flac, yields SanctSoundClip per file."""

    source_id = "sanctsound"
    license_name = SANCTSOUND_LICENSE

    def __init__(self, root: Path) -> None:
        self._root = root

    def exists(self) -> bool:
        return self._root.exists() and any(self._root.iterdir())

    def enumerate(self) -> Iterator[SanctSoundClip]:
        """Yield one clip per FLAC found under data/sanctsound/<site>/."""
        if not self.exists():
            log.info("sanctsound_dir_missing", path=str(self._root))
            return
        for site_dir in sorted(p for p in self._root.iterdir() if p.is_dir()):
            site = site_dir.name.lower()
            subclass, sanctuary = SITE_MAP.get(site, (site, None))
            for flac in sorted(site_dir.rglob("*.flac")):
                try:
                    samples, sr = sf.read(flac, dtype="float32")
                    if samples.ndim > 1:
                        samples = samples.mean(axis=1)
                    if sr != TARGET_SAMPLE_RATE:
                        # Lazy import — librosa is heavy, only pull it in
                        # when we actually need resampling.
                        import librosa
                        samples = librosa.resample(
                            samples, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE,
                        ).astype(np.float32, copy=False)
                    log.info(
                        "sanctsound_clip_loaded",
                        site=site, path=str(flac.name),
                        samples=len(samples), sr=TARGET_SAMPLE_RATE,
                    )
                    yield SanctSoundClip(
                        path=flac, site=site, subclass=subclass,
                        sanctuary=sanctuary, samples=samples,
                        sample_rate=TARGET_SAMPLE_RATE,
                    )
                except Exception as e:
                    log.warning(
                        "sanctsound_clip_skipped",
                        path=str(flac), error=str(e),
                    )
