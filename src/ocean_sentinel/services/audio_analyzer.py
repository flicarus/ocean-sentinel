from __future__ import annotations

from dataclasses import replace

import librosa
import numpy as np 
import structlog

from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import AudioSegment
from ocean_sentinel.exceptions import SpectrogramError

log = structlog.get_logger()

# Ship engine acoustics signature: dominant energy between 50-500Hz
ENGINE_BAND_LOW = 50
ENGINE_BAND_HIGH = 500


class AudioAnalyzer:
    """Generates spectrograms and extracts acoustics features from hydrophone audio. """

    def __init__(self, settings: Settings) -> None:
        self._n_mels = settings.spectrogram_n_mels     #128
        self._fmax = settings.spectrogram_fmax         #1000Hz


    def analyze(self, segment: AudioSegment) -> tuple[AudioSegment, dict]:
        """"Generate spectrogram + extract features for classification.
        
            Returns updated AudioSegment (with spectrogram) and a features dict.
         """
        try: 
            spectrogram = self._make_spectrogram(segment.samples, segment.sample_rate)
            features = self._extract_features(segment.samples, segment.sample_rate, spectrogram)

            updated = replace(segment, spectrogram=spectrogram)

            log.info(
                "audio_analyzed",
                source=segment.source_file,
                engine_band_energy=features["engine_band_energy_db"],
                peak_freq=features["peak_frequency_hz"],
            )
            return updated, features

        except Exception as e:
            raise SpectrogramError(
                code="spectrogram_failed",
                message=f"Failed to analyze audio from {segment.source_file}",
                details={"error": str(e)},
            ) from e


    def _make_spectrogram(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Raw samples -> mel spectrogram in absolute dB scale.

        ref=1.0 (not np.max) so engine energy is a consistent physical
        quantity across clips and environments. With ref=np.max the same
        engine sounds different depending on what else is in the clip;
        with ref=1.0 it's an absolute measurement the CNN can generalize on.
        """
        mel = librosa.feature.melspectrogram(
            y=samples,
            sr=sr,
            n_mels=self._n_mels,
            fmax=self._fmax,
        )
        return librosa.power_to_db(mel, ref=1.0)

    def _extract_features(
        self, samples: np.ndarray, sr: int, spectrogram: np.ndarray
    ) -> dict:
        """Pull out numbers that help Gemma decide if this is a ship.

        Features are computed on the FULL spectrum (0 to sr/2) so that
        engine band ratio is meaningful. The visual spectrogram for Gemma
        stays at fmax from config.
        """
        # Full-range spectrogram for feature analysis (not the visual one)
        n_mels_full = 256
        fmax_full = sr // 2  # 8000 Hz for 16kHz audio
        full_mel = librosa.feature.melspectrogram(
            y=samples, sr=sr, n_mels=n_mels_full, fmax=fmax_full,
        )
        full_db = librosa.power_to_db(full_mel, ref=np.max)

        freqs = librosa.mel_frequencies(n_mels=n_mels_full, fmax=fmax_full)
        engine_mask = (freqs >= ENGINE_BAND_LOW) & (freqs <= ENGINE_BAND_HIGH)
        non_engine_mask = ~engine_mask

        engine_energy = np.mean(full_db[engine_mask])
        non_engine_energy = np.mean(full_db[non_engine_mask])
        total_energy = np.mean(full_db)

        # Ratio: how much louder is engine band vs the rest
        # > 1.0 means engine band dominates, < 1.0 means background dominates
        engine_ratio = float(engine_energy / non_engine_energy) if non_engine_energy != 0 else 0.0

        # Peak frequency across full spectrum
        mean_per_band = np.mean(full_db, axis=1)
        peak_idx = int(np.argmax(mean_per_band))
        peak_freq = float(freqs[peak_idx])

        # Spectral flatness in engine band only — low = tonal (engine hum)
        engine_flatness = float(np.mean(
            librosa.feature.spectral_flatness(S=full_mel[engine_mask])
        ))

        rms = float(np.mean(librosa.feature.rms(y=samples)))

        is_engine_dominant = (
            peak_freq >= ENGINE_BAND_LOW
            and peak_freq <= ENGINE_BAND_HIGH
            and engine_ratio > 1.05
        )

        return {
            "engine_band_energy_db": round(float(engine_energy), 2),
            "non_engine_energy_db": round(float(non_engine_energy), 2),
            "total_energy_db": round(float(total_energy), 2),
            "engine_band_ratio": round(engine_ratio, 3),
            "peak_frequency_hz": round(peak_freq, 1),
            "spectral_flatness": round(engine_flatness, 4),
            "rms_energy": round(rms, 6),
            "is_engine_band_dominant": is_engine_dominant,
        }
    