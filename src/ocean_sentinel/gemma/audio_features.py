"""Real audio → spectral signature.

Replaces the random-Gaussian mock in tools.py with a librosa-based pipeline.
The signature is what we use to fingerprint a site for transfer-learning
lookup (compare_to_known_sites).

Design choices:
- 64 mel bands. Smaller than the CNN's 128-mel input but plenty for site
  fingerprinting. Cuts down compute + the JSON we send to Gemma.
- log-power dB scale, ref=1.0 (consistent across clips, like training).
- Per-band MEDIAN across time. Robust to transient events (passing vessel
  during ambient capture, snapping shrimp bursts) — what we want is the
  background statistics of the site, not its tail.
- Up to 5 minutes loaded (matching the prompt to the user). Longer is fine
  but yields no extra info for fingerprinting.
- Mono. 16 kHz resample (matches training corpus). 64-mel covers 0–8 kHz.

Returns a dict with the signature plus interpretable summary fields so
Gemma can narrate without reading 64 raw numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np

_TARGET_SR_HZ = 16_000
_FMAX_HZ = 8_000
_FMIN_HZ = 20
_N_MELS = 64
_HOP_LENGTH = 1024
_N_FFT = 4096
_DEFAULT_DURATION_S = 300.0


def _classify_ambient(
    mel_db_full: np.ndarray,
    mel_freqs: np.ndarray,
) -> tuple[str, dict[str, float]]:
    """Heuristic ambient class from the multi-band time-frequency surface.

    Display-only — never fed back to the model. Gemma uses this to narrate.

    The earlier version used only the dominant median band and would
    misclassify dense-transient sites (coral reef snapping shrimp) as
    "vessel band" because the median across time picks up the steady
    fish chorus and not the clicks. We now look at:

      - dominant frequency of the per-band median (steady background)
      - the gap between the per-band 95th percentile and the median in
        the high-frequency bands (>1500 Hz). A gap > ~15 dB indicates
        dense, sharp transient activity — the visual signature of
        snapping shrimp on a tropical reef.

    Both signals are returned alongside the label so the caller can
    persist them for the Step 3 OOD diagnostic.

    Thresholds based on Wenz 1962 / Hildebrand 2009 plus our own
    synthetic-reef calibration (scripts/test_drastically_different_site.py).
    """
    median = np.median(mel_db_full, axis=1)
    p95 = np.percentile(mel_db_full, 95, axis=1)

    high_band_mask = mel_freqs > 1500
    if high_band_mask.any():
        transient_gap_hf_db = float(
            (p95[high_band_mask] - median[high_band_mask]).mean()
        )
    else:
        transient_gap_hf_db = 0.0

    dominant_idx = int(np.argmax(median))
    dominant_hz = float(mel_freqs[dominant_idx])

    features = {
        "transient_gap_hf_db": round(transient_gap_hf_db, 2),
        "dominant_hz": round(dominant_hz, 1),
    }

    # Reef / dense-click ambient first: high-band transient activity wins
    # over whatever the steady median says.
    if transient_gap_hf_db > 15.0:
        return "biological · transient-rich (HF clicks)", features

    if dominant_hz < 80:
        return "infrasound · deep-water", features
    if dominant_hz < 500:
        return "low-frequency · vessel band", features
    if dominant_hz < 2000:
        return "mid-frequency · mixed", features
    return "high-frequency · biological", features


def _format_band(mel_freqs: np.ndarray, idx: int, span: int = 2) -> str:
    """Pretty 'lo-hi Hz' string around a center bin."""
    lo = int(mel_freqs[max(0, idx - span)])
    hi = int(mel_freqs[min(len(mel_freqs) - 1, idx + span)])
    return f"{lo}-{hi} Hz"


def compute_spectral_signature(
    audio_source: str,
    duration_s: float = _DEFAULT_DURATION_S,
    n_mels: int = _N_MELS,
) -> dict[str, Any]:
    """Load audio, compute a 64-band log-mel spectral signature.

    Returns dict with:
        ok: bool
        signature: list[float]    (n_mels elements, dB)
        median_psd_db: float
        dominant_band_hz: str     (e.g. "80-200 Hz")
        ambient_class: str
        duration_loaded_s: float
        sample_rate_hz: int
        summary: str              (one-line narration hint)
    """
    path = Path(audio_source)
    if not path.exists():
        return {"ok": False, "error": f"audio file not found: {audio_source}"}

    try:
        y, sr = librosa.load(
            str(path), sr=_TARGET_SR_HZ, mono=True, duration=duration_s,
        )
    except Exception as e:
        return {"ok": False, "error": f"failed to load audio: {type(e).__name__}: {e}"}

    if y.size == 0:
        return {"ok": False, "error": "audio file decoded to empty buffer"}

    # Mel power spectrogram → log-dB.
    mel_power = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=_N_FFT, hop_length=_HOP_LENGTH,
        fmin=_FMIN_HZ, fmax=min(_FMAX_HZ, sr // 2),
    )
    mel_db = librosa.power_to_db(mel_power, ref=1.0)

    # Per-band median across time — robust to transients.
    signature = np.median(mel_db, axis=1)
    median_psd = float(np.median(signature))

    mel_freqs = librosa.mel_frequencies(
        n_mels=n_mels, fmin=_FMIN_HZ, fmax=min(_FMAX_HZ, sr // 2),
    )
    ambient_class, class_features = _classify_ambient(mel_db, mel_freqs)

    dominant_idx = int(np.argmax(signature))
    dominant_freq_hz = float(mel_freqs[dominant_idx])
    dominant_band_str = _format_band(mel_freqs, dominant_idx)

    snapping_shrimp = ambient_class.startswith("biological · transient-rich")

    return {
        "ok": True,
        "signature": [round(float(x), 3) for x in signature.tolist()],
        "n_bands": n_mels,
        "median_psd_db": round(median_psd, 2),
        "dominant_band_hz": dominant_band_str,
        "dominant_freq_hz": round(dominant_freq_hz, 1),
        "ambient_class": ambient_class,
        "transient_gap_hf_db": class_features["transient_gap_hf_db"],
        "duration_loaded_s": round(float(len(y) / sr), 1),
        "sample_rate_hz": int(sr),
        "snapping_shrimp": snapping_shrimp,
        "summary": (
            f"dominant {dominant_band_str} · {ambient_class} · "
            f"median {median_psd:.1f} dB"
        ),
    }


def signature_from_spec_array(spec: np.ndarray, n_bands: int = _N_MELS) -> list[float]:
    """Re-extract a signature from a pre-computed (n_mels x T) log-dB spec.

    Used by scripts/precompute_signatures.py to bootstrap known_sites.json
    from training data's .npy spectrograms — much faster than re-decoding WAVs.

    Source specs are typically 128-mel; we downsample by averaging adjacent
    bands to match the 64-band runtime signature shape.
    """
    spec = np.asarray(spec, dtype=np.float32)
    if spec.ndim != 2:
        raise ValueError(f"expected 2D spec, got shape {spec.shape}")

    src_bands = spec.shape[0]
    if src_bands == n_bands:
        return [round(float(x), 3) for x in np.median(spec, axis=1).tolist()]
    if src_bands % n_bands == 0:
        factor = src_bands // n_bands
        grouped = spec.reshape(n_bands, factor, -1).mean(axis=1)
        return [round(float(x), 3) for x in np.median(grouped, axis=1).tolist()]
    # Fall back: pick n_bands evenly spaced rows
    idx = np.linspace(0, src_bands - 1, n_bands).astype(int)
    return [round(float(x), 3) for x in np.median(spec[idx], axis=1).tolist()]
