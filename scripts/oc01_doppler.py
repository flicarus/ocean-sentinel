"""OC01 Doppler / approach-peak-recede signature analysis.

A vessel passing a hydrophone leaves a characteristic energy curve:
quiet → rising → peak (closest point of approach) → falling → quiet.
Stationary noise sources (electrical, geological, biological steady state)
do not. This script computes the engine-band (50-500 Hz) energy
time-series for the 12-min suspect window and the 5-min control, and
plots them side-by-side.

If the suspect window shows one or more clear approach-peak-recede
arcs aligned with the chunks our CNN flagged, that is independent
physical evidence that the source was moving — i.e. a vessel, not
ambient noise.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


AUDIO_DIR = Path("data/diagnostic/sanctsound/audio")
SUSPECT_WAV = AUDIO_DIR / "suspect_window_12min.wav"
CONTROL_WAV = AUDIO_DIR / "control_5min.wav"
OUT_PNG = AUDIO_DIR / "doppler_energy_curves.png"

ENGINE_BAND_LOW_HZ = 50.0
ENGINE_BAND_HIGH_HZ = 500.0

# Suspect window starts at offset 2040 s into the FLAC (chunk 34).
# Chunk N's start in this window = (N - 34) * 60 seconds.
MISS_CHUNKS = [34, 35, 36, 37, 38, 42, 43, 44, 45]
MISS_CONFIDENCES = {
    34: 0.734, 35: 0.967, 36: 0.652, 37: 0.597, 38: 0.568,
    42: 0.529, 43: 0.946, 44: 0.852, 45: 0.805,
}


def engine_band_energy(samples: np.ndarray, sr: int,
                       smoothing_s: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Time-series of mean energy (dB) in the 50-500 Hz band.

    Uses a wide-spectrum mel basis (n_mels=256, fmax=sr/2) — matches
    AudioAnalyzer._extract_features so the band carries real meaning.
    Returns (time_seconds, energy_db).
    """
    n_mels = 256
    fmax = sr // 2
    mel = librosa.feature.melspectrogram(
        y=samples, sr=sr, n_mels=n_mels, fmax=fmax, hop_length=512,
    )
    db = librosa.power_to_db(mel, ref=1.0)

    freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=fmax)
    band_mask = (freqs >= ENGINE_BAND_LOW_HZ) & (freqs <= ENGINE_BAND_HIGH_HZ)
    band_energy = db[band_mask].mean(axis=0)

    times = librosa.frames_to_time(
        np.arange(len(band_energy)), sr=sr, hop_length=512,
    )

    # Smooth with a moving average to surface the slow envelope.
    if smoothing_s > 0:
        frame_dt = times[1] - times[0] if len(times) > 1 else 1 / sr
        win = max(1, int(smoothing_s / frame_dt))
        kernel = np.ones(win) / win
        band_energy = np.convolve(band_energy, kernel, mode="same")

    return times, band_energy


def main() -> None:
    if not SUSPECT_WAV.exists():
        raise SystemExit(f"missing {SUSPECT_WAV}")
    if not CONTROL_WAV.exists():
        raise SystemExit(f"missing {CONTROL_WAV}")

    suspect_samples, sr_s = sf.read(str(SUSPECT_WAV), dtype="float32")
    control_samples, sr_c = sf.read(str(CONTROL_WAV), dtype="float32")
    if suspect_samples.ndim > 1:
        suspect_samples = suspect_samples.mean(axis=1)
    if control_samples.ndim > 1:
        control_samples = control_samples.mean(axis=1)

    t_s, e_s = engine_band_energy(suspect_samples, sr_s)
    t_c, e_c = engine_band_energy(control_samples, sr_c)

    print(f"Suspect window: {len(suspect_samples) / sr_s:.1f} s, "
          f"engine band energy mean={e_s.mean():.2f} dB, "
          f"std={e_s.std():.2f} dB, "
          f"range=[{e_s.min():.2f}, {e_s.max():.2f}]")
    print(f"Control window: {len(control_samples) / sr_c:.1f} s, "
          f"engine band energy mean={e_c.mean():.2f} dB, "
          f"std={e_c.std():.2f} dB, "
          f"range=[{e_c.min():.2f}, {e_c.max():.2f}]")
    print()

    # Identify peaks in suspect curve — the smoking gun is an approach-peak-recede arc.
    median_s = float(np.median(e_s))
    threshold_s = median_s + 1.5 * float(np.std(e_s))
    above = e_s > threshold_s
    print(f"Suspect: median {median_s:.2f} dB, "
          f"threshold (median + 1.5σ) {threshold_s:.2f} dB")
    print(f"  frames above threshold: {int(above.sum())}/{len(above)} "
          f"({above.mean():.1%})")

    # Window-level: split suspect into 60s windows, compute mean per window
    win_s = 60.0
    n_win = int(t_s.max() // win_s)
    window_means = []
    for i in range(n_win):
        mask = (t_s >= i * win_s) & (t_s < (i + 1) * win_s)
        if mask.any():
            window_means.append(float(e_s[mask].mean()))

    print()
    print("Suspect 60-s window means (chunk N corresponds to window N-34):")
    print(f"  {'window':>6s}  {'chunk':>5s}  {'mean dB':>8s}  {'CNN p_ship':>11s}")
    for i, m in enumerate(window_means):
        chunk = 34 + i
        cnn = MISS_CONFIDENCES.get(chunk, None)
        cnn_str = f"{cnn:.2f}" if cnn is not None else "-"
        print(f"  {i:>6d}  {chunk:>5d}  {m:>8.2f}  {cnn_str:>11s}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharey=True)

    ax = axes[0]
    ax.plot(t_s, e_s, color="crimson", linewidth=0.8, alpha=0.9,
            label="Smoothed engine-band energy (5-s MA)")
    ax.axhline(median_s, color="grey", linestyle=":", alpha=0.6, label="median")
    ax.axhline(threshold_s, color="orange", linestyle="--", alpha=0.6,
               label="median + 1.5σ")
    for c in MISS_CHUNKS:
        offset = (c - 34) * 60
        ax.axvline(offset, color="cyan", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(offset, e_s.max(), f"c{c}", color="cyan", fontsize=8,
                rotation=90, va="top", ha="right")
    ax.set_title(
        "OC01 2019-03-09 12:33-12:45 UTC — suspect window\n"
        f"Engine band 50-500 Hz energy (mean {e_s.mean():.2f} dB)",
        fontsize=11,
    )
    ax.set_xlabel("Seconds within window")
    ax.set_ylabel("Energy (dB, absolute)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_c, e_c, color="steelblue", linewidth=0.8, alpha=0.9)
    ax.axhline(float(np.median(e_c)), color="grey", linestyle=":", alpha=0.6)
    ax.set_title(
        "OC01 2019-03-09 13:39-13:44 UTC — control window\n"
        f"Engine band 50-500 Hz energy (mean {e_c.mean():.2f} dB)",
        fontsize=11,
    )
    ax.set_xlabel("Seconds within window")
    ax.set_ylabel("Energy (dB, absolute)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()
