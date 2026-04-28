"""OC01 narrow-band tonal analysis — power spectral density comparison.

The energy-vs-time analysis (oc01_doppler.py) showed only ~2 dB
modulation between miss clusters and quiet gaps. The vessel must be
distant — its loudness is barely above ambient. But cargo ships have
characteristic narrow-band tonal emissions (propeller blade-rate
harmonics in 5-50 Hz, diesel engine harmonics in 50-500 Hz) that are
distinctive even at low source levels.

This script computes Welch's PSD over the full suspect window vs the
control window and plots them on the same axes. If the suspect PSD
shows narrow peaks at specific low frequencies that the control does
not — those are tonal vessel signatures, the smoking gun.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import soundfile as sf


AUDIO_DIR = Path("data/diagnostic/sanctsound/audio")
SUSPECT_WAV = AUDIO_DIR / "suspect_window_12min.wav"
CONTROL_WAV = AUDIO_DIR / "control_5min.wav"
OUT_PNG = AUDIO_DIR / "psd_comparison.png"


def main() -> None:
    suspect, sr_s = sf.read(str(SUSPECT_WAV), dtype="float32")
    control, sr_c = sf.read(str(CONTROL_WAV), dtype="float32")
    if suspect.ndim > 1:
        suspect = suspect.mean(axis=1)
    if control.ndim > 1:
        control = control.mean(axis=1)

    # Welch's PSD with long enough segments to resolve sub-1Hz features at low freqs.
    nperseg = 1 << 14  # 16384 samples = 1.024 s at 16 kHz, 0.98 Hz bins
    f_s, p_s = welch(suspect, fs=sr_s, nperseg=nperseg, noverlap=nperseg // 2)
    f_c, p_c = welch(control, fs=sr_c, nperseg=nperseg, noverlap=nperseg // 2)

    # Convert to dB. Add tiny epsilon to avoid log(0).
    p_s_db = 10 * np.log10(p_s + 1e-20)
    p_c_db = 10 * np.log10(p_c + 1e-20)

    # Difference: suspect - control. Positive = suspect is louder at that frequency.
    # Reinterpolate control onto suspect freq grid (they should match here, sr matches).
    diff = p_s_db - np.interp(f_s, f_c, p_c_db)

    # Find narrow peaks in the difference, focused on cargo-ship-relevant bands.
    BAND_LO, BAND_HI = 5.0, 500.0
    band_mask = (f_s >= BAND_LO) & (f_s <= BAND_HI)
    f_band = f_s[band_mask]
    diff_band = diff[band_mask]

    # Top 10 peaks where suspect exceeds control by the largest dB margin.
    top_idx = np.argsort(diff_band)[-10:][::-1]
    print(f"Top 10 frequencies where suspect > control (dB margin):")
    print(f"  {'freq Hz':>10s} {'suspect dB':>12s} {'control dB':>12s} "
          f"{'diff dB':>9s}")
    for idx in top_idx:
        f0 = f_band[idx]
        idx_global = np.searchsorted(f_s, f0)
        print(
            f"  {f0:>10.2f} {p_s_db[idx_global]:>12.2f} "
            f"{p_c_db[idx_global]:>12.2f} {diff_band[idx]:>9.2f}"
        )

    print()
    print(f"Mean suspect-control diff in 5-50 Hz (blade-rate band): "
          f"{diff_band[(f_band >= 5) & (f_band <= 50)].mean():.2f} dB")
    print(f"Mean suspect-control diff in 50-500 Hz (engine band): "
          f"{diff_band[(f_band >= 50) & (f_band <= 500)].mean():.2f} dB")
    print(f"Mean suspect-control diff in 500-1000 Hz: "
          f"{diff_band[(f_band >= 500) & (f_band <= 1000)].mean():.2f} dB"
          if (f_band >= 500).any() else "")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    ax = axes[0]
    plot_mask = (f_s >= 1) & (f_s <= 1000)
    ax.semilogx(f_s[plot_mask], p_s_db[plot_mask],
                color="crimson", linewidth=1.0, label="Suspect (12 min, vessel?)")
    ax.semilogx(f_c[plot_mask], np.interp(f_s, f_c, p_c_db)[plot_mask],
                color="steelblue", linewidth=1.0, label="Control (5 min, ambient)")
    ax.axvspan(5, 50, alpha=0.08, color="orange",
               label="Cargo blade-rate band 5-50 Hz")
    ax.axvspan(50, 500, alpha=0.05, color="red",
               label="Engine harmonic band 50-500 Hz")
    ax.set_title(
        "OC01 2019-03-09 — power spectral density comparison",
        fontsize=12,
    )
    ax.set_xlabel("Frequency (Hz, log scale)")
    ax.set_ylabel("PSD (dB / Hz)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(which="both", alpha=0.3)

    ax = axes[1]
    ax.semilogx(f_s[plot_mask], diff[plot_mask],
                color="purple", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvspan(5, 50, alpha=0.08, color="orange")
    ax.axvspan(50, 500, alpha=0.05, color="red")
    ax.set_title("Suspect − Control (positive = vessel-like excess in suspect)")
    ax.set_xlabel("Frequency (Hz, log scale)")
    ax.set_ylabel("Δ PSD (dB)")
    ax.grid(which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()
