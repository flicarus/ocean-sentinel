"""Render the full 12-min suspect window (chunks 34-45, 12:33-12:45 UTC)
as one continuous high-resolution mel spectrogram. Marks the 10 miss
chunk boundaries on the time axis so a vessel passage (if present) can
be seen rolling across the spectrogram instead of sliced into 5-second
PNGs.

Also renders the 5-min control for side-by-side comparison.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


OUT_DIR = Path("data/diagnostic/sanctsound/audio")
SUSPECT_WAV = OUT_DIR / "suspect_window_12min.wav"
CONTROL_WAV = OUT_DIR / "control_5min.wav"

# Chunk N starts at offset N*60s within the FLAC. Suspect window starts at
# offset 2040s (chunk 34). So chunk M's offset within the suspect window
# is (M-34) * 60 seconds.
MISS_CHUNKS = [34, 35, 36, 37, 38, 42, 43, 44, 45]
MISS_CONFIDENCES = {
    34: 0.734, 35: 0.967, 36: 0.652, 37: 0.597, 38: 0.568,
    42: 0.529, 43: 0.946, 44: 0.852, 45: 0.805,
}


def _render(wav_path: Path, title: str, out_path: Path,
            mark_chunks: list[int] | None = None) -> None:
    samples, sr = sf.read(str(wav_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    mel = librosa.feature.melspectrogram(
        y=samples, sr=sr, n_mels=128, fmax=1000,
    )
    db = librosa.power_to_db(mel, ref=1.0)

    duration_s = len(samples) / sr
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    img = ax.imshow(
        db, aspect="auto", origin="lower", cmap="magma",
        extent=[0, duration_s, 0, 1000],
    )
    ax.set_xlabel("Seconds within window")
    ax.set_ylabel("Frequency (Hz, mel scale)")
    ax.set_title(title, fontsize=11)
    plt.colorbar(img, ax=ax, label="dB (absolute)")

    if mark_chunks:
        for m in mark_chunks:
            offset = (m - 34) * 60
            conf = MISS_CONFIDENCES.get(m, 0.0)
            ax.axvline(offset, color="cyan", linestyle="--", alpha=0.6,
                       linewidth=1)
            ax.text(
                offset, 950, f"chunk {m}\n{conf:.2f}",
                color="cyan", ha="left", va="top", fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    if not SUSPECT_WAV.exists():
        raise SystemExit(f"missing {SUSPECT_WAV} — run the ffmpeg slicer first")

    _render(
        SUSPECT_WAV,
        "OC01 2019-03-09 12:33-12:45 UTC — suspect window (CNN flagged 9 chunks)",
        OUT_DIR / "suspect_window_12min.png",
        mark_chunks=MISS_CHUNKS,
    )
    _render(
        CONTROL_WAV,
        "OC01 2019-03-09 13:39-13:44 UTC — control window (CNN said correctly: ambient)",
        OUT_DIR / "control_5min.png",
    )


if __name__ == "__main__":
    main()
