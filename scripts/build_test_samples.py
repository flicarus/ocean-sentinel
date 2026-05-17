"""Build the bundled test_samples directory shipped with Ocean Sentinel.

Produces 5 clips (~30s each, 16 kHz mono, ~1 MB each) under
src/ocean_sentinel/test_samples/, plus a manifest.yaml describing the
expected label for each. After install, users can run

    os test <site_id>

to verify their calibrated pipeline produces sensible decisions on
known-label samples — proof that the system isn't a mock and that
their site's adapter+threshold work end-to-end.

Vessel sources: DeepShip (Cargo, Tug, Passengership). Each clip
trimmed to a 30 s representative window starting at 30 s offset
(skipping any silent intro), resampled to 16 kHz mono to match the
CNN's training preprocessing.

Ambient sources: synthesised deterministically (fixed seed) so the
labels are guaranteed correct without depending on a curation step
we can't fully verify.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16_000
CLIP_S = 30
OUT = Path("src/ocean_sentinel/test_samples")
OUT.mkdir(parents=True, exist_ok=True)


def _trim_and_resample(src: str, out: Path, start_s: float = 30.0) -> None:
    y, _ = librosa.load(src, sr=SR, mono=True,
                        offset=start_s, duration=CLIP_S)
    sf.write(str(out), y, SR)


def _synth_ambient_quiet(out: Path) -> None:
    """Calm, broadband-low ambient — open ocean with light wind."""
    rng = np.random.RandomState(101)
    n = SR * CLIP_S
    # Pink-ish noise: white noise filtered to emphasise low end.
    white = rng.randn(n).astype(np.float32) * 0.05
    Y = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    # 1/f^0.5 envelope — gentle pink tilt
    weights = 1.0 / (1.0 + freqs / 100.0) ** 0.5
    weights[freqs < 5] = 0.0  # high-pass DC and sub-acoustic
    y = np.fft.irfft(Y * weights.astype("float32"), n=n).astype(np.float32)
    y = y / max(1.0, float(np.max(np.abs(y))) * 1.05)
    sf.write(str(out), y * 0.4, SR)


def _synth_ambient_busy(out: Path) -> None:
    """Slightly more textured ambient — choppy weather, rain on surface."""
    rng = np.random.RandomState(202)
    n = SR * CLIP_S
    # Louder broadband + occasional rain-tick transients spread across band
    base = rng.randn(n).astype(np.float32) * 0.1
    Y = np.fft.rfft(base)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    weights = (freqs > 50).astype("float32") * 0.7 + 0.3
    base = np.fft.irfft(Y * weights, n=n).astype(np.float32)

    # Sparse rain-tick clicks
    n_ticks = 200
    positions = rng.randint(0, n - SR // 50, size=n_ticks)
    template = np.exp(-np.linspace(0, 8, SR // 50)).astype(np.float32)
    template *= rng.randn(SR // 50).astype(np.float32) * 0.3
    for pos in positions:
        end = pos + len(template)
        if end < n:
            base[pos:end] += template * rng.uniform(0.3, 0.7)

    y = base / max(1.0, float(np.max(np.abs(base))) * 1.05)
    sf.write(str(out), y * 0.5, SR)


def main() -> None:
    print(f"writing test samples to {OUT}")

    # ALL three vessel clips below are *unseen by CNN v7.4* — their
    # (class, number) tuples do NOT appear in any training JSONL under
    # data/training/. Verified by scripts/evaluate_unseen.py. The
    # earlier bundle accidentally used Tug/49 and Cargo/103 which WERE
    # in training, defeating the point of `os test`.
    samples = [
        ("vessel_cargo.wav",
         "data/deepship/Cargo/38.wav", 30.0,
         "DeepShip / Cargo / 38.wav (held-out, CC-BY-NC 4.0 research use)"),
        ("vessel_passenger.wav",
         "data/deepship/Passengership/4.wav", 30.0,
         "DeepShip / Passengership / 4.wav (held-out, CC-BY-NC 4.0 research use)"),
        ("vessel_tanker.wav",
         "data/deepship/Tanker/5.wav", 5.0,
         "DeepShip / Tanker / 5.wav (held-out, CC-BY-NC 4.0 research use)"),
    ]

    for name, src, offset, attribution in samples:
        out = OUT / name
        if not Path(src).exists():
            print(f"  SKIP {name} — source missing: {src}")
            continue
        _trim_and_resample(src, out, start_s=offset)
        size = out.stat().st_size / 1024
        print(f"  ok   {name} ({size:.0f} KB)  ← {attribution}")

    _synth_ambient_quiet(OUT / "ambient_quiet.wav")
    print(f"  ok   ambient_quiet.wav (synthetic, deterministic seed=101)")
    _synth_ambient_busy(OUT / "ambient_busy.wav")
    print(f"  ok   ambient_busy.wav (synthetic, deterministic seed=202)")

    manifest = OUT / "manifest.yaml"
    manifest.write_text(
        """\
# Bundled test samples for `os test`.
#
# Each sample below has a known label so the test command can verify
# the calibrated pipeline produces a sensible decision on real audio.
#
# IMPORTANT — held-out integrity:
# All vessel clips below are confirmed UNSEEN by the trained CNN v7.4.
# Their (class, clip_id) tuples do NOT appear in any training JSONL
# under data/training/. This makes `os test` a real generalisation
# probe rather than a memorisation check.
# Verified by scripts/evaluate_unseen.py.
#
# Vessel clips are trimmed from DeepShip (https://www.kaggle.com/datasets/
# pranabkumarbose/deepship-data), used here under research / fair use
# for product validation. Ambient clips are synthesised deterministically
# so labels are guaranteed correct.

samples:
  - path: vessel_cargo.wav
    expected_label: ship
    duration_s: 30
    source: DeepShip Cargo clip 38.wav, 30 s window from offset 30 s
    held_out: true
    notes: cargo-class signature, broadband below 1 kHz, never in training

  - path: vessel_passenger.wav
    expected_label: ship
    duration_s: 30
    source: DeepShip Passengership clip 4.wav, 30 s window from offset 30 s
    held_out: true
    notes: passenger-vessel signature, never in training

  - path: vessel_tanker.wav
    expected_label: ship
    duration_s: 30
    source: DeepShip Tanker clip 5.wav, 30 s window from offset 5 s
    held_out: true
    notes: tanker-class signature, never in training

  - path: ambient_quiet.wav
    expected_label: not_ship
    duration_s: 30
    source: synthetic broadband ambient (deterministic seed=101)
    notes: pink-tilt noise, no vessel signature, no transients

  - path: ambient_busy.wav
    expected_label: not_ship
    duration_s: 30
    source: synthetic textured ambient (deterministic seed=202)
    notes: broadband + sparse rain-tick transients, no vessel signature
"""
    )
    print(f"  ok   manifest.yaml")


if __name__ == "__main__":
    main()
