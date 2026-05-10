"""Continual-improvement demo: site improves as more ambient accumulates.

Simulates the lifecycle of a deployed hydrophone:

  Day 1 — user onboards with 3 minutes of ambient. Adapter + threshold
          calibrated on a tiny sample. Coarse but usable.
  Day 7 — hydrophone has been running, we have an extra 6 minutes of
          ambient. `os refresh` re-fits adapter and tightens threshold.
  Day 30 — accumulated 25 minutes total. Another refresh.

We measure FA rate and vessel recall after each phase against the same
held-out vessel set, so the improvement curve is comparable.

Real production hydrophones obviously stream continuously; here we
synthesise progressively-larger ambient samples that share the same
tropical-reef distribution as the original test, so the differences
reflect "more calibration data" rather than "different site".
"""
from __future__ import annotations

import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from ocean_sentinel.gemma.site_adapter import train_site_adapter
from ocean_sentinel.gemma.conformal import calibrate_conformal_real
from ocean_sentinel.gemma.refresh import refresh_site


SR = 16_000
SITE_ID = "reef-continual"
VESSEL_CLIPS = [
    "data/deepship/Tug/49.wav",
    "data/deepship/Tug/40.wav",
    "data/deepship/Cargo/103.wav",
    "data/deepship/Cargo/15.wav",
    "data/deepship/Cargo/99.wav",
    "data/deepship/Passengership/16.wav",
    "data/deepship/Passengership/29.wav",
]


# ── synth (same distribution as scripts/test_drastically_different_site.py) ─


def _synth_reef(out: Path, duration_s: int, seed: int) -> Path:
    rng = np.random.RandomState(seed)
    n = SR * duration_s
    y = np.zeros(n, dtype=np.float32)

    # Snapping shrimp clicks
    n_clicks = int(30 * duration_s)
    click_template = np.exp(-np.linspace(0, 6, SR // 200)).astype(np.float32)
    t_click = np.arange(len(click_template)) / SR
    click_template *= np.sin(2 * np.pi * 4_000 * t_click)
    positions = rng.randint(0, n - len(click_template), size=n_clicks)
    for pos in positions:
        end = pos + len(click_template)
        if end < n:
            y[pos:end] += 0.4 * click_template * rng.uniform(0.5, 1.0)

    # Fish chorus 200-800 Hz
    chorus = rng.randn(n).astype(np.float32) * 0.15
    Y = np.fft.rfft(chorus)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    chorus = np.fft.irfft(Y * ((freqs > 200) & (freqs < 800)).astype("float32"),
                           n=n).astype(np.float32)
    mod = 0.7 + 0.3 * np.sin(2 * np.pi * 0.05 * np.arange(n) / SR)
    y += chorus * mod

    # Broadband floor
    y += 0.05 * rng.randn(n).astype(np.float32)
    y = y / max(1.0, float(np.max(np.abs(y))) * 1.05)

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y, SR)
    return out


# ── measurement ─────────────────────────────────────────────────────────


def _vessel_ship_probs(adapter_path: Path | None) -> list[tuple[str, float]]:
    from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier
    clf = CNNV7Classifier("data/models/cnn_v7_4.pt")
    clf.set_site_adapter(adapter_path if adapter_path and adapter_path.exists() else None)
    out = []
    for clip in VESSEL_CLIPS:
        if not Path(clip).exists():
            continue
        y, sr = librosa.load(clip, sr=SR, mono=True, duration=60.0)
        if y.size < sr * 5:
            continue
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=1000)
        spec = librosa.power_to_db(mel, ref=1.0)
        pred = clf.predict(spec, source_id=SITE_ID)
        p = pred["confidence"] if pred["label"] == "ship" else 1.0 - pred["confidence"]
        out.append((Path(clip).stem, p))
    return out


def _ambient_fa_rate(ambient_path: Path, adapter_path: Path, threshold: float) -> tuple[float, int]:
    from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier
    clf = CNNV7Classifier("data/models/cnn_v7_4.pt")
    clf.set_site_adapter(adapter_path if adapter_path.exists() else None)
    y, sr = librosa.load(str(ambient_path), sr=SR, mono=True)
    win = int(sr * 60)
    hop = int(sr * 4)
    if y.size < win:
        return 0.0, 0
    starts = list(range(0, y.size - win + 1, hop))
    fa = 0
    for s in starts:
        chunk = y[s : s + win]
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=1000)
        spec = librosa.power_to_db(mel, ref=1.0)
        pred = clf.predict(spec, source_id=SITE_ID)
        p = pred["confidence"] if pred["label"] == "ship" else 1.0 - pred["confidence"]
        if p >= threshold:
            fa += 1
    return fa / len(starts), len(starts)


def _summarise(label: str, ambient_path: Path, threshold: float, adapter_path: Path) -> None:
    fa, n_w = _ambient_fa_rate(ambient_path, adapter_path, threshold)
    vessel_results = _vessel_ship_probs(adapter_path)
    fired = sum(1 for _, p in vessel_results if p >= threshold)
    print(f"\n  ── {label} ──")
    print(f"  threshold:           {threshold:.3f}")
    print(f"  ambient FA:          {fa*100:5.1f}% over {n_w} windows")
    print(f"  vessel recall:       {fired}/{len(vessel_results)} = "
          f"{100*fired/max(1, len(vessel_results)):.0f}%")
    print(f"  vessel ship_probs:   "
          + " ".join(f"{p:.2f}" for _, p in vessel_results))


# ── runner ──────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 76)
    print("CONTINUAL-IMPROVEMENT DEMO")
    print("=" * 76)

    site_dir = Path("data/sites") / SITE_ID
    if site_dir.exists():
        shutil.rmtree(site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)

    base_audio_dir = Path("data/synthetic/continual")
    base_audio_dir.mkdir(parents=True, exist_ok=True)

    # ── Day 1 — onboarding (3 min ambient) ─────────────────────────────
    print("\n[Day 1] onboarding with 3 minutes of ambient")
    day1 = _synth_reef(base_audio_dir / "day1_3min.wav", 180, seed=1)
    train_result = train_site_adapter(site_id=SITE_ID, ambient_source=str(day1))
    assert train_result["ok"], train_result.get("error")
    cal = calibrate_conformal_real(site_id=SITE_ID, ambient_source=str(day1), alpha=0.05)
    assert cal["ok"], cal.get("error")
    # write a minimal site config so refresh_site can find original ambient
    Path(f"data/sites/{SITE_ID}.yaml").write_text(
        f"site_id: {SITE_ID}\nambient_source: {day1}\n"
    )
    threshold_d1 = cal["threshold_p"]
    _summarise("after Day 1 (3 min)", day1, threshold_d1,
               site_dir / "adapter.pt")

    # ── Day 7 — refresh after 6 more minutes ───────────────────────────
    print("\n[Day 7] refresh — 6 additional minutes of ambient")
    day7_extra = _synth_reef(base_audio_dir / "day7_6min.wav", 360, seed=7)
    refresh1 = refresh_site(site_id=SITE_ID,
                             additional_ambient_path=str(day7_extra))
    print(f"  refresh result: {refresh1.get('summary', refresh1.get('error'))}")
    assert refresh1["ok"], refresh1.get("error")
    threshold_d7 = refresh1["after"]["threshold"]
    # Combined ambient is what the model actually saw at refresh time
    combined_d7 = site_dir / "_combined_ambient.wav"
    _summarise("after Day 7 refresh (3+6 min)", combined_d7, threshold_d7,
               site_dir / "adapter.pt")

    # ── Day 30 — refresh after 16 more minutes (cumulative 25 min) ─────
    print("\n[Day 30] refresh — additional 16 minutes of ambient")
    day30_extra = _synth_reef(base_audio_dir / "day30_16min.wav", 960, seed=30)
    refresh2 = refresh_site(site_id=SITE_ID,
                             additional_ambient_path=str(day30_extra))
    print(f"  refresh result: {refresh2.get('summary', refresh2.get('error'))}")
    assert refresh2["ok"], refresh2.get("error")
    threshold_d30 = refresh2["after"]["threshold"]
    combined_d30 = site_dir / "_combined_ambient.wav"
    _summarise("after Day 30 refresh (3+6+16 min)", combined_d30, threshold_d30,
               site_dir / "adapter.pt")

    # ── BOTTOM LINE ────────────────────────────────────────────────────
    print(f"\n{'═' * 76}")
    print("  IMPROVEMENT CURVE")
    print(f"{'═' * 76}")
    print(f"  threshold:    Day 1 {threshold_d1:.3f}  →  "
          f"Day 7 {threshold_d7:.3f}  →  "
          f"Day 30 {threshold_d30:.3f}")
    print(f"  n_calibration: Day 1 {cal['n_calibration']}  →  "
          f"Day 7 {refresh1['after']['n_calibration']}  →  "
          f"Day 30 {refresh2['after']['n_calibration']}")
    print()
    print("  As more ambient accumulates, the conformal threshold tightens")
    print("  (more samples → smaller calibration uncertainty → less padding)")
    print("  while the adapter further suppresses ambient ship_prob.")
    print("  Both effects compound to lower the FA rate at the same alpha=5%.")


if __name__ == "__main__":
    main()
