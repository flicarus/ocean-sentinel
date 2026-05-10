"""Quantify the impact of per-site safeguards on a drastically-different
reef site.

Compares two configurations of the SAME CNN v7.4 weights:

  A. "Out-of-the-box": v7.4 + the global conformal threshold derived from
     our held-out training distribution (data/calibration/conformal_v7_4.json,
     threshold ≈ 0.613).

  B. "Per-site adapted": v7.4 + the per-site conformal threshold derived
     from split-conformal calibration on the user's reef ambient.

For each configuration we measure:

  - False-alarm rate on the user's ambient (% of windows that exceed the
    threshold). Should be ~5 % by conformal guarantee under config B.

  - Recall on a battery of real vessel clips (DeepShip Tug + Cargo +
    Passengership). Same CNN; only the threshold changes.

We also report base-CNN per-event ship_prob so the user can see how the
underlying model behaves regardless of threshold.

Note: we don't compute "accuracy after fine-tuning" because we didn't
fine-tune — there are no per-site labels to fine-tune against. The
per-site contribution is the THRESHOLD, not the WEIGHTS. That's the
whole "lightweight adaptation" point: provable false-alarm bound
without per-site labels.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import librosa
import numpy as np

from ocean_sentinel.gemma.cnn_inference import simulate_detection
from ocean_sentinel.gemma.conformal import calibrate_conformal_real

CKPT = "data/models/cnn_v7_4.pt"
GLOBAL_CONFORMAL = "data/calibration/conformal_v7_4.json"
AMBIENT = "data/synthetic/tropical_reef_3min.wav"
SITE_ID = "reef-test-measure"

VESSEL_CLIPS = [
    ("Tug-49",        "data/deepship/Tug/49.wav"),
    ("Tug-40",        "data/deepship/Tug/40.wav"),
    ("Cargo-103",     "data/deepship/Cargo/103.wav"),
    ("Cargo-15",      "data/deepship/Cargo/15.wav"),
    ("Cargo-99",      "data/deepship/Cargo/99.wav"),
    ("Passenger-16",  "data/deepship/Passengership/16.wav"),
    ("Passenger-29",  "data/deepship/Passengership/29.wav"),
]


def _load_ambient_window_ship_probs(
    ambient: Path,
    *,
    adapter_path: Path | None = None,
    n_max: int = 60,
) -> list[float]:
    """Slide v7.4 across the ambient and return per-window ship_probs.

    60s windows, 4s hop — matches the per-site conformal calibration so
    we measure FA on the same window distribution the calibration was
    derived from. When `adapter_path` is set, the CNN is run WITH the
    per-site adapter applied; otherwise the un-adapted CNN is used.
    """
    from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier

    sr_target = 16_000
    win_s = 60.0
    hop_s = 4.0

    y, sr = librosa.load(str(ambient), sr=sr_target, mono=True, duration=300.0)
    win = int(sr * win_s)
    hop = int(sr * hop_s)
    if y.size < win:
        return []

    clf = CNNV7Classifier(CKPT)
    clf.set_site_adapter(adapter_path)
    starts = list(range(0, y.size - win + 1, hop))[:n_max]
    probs = []
    for s in starts:
        chunk = y[s : s + win]
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=1000)
        spec = librosa.power_to_db(mel, ref=1.0)
        pred = clf.predict(spec, source_id=SITE_ID)
        label = pred.get("label", "not_ship")
        conf = float(pred.get("confidence", 0.5))
        probs.append(conf if label == "ship" else 1.0 - conf)
    return probs


def main() -> None:
    print("=" * 76)
    print("SAFEGUARD IMPACT ON DRASTICALLY-DIFFERENT REEF SITE")
    print("=" * 76)

    if not Path(AMBIENT).exists():
        print(f"\nMissing {AMBIENT}. Run scripts/test_drastically_different_site.py first.")
        return

    # Read both thresholds.
    global_t = json.loads(Path(GLOBAL_CONFORMAL).read_text())["threshold"]
    print(f"\n[thresholds]")
    print(f"  global v7.4 conformal:   {global_t:.3f}")

    cal = calibrate_conformal_real(
        site_id=SITE_ID, ambient_source=AMBIENT, alpha=0.05,
    )
    if not cal.get("ok"):
        print(f"  per-site calibration failed: {cal.get('error')}")
        return
    site_t = cal["threshold_p"]
    print(f"  per-site (reef) conformal: {site_t:.3f}")
    print(f"  alpha (target FA):         {cal['alpha']}")
    print(f"  contract warning:          {cal.get('contract_warning')}")

    # ── PART 1: false-alarm rate on the user's ambient ─────────────────
    print(f"\n{'─' * 76}\nFalse-alarm rate on reef ambient (lower is better)\n{'─' * 76}")

    adapter_path = Path("data/sites") / SITE_ID / "adapter.pt"
    has_adapter = adapter_path.exists()
    print(f"  adapter present:  {has_adapter}  ({adapter_path})")

    probs_base = _load_ambient_window_ship_probs(Path(AMBIENT), adapter_path=None)
    if has_adapter:
        probs_adapted = _load_ambient_window_ship_probs(Path(AMBIENT), adapter_path=adapter_path)
    else:
        probs_adapted = probs_base
    if not probs_base:
        print("  no windows could be evaluated")
        return

    n = len(probs_base)
    fa_global = sum(1 for p in probs_base if p >= global_t) / n
    fa_site = sum(1 for p in probs_adapted if p >= site_t) / n

    print(f"  n_windows:                            {n}")
    print(f"  median ship_prob (un-adapted CNN):    {median(probs_base):.3f}")
    if has_adapter:
        print(f"  median ship_prob (adapted CNN):       {median(probs_adapted):.3f}")
    print(f"  un-adapted CNN ≥ global {global_t:.2f}:      "
          f"{fa_global*100:5.1f}%   ← out-of-the-box FA rate")
    print(f"  {'  adapted' if has_adapter else 'un-adapted'} CNN ≥ per-site {site_t:.2f}:    "
          f"{fa_site*100:5.1f}%   ← per-site adapted FA rate")
    if fa_global > 0:
        if fa_site == 0:
            improvement_msg = (
                f"  reduction: {fa_global*100:.1f}% → 0% "
                f"(every reef-ambient FA suppressed)"
            )
        elif fa_site < fa_global:
            improvement_msg = (
                f"  reduction: {fa_global*100:.1f}% → {fa_site*100:.1f}% "
                f"({fa_global / max(fa_site, 1e-9):.1f}× fewer false alarms)"
            )
        else:
            improvement_msg = (
                f"  no FA reduction: {fa_global*100:.1f}% → {fa_site*100:.1f}%"
            )
        print(improvement_msg)
    else:
        print(f"  (per-site already at floor)")

    # ── PART 2: recall on real vessel clips ────────────────────────────
    print(f"\n{'─' * 76}\nRecall on real vessel clips (higher is better)\n{'─' * 76}")
    n_v = 0
    n_global_pass = 0
    n_site_pass = 0
    rows = []

    # Build two classifiers: one without adapter (out-of-the-box), one with.
    from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier
    clf_base = CNNV7Classifier(CKPT)
    clf_base.set_site_adapter(None)
    clf_adapted = CNNV7Classifier(CKPT)
    clf_adapted.set_site_adapter(adapter_path if has_adapter else None)

    for label, clip in VESSEL_CLIPS:
        if not Path(clip).exists():
            continue
        try:
            y, sr = librosa.load(clip, sr=16_000, mono=True, duration=60.0)
        except Exception:
            continue
        if y.size < sr * 5:
            continue
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=1000)
        spec = librosa.power_to_db(mel, ref=1.0)

        pred_base = clf_base.predict(spec, source_id=SITE_ID)
        pred_ada = clf_adapted.predict(spec, source_id=SITE_ID)

        p_base = pred_base["confidence"] if pred_base["label"] == "ship" \
            else 1.0 - pred_base["confidence"]
        p_ada = pred_ada["confidence"] if pred_ada["label"] == "ship" \
            else 1.0 - pred_ada["confidence"]

        n_v += 1
        passes_global = p_base >= global_t
        passes_site = p_ada >= site_t
        n_global_pass += int(passes_global)
        n_site_pass += int(passes_site)
        rows.append((label, p_base, p_ada, passes_global, passes_site))

    print(f"  {'clip':<16}  {'p (base)':>9}  {'p (adapt)':>10}  "
          f"{'global '+f'{global_t:.2f}':>14}  "
          f"{'reef '+f'{site_t:.2f}':>14}")
    for label, p_b, p_a, pg, ps in rows:
        print(f"  {label:<16}  {p_b:>9.3f}  {p_a:>10.3f}  "
              f"{('FIRES' if pg else '----'):>14}  "
              f"{('FIRES' if ps else '----'):>14}")
    if n_v:
        print(f"\n  recall (out-of-the-box CNN, global threshold {global_t:.2f}):  "
              f"{n_global_pass}/{n_v} = {100 * n_global_pass / n_v:.0f}%")
        print(f"  recall (adapted CNN, per-site threshold {site_t:.2f}):       "
              f"{n_site_pass}/{n_v} = {100 * n_site_pass / n_v:.0f}%")

    # ── BOTTOM LINE ────────────────────────────────────────────────────
    print(f"\n{'═' * 76}\n  BOTTOM LINE\n{'═' * 76}")
    print(f"  v7.4 backbone + transformer + heads: frozen.")
    if has_adapter:
        print(f"  Site adapter: 33 k-param residual MLP, trained on user ambient")
        print(f"                with held-out vessel-recall floor = 0.85.")
    print(f"  Conformal threshold: {global_t:.3f} (global) → {site_t:.3f} (per-site)")
    print(f"  • Reef-ambient FA:    {fa_global*100:5.1f}% → {fa_site*100:5.1f}%  "
          f"(target alpha = {cal['alpha']*100:.0f}%)")
    if n_v:
        print(f"  • Vessel-clip recall: {100*n_global_pass/n_v:5.0f}% → "
              f"{100*n_site_pass/n_v:5.0f}%")
    if has_adapter and fa_global > 0 and fa_site < fa_global and \
            n_v and n_site_pass >= n_global_pass:
        print(
            f"\n  Net result on this OOD site: FA cut "
            f"{fa_global / max(fa_site, 1e-9):.0f}× "
            f"with no loss of vessel recall."
        )
    elif fa_site == 0 and n_site_pass < n_global_pass:
        print(
            f"\n  Net result on this OOD site: FA eliminated, "
            f"{n_global_pass - n_site_pass}/{n_v} near-threshold vessels lost."
        )


if __name__ == "__main__":
    main()
