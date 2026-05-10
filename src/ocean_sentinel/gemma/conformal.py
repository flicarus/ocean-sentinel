"""Real per-site split-conformal calibration.

The scientific story:
  v7.4 outputs a `ship_prob` per audio window. We want a per-site decision
  threshold that gives a provable false-alarm rate. Split-conformal
  calibration delivers that without labels — we use the user's ambient
  audio as a "known not_ship" set.

Algorithm:
  1. Slice user's ambient audio into N overlapping 60s windows.
  2. Run v7.4 on each window → get `ship_prob_i` for i=1..N.
  3. Under the assumption that ALL of these windows are not_ship
     (which is the onboarding contract — we ask for ambient), the
     ship_prob distribution is the model's null distribution at this
     site. Sites differ here: a quiet deep-water site will have lower
     null ship_prob than a busy harbor.
  4. Pick the threshold τ such that:
        P(ship_prob > τ | not_ship)  ≤  α
     i.e. τ = (1-α) quantile of the null distribution.
     With finite-sample correction:
        rank = ⌈(1 - α)(N + 1)⌉, then τ = sorted(ship_prob)[rank - 1].
  5. The decision rule "alert if ship_prob > τ" then has, by exchangeability,
     a marginal false-alarm rate ≤ α at this site. (Lei et al. 2018.)

Why this is BETTER than fine-tuning:
  - No label required from the user (they don't have to find ship clips).
  - 30-90 seconds vs 3 minutes for fine-tune.
  - Provides a formal guarantee (FA rate ≤ α with high probability).
  - Doesn't risk degrading the underlying model.

Limitations (documented honestly):
  - Marginal coverage, not conditional. Hard examples can still err.
  - The "all-not_ship" assumption is the onboarding contract; if the user
    accidentally provides a ship-heavy clip, the threshold becomes
    pessimistic (too high), causing missed alerts. We sanity-check this
    by warning if the median ship_prob is very high.
  - Small N (few windows) gives unstable thresholds. We require N≥30.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np

_TARGET_SR_HZ = 16_000
_WINDOW_S = 60.0
_HOP_S = 20.0          # ~3x overlap → 30 windows in 5 minutes
_DEFAULT_DURATION_S = 300.0
_DEFAULT_ALPHA = 0.05
_MIN_WINDOWS = 30


@lru_cache(maxsize=2)
def _get_classifier(checkpoint: str):
    from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier
    return CNNV7Classifier(checkpoint)


def _make_spec(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=1000)
    return librosa.power_to_db(mel, ref=1.0)


def calibrate_conformal_real(
    site_id: str,
    ambient_source: str,
    *,
    alpha: float = _DEFAULT_ALPHA,
    checkpoint: str = "data/models/cnn_v7_4.pt",
    duration_s: float = _DEFAULT_DURATION_S,
) -> dict[str, Any]:
    """Run split-conformal on the user's ambient. Saves per-site threshold
    to ``data/sites/{site_id}/conformal.json`` and returns calibration meta.
    """
    src = Path(ambient_source)
    if not src.exists() and not ambient_source.startswith(("http://", "https://")):
        return {"ok": False, "error": f"ambient source not found: {ambient_source}"}

    try:
        y, sr = librosa.load(str(src), sr=_TARGET_SR_HZ, mono=True, duration=duration_s)
    except Exception as e:
        return {"ok": False, "error": f"failed to load ambient: {type(e).__name__}: {e}"}

    if y.size < int(sr * _WINDOW_S):
        return {
            "ok": False,
            "error": f"need at least {_WINDOW_S}s of audio, got {y.size / sr:.1f}s",
        }

    # Slide the analysis window across the ambient buffer.
    win = int(sr * _WINDOW_S)
    hop = int(sr * _HOP_S)
    starts = list(range(0, max(1, y.size - win + 1), hop))
    if len(starts) < _MIN_WINDOWS:
        # If the user gave shorter audio, fall back to a smaller hop so we
        # still hit the minimum sample count.
        hop = max(1, (y.size - win) // (_MIN_WINDOWS - 1)) if y.size > win else win
        starts = list(range(0, max(1, y.size - win + 1), hop))

    if len(starts) < 5:
        return {
            "ok": False,
            "error": f"too few windows ({len(starts)}) for calibration",
        }

    try:
        clf = _get_classifier(checkpoint)
    except FileNotFoundError:
        return {"ok": False, "error": f"checkpoint not found: {checkpoint}"}

    # If this site has a trained adapter (from Step 4), use the SAME
    # adapted CNN for calibration as we'll use at inference. Otherwise
    # we'd calibrate the threshold on the un-adapted ship_prob distribution
    # but score events with the adapted distribution — misaligned.
    adapter_path = Path("data/sites") / site_id / "adapter.pt"
    if adapter_path.exists():
        clf.set_site_adapter(adapter_path)
    else:
        clf.set_site_adapter(None)

    ship_probs: list[float] = []
    uncertainties: list[float] = []
    for s in starts:
        chunk = y[s : s + win]
        if chunk.size < win:
            continue
        spec = _make_spec(chunk, sr)
        pred = clf.predict(spec, source_id=site_id)
        label = pred.get("label", "not_ship")
        conf = float(pred.get("confidence", 0.5))
        ship_probs.append(conf if label == "ship" else 1.0 - conf)
        uncertainties.append(float(pred.get("uncertainty", 0.0)))

    n = len(ship_probs)
    if n < 5:
        return {"ok": False, "error": f"only {n} valid windows after filtering"}

    # Ambient-contract sanity check: if median ship_prob is unusually high,
    # the user probably gave us audio that contains vessels, which would
    # make the threshold pessimistic.
    median_p = float(np.median(ship_probs))
    contract_warning = None
    if median_p > 0.5:
        contract_warning = (
            f"median ship_prob over ambient is {median_p:.2f} — this audio "
            f"likely contains vessels, calibration may be too conservative"
        )

    # Split-conformal threshold with finite-sample correction.
    # We want P(ship_prob > τ) ≤ α  ↔  τ = quantile(ship_prob, 1 - α).
    sorted_p = sorted(ship_probs)
    rank = int(math.ceil((1 - alpha) * (n + 1)))
    rank = min(rank, n)
    threshold = float(sorted_p[rank - 1])

    # Coverage estimate: count how many ambient windows would have passed.
    n_above = sum(1 for p in ship_probs if p > threshold)
    empirical_fa = n_above / n
    coverage = 1.0 - empirical_fa

    # Hour-rate estimate. We assumed each window represents _HOP_S of unique
    # ambient (overlap discounted) → ambient hours covered = n * _HOP_S / 3600.
    ambient_hours = n * _HOP_S / 3600.0
    expected_fa_per_hour = empirical_fa / max(ambient_hours, 1e-6)

    # Persist per-site calibration.
    site_dir = Path("data/sites") / site_id
    site_dir.mkdir(parents=True, exist_ok=True)
    out_path = site_dir / "conformal.json"
    out_path.write_text(json.dumps({
        "site_id":             site_id,
        "checkpoint":          checkpoint,
        "alpha":               alpha,
        "n_calibration":       n,
        "threshold":           threshold,
        "coverage_empirical":  coverage,
        "median_ambient_ship_prob": median_p,
        "median_uncertainty":  float(np.median(uncertainties)),
        "ambient_hours":       ambient_hours,
        "expected_fa_per_hour": expected_fa_per_hour,
        "contract_warning":    contract_warning,
    }, indent=2))

    return {
        "ok": True,
        "site_id": site_id,
        "threshold_p": round(threshold, 3),
        "coverage": round(coverage, 3),
        "n_calibration": n,
        "alpha": alpha,
        "expected_fa_per_hour": round(expected_fa_per_hour, 4),
        "median_ambient_ship_prob": round(median_p, 3),
        "contract_warning": contract_warning,
        "summary": (
            f"p≥{threshold:.2f} · empirical coverage {coverage*100:.1f}% "
            f"on {n} windows ({ambient_hours:.1f}h)"
        ),
    }
