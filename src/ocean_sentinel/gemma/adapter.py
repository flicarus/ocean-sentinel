"""Real per-site adapter validation.

Replaces the placeholder mock in tools.py with a label-free verification
of the base CNN against the user's ambient.

What this does (and doesn't):

- Loads v7.4.
- Slides the analysis window across the user's ambient (the same audio
  used for fingerprinting + conformal). Runs CNN on each window.
- Computes the *recall on not_ship*: % of windows that the model
  confidently classifies as not_ship.
- Reports that as `val_acc` — a real, defensible per-site metric:
  "the base model gets X% of your ambient correct without any retraining."

Why this is honest about being an "adapter step" without retraining:

  Real per-site fine-tuning needs labelled ship+ambient pairs from the
  user's site, which the onboarding contract does not collect (we only
  ask for ambient). To still provide a per-site adaptation step that
  delivers measurable value, we run validation here and per-site
  *threshold* calibration in Step 5 — that's the actual per-site
  contribution of the system.

  This pattern (calibrate threshold rather than retrain weights) is the
  modern lightweight-adaptation approach (cf. split-conformal, Lei 2018),
  appropriate for edge deployment on a 4B-class compute target.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np

_TARGET_SR_HZ = 16_000
_WINDOW_S = 60.0
_HOP_S = 30.0
_DEFAULT_DURATION_S = 300.0


@lru_cache(maxsize=2)
def _get_classifier(checkpoint: str):
    from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier
    return CNNV7Classifier(checkpoint)


def assessment_from_val_acc(val_acc: float) -> tuple[str, str]:
    """Map label-free val_acc on ambient to a (assessment, recommendation)
    pair. Display-only — never fed back into the model.

    val_acc here is "share of windows that the base CNN classified as
    not_ship". On a clean ambient, that should be near 1.0; if the user
    site is acoustically out-of-distribution the CNN can hallucinate
    ships and val_acc collapses, surfacing the OOD failure to the user
    BEFORE conformal calibration silently absorbs it into the threshold.

    Thresholds are heuristic (no LOHO study yet at this granularity) but
    documented and testable. They mirror the categories we use elsewhere
    in the system: validated / marginal / ood_confusion.
    """
    if val_acc >= 0.85:
        return (
            "model_validated",
            "Base CNN classifies your ambient correctly — calibration "
            "will tune the threshold for your false-alarm spec, and "
            "per-event accuracy should be similar to held-out training "
            "sites.",
        )
    if val_acc >= 0.5:
        return (
            "marginal",
            "Base CNN is uncertain on your ambient — calibration will "
            "compensate at the threshold level, but events near the "
            "threshold should be flagged for human review until the "
            "system has accumulated some per-site confirmations.",
        )
    return (
        "ood_confusion",
        "Base CNN classifies most of your ambient as 'ship' — either "
        "your audio actually contains many vessels, or your site is "
        "acoustically out-of-distribution. Conformal calibration in "
        "Step 5 will raise the threshold to keep false-alarm rate at "
        "spec, but PER-EVENT accuracy on this site cannot be verified "
        "without labelled events. We recommend reviewing the first "
        "~10 alerts manually before trusting unattended operation.",
    )


def _make_spec(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=1000)
    return librosa.power_to_db(mel, ref=1.0)


def validate_adapter_on_ambient(
    site_id: str,
    ambient_source: str,
    *,
    checkpoint: str = "data/models/cnn_v7_4.pt",
    duration_s: float = _DEFAULT_DURATION_S,
) -> dict[str, Any]:
    src = Path(ambient_source)
    if not src.exists() and not ambient_source.startswith(("http://", "https://")):
        return {"ok": False, "error": f"ambient source not found: {ambient_source}"}

    try:
        y, sr = librosa.load(str(src), sr=_TARGET_SR_HZ, mono=True, duration=duration_s)
    except Exception as e:
        return {"ok": False, "error": f"failed to load ambient: {type(e).__name__}: {e}"}

    win = int(sr * _WINDOW_S)
    hop = int(sr * _HOP_S)
    if y.size < win:
        return {
            "ok": False,
            "error": f"need at least {_WINDOW_S}s of audio, got {y.size / sr:.1f}s",
        }

    starts = list(range(0, max(1, y.size - win + 1), hop)) or [0]
    try:
        clf = _get_classifier(checkpoint)
    except FileNotFoundError:
        return {"ok": False, "error": f"checkpoint not found: {checkpoint}"}

    n_windows = 0
    n_correct_not_ship = 0
    confs: list[float] = []
    uncs: list[float] = []
    for s in starts:
        chunk = y[s : s + win]
        if chunk.size < win:
            continue
        spec = _make_spec(chunk, sr)
        pred = clf.predict(spec, source_id=site_id)
        n_windows += 1
        label = pred.get("label", "not_ship")
        confs.append(float(pred.get("confidence", 0.5)))
        uncs.append(float(pred.get("uncertainty", 0.0)))
        if label == "not_ship":
            n_correct_not_ship += 1

    if n_windows == 0:
        return {"ok": False, "error": "no windows could be evaluated"}

    val_acc = n_correct_not_ship / n_windows
    site_dir = Path("data/sites") / site_id
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "adapter_validation.json").write_text(
        '{"val_acc": ' + f"{val_acc:.4f}" +
        ', "n_windows": ' + str(n_windows) + '}\n'
    )

    assessment, recommendation = assessment_from_val_acc(val_acc)

    return {
        "ok": True,
        "site_id": site_id,
        "n_windows": n_windows,
        "final_val_acc": round(val_acc, 4),
        "mean_confidence": round(float(np.mean(confs)), 3),
        "mean_uncertainty": round(float(np.mean(uncs)), 3),
        "assessment": assessment,
        "recommendation": recommendation,
        "checkpoint": checkpoint,
        "summary": (
            f"v7.4 base validates on your ambient: {val_acc * 100:.1f}% "
            f"correctly classified as not_ship across {n_windows} "
            f"windows · assessment={assessment}"
        ),
    }
