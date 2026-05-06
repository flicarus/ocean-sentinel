"""Layer 3 — split-conformal prediction.

Why conformal
-------------
v7's evidential head emits a saturated confidence (~0.975 across all
inputs); the only calibrated signal it gives is `uncertainty`, and even
that's only ordinally useful, not a probability. Conformal prediction
sits on top and gives us a *statistically guaranteed* lower bound:

    "lower_bound(p) = max(0, p - q)"

where `q` is the (1 - alpha)-quantile of nonconformity scores observed on
a held-out calibration set, and the guarantee is

    P(predicted class is correct | nonconformity ≤ q) ≥ 1 - alpha

across IID samples. Concretely, with alpha = 0.10 we want lower_bound to
mean "the true class has at least this probability with ≥ 90% coverage".

This module is the *infrastructure*. The actual threshold has to come
from running the model on a labelled calibration set the model never saw
in training (e.g. sanctsound_corrected). `scripts/calibrate_conformal.py`
does that and writes a JSON the engine loads at startup.

Design notes
------------
- `from_calibration_set` does the math. It takes the predicted-class
  probabilities for each calibration sample (so for sample i the model
  said class c_i with probability p_i) and the true labels. The score
  is `1 - p_i` when c_i == true_i (the model was right but maybe not
  by much) and `1` when c_i != true_i (max nonconformity for a miss).
- The finite-sample correction `(n+1)/n` (Vovk et al.) prevents the
  empirical quantile from under-covering on small calibration sets.
- We persist (alpha, threshold, n_calibration, model_checkpoint) so the
  audit log can reproduce exactly which calibrator was active for any
  decision.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConformalPredictor:
    """Frozen calibrated thresholder. Build with `from_calibration_set`,
    then call `lower_bound(p)` to get a coverage-guaranteed lower bound.
    """

    alpha: float                       # nominal miscoverage rate, e.g. 0.10
    threshold: float                   # (1-alpha)-quantile of nonconformity scores
    n_calibration: int                 # so we can audit the sample size
    model_checkpoint: str | None = None  # which checkpoint we calibrated against

    def lower_bound(self, predicted_prob: float) -> float:
        """Coverage-guaranteed lower bound on the predicted class probability.

        Caller passes the raw model probability for the predicted class.
        Returns max(0, p - threshold). If the result is below 0.5, the
        decision engine should treat the prediction as not statistically
        differentiated from a coin flip on this calibration distribution.
        """
        return max(0.0, float(predicted_prob) - self.threshold)

    @classmethod
    def from_calibration_set(
        cls,
        true_class_probs: list[float],
        alpha: float = 0.10,
        model_checkpoint: str | None = None,
    ) -> "ConformalPredictor":
        """Build the predictor from a labelled calibration set.

        `true_class_probs[i]` is the probability the model assigned to
        sample i's *true* class — not the predicted class. For binary
        ship/not_ship that's `verdict["probabilities"][true_label]` from
        v7's predict(). The nonconformity score per sample is

            score_i = 1 - true_class_probs[i]

        which is in [0, 1]: 0 means model was perfectly confident on the
        right answer, 1 means it was perfectly confident on the wrong one.
        Using this score (rather than 1 - max_prob_of_predicted) keeps the
        distribution smooth even when the model misclassifies, which lets
        the (1-alpha) quantile actually move with the data.

        The threshold is the ceil((n+1)*(1-alpha)) / n quantile of those
        scores — that's the finite-sample correction (Vovk et al.) so
        coverage holds at small n.
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        n = len(true_class_probs)
        if n == 0:
            raise ValueError("calibration set must be non-empty")

        scores = 1.0 - np.asarray(true_class_probs, dtype=np.float64)
        if scores.min() < 0 or scores.max() > 1:
            raise ValueError(
                f"true_class_probs must be in [0, 1]; got range "
                f"[{1 - scores.max():.3f}, {1 - scores.min():.3f}]"
            )

        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        threshold = float(np.quantile(scores, q_level))

        return cls(
            alpha=alpha,
            threshold=threshold,
            n_calibration=n,
            model_checkpoint=model_checkpoint,
        )

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "n_calibration": self.n_calibration,
            "model_checkpoint": self.model_checkpoint,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ConformalPredictor":
        data = json.loads(Path(path).read_text())
        return cls(
            alpha=float(data["alpha"]),
            threshold=float(data["threshold"]),
            n_calibration=int(data["n_calibration"]),
            model_checkpoint=data.get("model_checkpoint"),
        )

    @classmethod
    def uncalibrated(cls, model_checkpoint: str | None = None) -> "ConformalPredictor":
        """Pass-through predictor for the bootstrap window before
        calibration is run. Threshold = 0 means lower_bound(p) = p, i.e.
        the engine sees the raw probability with no statistical guarantee.
        Logged so the audit trail records that no calibration was active.
        """
        return cls(
            alpha=1.0,
            threshold=0.0,
            n_calibration=0,
            model_checkpoint=model_checkpoint,
        )

    @property
    def is_calibrated(self) -> bool:
        return self.n_calibration > 0
