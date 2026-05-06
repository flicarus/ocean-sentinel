"""Compute conformal threshold for v7 against a labelled calibration set.

Usage:
    PYTHONPATH=src venv/bin/python scripts/calibrate_conformal.py \\
        --model data/models/cnn_v7_holdout.pt \\
        --calibration-set data/training/sanctsound_corrected.jsonl \\
        --alpha 0.10 \\
        --out data/calibration/conformal.json

What it does
------------
Runs the model on a held-out labelled set, scores each prediction by
nonconformity (1 - P(true_class)), takes the (1-alpha)-quantile with
finite-sample correction, and writes a JSON the DecisionEngine loads
at app startup.

Held-out is critical: any sample the model saw during training would
make the threshold too tight. SanctSound was explicitly excluded from
v7's training (--exclude-sanctsound), so sanctsound_corrected is a
clean calibration corpus for v7.

Why this matters
----------------
v7's evidential head outputs a saturated confidence (~0.975 for almost
every input), so the raw `confidence` field is useless as a calibration
signal. The conformal lower bound replaces it: for any new prediction,
`lower_bound(p) = max(0, p - threshold)` carries a mathematical
guarantee that the predicted class is correct with probability ≥ 1-alpha
on samples drawn from the calibration distribution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import structlog

sys.path.insert(0, "src")

from ocean_sentinel.decision.conformal import ConformalPredictor
from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier

log = structlog.get_logger()


def _load_calibration_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/cnn_v7_holdout.pt")
    p.add_argument(
        "--calibration-set",
        default="data/training/sanctsound_corrected.jsonl",
        help="JSONL with held-out labelled samples; each row needs "
             "spectrogram_path and label (ship/not_ship).",
    )
    p.add_argument(
        "--alpha", type=float, default=0.10,
        help="Nominal miscoverage. 0.10 = 90% coverage guarantee.",
    )
    p.add_argument(
        "--out", default="data/calibration/conformal.json",
        help="Where to save the calibrated predictor JSON.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap calibration samples for quick smoke tests; "
             "leave unset for the full set.",
    )
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"model not found: {model_path}")
    cal_path = Path(args.calibration_set)
    if not cal_path.exists():
        sys.exit(f"calibration set not found: {cal_path}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clf = CNNV7Classifier(model_path)
    rows = _load_calibration_rows(cal_path)
    if args.limit:
        rows = rows[: args.limit]

    log.info("calibration_start", n_rows=len(rows), model=str(model_path))

    true_class_probs: list[float] = []
    skipped = 0
    for i, r in enumerate(rows):
        spec_path = Path(r["spectrogram_path"])
        truth = r.get("label")
        if not spec_path.exists() or truth not in ("ship", "not_ship"):
            skipped += 1
            continue
        try:
            spec = np.load(spec_path)
            verdict = clf.predict(spec)
        except Exception as e:
            log.warning("calibration_sample_failed", path=str(spec_path), error=str(e))
            skipped += 1
            continue
        # We need P(true_class), not P(predicted_class). predict() returns
        # the full per-class distribution under "probabilities".
        prob = float(verdict["probabilities"].get(truth, 0.0))
        true_class_probs.append(prob)
        if (i + 1) % 100 == 0:
            log.info("calibration_progress", processed=i + 1, total=len(rows))

    if not true_class_probs:
        sys.exit("no usable calibration samples")

    predictor = ConformalPredictor.from_calibration_set(
        true_class_probs=true_class_probs,
        alpha=args.alpha,
        model_checkpoint=str(model_path),
    )
    predictor.save(out_path)

    log.info(
        "calibration_done",
        n_used=len(true_class_probs),
        n_skipped=skipped,
        threshold=round(predictor.threshold, 4),
        alpha=predictor.alpha,
        median_true_prob=round(float(np.median(true_class_probs)), 4),
        out=str(out_path),
    )

    # Print a small summary so users see immediate feedback.
    print(f"\nCalibration complete:")
    print(f"  samples used: {len(true_class_probs)}  (skipped {skipped})")
    print(f"  alpha:        {predictor.alpha}")
    print(f"  threshold:    {predictor.threshold:.4f}")
    print(f"  saved to:     {out_path}")
    print(f"\nlower_bound semantics (for sanity):")
    for p in [0.99, 0.85, 0.65, 0.50]:
        print(f"  raw_prob={p:.2f}  ->  lower_bound={predictor.lower_bound(p):.3f}")


if __name__ == "__main__":
    main()
