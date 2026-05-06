"""Balanced ship/ambient eval — fair comparison of v6, v7_holdout, v7_aug.

The OOD eval on sanctsound_corrected (640 chunks, all AIS-positive) only
measures recall. A model that always predicts "ship" would score 100% on
that test but be useless in production.

This script picks a class-balanced eval: 320 ship-positive (from
sanctsound_corrected) + 320 ambient (from MBARI, deep canyon with no surface
vessels). Then we report:
  - accuracy (overall %)
  - recall on ship (sensitivity)
  - precision on ship (when model says ship, is it right?)
  - F1 (harmonic mean of recall + precision)
  - false positive rate (recall on ambient flipped)

This separates "good detector" from "ship-everything detector".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from ocean_sentinel.services.cnn_classifier import CNNClassifier
from ocean_sentinel.models.cnn_v7 import OceanSentinelV7

CORRECTED = Path("data/training/sanctsound_corrected.jsonl")
PRIMARY = Path("data/training/gemma_labels.v7.jsonl")
N_SHIP = 320
N_AMBIENT = 320
SEED = 42

MEL_N = 128
MEL_FREQS = librosa.mel_frequencies(n_mels=MEL_N, fmax=1000.0)
LOW_FREQ_MASK = MEL_FREQS < 80.0
TARGET_FRAMES = 313


def preprocess_v7(spec: np.ndarray) -> torch.Tensor:
    spec = spec.astype(np.float32)
    if spec.shape[1] > TARGET_FRAMES:
        s = (spec.shape[1] - TARGET_FRAMES) // 2
        spec = spec[:, s:s + TARGET_FRAMES]
    elif spec.shape[1] < TARGET_FRAMES:
        spec = np.pad(spec, ((0, 0), (0, TARGET_FRAMES - spec.shape[1])), mode="edge")
    spec = spec.copy()
    high_mean = float(spec[~LOW_FREQ_MASK].mean())
    spec[LOW_FREQ_MASK, :] = high_mean
    spec = (spec - spec.mean()) / (spec.std() + 1e-8)
    return torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()


def metrics(y_true: list[int], y_pred: list[int]) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = len(y_true)
    accuracy = (tp + tn) / n
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return dict(
        accuracy=accuracy, recall=recall, precision=precision, f1=f1,
        fpr=fpr, tp=tp, fp=fp, tn=tn, fn=fn,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", type=Path, default=Path("data/models/cnn_v6.pt"))
    ap.add_argument("--v7-holdout", type=Path,
                    default=Path("data/models/cnn_v7_holdout.pt"))
    ap.add_argument("--v7-aug", type=Path,
                    default=Path("data/models/cnn_v7_aug_holdout.pt"))
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)

    # Load ship positives from sanctsound_corrected
    ship_rows = [r for r in (json.loads(l) for l in CORRECTED.open())
                 if r.get("sanctsound_ais_label") in ("ship", "ship_distant")]
    rng.shuffle(ship_rows)
    ship_eval = ship_rows[:N_SHIP]

    # Load ambient from MBARI (in primary jsonl, source_id=mbari)
    ambient_rows = []
    with PRIMARY.open() as f:
        for line in f:
            r = json.loads(line)
            if (r.get("provenance") or {}).get("source_id") == "mbari":
                ambient_rows.append(r)
    rng.shuffle(ambient_rows)
    ambient_eval = ambient_rows[:N_AMBIENT]

    print(f"Eval set: {len(ship_eval)} ship + {len(ambient_eval)} ambient = "
          f"{len(ship_eval) + len(ambient_eval)} balanced")
    print()

    eval_set = [(r, 1) for r in ship_eval] + [(r, 0) for r in ambient_eval]

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    def run_v7(ckpt: Path, label: str) -> None:
        model = OceanSentinelV7().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for r, true_label in eval_set:
                spec = np.load(r["spectrogram_path"]).astype(np.float32)
                x = preprocess_v7(spec).to(device)
                out = model(x)
                alpha = F.softplus(out["evidence"]) + 1.0
                p = (alpha / alpha.sum(dim=1, keepdim=True))[0]
                y_pred.append(int(p.argmax().item()))
                y_true.append(true_label)
        m = metrics(y_true, y_pred)
        print(f"=== {label} ===")
        print(f"  accuracy:  {m['accuracy']:.1%}")
        print(f"  recall:    {m['recall']:.1%}  (sensitivity on ship)")
        print(f"  precision: {m['precision']:.1%}  (when says ship, is right)")
        print(f"  F1:        {m['f1']:.3f}")
        print(f"  FPR:       {m['fpr']:.1%}  (false alarms on ambient)")
        print(f"  confusion: TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
        print()

    def run_v6(ckpt: Path, label: str) -> None:
        clf = CNNClassifier(ckpt)
        y_true, y_pred = [], []
        for r, true_label in eval_set:
            spec = np.load(r["spectrogram_path"]).astype(np.float32)
            sid = (r.get("provenance") or {}).get("source_id", "")
            sid_for_v6 = "sanctsound" if sid.startswith("sanctsound") else sid
            out = clf.predict(spec, source_id=sid_for_v6)
            y_pred.append(1 if out["label"] == "ship" else 0)
            y_true.append(true_label)
        m = metrics(y_true, y_pred)
        print(f"=== {label} ===")
        print(f"  accuracy:  {m['accuracy']:.1%}")
        print(f"  recall:    {m['recall']:.1%}")
        print(f"  precision: {m['precision']:.1%}")
        print(f"  F1:        {m['f1']:.3f}")
        print(f"  FPR:       {m['fpr']:.1%}")
        print(f"  confusion: TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
        print()

    if args.v6.exists():
        run_v6(args.v6, "v6 (legacy)")
    if args.v7_holdout.exists():
        run_v7(args.v7_holdout, "v7_holdout (no aug, no diverse)")
    if args.v7_aug.exists():
        run_v7(args.v7_aug, "v7_aug (augment + diverse)")


if __name__ == "__main__":
    main()
