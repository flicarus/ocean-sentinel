"""Balanced ship/ambient eval at 60s context — fair comparison for v8/v9.

Uses the held-out OOD test set: oc01 + sb01 chunks from sanctsound_60s.jsonl
(filtered out of training when --exclude-sanctsound is set).

Eval set: N ship-positive + N ambient (from oc01/sb01 quiet hours,
AIS-confirmed no vessel within 50km), balanced 50/50.
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
from ocean_sentinel.models.cnn_v7 import OceanSentinelV7

SANCT_60S = Path("data/training/sanctsound_60s.jsonl")
SEED = 42

MEL_N = 128
MEL_FREQS = librosa.mel_frequencies(n_mels=MEL_N, fmax=1000.0)
LOW_FREQ_MASK = MEL_FREQS < 80.0
TARGET_FRAMES = 1876  # 60s context


def preprocess_v8(spec: np.ndarray) -> torch.Tensor:
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
    return dict(accuracy=accuracy, recall=recall, precision=precision, f1=f1,
                fpr=fpr, tp=tp, fp=fp, tn=tn, fn=fn)


def load_oc_sb_eval(n_per_class: int) -> tuple[list, list]:
    ship_rows: list = []
    ambient_rows: list = []
    with SANCT_60S.open() as f:
        for line in f:
            r = json.loads(line)
            sid = (r.get("provenance") or {}).get("source_id", "")
            if "oc01" not in sid and "sb01" not in sid:
                continue
            ais = r.get("sanctsound_ais_label", "")
            if ais == "ship":
                ship_rows.append(r)
            elif ais == "ambient":
                ambient_rows.append(r)
    rng = np.random.default_rng(SEED)
    rng.shuffle(ship_rows)
    rng.shuffle(ambient_rows)
    n = min(n_per_class, len(ship_rows), len(ambient_rows))
    return ship_rows[:n], ambient_rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True,
                    help="Path to v8/v9 .pt checkpoint")
    ap.add_argument("--n-per-class", type=int, default=240,
                    help="Number of ship and ambient samples (max 240)")
    ap.add_argument("--label", type=str, default="model",
                    help="Display label for the model")
    args = ap.parse_args()

    ship_eval, ambient_eval = load_oc_sb_eval(args.n_per_class)
    print(f"Eval set: {len(ship_eval)} ship + {len(ambient_eval)} ambient = "
          f"{len(ship_eval) + len(ambient_eval)} balanced (oc01+sb01 OOD)")
    print()
    eval_set = [(r, 1) for r in ship_eval] + [(r, 0) for r in ambient_eval]

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    model = OceanSentinelV7().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    y_true, y_pred, confidences = [], [], []
    with torch.no_grad():
        for r, true_label in eval_set:
            spec = np.load(r["spectrogram_path"]).astype(np.float32)
            x = preprocess_v8(spec).to(device)
            out = model(x)
            alpha = F.softplus(out["evidence"]) + 1.0
            p = (alpha / alpha.sum(dim=1, keepdim=True))[0]
            y_pred.append(int(p.argmax().item()))
            y_true.append(true_label)
            confidences.append(float(p.max().item()))

    m = metrics(y_true, y_pred)
    print(f"=== {args.label} ({args.model.name}) ===")
    print(f"  accuracy:  {m['accuracy']:.1%}")
    print(f"  recall:    {m['recall']:.1%}  (sensitivity on ship)")
    print(f"  precision: {m['precision']:.1%}  (when says ship, is right)")
    print(f"  F1:        {m['f1']:.3f}")
    print(f"  FPR:       {m['fpr']:.1%}  (false alarms on ambient)")
    print(f"  confusion: TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    print(f"  mean conf: {np.mean(confidences):.3f}")


if __name__ == "__main__":
    main()
