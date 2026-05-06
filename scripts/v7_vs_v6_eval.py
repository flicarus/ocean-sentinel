"""Side-by-side eval: v7 vs v6 on AIS-corrected SanctSound.

This is the headline number for whether v7 actually improved generalization.
v6 baseline (already measured): 12% recall on AIS-confirmed ship chunks.

For v7 we want to see this number go up substantially. The evidential head
also lets us threshold by uncertainty — if v7 is uncertain on hard cases
but confident on easy ones, we can quote both raw recall and "confident
recall" (only count predictions where evidence is high enough).

Usage:
    PYTHONPATH=src venv/bin/python scripts/v7_vs_v6_eval.py \\
        --v6 data/models/cnn_v6.pt --v7 data/models/cnn_v7.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from ocean_sentinel.services.cnn_classifier import CNNClassifier
from ocean_sentinel.models.cnn_v7 import OceanSentinelV7

CORRECTED = Path("data/training/sanctsound_corrected.jsonl")

MEL_N = 128
MEL_FREQS = librosa.mel_frequencies(n_mels=MEL_N, fmax=1000.0)
HIGH_PASS_CUTOFF_HZ = 80.0
LOW_FREQ_MASK = MEL_FREQS < HIGH_PASS_CUTOFF_HZ
TARGET_FRAMES = 313


def preprocess_for_v7(spec: np.ndarray) -> torch.Tensor:
    """Mirror SpecDataset preprocessing in train_cnn_v7.py."""
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
    return torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()  # (1,1,128,T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", type=Path, default=Path("data/models/cnn_v6.pt"))
    ap.add_argument("--v7", type=Path, default=Path("data/models/cnn_v7.pt"))
    ap.add_argument("--evidence-threshold", type=float, default=0.0,
                    help="Optional: only count v7 predictions where total "
                         "evidence (sum of alphas - K) exceeds this threshold")
    args = ap.parse_args()

    rows = [json.loads(l) for l in CORRECTED.open()]
    print(f"Eval set: {len(rows)} AIS-corrected SanctSound chunks")

    # v6
    print("\n--- v6 ---")
    v6 = CNNClassifier(args.v6)
    v6_correct = 0
    v6_confidences = []
    for r in rows:
        spec = np.load(r["spectrogram_path"]).astype(np.float32)
        ais_label = r.get("sanctsound_ais_label", r["label"])
        true = "ship" if ais_label in ("ship", "ship_distant") else "ambient"
        pred = v6.predict(spec, source_id="sanctsound")
        pred_label = "ship" if pred["label"] == "ship" else "ambient"
        if pred_label == true:
            v6_correct += 1
        v6_confidences.append(pred["probabilities"]["ship"])
    v6_recall = v6_correct / len(rows)
    print(f"  recall on ship: {v6_recall:.1%}  ({v6_correct}/{len(rows)})")

    # v7
    if not args.v7.exists():
        print(f"\n[v7 checkpoint missing: {args.v7}]")
        return

    print("\n--- v7 ---")
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    v7 = OceanSentinelV7().to(device)
    v7.load_state_dict(torch.load(args.v7, map_location=device))
    v7.eval()

    v7_correct = 0
    v7_correct_confident = 0
    v7_n_confident = 0
    v7_evidence_sums = []
    distance_correct = 0
    distance_total = 0

    DIST_TO_IDX = {"none": 0, "far": 1, "medium": 2, "close": 3}

    with torch.no_grad():
        for r in rows:
            spec_np = np.load(r["spectrogram_path"]).astype(np.float32)
            x = preprocess_for_v7(spec_np).to(device)
            out = v7(x)
            alpha = F.softplus(out["evidence"]) + 1.0
            S = alpha.sum(dim=1)
            evidence_sum = (alpha.sum(dim=1) - 2).item()  # total evidence
            v7_evidence_sums.append(evidence_sum)
            p = (alpha / alpha.sum(dim=1, keepdim=True))[0]
            pred_idx = int(p.argmax().item())
            pred_label = "ship" if pred_idx == 1 else "ambient"

            ais_label = r.get("sanctsound_ais_label", r["label"])
            true = "ship" if ais_label in ("ship", "ship_distant") else "ambient"
            if pred_label == true:
                v7_correct += 1
                if evidence_sum > args.evidence_threshold:
                    v7_correct_confident += 1
            if evidence_sum > args.evidence_threshold:
                v7_n_confident += 1

            # Distance head check
            target_dist = DIST_TO_IDX.get(r.get("distance_bucket"))
            if target_dist is not None:
                dist_pred = int(out["distance"][0].argmax().item())
                if dist_pred == target_dist:
                    distance_correct += 1
                distance_total += 1

    v7_recall = v7_correct / len(rows)
    print(f"  recall on ship:        {v7_recall:.1%}  ({v7_correct}/{len(rows)})")
    if v7_n_confident:
        v7_recall_conf = v7_correct_confident / v7_n_confident
        coverage = v7_n_confident / len(rows)
        print(f"  confident recall:      {v7_recall_conf:.1%} (coverage {coverage:.1%}, "
              f"evidence > {args.evidence_threshold})")
    if distance_total:
        print(f"  distance head accuracy: {distance_correct/distance_total:.1%} "
              f"({distance_correct}/{distance_total})")

    arr = np.array(v7_evidence_sums)
    print(f"  evidence stats: mean={arr.mean():.2f}  median={np.median(arr):.2f}  "
          f"5%={np.quantile(arr, 0.05):.2f}  95%={np.quantile(arr, 0.95):.2f}")

    print("\n=== headline ===")
    print(f"v6 recall:           {v6_recall:.1%}")
    print(f"v7 recall:           {v7_recall:.1%}")
    print(f"v7 - v6:             {(v7_recall - v6_recall) * 100:+.1f} pp")


if __name__ == "__main__":
    main()
