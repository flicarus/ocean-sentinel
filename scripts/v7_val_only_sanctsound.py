"""Reproduce the v7 train/val split, score v7 on the val-only sanctsound subset.

This is a quick directional check while we wait for the proper held-out
retrain (cnn_v7_holdout.pt) to finish. The v7 we just trained saw ~85% of
the 640 corrected SanctSound chunks during training, so its 82.3% on the
full set is contaminated. This script extracts only the sanctsound chunks
that ended up in the val subset and scores v7 on those — gives an honest
within-distribution number.

For the full OOD held-out claim, wait for cnn_v7_holdout.pt then run
v7_vs_v6_eval.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from ocean_sentinel.models.cnn_v7 import OceanSentinelV7
from ocean_sentinel.services.cnn_classifier import CNNClassifier

PRIMARY = Path("data/training/gemma_labels.v7.jsonl")
SANCT = Path("data/training/sanctsound_corrected.jsonl")
BULK_DIR = Path("data/training/v7_bulk")
V7_CKPT = Path("data/models/cnn_v7.pt")
V6_CKPT = Path("data/models/cnn_v6.pt")
SEED = 42
VAL_FRAC = 0.15

MEL_N = 128
MEL_FREQS = librosa.mel_frequencies(n_mels=MEL_N, fmax=1000.0)
LOW_FREQ_MASK = MEL_FREQS < 80.0
TARGET_FRAMES = 313


def load_rows() -> list[dict]:
    rows: list[dict] = []
    paths = [PRIMARY] + sorted(BULK_DIR.glob("*.jsonl") if BULK_DIR.exists() else [])
    paths.append(SANCT)
    for p in paths:
        if not p.exists():
            continue
        is_primary = p.name == "gemma_labels.v7.jsonl"
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "spectrogram_path" not in r or "label" not in r:
                    continue
                if (
                    is_primary
                    and (r.get("provenance") or {}).get("source_id") == "sanctsound"
                ):
                    continue
                rows.append(r)
    return rows


def preprocess(spec: np.ndarray) -> torch.Tensor:
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


def main() -> None:
    rows = load_rows()
    print(f"Loaded {len(rows)} total rows (mirrors v7 SpecDataset)")

    rng = np.random.default_rng(SEED)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    val_n = int(len(rows) * VAL_FRAC)
    val_idx = set(indices[:val_n].tolist())
    print(f"Random val split: {len(val_idx)} rows (15%)")

    val_sanct = [
        rows[i] for i in val_idx
        if "sanctsound-corrected" in (rows[i].get("provenance") or {}).get("source_id", "")
    ]
    print(f"Val-only sanctsound chunks: {len(val_sanct)}")

    if not val_sanct:
        print("No sanctsound rows in val set. Adjust seed or VAL_FRAC.")
        return

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    v7 = OceanSentinelV7().to(device)
    v7.load_state_dict(torch.load(V7_CKPT, map_location=device))
    v7.eval()

    v6 = CNNClassifier(V6_CKPT)

    v7_correct = 0
    v6_correct = 0
    total = 0
    confusion_v7 = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    confusion_v6 = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    distance_correct = 0
    distance_total = 0
    DIST_TO_IDX = {"none": 0, "far": 1, "medium": 2, "close": 3}

    for r in val_sanct:
        spec_np = np.load(r["spectrogram_path"]).astype(np.float32)
        true_label = r.get("label")
        true_is_ship = true_label == "ship"

        # v7
        with torch.no_grad():
            x = preprocess(spec_np).to(device)
            out = v7(x)
            alpha = F.softplus(out["evidence"]) + 1.0
            probs = (alpha / alpha.sum(dim=1, keepdim=True))[0]
            pred_idx = int(probs.argmax().item())
        v7_pred_is_ship = pred_idx == 1
        if v7_pred_is_ship == true_is_ship:
            v7_correct += 1
        if true_is_ship and v7_pred_is_ship:    confusion_v7["tp"] += 1
        elif not true_is_ship and not v7_pred_is_ship: confusion_v7["tn"] += 1
        elif true_is_ship and not v7_pred_is_ship: confusion_v7["fn"] += 1
        elif not true_is_ship and v7_pred_is_ship: confusion_v7["fp"] += 1

        # v6
        v6_out = v6.predict(spec_np, source_id="sanctsound")
        v6_pred_is_ship = v6_out["label"] == "ship"
        if v6_pred_is_ship == true_is_ship:
            v6_correct += 1
        if true_is_ship and v6_pred_is_ship:    confusion_v6["tp"] += 1
        elif not true_is_ship and not v6_pred_is_ship: confusion_v6["tn"] += 1
        elif true_is_ship and not v6_pred_is_ship: confusion_v6["fn"] += 1
        elif not true_is_ship and v6_pred_is_ship: confusion_v6["fp"] += 1

        # distance
        dist_target = DIST_TO_IDX.get(r.get("distance_bucket"))
        if dist_target is not None:
            with torch.no_grad():
                dist_pred = int(out["distance"][0].argmax().item())
            if dist_pred == dist_target:
                distance_correct += 1
            distance_total += 1

        total += 1

    print(f"\n=== Val-only sanctsound ({total} chunks) ===")
    print(f"v6 accuracy: {v6_correct/total:.1%} ({v6_correct}/{total})")
    print(f"  v6 confusion: {confusion_v6}")
    print(f"v7 accuracy: {v7_correct/total:.1%} ({v7_correct}/{total})")
    print(f"  v7 confusion: {confusion_v7}")
    print(f"v7 distance head: {distance_correct/distance_total:.1%} ({distance_correct}/{distance_total})")
    print(f"\nGap: v7 - v6 = {(v7_correct-v6_correct)/total*100:+.1f} pp")


if __name__ == "__main__":
    main()
