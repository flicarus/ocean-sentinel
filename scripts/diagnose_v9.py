"""Diagnose why v9 fails on OOD (oc01/sb01) but does 85% on val.

Test 1 — per-site eval:
  Run v9 on each site separately (using only chunks from sanctsound_60s +
  sanctsound_more_60s). For sites in training, randomly hold out 30 ship
  + 30 ambient. For oc01/sb01 (truly held out), use all available.

If v9 nails in-training sites but fails on oc01/sb01 → distribution shift.
If v9 fails on most sites → architectural / training problem.

Test 2 — confidence calibration:
  Bin predictions by confidence (0.5-0.6, 0.6-0.7, ..., 0.9-1.0) and report
  accuracy per bin. A well-calibrated model: high conf → high acc.

Test 3 — spectrogram statistics per site:
  Mean / std / energy / peak-freq distribution per site. Quantifies how
  different oc01/sb01 are from training.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from ocean_sentinel.models.cnn_v7 import OceanSentinelV7

CKPT = Path("data/models/cnn_v9.pt")
PATHS = [
    Path("data/training/sanctsound_60s.jsonl"),
    Path("data/training/sanctsound_more_60s.jsonl"),
]
TARGET_FRAMES = 1876
MEL_FREQS = librosa.mel_frequencies(n_mels=128, fmax=1000.0)
LOW_FREQ_MASK = MEL_FREQS < 80.0


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


def site_of(row: dict) -> str:
    sid = (row.get("provenance") or {}).get("source_id", "")
    for s in ["oc01", "sb01", "sb02", "sb03", "fk01", "fk02", "fk03",
             "hi01", "hi03", "hi04", "hi06", "mb01", "mb02", "gr01",
             "ci01", "ci02", "ci04", "oc02"]:
        if s in sid:
            return s
    return "unknown"


def main() -> None:
    rows: list[dict] = []
    for p in PATHS:
        with p.open() as f:
            for line in f:
                rows.append(json.loads(line))

    by_site: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_site[site_of(r)].append(r)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model = OceanSentinelV7().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    print("=" * 80)
    print("PER-SITE EVAL — v9 predictions vs AIS labels")
    print("=" * 80)
    print(f"{'site':<8} {'n_ship':>6} {'n_amb':>6} {'ship_acc':>9} {'amb_acc':>9} "
          f"{'mean_conf':>9}  in-train?")
    print("-" * 80)

    rng = np.random.default_rng(42)
    all_preds: list[tuple[float, int]] = []  # (confidence, correct)

    for site in sorted(by_site.keys()):
        ship_rows = [r for r in by_site[site] if r.get("sanctsound_ais_label") == "ship"]
        amb_rows = [r for r in by_site[site] if r.get("sanctsound_ais_label") == "ambient"]
        # Sample up to 60 each to keep it fast
        rng.shuffle(ship_rows)
        rng.shuffle(amb_rows)
        ship_rows = ship_rows[:60]
        amb_rows = amb_rows[:60]

        ship_correct = 0
        amb_correct = 0
        confs: list[float] = []

        with torch.no_grad():
            for r in ship_rows:
                spec = np.load(r["spectrogram_path"]).astype(np.float32)
                x = preprocess(spec).to(device)
                out = model(x)
                alpha = F.softplus(out["evidence"]) + 1.0
                p = (alpha / alpha.sum(dim=1, keepdim=True))[0]
                pred = int(p.argmax().item())
                conf = float(p.max().item())
                if pred == 1:
                    ship_correct += 1
                confs.append(conf)
                all_preds.append((conf, int(pred == 1)))
            for r in amb_rows:
                spec = np.load(r["spectrogram_path"]).astype(np.float32)
                x = preprocess(spec).to(device)
                out = model(x)
                alpha = F.softplus(out["evidence"]) + 1.0
                p = (alpha / alpha.sum(dim=1, keepdim=True))[0]
                pred = int(p.argmax().item())
                conf = float(p.max().item())
                if pred == 0:
                    amb_correct += 1
                confs.append(conf)
                all_preds.append((conf, int(pred == 0)))

        ship_acc = ship_correct / max(len(ship_rows), 1)
        amb_acc = amb_correct / max(len(amb_rows), 1)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        in_train = "NO" if site in ("oc01", "sb01") else "yes"
        print(f"{site:<8} {len(ship_rows):>6} {len(amb_rows):>6} "
              f"{ship_acc:>8.1%}  {amb_acc:>8.1%}  {mean_conf:>9.3f}  {in_train}")

    print()
    print("=" * 80)
    print("CONFIDENCE CALIBRATION (all sites)")
    print("=" * 80)
    print(f"{'conf bin':<12} {'n':>6} {'accuracy':>9}")
    print("-" * 40)
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        bucket = [c for c in all_preds if lo <= c[0] < hi]
        n = len(bucket)
        if n == 0:
            continue
        acc = np.mean([c[1] for c in bucket])
        print(f"{lo:.1f}-{hi:.2f}    {n:>6} {acc:>8.1%}")

    print()
    print("=" * 80)
    print("SPECTROGRAM STATS PER SITE (mean energy in dB, ambient samples only)")
    print("=" * 80)
    print(f"{'site':<8} {'n':>6} {'mean':>8} {'std':>8} {'engine_band':>12}")
    print("-" * 48)
    for site in sorted(by_site.keys()):
        amb = [r for r in by_site[site] if r.get("sanctsound_ais_label") == "ambient"][:30]
        if not amb:
            continue
        means: list[float] = []
        stds: list[float] = []
        engine_band: list[float] = []  # 50-300 Hz mean (vessel engine band)
        for r in amb:
            spec = np.load(r["spectrogram_path"]).astype(np.float32)
            means.append(float(spec.mean()))
            stds.append(float(spec.std()))
            band_mask = (MEL_FREQS >= 50) & (MEL_FREQS <= 300)
            engine_band.append(float(spec[band_mask].mean()))
        print(f"{site:<8} {len(amb):>6} {np.mean(means):>8.1f} {np.mean(stds):>8.1f} "
              f"{np.mean(engine_band):>12.1f}")


if __name__ == "__main__":
    main()
