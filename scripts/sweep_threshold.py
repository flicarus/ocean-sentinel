"""Sweep the fast-path skip-Gemma threshold.

For each threshold τ in [0.50, 0.99], compute on the same val split that
training/calibration used:
  - coverage: % of val chunks that would skip Gemma
                (= chunks predicted `not_ship` with confidence >= τ)
  - false-NONE rate: of those skipped chunks, how many were truly `ship`
                (= ships missed by the fast path)

Pick the highest τ where false-NONE rate is acceptable (< 1-2%) — that
gives maximum Gemma savings without losing real ships.

Reads the calibrated `<ckpt>.temperature.json` if present so the sweep
operates on the same confidences the live system reports.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ocean_sentinel.models.cnn import OceanSentinelCNN
from train_cnn import (
    LABEL_MAP, SpecDataset,
    _split_session_indices, compute_source_freq_profiles,
)


@torch.no_grad()
def collect_predictions(
    model: OceanSentinelCNN,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (probs[N, 2], truths[N]) using calibrated confidences."""
    all_probs, all_truths = [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x)["vessel"] / temperature
        all_probs.append(F.softmax(logits, dim=1))
        all_truths.append(y)
    return torch.cat(all_probs).cpu(), torch.cat(all_truths).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path,
                    default=Path("data/training/gemma_labels.v4.jsonl"))
    ap.add_argument("--ckpt", type=Path,
                    default=Path("data/models/cnn_v6.pt"))
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Same val construction as training/calibration.
    ds_train = SpecDataset(
        str(args.jsonl), representation_version="abs_db_v1", augment=False,
    )
    ds_val = SpecDataset(
        str(args.jsonl), representation_version="abs_db_v1", augment=False,
    )
    train_idx, val_idx = _split_session_indices(ds_train.entries)
    train_entries = [ds_train.entries[i] for i in train_idx]
    profiles = compute_source_freq_profiles(train_entries)
    ds_train.set_source_profiles(profiles)
    ds_val.set_source_profiles(profiles)

    val_loader = DataLoader(Subset(ds_val, val_idx), batch_size=32, shuffle=False)

    model = OceanSentinelCNN().to(device)
    model.load_state_dict(torch.load(str(args.ckpt), map_location=device))
    model.eval()

    temperature_path = args.ckpt.with_suffix(".temperature.json")
    if temperature_path.exists():
        T = float(json.loads(temperature_path.read_text())["temperature"])
        print(f"Using calibrated T = {T:.4f}")
    else:
        T = 1.0
        print("No temperature.json found — using raw softmax (T=1.0)")

    probs, truths = collect_predictions(model, val_loader, device, T)
    confidences, preds = probs.max(dim=1)

    not_ship_idx = LABEL_MAP["not_ship"]
    ship_idx = LABEL_MAP["ship"]
    n_total = truths.size(0)
    n_ship_total = int((truths == ship_idx).sum().item())

    print(f"\nVal set: {n_total} chunks ({n_ship_total} ship, "
          f"{n_total - n_ship_total} not_ship)\n")

    print(f"{'τ':>6s}  {'fast-path':>10s}  {'coverage':>10s}  "
          f"{'false-NONE':>12s}  {'ships missed':>13s}")
    print("-" * 60)

    for tau in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85,
                0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
        # Fast-path triggers iff: prediction is not_ship AND confidence >= tau
        fast_path_mask = (preds == not_ship_idx) & (confidences >= tau)
        n_fast = int(fast_path_mask.sum().item())
        coverage = n_fast / n_total

        # False-NONE: fast-path engaged but truth was actually ship
        ships_in_fast = int(((truths == ship_idx) & fast_path_mask).sum().item())
        false_none_rate = ships_in_fast / n_fast if n_fast else 0.0

        print(
            f"{tau:>6.2f}  {n_fast:>10d}  {coverage:>9.1%}  "
            f"{false_none_rate:>11.2%}  {ships_in_fast:>13d}/{n_ship_total}"
        )

    print("\nReadout:")
    print("  - 'coverage' = fraction of val chunks that would skip Gemma")
    print("  - 'false-NONE rate' = of skipped chunks, fraction that were "
          "actually ships (missed!)")
    print("  - Pick highest τ where false-NONE rate is acceptable (< 1-2%)")


if __name__ == "__main__":
    main()
