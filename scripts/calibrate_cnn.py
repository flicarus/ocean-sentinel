"""Post-hoc temperature scaling for the CNN's confidence calibration.

After training, the model's softmax confidences may not match its accuracy
(typically overconfident under CrossEntropyLoss + Mixup). Temperature scaling
fits a single scalar T so that softmax(logits / T) is calibrated. T is
optimized on the validation split via L-BFGS minimizing NLL — the model's
weights are NOT modified.

Outputs:
  <ckpt>.temperature.json
      {"temperature": T, "nll_before": ..., "nll_after": ...,
       "ece_before": ..., "ece_after": ...}

CNNClassifier auto-loads this file and divides logits by T at inference.
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
    SpecDataset,
    _split_session_indices,
    compute_source_freq_profiles,
)


def expected_calibration_error(
    probs: torch.Tensor, truths: torch.Tensor, n_bins: int = 15,
) -> float:
    """Expected Calibration Error (Naeini et al. 2015).

    Bins predictions by max-confidence; for each bin, computes the gap
    between mean confidence and actual accuracy, weighted by bin size.
    ECE=0 means a model whose stated confidence equals real accuracy.
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(truths).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = truths.size(0)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.float().sum() / n).item() * float(
                (bin_conf - bin_acc).abs()
            )
    return ece


@torch.no_grad()
def collect_val_logits(
    model: OceanSentinelCNN, val_loader: DataLoader, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run val once, stack all raw logits and ground-truth labels."""
    all_logits, all_truths = [], []
    for batch in val_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        all_logits.append(model(x)["vessel"])
        all_truths.append(y)
    return torch.cat(all_logits), torch.cat(all_truths)


def fit_temperature(
    logits: torch.Tensor, truths: torch.Tensor, max_iter: int = 100,
) -> float:
    """L-BFGS over a single scalar T minimizing NLL(logits / T, truths).

    L-BFGS is the standard for low-dim convex problems like this.
    Initial T=1.0 (no scaling) so the optimum is reachable in either direction.
    """
    T = torch.nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / T, truths)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path,
                    default=Path("data/training/gemma_labels.v4.jsonl"))
    ap.add_argument("--ckpt", type=Path,
                    default=Path("data/models/cnn_v6.pt"))
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Reproduce v6's exact val split + source-profile state at training time.
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

    logits, truths = collect_val_logits(model, val_loader, device)

    # L-BFGS doesn't play with MPS — move to CPU for the scalar fit.
    logits, truths = logits.cpu(), truths.cpu()

    probs_before = F.softmax(logits, dim=1)
    nll_before = F.cross_entropy(logits, truths).item()
    ece_before = expected_calibration_error(probs_before, truths)

    T = fit_temperature(logits, truths)

    probs_after = F.softmax(logits / T, dim=1)
    nll_after = F.cross_entropy(logits / T, truths).item()
    ece_after = expected_calibration_error(probs_after, truths)

    out_path = args.ckpt.with_suffix(".temperature.json")
    out_path.write_text(json.dumps({
        "temperature": T,
        "nll_before": nll_before, "nll_after": nll_after,
        "ece_before": ece_before, "ece_after": ece_after,
    }, indent=2))

    print(f"Temperature: T = {T:.4f}")
    print(f"NLL: {nll_before:.4f} -> {nll_after:.4f}")
    print(f"ECE: {ece_before:.4f} -> {ece_after:.4f} "
          f"(lower = better calibrated)")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
