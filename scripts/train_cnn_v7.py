"""Train OceanSentinelV7 — temporal-aware ship/ambient classifier with
evidential uncertainty and optional auxiliary heads.

Inputs (read from JSONL):
  - existing data/training/gemma_labels.v7.jsonl (5499 rows)
  - new   data/training/v7_bulk/*.jsonl         (per-node bulk pulls)
  - future data/training/sanctsound_corrected.jsonl (when re-labeling lands)

Each spectrogram is center-cropped/padded to a fixed-length window before
training. We standardize on 313 frames (~10s at 16 kHz / 512 hop) because
that's the largest size we can reliably extract today across all sources.
The model architecture handles variable-length inputs but training is
faster with fixed shapes.

Multi-task heads:
  - vessel head (primary): binary ship / not_ship, evidential output
  - distance head (auxiliary): close / medium / far / none — only loaded
    when row has 'distance_bucket' field; otherwise masked out
  - vessel_type head (auxiliary): cargo/tanker/fishing/passenger/none —
    only loaded when row has vessel type info

Evidential loss (Sensoy 2018): alpha = softplus(logits) + 1; loss = MSE
between target one-hot and alpha/S, plus a KL term that pushes alpha
toward 1 (uniform/uncertain) when the model is wrong. This gives the
model a way to express "I don't know" via flat alphas.

Usage:
    PYTHONPATH=src venv/bin/python scripts/train_cnn_v7.py \\
        --epochs 30 --batch-size 32 --lr 3e-4 \\
        --out data/models/cnn_v7.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, "src")

from ocean_sentinel.models.cnn_v7 import OceanSentinelV7

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Constants matching CNNClassifier preprocessing.
# ---------------------------------------------------------------------------

MEL_N = 128
MEL_FMAX = 1000.0
MEL_FREQS = librosa.mel_frequencies(n_mels=MEL_N, fmax=MEL_FMAX)
HIGH_PASS_CUTOFF_HZ = 80.0
LOW_FREQ_MASK = MEL_FREQS < HIGH_PASS_CUTOFF_HZ
LABEL_TO_IDX = {"not_ship": 0, "ship": 1}
DISTANCE_TO_IDX = {"none": 0, "far": 1, "medium": 2, "close": 3}
VESSEL_TYPE_TO_IDX = {
    "none": 0, "cargo_ship": 1, "tanker": 2,
    "fishing_vessel": 3, "passenger_vessel": 4,
}

# Default target frames matches the 5-10s spectrograms we used through v7.
# For 60s windows (v8) override via SpecDataset(..., target_frames=1876).
DEFAULT_TARGET_FRAMES = 313


# ---------------------------------------------------------------------------
# Dataset.
# ---------------------------------------------------------------------------

def specaugment(spec: np.ndarray, time_mask_max: int = 30, freq_mask_max: int = 18,
                n_time_masks: int = 2, n_freq_masks: int = 2) -> np.ndarray:
    """SpecAugment (Park et al. 2019) — replace random time/freq strips with
    the mean. Forces the model to use redundant cues across time + frequency
    instead of memorizing exact features. Standard for audio classification.
    """
    spec = spec.copy()
    fill = float(spec.mean())
    T = spec.shape[1]
    F_ = spec.shape[0]
    for _ in range(n_time_masks):
        if T > 0:
            t = np.random.randint(0, time_mask_max + 1)
            t0 = np.random.randint(0, max(1, T - t))
            spec[:, t0:t0 + t] = fill
    for _ in range(n_freq_masks):
        if F_ > 0:
            f = np.random.randint(0, freq_mask_max + 1)
            f0 = np.random.randint(0, max(1, F_ - f))
            spec[f0:f0 + f, :] = fill
    return spec


class SpecDataset(Dataset):
    """Loads (spectrogram, labels) tuples from one or more JSONL manifest files.

    Spectrograms are standardized at __getitem__ time:
      - center-crop or right-pad to self.target_frames
      - high-pass below 80 Hz
      - z-score normalize per sample
      - (training only) SpecAugment time + freq masking

    Auxiliary labels are emitted as -100 (loss ignore_index) when missing.
    """

    def __init__(self, jsonl_paths: list[Path], skip_sanctsound_in_primary: bool = True,
                 augment: bool = False, target_frames: int = DEFAULT_TARGET_FRAMES,
                 exclude_source_substrings: tuple[str, ...] = ()) -> None:
        self.augment = augment
        self.target_frames = target_frames
        self.rows: list[dict] = []
        n_filtered = 0
        for p in jsonl_paths:
            if not p.exists():
                continue
            is_primary = "v7.jsonl" in p.name
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "spectrogram_path" not in row or "label" not in row:
                        continue
                    if (
                        is_primary
                        and skip_sanctsound_in_primary
                        and (row.get("provenance") or {}).get("source_id") == "sanctsound"
                    ):
                        continue
                    if exclude_source_substrings:
                        sid = (row.get("provenance") or {}).get("source_id", "")
                        if any(s in sid for s in exclude_source_substrings):
                            n_filtered += 1
                            continue
                    self.rows.append(row)
        if n_filtered:
            log.info("dataset_filtered", n_filtered=n_filtered,
                     exclude=list(exclude_source_substrings))
        log.info("dataset_loaded", n=len(self.rows), files=len(jsonl_paths))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        spec = np.load(row["spectrogram_path"]).astype(np.float32)

        # Center-crop or pad to self.target_frames.
        if spec.shape[1] > self.target_frames:
            start = (spec.shape[1] - self.target_frames) // 2
            spec = spec[:, start:start + self.target_frames]
        elif spec.shape[1] < self.target_frames:
            pad = self.target_frames - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad)), mode="edge")

        # High-pass: zero out sub-80 Hz (replace with mean of >80 Hz).
        spec = spec.copy()
        high_mean = float(spec[~LOW_FREQ_MASK].mean())
        spec[LOW_FREQ_MASK, :] = high_mean

        # Z-score normalize per sample.
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)

        # Training-only augmentation.
        if self.augment:
            spec = specaugment(spec)

        binary_idx = LABEL_TO_IDX.get(row.get("label"), -100)
        distance_idx = DISTANCE_TO_IDX.get(row.get("distance_bucket"), -100)

        vessel_type_str = (row.get("gemma_verdict") or {}).get("vessel_type")
        vessel_type_idx = VESSEL_TYPE_TO_IDX.get(vessel_type_str, -100)

        source_id = (row.get("provenance") or {}).get("source_id") or row.get("source_id") or "unknown"

        return {
            "spec": torch.from_numpy(spec).unsqueeze(0).float(),  # (1, 128, T)
            "binary": torch.tensor(binary_idx, dtype=torch.long),
            "distance": torch.tensor(distance_idx, dtype=torch.long),
            "vessel_type": torch.tensor(vessel_type_idx, dtype=torch.long),
            "source_id": source_id,
        }


# ---------------------------------------------------------------------------
# Evidential loss (Sensoy 2018).
# ---------------------------------------------------------------------------

def evidential_loss(
    evidence: torch.Tensor, target: torch.Tensor, num_classes: int = 2,
    annealing_coef: float = 1.0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sensoy et al. 2018 — evidential MSE + KL regularization.

    evidence: (B, K) raw output; we apply softplus+1 to get Dirichlet alphas.
    target:   (B,) class indices.
    class_weights: optional (K,) tensor, broadcasted to per-sample weight at
                   the target class. Used to counter dataset imbalance — our
                   v7 corpus skews ~75/25 ship/ambient, so ambient examples
                   should weigh more.

    The MSE term pulls the predicted mean toward the true class. The KL term
    penalizes evidence assigned to the wrong class — when the model is right,
    it can be confident; when wrong, alphas should collapse toward 1 (uniform).
    """
    alpha = F.softplus(evidence) + 1.0          # (B, K)
    S = alpha.sum(dim=1, keepdim=True)          # (B, 1)
    p = alpha / S                                # mean prob

    y = F.one_hot(target, num_classes=num_classes).float()  # (B, K)

    # MSE between target and predicted mean, plus variance term.
    err = (y - p).pow(2).sum(dim=1)
    var = (p * (1 - p) / (S + 1)).sum(dim=1)
    mse = err + var

    # KL divergence to uniform Dirichlet on the WRONG-class evidence.
    alpha_tilde = y + (1 - y) * alpha
    K = num_classes
    sum_alpha_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(sum_alpha_tilde)
        - torch.lgamma(torch.tensor(float(K)))
        - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
        + ((alpha_tilde - 1) * (
            torch.digamma(alpha_tilde) - torch.digamma(sum_alpha_tilde)
        )).sum(dim=1, keepdim=True)
    ).squeeze(1)

    per_sample = mse + annealing_coef * kl
    if class_weights is not None:
        w = class_weights[target]
        per_sample = per_sample * w
    return per_sample.mean()


def compute_class_weights(dataset: SpecDataset, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency weights so under-represented classes carry more loss."""
    counts = torch.zeros(num_classes)
    for r in dataset.rows:
        idx = LABEL_TO_IDX.get(r.get("label"), -1)
        if 0 <= idx < num_classes:
            counts[idx] += 1
    # Avoid div by zero on missing classes.
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------

def evaluate(
    model: OceanSentinelV7, loader: DataLoader, device: torch.device,
) -> dict:
    model.eval()
    correct = 0
    total = 0
    confusion = Counter()
    aux_distance_correct = 0
    aux_distance_total = 0
    with torch.no_grad():
        for batch in loader:
            spec = batch["spec"].to(device)
            target = batch["binary"].to(device)
            mask = target != -100
            if mask.sum() == 0:
                continue

            out = model(spec)
            alpha = F.softplus(out["evidence"]) + 1.0
            S = alpha.sum(dim=1, keepdim=True)
            preds = (alpha / S).argmax(dim=1)
            correct += (preds[mask] == target[mask]).sum().item()
            total += mask.sum().item()
            for p, t in zip(preds[mask].cpu().tolist(), target[mask].cpu().tolist()):
                confusion[(t, p)] += 1

            # Auxiliary distance accuracy where label is present.
            dist_target = batch["distance"].to(device)
            dmask = dist_target != -100
            if dmask.sum():
                dist_preds = out["distance"].argmax(dim=1)
                aux_distance_correct += (dist_preds[dmask] == dist_target[dmask]).sum().item()
                aux_distance_total += dmask.sum().item()

    acc = correct / max(total, 1)
    aux_acc = aux_distance_correct / aux_distance_total if aux_distance_total else 0.0
    return {
        "binary_acc": acc,
        "binary_n": total,
        "distance_acc": aux_acc,
        "distance_n": aux_distance_total,
        "confusion": dict(confusion),
    }


def train(args) -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("device", device=str(device))

    # Discover JSONL files.
    paths = [Path(args.primary_jsonl)]
    bulk_dir = Path("data/training/v7_bulk")
    if bulk_dir.exists():
        paths.extend(sorted(bulk_dir.glob("*.jsonl")))
    sanct_corrected = Path("data/training/sanctsound_corrected.jsonl")
    if sanct_corrected.exists() and not args.exclude_sanctsound:
        paths.append(sanct_corrected)
    # Diverse SanctSound (sb02/fk01/hi01/mb01/gr01) is geographically
    # separate from the held-out test set (oc01/sb01), so include it even
    # when --exclude-sanctsound is set. The flag is only meant to keep
    # sanctsound_corrected (the test set) out of training.
    sanct_diverse = Path("data/training/sanctsound_diverse.jsonl")
    if sanct_diverse.exists():
        paths.append(sanct_diverse)

    # v8 long-context (60s) sources, used when --target-frames is set high.
    if args.target_frames >= 600:
        paths = []  # discard 5-10s sources; they don't pad cleanly to 60s
        v8_bulk_dir = Path("data/training/v8_bulk")
        if v8_bulk_dir.exists():
            paths.extend(sorted(v8_bulk_dir.glob("*.jsonl")))
        sanct_60s = Path("data/training/sanctsound_60s.jsonl")
        if sanct_60s.exists():
            paths.append(sanct_60s)
        sanct_more_60s = Path("data/training/sanctsound_more_60s.jsonl")
        if sanct_more_60s.exists():
            paths.append(sanct_more_60s)
    log.info("training_files", files=[str(p) for p in paths])

    # When --exclude-sanctsound is set in 60s mode, also drop oc01/sb01
    # entries from sanctsound_60s — those are our held-out OOD test set.
    exclude_substrings: tuple[str, ...] = ()
    if args.exclude_sanctsound and args.target_frames >= 600:
        exclude_substrings = ("oc01", "sb01")

    # Two dataset instances so augmentation only fires on training.
    train_dataset = SpecDataset(
        paths, augment=args.augment, target_frames=args.target_frames,
        exclude_source_substrings=exclude_substrings,
    )
    val_dataset = SpecDataset(
        paths, augment=False, target_frames=args.target_frames,
        exclude_source_substrings=exclude_substrings,
    )
    if len(train_dataset) == 0:
        sys.exit("no training data found")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(train_dataset))
    rng.shuffle(indices)
    val_n = int(len(train_dataset) * args.val_frac)
    val_indices = indices[:val_n].tolist()
    train_indices = indices[val_n:].tolist()
    train_set = torch.utils.data.Subset(train_dataset, train_indices)
    val_set = torch.utils.data.Subset(val_dataset, val_indices)
    log.info("split", train=len(train_set), val=len(val_set), augment=args.augment)
    dataset = train_dataset  # for class_weights compatibility

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = OceanSentinelV7().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model", params=f"{n_params:,}")

    class_weights = compute_class_weights(dataset).to(device)
    log.info("class_weights", values=class_weights.cpu().tolist())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    best_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        # Anneal evidential KL coefficient up to 1 over the first 10 epochs
        # (helps avoid the network shutting down evidence too early).
        anneal = min(1.0, (epoch + 1) / 10.0)

        for batch in train_loader:
            spec = batch["spec"].to(device)
            binary = batch["binary"].to(device)
            distance = batch["distance"].to(device)

            mask_b = binary != -100
            if mask_b.sum() == 0:
                continue

            out = model(spec)

            loss_b = evidential_loss(
                out["evidence"][mask_b], binary[mask_b],
                annealing_coef=anneal,
                class_weights=class_weights,
            )

            mask_d = distance != -100
            loss_d = torch.tensor(0.0, device=device)
            if mask_d.sum() > 0:
                loss_d = F.cross_entropy(
                    out["distance"][mask_d], distance[mask_d],
                )

            loss = loss_b + 0.3 * loss_d

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        train_loss = epoch_loss / max(n_batches, 1)
        metrics = evaluate(model, val_loader, device)
        log.info(
            "epoch_done",
            epoch=epoch + 1, train_loss=round(train_loss, 4),
            val_binary_acc=round(metrics["binary_acc"], 4),
            val_distance_acc=round(metrics["distance_acc"], 4),
            val_binary_n=metrics["binary_n"],
            val_distance_n=metrics["distance_n"],
            anneal=round(anneal, 2),
        )

        if metrics["binary_acc"] > best_acc:
            best_acc = metrics["binary_acc"]
            torch.save(model.state_dict(), out_path)
            log.info("checkpoint_saved", path=str(out_path), acc=best_acc)

    print(f"\nBest val binary accuracy: {best_acc:.4f}")
    print(f"Saved to: {out_path}")


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-jsonl", type=str,
                    default="data/training/gemma_labels.v7.jsonl")
    ap.add_argument("--out", type=str, default="data/models/cnn_v7.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--exclude-sanctsound", action="store_true",
        help="Exclude sanctsound_corrected.jsonl from training so it can "
             "be used as a clean held-out test set",
    )
    ap.add_argument(
        "--augment", action="store_true",
        help="Enable SpecAugment (time + freq masking) on training samples",
    )
    ap.add_argument(
        "--target-frames", type=int, default=DEFAULT_TARGET_FRAMES,
        help="Spectrogram width to crop/pad to. 313 = ~10s (v7 default), "
             "1876 = ~60s (v8). Higher values switch to v8_bulk + sanctsound_60s.",
    )
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    cli()
