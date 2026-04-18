from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torch.utils.data import DataLoader, random_split


LABEL_MAP = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


class SpecDataset(Dataset):
    """Reads one training pair per JSONL line: (.npy spectrogram, threat label)."""

    def __init__(self, jsonl_path: str) -> None:
        lines = Path(jsonl_path).read_text().splitlines()
        all_entries = [json.loads(l) for l in lines if l.strip()]
        # Drop entries where Gemma returned invalid JSON (no valid threat_level).
        self.entries = [
            e for e in all_entries
            if e.get("gemma_verdict", {}).get("threat_level") in LABEL_MAP
        ]
        skipped = len(all_entries) - len(self.entries)
        if skipped:
            print(f"SpecDataset: skipped {skipped} malformed entries")


    def __len__(self) -> int:
        return len(self.entries)
    

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        entry = self.entries[idx]
        spec = np.load(entry["spectrogram_path"])
        tensor = torch.from_numpy(spec).unsqueeze(0).float()  # (1, 128, T)
        # Per-sample z-score normalization: makes inputs comparable across
        # sources and stabilizes training. Epsilon guards against silent chunks.
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        label = LABEL_MAP[entry["gemma_verdict"]["threat_level"]]
        return tensor, label


def make_loaders(dataset: SpecDataset, batch_size: int = 16, val_frac: float = 0.2):
    """80/20 train-val split, seeded for reproducibility."""
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader



def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()                  # clear gradients from prev step
        logits = model(x)["vessel"]            # (B, 5) raw class scores
        loss = loss_fn(logits, y)
        loss.backward()                        # compute gradients
        optimizer.step()                       # update weights
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

@torch.no_grad()                               # no gradients in eval (saves memory)
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)["vessel"]
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def main() -> None:
    from ocean_sentinel.models.cnn import OceanSentinelCNN

    dataset = SpecDataset("data/training/gemma_labels.jsonl")
    train_loader, val_loader = make_loaders(dataset, batch_size=16)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = OceanSentinelCNN().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Save on lowest val_loss (not highest val_acc). Loss keeps decreasing
    # even after acc plateaus/ties, so we capture the most-converged model.
    best_val_loss = float("inf")
    best_val_acc = 0.0
    ckpt_path = Path("data/models/cnn_v1.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = 20
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(
            f"epoch {epoch:2d}/{epochs}  "
            f"train_loss={train_loss:.3f} train_acc={train_acc:.1%}  "
            f"val_loss={val_loss:.3f} val_acc={val_acc:.1%}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved checkpoint (val_loss={val_loss:.3f}, val_acc={val_acc:.1%})")

    print(f"\nBest val_loss: {best_val_loss:.3f}  (val_acc={best_val_acc:.1%})")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
