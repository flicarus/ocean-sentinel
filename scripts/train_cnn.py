from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torch.utils.data import DataLoader, Subset


LABEL_MAP = {"not_ship": 0, "ship": 1}


class SpecDataset(Dataset):
    """Reads one training pair per JSONL line: (.npy spectrogram, binary label).

    `sources` filters by `provenance.source_id` in the new rich schema.
    """

    def __init__(
        self,
        jsonl_path: str,
        sources: set[str] | None = None,
        representation_version: str | None = None,
    ) -> None:
        lines = Path(jsonl_path).read_text().splitlines()
        all_entries = [json.loads(l) for l in lines if l.strip()]

        def keep(e: dict) -> bool:
            if e.get("label") not in LABEL_MAP:
                return False
            if sources is not None and e.get("provenance", {}).get("source_id") not in sources:
                return False
            if (
                representation_version is not None
                and e.get("representation_version") != representation_version
            ):
                return False
            return True

        self.entries = [e for e in all_entries if keep(e)]
        skipped = len(all_entries) - len(self.entries)
        if skipped:
            print(f"SpecDataset: skipped {skipped} entries (filter + malformed)")
        label_dist = {}
        for e in self.entries:
            lbl = e["label"]
            label_dist[lbl] = label_dist.get(lbl, 0) + 1
        print(f"SpecDataset: loaded {len(self.entries)} pairs, distribution: {label_dist}")


    def __len__(self) -> int:
        return len(self.entries)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        entry = self.entries[idx]
        spec = np.load(entry["spectrogram_path"])
        tensor = torch.from_numpy(spec).unsqueeze(0).float()  # (1, 128, T)
        # Per-sample z-score normalization: makes inputs comparable across
        # sources and stabilizes training. Epsilon guards against silent chunks.
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        label = LABEL_MAP[entry["label"]]
        return tensor, label


def _session_key(entry: dict) -> str:
    """Canonical 'session' key per source.

    Sources name their chunks differently, so session grouping is source-aware:
      - ShipsEar chunks (shipsear/0_0/0_0_N.wav) share a parent dir per
        recording; group by that parent.
      - MBARI chunks carry their offset in the URL (...wav#offset=34522s),
        so source_file is already per 60s sample.
      - Anything else: trust source_file as-is.
    """
    src_id = entry["provenance"]["source_id"]
    src_file = entry["provenance"]["source_file"]
    if src_id == "shipsear":
        return src_file.rsplit("/", 1)[0]
    return src_file


def _group_indices_by_session(entries: list[dict]) -> dict[str, list[int]]:
    """Group chunk indices by original recording ('session').

    All 5s chunks carved from the same recording share a group, so the
    session-level split can keep them on the same side of train/val.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        groups[_session_key(entry)].append(idx)
    return dict(groups)


def make_loaders(dataset: SpecDataset, batch_size: int = 16, val_frac: float = 0.2):
    """Session-level 80/20 split, stratified by binary label.

    Within each label (ship / not_ship), shuffle sessions with seed=42 and
    fill val to ~val_frac of that label's chunks. A 1.5x target cap prevents
    a single giant session (e.g. ShipsEar class C, ~843 chunks) from
    dominating val and starving train of that recording's signal.
    """
    groups = _group_indices_by_session(dataset.entries)
    group_label = {k: dataset.entries[v[0]]["label"] for k, v in groups.items()}

    by_label: dict[str, list[str]] = defaultdict(list)
    for k, lbl in group_label.items():
        by_label[lbl].append(k)

    rng = random.Random(42)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for lbl in sorted(by_label):
        keys = sorted(by_label[lbl])
        rng.shuffle(keys)
        total = sum(len(groups[k]) for k in keys)
        target = int(total * val_frac)
        cap = int(target * 1.5)
        val_so_far = 0
        for k in keys:
            size = len(groups[k])
            if val_so_far < target and val_so_far + size <= cap:
                val_indices.extend(groups[k])
                val_so_far += size
            else:
                train_indices.extend(groups[k])

    train_labels = Counter(dataset.entries[i]["label"] for i in train_indices)
    val_labels = Counter(dataset.entries[i]["label"] for i in val_indices)
    print(
        f"Session-level stratified split: {len(groups)} sessions -> "
        f"train={len(train_indices)} {dict(train_labels)}, "
        f"val={len(val_indices)} {dict(val_labels)}"
    )

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
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
        logits = model(x)["vessel"]            # (B, 2) raw class scores
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


@torch.no_grad()
def _confusion_matrix(model, loader, device, n_classes: int = 2) -> np.ndarray:
    """Build a raw NxN confusion matrix over a dataloader."""
    model.train(False)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)["vessel"].argmax(dim=1)
        for t, p in zip(y.tolist(), preds.tolist()):
            cm[t, p] += 1
    return cm


def _print_confusion(cm: np.ndarray) -> None:
    inv = {v: k for k, v in LABEL_MAP.items()}
    labels = [inv[i] for i in range(cm.shape[0])]
    header = "truth\\pred".ljust(12) + "".join(f"{l:>9s}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        total = int(row.sum())
        cells = "".join(f"{c:>9d}" for c in row)
        acc = (row[i] / total) if total else 0.0
        print(f"{labels[i]:<12s}{cells}   ({int(row[i])}/{total} = {acc:.1%})")


def main() -> None:
    import argparse

    from ocean_sentinel.models.cnn import OceanSentinelCNN

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/training/gemma_labels.jsonl")
    ap.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Filter by provenance.source_id. Default: all sources.",
    )
    ap.add_argument("--representation", default="abs_db_v1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ckpt", default="data/models/cnn_v2.pt")
    args = ap.parse_args()

    dataset = SpecDataset(
        args.jsonl,
        sources=set(args.sources) if args.sources else None,
        representation_version=args.representation,
    )
    if len(dataset) < 20:
        raise SystemExit(
            f"Dataset too small ({len(dataset)}). Run bootstrap_shipsear.py first."
        )

    train_loader, val_loader = make_loaders(dataset, batch_size=args.batch_size)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = OceanSentinelCNN().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(
            f"epoch {epoch:2d}/{args.epochs}  "
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

    # Reload best weights, build confusion matrix on validation set
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    cm = _confusion_matrix(model, val_loader, device)
    print("\nValidation confusion matrix:")
    _print_confusion(cm)


if __name__ == "__main__":
    main()
