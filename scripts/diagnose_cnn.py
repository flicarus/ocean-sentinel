"""Diagnose the binary CNN: per-source accuracy on the validation split.
Rebuilds the exact same train/val split used during training (seed=42),
then groups val predictions by provenance.source_id + taxonomy.shipsear_class.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader

from ocean_sentinel.models.cnn import OceanSentinelCNN
from train_cnn import LABEL_MAP, SpecDataset, make_loaders

CKPT = Path("data/models/cnn_v2.pt")
JSONL = Path("data/training/gemma_labels.v2.jsonl")

def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds = SpecDataset(str(JSONL), representation_version="abs_db_v1")
    _, val_loader = make_loaders(ds, batch_size=32, val_frac=0.2)

    #Grab the exact val subset indices so we can look up source metadata
    val_subset = val_loader.dataset
    val_indices = val_subset.indices

    model = OceanSentinelCNN().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    inv_label = {v: k for k, v in LABEL_MAP.items()}
    per_source: dict[str, Counter] = {}

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device)
            preds = model(x)["vessel"].argmax(dim=1).cpu().tolist()
            truths = y.tolist()

            for i, (pred, truth) in enumerate(zip(preds, truths)):
                global_idx = val_indices[batch_idx * 32 + i]
                entry = ds.entries[global_idx]
                src = entry["provenance"]["source_id"]
                ss_class = entry["taxonomy"].get("shipsear_class") or "-"
                key = f"{src} / {ss_class}"

                bucket = per_source.setdefault(key, Counter())
                bucket["total"] += 1
                if pred == truth:
                    bucket["correct"] += 1
                bucket[f"truth={inv_label[truth]}"] += 1


    print(f"{'source / class':<25s} {'correct':>8s} {'total':>6s} {'acc':>8s}   distribution")
    print("-" * 80)
    for key in sorted(per_source):
        c = per_source[key]
        total = c["total"]
        correct = c["correct"]
        acc = correct / total if total else 0.0
        dist = {k: v for k, v in c.items()
                if k.startswith("truth=")}
        print(f"{key:<25s} {correct:>8d} {total:>6d} {acc:>7.1%}   {dict(dist)}")


if __name__ == "__main__":
    main()
