"""Diagnose the binary CNN: per-source accuracy on the validation split.
Rebuilds the exact same train/val split used during training (seed=42),
then groups val predictions by provenance.source_id + taxonomy.shipsear_class.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader

from ocean_sentinel.models.cnn import OceanSentinelCNN
from train_cnn import (
    LABEL_MAP, SpecDataset,
    _split_held_out_sources, _split_session_indices,
    compute_source_freq_profiles, make_loaders,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path,
                    default=Path("data/training/gemma_labels.v5.jsonl"))
    ap.add_argument("--ckpt", type=Path,
                    default=Path("data/models/cnn_v6.pt"))
    ap.add_argument("--sources", nargs="+", default=None,
                    help="Filter by provenance.source_id. Default: all.")
    ap.add_argument("--held-out-sources", nargs="+", default=None,
                    help="LOHO mode: val = these sources only.")
    ap.add_argument("--no-source-norm", action="store_true",
                    help="Disable per-source freq-profile subtraction.")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sources = set(args.sources) if args.sources else None
    held_out = set(args.held_out_sources) if args.held_out_sources else None
    # augment=False on both: diagnosis is evaluation-only, no training here
    ds_train = SpecDataset(
        str(args.jsonl), sources=sources,
        representation_version="abs_db_v1", augment=False,
    )
    ds_val = SpecDataset(
        str(args.jsonl), sources=sources,
        representation_version="abs_db_v1", augment=False,
    )

    # Recompute training indices (for profile computation) matching whichever
    # split mode the training run used — deterministic via seed=42.
    if held_out:
        train_idx, _ = _split_held_out_sources(ds_train.entries, held_out)
    else:
        train_idx, _ = _split_session_indices(ds_train.entries)

    if not args.no_source_norm:
        train_entries = [ds_train.entries[i] for i in train_idx]
        profiles = compute_source_freq_profiles(train_entries)
        ds_train.set_source_profiles(profiles)
        ds_val.set_source_profiles(profiles)

    _, val_loader = make_loaders(
        ds_train, ds_val, batch_size=32, val_frac=0.2,
        held_out_sources=held_out,
    )

    #Grab the exact val subset indices so we can look up source metadata
    val_subset = val_loader.dataset
    val_indices = val_subset.indices
    ds = ds_val  # alias used below to look up source metadata

    model = OceanSentinelCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    inv_label = {v: k for k, v in LABEL_MAP.items()}
    per_source: dict[str, Counter] = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x, y = batch[0], batch[1]
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
