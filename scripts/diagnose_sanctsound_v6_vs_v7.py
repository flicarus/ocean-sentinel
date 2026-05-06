"""Compare v6 and v7 CNN predictions on the same SanctSound val chunks.

Goal: understand the SanctSound 83% -> 20% regression in v7.  Two
hypotheses we want to discriminate:

  A) v7 over-generalised after DeepShip — domain shift makes any cargo-
     like spectral signature read as ship.  If true, regression should
     be ROUGHLY UNIFORM across all SanctSound sites.

  B) SanctSound has more mislabels than the 10 OC01 chunks we already
     surfaced.  If true, regression should CONCENTRATE on sites with
     known shipping lanes (OC01, SB01).
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "src")

from ocean_sentinel.services.cnn_classifier import CNNClassifier
from train_cnn import LABEL_MAP, SpecDataset, _split_session_indices

JSONL = "data/training/gemma_labels.v7.jsonl"
V6_CKPT = "data/models/cnn_v6.pt"
V7_CKPT = "data/models/cnn_v7.pt"

LABELS = ["not_ship", "ship"]


def site_from_event_id(event_id: str) -> str:
    # event_id: sanctsound_<site>_SanctSound_<SITE>_..._chunk
    m = re.match(r"sanctsound_([a-z]+\d+)_", event_id)
    return m.group(1) if m else "unknown"


def main() -> None:
    ds = SpecDataset(
        JSONL, sources={"sanctsound"},
        representation_version="abs_db_v1", augment=False,
    )
    # Same split as training (seed=42, session-level).  Actually we want
    # only val indices, but the split is computed over all sanctsound-
    # filtered entries. _split_session_indices takes the entries directly.
    train_idx, val_idx = _split_session_indices(ds.entries)

    v6 = CNNClassifier(V6_CKPT)
    v7 = CNNClassifier(V7_CKPT)

    # Per-site tallies.
    sites: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "v6_ship": 0, "v7_ship": 0, "agree": 0, "v6_only_ship": 0, "v7_only_ship": 0}
    )

    for i in val_idx:
        entry = ds.entries[i]
        event_id = entry["event_id"]
        site = site_from_event_id(event_id)
        truth = entry["label"]

        spec = np.load(entry["spectrogram_path"]).astype(np.float32)

        # Run both classifiers via their public predict(). source_id makes
        # CNNClassifier apply the right per-source profile.
        r6 = v6.predict(spec, source_id="sanctsound")
        r7 = v7.predict(spec, source_id="sanctsound")

        s = sites[site]
        s["n"] += 1
        v6_pred = r6["label"]
        v7_pred = r7["label"]
        if v6_pred == "ship":
            s["v6_ship"] += 1
        if v7_pred == "ship":
            s["v7_ship"] += 1
        if v6_pred == v7_pred:
            s["agree"] += 1
        elif v6_pred == "ship" and v7_pred == "not_ship":
            s["v6_only_ship"] += 1
        elif v7_pred == "ship" and v6_pred == "not_ship":
            s["v7_only_ship"] += 1

    # Pretty print.
    print(f"\nSanctSound val split: {len(val_idx)} chunks across {len(sites)} sites")
    print(f"All chunks labeled `not_ship` in dataset (truth = ambient).")
    print()
    header = (
        f"{'site':6s} {'n':>4s}  "
        f"{'v6→ship':>8s} {'v7→ship':>8s}  "
        f"{'agree':>6s}  {'v6 only':>7s} {'v7 only':>7s}"
    )
    print(header)
    print("-" * len(header))

    totals = {"n": 0, "v6_ship": 0, "v7_ship": 0, "agree": 0, "v6_only_ship": 0, "v7_only_ship": 0}
    for site in sorted(sites):
        s = sites[site]
        for k in totals:
            totals[k] += s[k]
        v6_pct = s["v6_ship"] / s["n"] * 100 if s["n"] else 0.0
        v7_pct = s["v7_ship"] / s["n"] * 100 if s["n"] else 0.0
        agree_pct = s["agree"] / s["n"] * 100 if s["n"] else 0.0
        print(
            f"{site:6s} {s['n']:>4d}  "
            f"{s['v6_ship']:>4d} ({v6_pct:>4.0f}%)  "
            f"{s['v7_ship']:>4d} ({v7_pct:>4.0f}%)  "
            f"{agree_pct:>5.0f}%  "
            f"{s['v6_only_ship']:>7d} {s['v7_only_ship']:>7d}"
        )
    print("-" * len(header))
    v6_pct = totals["v6_ship"] / totals["n"] * 100
    v7_pct = totals["v7_ship"] / totals["n"] * 100
    agree_pct = totals["agree"] / totals["n"] * 100
    print(
        f"{'TOTAL':6s} {totals['n']:>4d}  "
        f"{totals['v6_ship']:>4d} ({v6_pct:>4.0f}%)  "
        f"{totals['v7_ship']:>4d} ({v7_pct:>4.0f}%)  "
        f"{agree_pct:>5.0f}%  "
        f"{totals['v6_only_ship']:>7d} {totals['v7_only_ship']:>7d}"
    )
    print()
    print("Hypothesis A (over-generalisation) → regression UNIFORM across sites.")
    print("Hypothesis B (more mislabels)      → regression CONCENTRATED on shipping-lane sites.")


if __name__ == "__main__":
    main()
