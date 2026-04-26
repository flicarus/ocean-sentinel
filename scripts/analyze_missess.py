"""Sample-level analysis of CNN val misses.

Runs the full val split through the production CNNClassifier (profiles +
temperature) and grinds the misclassifications:
  - per-source miss counts and rates
  - confidence distribution of misses (overconfident wrong predictions are
    the dangerous ones — fast-path may swallow them)
  - per-truth-class breakdown (false-ship vs missed-ship)
  - top-K hardest misses (highest-confidence wrong) for manual inspection

Output gives a concrete punch list: where is the model failing, on which
class, with what confidence — informs whether the next move is data
cleanup (B), online adaptation (C), or DeepShip retrain (D).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from train_cnn import LABEL_MAP, SpecDataset, _split_session_indices
from ocean_sentinel.services.cnn_classifier import CNNClassifier


def main() -> None:
    ds = SpecDataset(
        "data/training/gemma_labels.v4.jsonl",
        representation_version="abs_db_v1",
        augment=False,
    )
    _, val_idx = _split_session_indices(ds.entries)

    clf = CNNClassifier("data/models/cnn_v6.pt")

    inv_label = {v: k for k, v in LABEL_MAP.items()}
    rows: list[dict] = []

    for i in val_idx:
        e = ds.entries[i]
        spec = np.load(e["spectrogram_path"]).astype(np.float32)
        out = clf.predict(spec, source_id=e["provenance"]["source_id"])
        truth_label = e["label"]
        pred_label = out["label"]
        rows.append({
            "source": e["provenance"]["source_id"],
            "shipsear_class": e["taxonomy"].get("shipsear_class") or "-",
            "truth": truth_label,
            "pred": pred_label,
            "correct": truth_label == pred_label,
            "confidence": out["confidence"],
            "p_ship": out["probabilities"]["ship"],
            "p_not_ship": out["probabilities"]["not_ship"],
            "spec_path": e["spectrogram_path"],
            "event_id": e["event_id"],
        })

    misses = [r for r in rows if not r["correct"]]
    n_total = len(rows)
    n_miss = len(misses)
    print()
    print(f"=== Total: {n_total} val samples, {n_miss} misses "
          f"({n_miss / n_total:.1%}) ===")
    print()

    # 1. Per-source breakdown
    print("--- Per-source miss rate ---")
    print(f"{'source':<35s} {'misses':>7s}/{'total':<7s} "
          f"{'rate':>6s}  {'avg_conf_on_miss':>17s}")
    print("-" * 80)
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    for src, items in sorted(by_source.items()):
        tot = len(items)
        miss = [r for r in items if not r["correct"]]
        avg_conf = (
            sum(r["confidence"] for r in miss) / len(miss)
            if miss else float("nan")
        )
        print(
            f"{src:<35s} {len(miss):>7d}/{tot:<7d} "
            f"{len(miss) / tot:>5.1%}   "
            f"{avg_conf:>17.3f}"
            if miss else
            f"{src:<35s} {len(miss):>7d}/{tot:<7d} "
            f"{len(miss) / tot:>5.1%}   {'-':>17s}"
        )
    print()

    # 2. Direction of misses: false-ship vs missed-ship
    print("--- Direction of misses ---")
    direction = Counter(
        (r["truth"], r["pred"]) for r in misses
    )
    for (truth, pred), n in direction.most_common():
        print(f"  truth={truth} predicted={pred}: {n}")
    print()

    # 3. Confidence distribution of misses
    if misses:
        print("--- Confidence on misses (lower = model was uncertain, "
              "higher = scary) ---")
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.01]
        for lo, hi in zip(bins[:-1], bins[1:]):
            n = sum(1 for r in misses if lo <= r["confidence"] < hi)
            bar = "█" * n
            print(f"  [{lo:.2f}, {hi:.2f}): {n:>3d}  {bar}")
        print()

    # 4. Top-K scariest misses (highest confidence wrong)
    K = 15
    scary = sorted(misses, key=lambda r: -r["confidence"])[:K]
    if scary:
        print(f"--- Top-{K} scariest misses (model was confident AND wrong) ---")
        print(
            f"{'source':<32s} {'class':>6s} "
            f"{'truth→pred':<22s} {'conf':>6s}  spec"
        )
        for r in scary:
            print(
                f"{r['source']:<32s} {r['shipsear_class']:>6s} "
                f"{r['truth']+'→'+r['pred']:<22s} "
                f"{r['confidence']:>6.3f}  {r['spec_path']}"
            )

    # 5. Fast-path safety check at τ=0.98
    fast_path = [r for r in rows if r["pred"] == "not_ship" and r["confidence"] >= 0.98]
    fast_path_misses = [r for r in fast_path if not r["correct"]]
    print()
    print(f"--- Fast-path safety (τ=0.98) ---")
    print(f"  fast-path triggered: {len(fast_path)}/{n_total} = "
          f"{len(fast_path) / n_total:.1%}")
    print(f"  ships missed in fast-path: {len(fast_path_misses)}")
    if fast_path_misses:
        for r in fast_path_misses:
            print(f"    {r['source']:<32s} {r['shipsear_class']:>6s} "
                  f"truth={r['truth']} conf={r['confidence']:.3f}")


if __name__ == "__main__":
    main()
