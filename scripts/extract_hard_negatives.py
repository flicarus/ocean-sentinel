"""Hard-negative mining: find training rows v7.1 gets wrong, save them to
a separate JSONL so v7.2 can oversample them.

The intuition: a generic SGD pass weights every sample equally, so the
model spends most updates on easy ships and barely sees the rows it
struggles with (label-noisy oc01, mb01 ambient that sounds vessel-like,
low-confidence sb02). Mining the errors and replaying them at higher
frequency forces the next training run to actually learn them.

This is *not* the same as just oversampling minority class — we
specifically pull the rows the *current model* gets wrong, regardless of
class. Output JSONL is a drop-in additional manifest for the v7.2
trainer (`--hard-negatives data/training/v7_bulk/hard_negatives.jsonl`).

Usage:
    PYTHONPATH=src venv/bin/python scripts/extract_hard_negatives.py \\
        --model data/models/cnn_v7_1.pt \\
        --out data/training/v7_bulk/hard_negatives.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import structlog

sys.path.insert(0, "src")

from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier

log = structlog.get_logger()


DEFAULT_FILES = [
    "data/training/gemma_labels.v7.jsonl",
    "data/training/sanctsound_corrected.jsonl",
    "data/training/sanctsound_diverse.jsonl",
]


def site_key(row: dict) -> str:
    src = (row.get("provenance") or {}).get("source_id", "unknown")
    if "sanctsound" in src and "-" in src:
        return src.rsplit("-", 1)[-1]
    return src


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/cnn_v7_1.pt")
    p.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    p.add_argument("--out", default="data/training/v7_bulk/hard_negatives.jsonl")
    p.add_argument(
        "--max-per-site", type=int, default=200,
        help="Cap hard negatives per site so one noisy site can't dominate "
             "the oversampling pool (default 200).",
    )
    p.add_argument(
        "--min-confidence", type=float, default=0.0,
        help="Only count errors where model was at least this confident — "
             "filters out genuine 'don't know' predictions where the model "
             "abstained mentally. Default 0 = include all errors.",
    )
    args = p.parse_args()

    clf = CNNV7Classifier(args.model)

    # Collect errors per site
    errors_by_site: dict[str, list[dict]] = defaultdict(list)
    n_total = 0
    n_processed = 0

    for fp in args.files:
        path = Path(fp)
        if not path.exists():
            log.warning("file_missing", path=fp)
            continue
        log.info("scanning", file=fp)
        with path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_total += 1

                spec_path = Path(r.get("spectrogram_path", ""))
                truth = r.get("label")
                if not spec_path.exists() or truth not in ("ship", "not_ship"):
                    continue

                try:
                    spec = np.load(spec_path)
                    v = clf.predict(spec)
                except Exception:
                    continue
                n_processed += 1

                # Wrong if predicted label doesn't match truth.
                if v["label"] == truth:
                    continue
                if float(v["confidence"]) < args.min_confidence:
                    continue

                # Mark and store.
                row = {**r, "is_hard_negative": True,
                       "hard_neg_pred": v["label"],
                       "hard_neg_pred_conf": float(v["confidence"]),
                       "hard_neg_pred_unc": float(v["uncertainty"])}
                errors_by_site[site_key(r)].append(row)

    # Cap per site for fairness — otherwise one noisy site dominates.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    site_summary: dict[str, int] = {}
    n_written = 0
    fp_pred = Counter()  # what was the wrong prediction
    with out_path.open("w") as out:
        for site, rows in sorted(errors_by_site.items()):
            if args.max_per_site > 0 and len(rows) > args.max_per_site:
                idx = rng.choice(len(rows), args.max_per_site, replace=False)
                rows = [rows[i] for i in idx]
            site_summary[site] = len(rows)
            for r in rows:
                fp_pred[(r.get("label"), r["hard_neg_pred"])] += 1
                out.write(json.dumps(r) + "\n")
                n_written += 1

    print(f"\nScanned: {n_total} rows, evaluated: {n_processed}")
    print(f"Hard negatives extracted: {n_written}\n")
    print(f"{'site':<30} {'errors':>7}")
    print("-" * 40)
    for site, n in sorted(site_summary.items(), key=lambda x: -x[1]):
        print(f"{site:<30} {n:>7}")

    print(f"\nError type breakdown (truth → predicted):")
    for (truth, pred), n in fp_pred.most_common():
        print(f"  {truth:>10} → {pred:<10} {n:>5}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
