"""Per-site OOD accuracy report — primary metric for plug-and-play target.

Aggregate accuracy hides which sites the model actually generalizes to.
This script breaks accuracy down by source_id (per-hydrophone), so we can
see whether v7.1's focal-loss + cross-site-mix recipe actually moved the
needle on never-seen sites.

Usage:
    PYTHONPATH=src venv/bin/python scripts/eval_per_site.py \\
        --model data/models/cnn_v7_holdout.pt \\
        --out data/eval/per_site_v7.json

To compare v7 vs v7.1 after retraining:
    diff -u data/eval/per_site_v7.json data/eval/per_site_v7_1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import structlog

sys.path.insert(0, "src")

from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier

log = structlog.get_logger()


# Default eval pool: every labelled JSONL we have (training & holdout).
# Per-site breakdown means trained sites still serve as sanity checks.
# sanctsound_more_60s.jsonl adds 7 SanctSound stations not in v7.4's training
# (hi04, hi06, oc02, ci01, ci02, mb02, sb03, …) — actual held-out OOD eval.
DEFAULT_FILES = [
    "data/training/sanctsound_corrected.jsonl",
    "data/training/sanctsound_diverse.jsonl",
    "data/training/sanctsound_more_60s.jsonl",
    "data/training/gemma_labels.v7.jsonl",
]


def site_key(row: dict) -> str:
    """Bucket per hydrophone. SanctSound rows encode site in source_id
    suffix (e.g. 'sanctsound-corrected-oc01' → 'oc01')."""
    src = (row.get("provenance") or {}).get("source_id", "unknown")
    if "sanctsound" in src and "-" in src:
        return src.rsplit("-", 1)[-1]
    return src


def evaluate_one(clf, rows: list[dict]) -> dict:
    correct = 0
    total = 0
    pred_ship = 0
    true_ship = 0
    pred_amb_correct = 0
    true_amb = 0
    confs: list[float] = []
    uncs: list[float] = []
    for r in rows:
        spec_path = Path(r["spectrogram_path"])
        truth = r.get("label")
        if not spec_path.exists() or truth not in ("ship", "not_ship"):
            continue
        try:
            spec = np.load(spec_path)
            src = (r.get("provenance") or {}).get("source_id")
            v = clf.predict(spec, source_id=src)
        except Exception:
            continue
        total += 1
        confs.append(float(v["confidence"]))
        uncs.append(float(v["uncertainty"]))
        if v["label"] == truth:
            correct += 1
        if truth == "ship":
            true_ship += 1
            if v["label"] == "ship":
                pred_ship += 1
        else:
            true_amb += 1
            if v["label"] == "not_ship":
                pred_amb_correct += 1

    acc = correct / total if total else 0.0
    recall = pred_ship / true_ship if true_ship else None
    fp = true_amb - pred_amb_correct
    prec = pred_ship / (pred_ship + fp) if (pred_ship + fp) > 0 else None
    return {
        "n": total,
        "accuracy": round(acc, 4),
        "true_ship": true_ship,
        "true_amb": true_amb,
        "recall": round(recall, 4) if recall is not None else None,
        "precision": round(prec, 4) if prec is not None else None,
        "conf_mean": round(float(np.mean(confs)), 4) if confs else None,
        "conf_std": round(float(np.std(confs)), 4) if confs else None,
        "unc_mean": round(float(np.mean(uncs)), 4) if uncs else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/cnn_v7_holdout.pt")
    p.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    p.add_argument("--out", default="data/eval/per_site.json")
    p.add_argument("--limit-per-site", type=int, default=300,
                   help="Cap each site for speed; set to 0 for no cap")
    args = p.parse_args()

    clf = CNNV7Classifier(args.model)

    # Group rows per site
    rows_by_site: dict[str, list[dict]] = defaultdict(list)
    for fp in args.files:
        path = Path(fp)
        if not path.exists():
            log.warning("file_missing", path=fp)
            continue
        with path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows_by_site[site_key(r)].append(r)

    # Cap per site for speed if requested.
    if args.limit_per_site > 0:
        rng = np.random.default_rng(42)
        for site, rows in rows_by_site.items():
            if len(rows) > args.limit_per_site:
                idx = rng.choice(len(rows), args.limit_per_site, replace=False)
                rows_by_site[site] = [rows[i] for i in idx]

    # Eval each site
    results: dict[str, dict] = {}
    for site in sorted(rows_by_site):
        results[site] = evaluate_one(clf, rows_by_site[site])

    # Aggregate
    total_n = sum(r["n"] for r in results.values())
    total_correct = sum(int(r["accuracy"] * r["n"]) for r in results.values())
    overall_acc = total_correct / total_n if total_n else 0.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model,
        "overall": {"n": total_n, "accuracy": round(overall_acc, 4)},
        "per_site": results,
    }, indent=2))

    # Pretty print
    print(f"\n{'site':<25} {'n':>6} {'acc':>7} {'recall':>8} {'prec':>7} "
          f"{'conf_mean':>10} {'conf_std':>10} {'unc_mean':>10}")
    print("-" * 95)
    for site, r in sorted(results.items(), key=lambda x: -x[1]["n"]):
        recall_str = f"{r['recall']:.1%}" if r['recall'] is not None else "  -  "
        prec_str = f"{r['precision']:.1%}" if r['precision'] is not None else "  -  "
        print(f"{site:<25} {r['n']:>6} {r['accuracy']:>7.1%} {recall_str:>8} "
              f"{prec_str:>7} {r['conf_mean']!s:>10} {r['conf_std']!s:>10} "
              f"{r['unc_mean']!s:>10}")
    print(f"\n{'OVERALL':<25} {total_n:>6} {overall_acc:>7.1%}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
