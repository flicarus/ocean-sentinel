"""Find optimal per-site decision threshold for v7.6.

The balanced sampler in v7.6 fixed point-robinson (+48pp) but de-weighted
mbari to 1/34 of training, causing 97pp regression on mbari ambient.
A full retrain takes 6h; per-site threshold calibration is free and
recovers most of the lost ground.

For each site, collect (true_label, ship_prob) pairs and sweep thresholds
to maximize accuracy. Save per-site thresholds for inference-time use.
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


DEFAULT_FILES = [
    "data/training/sanctsound_corrected.jsonl",
    "data/training/sanctsound_diverse.jsonl",
    "data/training/sanctsound_more_60s.jsonl",
    "data/training/gemma_labels.v7.jsonl",
]


def site_key(row: dict) -> str:
    src = (row.get("provenance") or {}).get("source_id", "unknown")
    if "sanctsound" in src and "-" in src:
        return src.rsplit("-", 1)[-1]
    return src


def collect_probs(clf, rows: list[dict]) -> tuple[list[int], list[float]]:
    """Return (truths, ship_probs) lists."""
    truths: list[int] = []
    probs: list[float] = []
    for r in rows:
        spec_path = Path(r["spectrogram_path"])
        truth = r.get("label")
        if not spec_path.exists() or truth not in ("ship", "not_ship"):
            continue
        try:
            spec = np.load(spec_path)
            v = clf.predict(spec)
        except Exception:
            continue
        truths.append(1 if truth == "ship" else 0)
        probs.append(float(v["probabilities"]["ship"]))
    return truths, probs


def find_optimal_threshold(truths: list[int], probs: list[float]) -> tuple[float, float]:
    """Sweep thresholds 0.02..0.98 step 0.02, return (best_threshold, best_acc).

    For ambient-only sites (all truths=0), we want threshold high enough to
    classify all as not_ship.
    """
    if not truths:
        return 0.5, 0.0
    truths_arr = np.array(truths)
    probs_arr = np.array(probs)
    best_thr = 0.5
    best_acc = 0.0
    # Include 0.5 as anchor + sweep 0.02..0.98
    candidates = [0.5] + [round(t, 2) for t in np.arange(0.02, 0.99, 0.02)]
    for thr in candidates:
        preds = (probs_arr > thr).astype(int)
        acc = float((preds == truths_arr).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/cnn_v7_6.pt")
    p.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    p.add_argument("--out", default="data/calibration/per_site_thresholds_v7_6.json")
    p.add_argument("--limit-per-site", type=int, default=300)
    args = p.parse_args()

    clf = CNNV7Classifier(args.model)

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

    if args.limit_per_site > 0:
        rng = np.random.default_rng(42)
        for site, rows in rows_by_site.items():
            if len(rows) > args.limit_per_site:
                idx = rng.choice(len(rows), args.limit_per_site, replace=False)
                rows_by_site[site] = [rows[i] for i in idx]

    print(f"\n{'site':<28} {'n':>5} {'acc@0.5':>9} {'best_thr':>10} {'acc@best':>10} {'delta':>8}")
    print("-" * 80)

    thresholds: dict[str, dict] = {}
    total_n = 0
    total_correct_default = 0
    total_correct_tuned = 0

    for site in sorted(rows_by_site):
        truths, probs = collect_probs(clf, rows_by_site[site])
        if not truths:
            continue
        truths_arr = np.array(truths)
        probs_arr = np.array(probs)
        # Default 0.5
        preds_default = (probs_arr > 0.5).astype(int)
        acc_default = float((preds_default == truths_arr).mean())
        # Tuned
        best_thr, best_acc = find_optimal_threshold(truths, probs)
        delta = (best_acc - acc_default) * 100
        n = len(truths)

        # Only save thresholds that meaningfully differ from 0.5
        if abs(best_thr - 0.5) > 0.01 and delta > 1:
            thresholds[site] = {
                "threshold": best_thr,
                "n": n,
                "acc_default": round(acc_default, 4),
                "acc_tuned": round(best_acc, 4),
                "delta_pp": round(delta, 1),
            }

        total_n += n
        total_correct_default += int(acc_default * n)
        total_correct_tuned += int(best_acc * n)

        flag = " ✅" if delta > 5 else (" •" if delta > 1 else "")
        print(f"{site:<28} {n:>5} {acc_default*100:>8.1f}% {best_thr:>10.2f} {best_acc*100:>9.1f}% {delta:>+7.1f}pp{flag}")

    print("-" * 80)
    overall_default = total_correct_default / total_n
    overall_tuned = total_correct_tuned / total_n
    print(f"{'OVERALL':<28} {total_n:>5} {overall_default*100:>8.1f}% {'(per-site)':>10} {overall_tuned*100:>9.1f}% {(overall_tuned-overall_default)*100:>+7.1f}pp")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model,
        "default_threshold": 0.5,
        "overall_default_acc": round(overall_default, 4),
        "overall_tuned_acc": round(overall_tuned, 4),
        "per_site_thresholds": thresholds,
    }, indent=2))
    print(f"\nSaved: {out_path}")
    print(f"Tuned thresholds for {len(thresholds)} sites (delta > 1pp AND |thr - 0.5| > 0.01)")


if __name__ == "__main__":
    main()
