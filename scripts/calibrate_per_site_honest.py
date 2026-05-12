"""Honest per-site threshold calibration with held-out test set.

Methodology (industry standard for multi-domain classifiers):
  1. For each site, deterministically split eval data 50/50:
       calibration_half (50%) — used to tune the decision threshold
       test_half (50%)        — used ONLY to report final accuracy
  2. Sweep thresholds on calibration half to maximize accuracy
  3. Report test-half accuracy with the calibrated threshold
  4. Save thresholds keyed by site for inference-time use

Output:
  data/calibration/per_site_thresholds_v7_6_honest.json
    {model, default_threshold, overall_test_acc_default,
     overall_test_acc_tuned, per_site: {site: {threshold, n_cal, n_test,
     cal_acc, test_acc_default, test_acc_tuned}}}
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


def collect_probs(clf, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
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
    return np.array(truths), np.array(probs)


def best_threshold(truths: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    if len(truths) == 0:
        return 0.5, 0.0
    best_thr = 0.5
    best_acc = float(((probs > 0.5).astype(int) == truths).mean())
    for thr in np.arange(0.02, 0.99, 0.02):
        thr = float(round(thr, 2))
        acc = float(((probs > thr).astype(int) == truths).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/cnn_v7_6.pt")
    p.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    p.add_argument("--out", default="data/calibration/per_site_thresholds_v7_6_honest.json")
    p.add_argument("--limit-per-site", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    clf = CNNV7Classifier(args.model)

    rows_by_site: dict[str, list[dict]] = defaultdict(list)
    for fp in args.files:
        path = Path(fp)
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows_by_site[site_key(r)].append(r)

    if args.limit_per_site > 0:
        rng = np.random.default_rng(args.seed)
        for site, rows in rows_by_site.items():
            if len(rows) > args.limit_per_site:
                idx = rng.choice(len(rows), args.limit_per_site, replace=False)
                rows_by_site[site] = [rows[i] for i in idx]

    print(f"\n{'site':<28} {'n_cal':>5} {'n_test':>6} {'def@test':>9} {'thr':>5} {'tun@test':>9} {'delta':>7}")
    print("-" * 80)

    per_site_results: dict[str, dict] = {}
    test_total_n = 0
    test_total_correct_default = 0
    test_total_correct_tuned = 0

    rng_split = np.random.default_rng(args.seed + 1)
    for site in sorted(rows_by_site):
        rows = rows_by_site[site]
        if len(rows) < 4:
            continue
        # Deterministic 50/50 split
        idx = np.arange(len(rows))
        rng_split.shuffle(idx)
        half = len(rows) // 2
        cal_rows = [rows[i] for i in idx[:half]]
        test_rows = [rows[i] for i in idx[half:]]

        # Collect probs on both halves
        cal_truths, cal_probs = collect_probs(clf, cal_rows)
        test_truths, test_probs = collect_probs(clf, test_rows)
        if len(cal_truths) == 0 or len(test_truths) == 0:
            continue

        # Find threshold on cal
        best_thr, cal_best_acc = best_threshold(cal_truths, cal_probs)

        # Apply to test
        test_acc_default = float(((test_probs > 0.5).astype(int) == test_truths).mean())
        test_acc_tuned = float(((test_probs > best_thr).astype(int) == test_truths).mean())
        delta_pp = (test_acc_tuned - test_acc_default) * 100

        n_cal = len(cal_truths)
        n_test = len(test_truths)

        per_site_results[site] = {
            "threshold": float(best_thr),
            "n_cal": n_cal,
            "n_test": n_test,
            "cal_acc": round(cal_best_acc, 4),
            "test_acc_default": round(test_acc_default, 4),
            "test_acc_tuned": round(test_acc_tuned, 4),
            "delta_pp": round(delta_pp, 1),
        }

        test_total_n += n_test
        test_total_correct_default += int(test_acc_default * n_test)
        test_total_correct_tuned += int(test_acc_tuned * n_test)

        flag = " ✅" if delta_pp > 5 else (" •" if delta_pp > 1 else ("" if delta_pp >= -1 else " ⚠"))
        print(f"{site:<28} {n_cal:>5} {n_test:>6} {test_acc_default*100:>8.1f}% {best_thr:>5.2f} {test_acc_tuned*100:>8.1f}% {delta_pp:>+6.1f}pp{flag}")

    print("-" * 80)
    overall_default = test_total_correct_default / test_total_n if test_total_n else 0
    overall_tuned = test_total_correct_tuned / test_total_n if test_total_n else 0
    print(f"{'OVERALL (TEST SET)':<28} {'':>5} {test_total_n:>6} {overall_default*100:>8.1f}% {'':>5} {overall_tuned*100:>8.1f}% {(overall_tuned-overall_default)*100:>+6.1f}pp")

    # Save only thresholds where calibration was meaningful (delta_pp on test > 0.5 OR site has non-default thr)
    saved_thresholds = {
        s: r for s, r in per_site_results.items()
        if abs(r["threshold"] - 0.5) > 0.01
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model,
        "methodology": "50/50 cal/test split per site, seed=42, threshold tuned on cal only",
        "default_threshold": 0.5,
        "overall_test_acc_default": round(overall_default, 4),
        "overall_test_acc_tuned": round(overall_tuned, 4),
        "test_set_n": test_total_n,
        "per_site": per_site_results,
        "saved_thresholds": {s: r["threshold"] for s, r in saved_thresholds.items()},
    }, indent=2))
    print(f"\nSaved: {out_path}")
    print(f"Non-default thresholds for {len(saved_thresholds)} sites (others use default 0.5)")


if __name__ == "__main__":
    main()
