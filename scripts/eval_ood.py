"""Out-of-distribution eval: rows added to gemma_labels.jsonl AFTER a baseline.

True held-out test: data pulled from known sites on dates NOT in v7.6's
training pool. Compares vanilla v7.6 (threshold 0.5) vs v7.6 + per-site
calibrated thresholds — answers "does the calibration generalise off the
training distribution?"
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


def get_source_id(row: dict) -> str:
    """source_id may live at the top level (live event log) or under
    provenance (training JSONL snapshots)."""
    if row.get("source_id"):
        return row["source_id"]
    return (row.get("provenance") or {}).get("source_id", "unknown")


def site_key(row: dict) -> str:
    src = get_source_id(row)
    if "sanctsound" in src and "-" in src:
        return src.rsplit("-", 1)[-1]
    return src


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/cnn_v7_6.pt")
    p.add_argument("--thresholds", default="data/calibration/per_site_thresholds_v7_6.json")
    p.add_argument("--file", default="data/training/gemma_labels.jsonl")
    p.add_argument("--skip-rows", type=int, required=True,
                   help="Skip first N rows (training cutoff)")
    p.add_argument("--site-filter", nargs="+", default=None,
                   help="Only eval these source_ids (substring match)")
    args = p.parse_args()

    # Load classifier WITHOUT thresholds first
    clf_vanilla = CNNV7Classifier(args.model)

    # Load again with thresholds
    clf_calibrated = CNNV7Classifier(args.model)
    clf_calibrated.set_site_thresholds(args.thresholds)

    # Read OOD rows
    rows: list[dict] = []
    with open(args.file) as f:
        for i, line in enumerate(f):
            if i < args.skip_rows:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.site_filter:
                src = (r.get("provenance") or {}).get("source_id", "")
                if not any(s in src for s in args.site_filter):
                    continue
            rows.append(r)

    print(f"OOD rows loaded: {len(rows)} (from line {args.skip_rows} onwards)")
    if not rows:
        print("No OOD rows found. Make sure fresh pulls completed.")
        return

    # Group per site
    by_site = defaultdict(list)
    for r in rows:
        by_site[site_key(r)].append(r)

    print(f"\n{'site':<28} {'n':>5} {'vanilla':>9} {'calibrated':>12} {'delta':>8}")
    print("-" * 70)

    total_n = 0
    total_v = 0
    total_c = 0

    for site in sorted(by_site):
        rows_s = by_site[site]
        correct_v = 0
        correct_c = 0
        n = 0
        for r in rows_s:
            sp = Path(r["spectrogram_path"])
            # Derive label: explicit `label` if present, otherwise
            # AIS-correlated rows are all "ship" (vessel within radius
            # determined by AIS) — threat_level encodes confidence.
            truth = r.get("label")
            if truth is None:
                verdict = r.get("gemma_verdict") or {}
                threat = verdict.get("threat_level")
                if threat in ("HIGH", "MEDIUM"):
                    truth = "ship"
                elif threat == "NONE":
                    truth = "not_ship"
                else:
                    continue
            if not sp.exists() or truth not in ("ship", "not_ship"):
                continue
            try:
                spec = np.load(sp)
                src = get_source_id(r)
                v_van = clf_vanilla.predict(spec, source_id=src)
                v_cal = clf_calibrated.predict(spec, source_id=src)
            except Exception:
                continue
            n += 1
            if v_van["label"] == truth:
                correct_v += 1
            if v_cal["label"] == truth:
                correct_c += 1
        if not n:
            continue
        acc_v = correct_v / n
        acc_c = correct_c / n
        delta = (acc_c - acc_v) * 100
        total_n += n
        total_v += correct_v
        total_c += correct_c
        flag = " ✅" if delta > 5 else (" •" if delta > 0 else ("" if delta == 0 else " ⚠"))
        print(f"{site:<28} {n:>5} {acc_v*100:>8.1f}% {acc_c*100:>11.1f}% {delta:>+7.1f}pp{flag}")

    overall_v = total_v / total_n if total_n else 0
    overall_c = total_c / total_n if total_n else 0
    print("-" * 70)
    print(f"{'OVERALL (OOD)':<28} {total_n:>5} {overall_v*100:>8.1f}% {overall_c*100:>11.1f}% {(overall_c-overall_v)*100:>+7.1f}pp")


if __name__ == "__main__":
    main()
