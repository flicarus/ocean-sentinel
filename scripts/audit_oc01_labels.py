"""Self-audit oc01 labels with the v7.6 production model.

The original v6-era case study found 10 mislabeled chunks at OC01 and
cross-referenced them to JOSCO HUIZHOU via GFW AIS. Since then the
relabel pass populated `sanctsound_corrected.jsonl` with 560
ambient→ship corrections (and kept 240 unchanged ambients).

This script asks v7.6 (96.4% honest test accuracy, per-site calibrated)
two questions on all 800 oc01 samples:

  1. Do the 560 prior corrections still hold? (label=ship, prob should be high
     at the calibrated threshold for oc01 = 0.14)
  2. Among the 240 unchanged ambients, does v7.6 surface NEW suspects
     (label=ambient but high ship_prob → likely additional label noise)?

Output: data/audit/oc01_suspects_v7_6.json + a tiny markdown summary.

Usage:
    PYTHONPATH=src venv/bin/python scripts/audit_oc01_labels.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CORPUS = ROOT / "data/training/sanctsound_corrected.jsonl"
MODEL = ROOT / "data/models/cnn_v7_6.pt"
THRESHOLDS = ROOT / "data/calibration/per_site_thresholds_v7_6.json"
OUT_JSON = ROOT / "data/audit/oc01_suspects_v7_6.json"
OUT_MD = ROOT / "data/audit/oc01_suspects_v7_6.md"


def main() -> int:
    from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier

    print(f"loading v7.6 model: {MODEL}")
    clf = CNNV7Classifier(str(MODEL))
    clf.set_site_thresholds(str(THRESHOLDS))
    active_thr = clf._threshold_for("oc01")
    print(f"active oc01 threshold: {active_thr:.3f}")

    samples = []
    with open(CORPUS) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            prov = d.get("provenance", {}).get("source_id", "")
            if "oc01" not in prov.lower():
                continue
            samples.append(d)
    print(f"oc01 samples: {len(samples)}")

    results = []
    n_load_fail = 0
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        if i % 100 == 0:
            print(f"  {i}/{len(samples)}  ({time.perf_counter() - t0:.1f}s)")
        path = ROOT / s["spectrogram_path"]
        try:
            spec = np.load(path)
        except Exception:
            n_load_fail += 1
            continue
        out = clf.predict(spec, source_id="oc01")
        ship_prob = float(out["probabilities"]["ship"])
        results.append({
            "event_id": s["event_id"],
            "spectrogram_path": s["spectrogram_path"],
            "source_file": s.get("provenance", {}).get("source_file", ""),
            "current_label": s["label"],
            "original_taxonomy": s.get("taxonomy", {}).get("category", ""),
            "ship_prob": ship_prob,
            "uncertainty": float(out["uncertainty"]),
            "model_label": out["label"],
            "fires_at_threshold": ship_prob > active_thr,
        })
    print(f"done in {time.perf_counter() - t0:.1f}s; load_fail={n_load_fail}")

    # Stratify
    corrections_confirmed = [
        r for r in results
        if r["current_label"] == "ship" and r["fires_at_threshold"]
    ]
    corrections_rejected = [
        r for r in results
        if r["current_label"] == "ship" and not r["fires_at_threshold"]
    ]
    new_suspects = [
        r for r in results
        if r["current_label"] == "not_ship" and r["fires_at_threshold"]
    ]
    confirmed_amb = [
        r for r in results
        if r["current_label"] == "not_ship" and not r["fires_at_threshold"]
    ]

    summary = {
        "model": str(MODEL.relative_to(ROOT)),
        "thresholds": str(THRESHOLDS.relative_to(ROOT)),
        "site": "oc01",
        "active_threshold": active_thr,
        "n_total": len(results),
        "n_load_fail": n_load_fail,
        "by_bucket": {
            "label_ship_v76_agrees": len(corrections_confirmed),
            "label_ship_v76_rejects": len(corrections_rejected),
            "label_ambient_v76_says_ship": len(new_suspects),  # ← NEW label noise candidates
            "label_ambient_v76_agrees": len(confirmed_amb),
        },
        "rates": {
            "prior_correction_validation": (
                len(corrections_confirmed) / max(1, len(corrections_confirmed) + len(corrections_rejected))
            ),
            "new_suspect_rate_in_unchanged_ambients": (
                len(new_suspects) / max(1, len(new_suspects) + len(confirmed_amb))
            ),
        },
        "top_new_suspects": sorted(new_suspects, key=lambda r: -r["ship_prob"])[:20],
        "top_rejected_corrections": sorted(corrections_rejected, key=lambda r: r["ship_prob"])[:20],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_JSON}")

    # Markdown summary
    md = []
    md.append("# OC01 label audit — v7.6 self-review\n")
    md.append(f"- **Model**: `{summary['model']}` (per-site calibrated, threshold={active_thr:.3f})\n")
    md.append(f"- **Samples audited**: {summary['n_total']}\n\n")
    md.append("## Validation of prior 560 ambient→ship corrections\n")
    md.append(f"- v7.6 fires above threshold: **{len(corrections_confirmed)}** / {len(corrections_confirmed) + len(corrections_rejected)} "
              f"({summary['rates']['prior_correction_validation']:.1%})\n")
    md.append(f"- v7.6 disagrees (would have stayed ambient): {len(corrections_rejected)}\n\n")
    md.append("## NEW label-noise candidates in unchanged ambients\n")
    md.append(f"- v7.6 surfaced **{len(new_suspects)}** additional suspects out of 240 unchanged ambients "
              f"({summary['rates']['new_suspect_rate_in_unchanged_ambients']:.1%})\n")
    md.append(f"- Top suspect ship_prob: {new_suspects[0]['ship_prob']:.3f}\n" if new_suspects else "")
    md.append("\n### Top-10 new suspects\n\n")
    md.append("| ship_prob | source_file | event_id |\n")
    md.append("|-----------|-------------|----------|\n")
    for r in sorted(new_suspects, key=lambda r: -r["ship_prob"])[:10]:
        md.append(f"| {r['ship_prob']:.3f} | `{r['source_file']}` | `{r['event_id']}` |\n")
    OUT_MD.write_text("".join(md))
    print(f"wrote {OUT_MD}")

    # Bottom-line print for terminal
    print()
    print(f"prior corrections validated: {len(corrections_confirmed)}/{len(corrections_confirmed)+len(corrections_rejected)} "
          f"({summary['rates']['prior_correction_validation']:.1%})")
    print(f"new label-noise candidates:  {len(new_suspects)}/240 unchanged ambients "
          f"({summary['rates']['new_suspect_rate_in_unchanged_ambients']:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
