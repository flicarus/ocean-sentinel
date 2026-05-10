"""Empirically derive site-classification thresholds from real data.

Inputs:
  - data/known_sites.json     (signatures from precompute_signatures.py)
  - data/eval/per_site_v7_4.json  (per-site accuracy of v7.4 from days 9-10 LOHO)

Outputs:
  - data/calibration/site_classification.yaml
  - prints a markdown report to stdout

What we derive:

  1. PAIRWISE COSINE DISTRIBUTION
     For all pairs of training-site signatures, compute cosine similarity.
     Look at the percentile distribution — this tells us what "similar"
     and "very similar" mean in practice for our corpus.

  2. ACCURACY-VS-SIMILARITY CORRELATION
     For every site that has a v7.4 LOHO accuracy:
       - find its nearest training-site neighbor (highest cosine to OTHER sites)
       - record (cosine_to_nearest, v7.4 accuracy) pair
     If v7.4 generalizes well at high cosine and degrades at low cosine,
     we have empirical evidence for a threshold. If accuracy is uniformly
     high, the threshold is permissive (we don't need fine-tune most of
     the time).

  3. AMBIENT BAND CUTOFFS
     For each site, locate the dominant Mel band → Hz. Pool across sites
     to see what bands actually carry energy in the wild — gives us
     informed cutoffs for the ambient_class display labels (instead of
     textbook values).

The output YAML config is consumed by gemma/audio_features.py at runtime,
so changes here flow without code edits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

KNOWN_SITES_JSON = Path("data/known_sites.json")
PER_SITE_EVAL = Path("data/eval/per_site_v7_4.json")
OUT_YAML = Path("data/calibration/site_classification.yaml")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score normalize. Removes absolute-energy bias from log-mel sigs
    so cosine measures spectral SHAPE rather than overall loudness."""
    a = arr.astype(np.float32)
    sd = float(a.std())
    return (a - a.mean()) / sd if sd > 0 else a - a.mean()


def _percentiles(values: list[float], pcts: list[int]) -> dict[int, float]:
    arr = np.asarray(values, dtype=np.float32)
    return {p: round(float(np.percentile(arr, p)), 3) for p in pcts}


def _normalize_site_id(sid: str) -> str:
    """Map per_site eval keys to known_sites.json ids where they differ."""
    if sid in {"mbari", "mbari-diverse"}:
        return "mbari-diverse"
    return sid


def main() -> None:
    if not KNOWN_SITES_JSON.exists():
        raise SystemExit(
            f"missing {KNOWN_SITES_JSON} — run scripts/precompute_signatures.py first"
        )

    raw = json.loads(KNOWN_SITES_JSON.read_text())
    sites = raw.get("sites", []) if isinstance(raw, dict) else raw
    by_id = {s["id"]: s for s in sites}
    ids = sorted(by_id.keys())

    print(f"\n[derive] {len(sites)} sites from known_sites.json\n")

    # ── 1. Pairwise cosine distribution ────────────────────────────────
    pair_cos = []
    nearest_per_site: dict[str, tuple[str, float]] = {}
    for sid in ids:
        sig_a = _zscore(np.asarray(by_id[sid]["signature"], dtype=np.float32))
        best_other = None
        best_sim = -2.0
        for other in ids:
            if other == sid:
                continue
            sig_b = _zscore(np.asarray(by_id[other]["signature"], dtype=np.float32))
            sim = _cosine(sig_a, sig_b)
            pair_cos.append(sim)
            if sim > best_sim:
                best_sim = sim
                best_other = other
        if best_other:
            nearest_per_site[sid] = (best_other, best_sim)

    cos_p = _percentiles(pair_cos, [10, 25, 50, 75, 90, 95, 99])
    print("Z-SCORED PAIRWISE COSINE DISTRIBUTION (all site pairs):")
    print("  (z-score removes absolute-energy bias; this measures spectral shape only)")
    for p, v in cos_p.items():
        print(f"  p{p:>2}: {v}")
    print()

    # ── 2. Accuracy vs nearest-neighbor cosine ─────────────────────────
    acc_correlation: list[tuple[str, float, float]] = []
    if PER_SITE_EVAL.exists():
        eval_data = json.loads(PER_SITE_EVAL.read_text()).get("per_site", {})
        print("ACCURACY × NEAREST-COSINE (v7.4 LOHO eval):")
        print(f"  {'site':<38s} {'acc':>6s}  {'nearest':<32s} {'cos':>6s}")
        for site_id, stats in eval_data.items():
            norm = _normalize_site_id(site_id)
            if norm not in nearest_per_site:
                # site we evaluated on but doesn't have a signature
                continue
            nearest, sim = nearest_per_site[norm]
            acc = stats.get("accuracy")
            if acc is None:
                continue
            acc_correlation.append((site_id, sim, float(acc)))
            print(f"  {site_id:<38s} {acc:>6.3f}  {nearest:<32s} {sim:>6.3f}")
        print()
    else:
        print(f"NOTE: {PER_SITE_EVAL} not found — skipping accuracy correlation\n")

    # ── 3. Threshold derivation ────────────────────────────────────────
    # Strategy:
    #   - use_existing_min_cos = p90 of pairwise distribution
    #     (only sites that genuinely look like a known site qualify)
    #   - finetune_min_cos = p50 (median) — sites at "average similarity"
    #     can still benefit from light fine-tune, below median needs more
    #
    # If we have accuracy data, refine via cutoffs that separate high- and
    # low-accuracy sites.
    use_existing = cos_p[90]
    finetune     = cos_p[50]

    accuracy_evidence: dict[str, Any] = {}
    if acc_correlation:
        sims_acc = sorted(acc_correlation, key=lambda r: r[1])
        # Find cosine cutoff where accuracy crosses 0.95 (good enough,
        # doesn't need fine-tune). Lowest cos with acc≥0.95.
        good = [r for r in sims_acc if r[2] >= 0.95]
        if good:
            empirical_use_existing = min(r[1] for r in good)
            use_existing = max(use_existing, round(empirical_use_existing, 3))
            accuracy_evidence["empirical_use_existing"] = round(empirical_use_existing, 3)
        # Below this, accuracy struggles → need fine-tune or full retrain
        bad = [r for r in sims_acc if r[2] < 0.85]
        if bad:
            empirical_finetune = max(r[1] for r in bad)
            finetune = max(min(finetune, round(empirical_finetune + 0.01, 3)),
                           cos_p[10])  # never below the 10th percentile
            accuracy_evidence["empirical_finetune_floor"] = round(empirical_finetune, 3)

    # ── 4. Write YAML ──────────────────────────────────────────────────
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "_meta": {
            "derived_from": {
                "known_sites": str(KNOWN_SITES_JSON),
                "per_site_eval": str(PER_SITE_EVAL) if PER_SITE_EVAL.exists() else None,
            },
            "n_sites": len(sites),
            "n_pairs_compared": len(pair_cos),
            "pairwise_cosine_percentiles": cos_p,
            "accuracy_evidence": accuracy_evidence,
        },
        # Display-only labels (Wenz 1962 / Hildebrand 2009 ocean acoustics).
        # Not derived empirically — these are textbook frequency bands; the
        # 64-dim signature is what actually drives the recommendation logic.
        "ambient_class_thresholds_hz": {
            "infrasound": [0,    80],
            "low_freq":   [80,   500],
            "mid_freq":   [500,  2000],
            "high_freq":  [2000, None],
        },
        "ambient_class_labels": {
            "infrasound": "infrasound · deep-water",
            "low_freq":   "low-frequency · vessel band",
            "mid_freq":   "mid-frequency · mixed",
            "high_freq":  "high-frequency · biological",
        },
        # Empirically derived from this corpus. See _meta.pairwise_cosine_percentiles
        # for the distribution context. accuracy_evidence shows the LOHO data
        # used to refine the cutoffs.
        "adapter_strategy_thresholds": {
            "use_existing_min_cos": use_existing,
            "finetune_min_cos":     finetune,
        },
    }
    OUT_YAML.write_text(yaml.safe_dump(config, sort_keys=False, indent=2))
    print(f"[derive] wrote {OUT_YAML}\n")
    print("FINAL THRESHOLDS:")
    print(f"  use_existing_min_cos = {use_existing}")
    print(f"  finetune_min_cos     = {finetune}")
    if accuracy_evidence:
        print(f"  evidence: {accuracy_evidence}")


if __name__ == "__main__":
    main()
