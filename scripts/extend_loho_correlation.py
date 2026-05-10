"""Extend the day-9 LOHO/cosine-correlation analysis from n=7 to n≈14.

Background
----------
docs/empirical_findings.md reports Pearson r(z-cos, v7.4_accuracy) = -0.57
on **n=7** sites. The reason it's not larger is that the original
known_sites.json only covered 7 of the 14 sites in
data/eval/per_site_v7_4.json. The remaining 7 (fk01, sb01, sb02, mb01,
hi01, gr01, oc01) are SanctSound stations that were trained on under
different `source_id` keys (e.g. `sanctsound60s-fk01`, `sanctsound-diverse-fk01`)
so the original precompute step skipped them.

What this script does
---------------------
1. Walks EVERY training JSONL (not just v7_bulk/) and aggregates rows
   per *normalised site_id* — collapsing any `sanctsound*-fk01` etc.
   under a single `fk01` bucket.
2. For each site with ≥ MIN_ROWS rows, computes a 64-band log-mel
   signature averaged across up to MAX_PER_SITE pre-computed .npy
   spectrograms.
3. Writes the extended registry to data/known_sites.json (overwriting),
   keeping a backup at data/known_sites.json.day13.bak.
4. Re-runs the cosine-vs-accuracy correlation against the same
   data/eval/per_site_v7_4.json and writes a fresh report at
   data/eval/loho_correlation_extended.json.
5. Prints a human-readable summary comparing n=7 (old) vs n=N (new)
   correlation strength.

Safe to run unattended: idempotent, no model retrain, no test changes,
all outputs go to known paths.

Usage
-----
    PYTHONPATH=src venv/bin/python scripts/extend_loho_correlation.py
    PYTHONPATH=src venv/bin/python scripts/extend_loho_correlation.py \\
        --max-per-site 200   # cap rows per site for speed (default 500)
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ocean_sentinel.gemma.audio_features import signature_from_spec_array


# ── source_id → normalised site_id ─────────────────────────────────────


_SANCTSOUND_PREFIXES = (
    "sanctsound60s-",
    "sanctsound-diverse-",
    "sanctsound-corrected-",
    "sanctsound-more-60s-",
)


def _normalize_source(source_id: str) -> str | None:
    """Map a training source_id to its physical site_id.

    Returns None for sources we don't want included in the per-site
    similarity registry (e.g. raw 'shipsear' / 'deepship' which mix
    many real sites under one umbrella, or generic 'mbari-diverse'
    that's collapsed under 'mbari').
    """
    s = source_id.lower()
    for pfx in _SANCTSOUND_PREFIXES:
        if s.startswith(pfx):
            return s[len(pfx):]   # e.g. "fk01"
    if s in {"mbari", "mbari-diverse"}:
        return "mbari"
    if s.startswith("ais-correlated-60s-"):
        # 60s variant of the ais-correlated streams — same physical site
        return "ais-correlated-" + s[len("ais-correlated-60s-"):]
    if s.startswith("ais-correlated-"):
        return s
    if s in {"shipsear", "shipsear-groundtruth"}:
        return "shipsear"
    if s == "sanctsound":
        return "sanctsound"
    if s == "deepship":
        return "deepship"
    return None


# ── walk training data ─────────────────────────────────────────────────


def _load_spec_safe(path: Path) -> np.ndarray | None:
    try:
        spec = np.load(str(path))
        if spec.ndim != 2:
            return None
        return spec
    except Exception:
        return None


def _bucketise_rows(
    *, max_per_site: int, min_rows_per_site: int = 5,
) -> dict[str, list[Path]]:
    """Walk every training JSONL, collect spectrogram_path per site_id."""
    buckets: dict[str, list[Path]] = defaultdict(list)
    for jsonl in sorted(Path("data/training").rglob("*.jsonl")):
        try:
            for line in jsonl.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                src = (
                    (row.get("provenance") or {}).get("source_id")
                    or row.get("source_id")
                    or row.get("source")
                )
                if not isinstance(src, str):
                    continue
                site = _normalize_source(src)
                if site is None:
                    continue
                if len(buckets[site]) >= max_per_site:
                    continue
                spec_path = row.get("spectrogram_path")
                if not isinstance(spec_path, str):
                    continue
                buckets[site].append(Path(spec_path))
        except Exception:
            continue
    # drop sites with too few rows
    return {site: paths for site, paths in buckets.items()
            if len(paths) >= min_rows_per_site}


def _compute_site_signatures(
    buckets: dict[str, list[Path]],
) -> list[dict[str, Any]]:
    out = []
    for site, paths in sorted(buckets.items()):
        sigs = []
        for p in paths:
            spec = _load_spec_safe(p)
            if spec is None:
                continue
            try:
                sigs.append(signature_from_spec_array(spec, n_bands=64))
            except Exception:
                continue
        if not sigs:
            continue
        arr = np.asarray(sigs, dtype=np.float32)
        mean_sig = arr.mean(axis=0).tolist()
        out.append({
            "id": site,
            "label": site,
            "n_rows_used": len(sigs),
            "signature": [round(float(x), 3) for x in mean_sig],
        })
    return out


# ── correlation analysis ───────────────────────────────────────────────


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _zscore(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    sd = float(a.std())
    return (a - a.mean()) / sd if sd > 0 else a - a.mean()


def _pearson(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Pearson r and a two-sided p-value approximation. We avoid scipy
    so this script has zero extra deps."""
    n = len(xs)
    if n < 3:
        return 0.0, 1.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(np.sqrt((xm ** 2).sum() * (ym ** 2).sum()))
    if denom == 0:
        return 0.0, 1.0
    r = float((xm * ym).sum() / denom)
    # Fisher's z transformation for an approx two-sided p-value
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - r ** 2))
    # crude two-sided p from t with n-2 dof — survival function approx
    # using normal tail (works decently for n>=8)
    from math import erf, sqrt
    z = abs(t)
    p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    return r, float(p)


def _spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    rx = np.argsort(np.argsort(xs)).astype(np.float64)
    ry = np.argsort(np.argsort(ys)).astype(np.float64)
    return _pearson(rx.tolist(), ry.tolist())


# ── runner ─────────────────────────────────────────────────────────────


def _run(max_per_site: int, min_rows: int) -> None:
    t0 = time.time()
    print("=" * 76)
    print("EXTENDED LOHO COSINE-vs-ACCURACY CORRELATION")
    print("=" * 76)
    print(f"  max rows per site:  {max_per_site}")
    print(f"  min rows per site:  {min_rows}")

    print("\n[1/4] walking training/**/*.jsonl ...")
    buckets = _bucketise_rows(max_per_site=max_per_site,
                                min_rows_per_site=min_rows)
    print(f"      {len(buckets)} sites with ≥{min_rows} rows")
    for site, paths in sorted(buckets.items(), key=lambda p: -len(p[1])):
        print(f"        {site:<40s}  {len(paths)} rows")

    print("\n[2/4] computing 64-band signatures ...")
    sites = _compute_site_signatures(buckets)
    print(f"      {len(sites)} sites with usable signatures")

    # Backup + write extended known_sites.json
    out_path = Path("data/known_sites.json")
    if out_path.exists():
        backup = Path("data/known_sites.json.day13.bak")
        if not backup.exists():
            shutil.copy2(out_path, backup)
            print(f"      backed up old registry to {backup}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"sites": sites}, indent=2))
    print(f"      wrote {out_path} ({len(sites)} sites)")

    # ── correlation ──────────────────────────────────────────────────
    print("\n[3/4] running cosine-vs-accuracy correlation ...")
    eval_path = Path("data/eval/per_site_v7_4.json")
    if not eval_path.exists():
        print(f"      MISSING {eval_path} — cannot compute correlation.")
        return
    eval_data = json.loads(eval_path.read_text()).get("per_site", {})

    sig_by_id: dict[str, np.ndarray] = {
        s["id"]: _zscore(np.asarray(s["signature"], dtype=np.float32))
        for s in sites
    }
    ids = sorted(sig_by_id.keys())

    # Find each site's nearest training neighbour (excluding self)
    nearest: dict[str, tuple[str, float]] = {}
    for sid in ids:
        a = sig_by_id[sid]
        best, best_sim = None, -2.0
        for other in ids:
            if other == sid:
                continue
            sim = _cosine(a, sig_by_id[other])
            if sim > best_sim:
                best, best_sim = other, sim
        if best:
            nearest[sid] = (best, best_sim)

    # Match eval site ids to known_sites ids
    rows: list[dict[str, Any]] = []
    for site_id, stats in eval_data.items():
        norm = site_id  # they should already match because of normaliser
        if norm not in nearest:
            continue
        acc = stats.get("accuracy")
        if acc is None:
            continue
        nb, sim = nearest[norm]
        rows.append({
            "site_id": site_id,
            "accuracy": float(acc),
            "nearest_site": nb,
            "z_cosine": round(float(sim), 4),
            "n_eval_clips": int(stats.get("n", 0)),
        })

    sims = [r["z_cosine"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    pearson_r, pearson_p = _pearson(sims, accs)
    spearman_r, spearman_p = _spearman(sims, accs)

    print(f"\n  matched {len(rows)} sites:")
    print(f"  {'site':<40s} {'acc':>6s}  {'nearest':<32s} {'z-cos':>7s}")
    for r in sorted(rows, key=lambda x: x["z_cosine"]):
        print(f"  {r['site_id']:<40s} {r['accuracy']:>6.3f}  "
              f"{r['nearest_site']:<32s} {r['z_cosine']:>7.3f}")

    print(f"\n  Pearson r:  {pearson_r:+.3f}  (p ≈ {pearson_p:.3f}, n={len(rows)})")
    print(f"  Spearman ρ: {spearman_r:+.3f}  (p ≈ {spearman_p:.3f}, n={len(rows)})")

    # ── save ─────────────────────────────────────────────────────────
    out_corr = Path("data/eval/loho_correlation_extended.json")
    out_corr.parent.mkdir(parents=True, exist_ok=True)
    out_corr.write_text(json.dumps({
        "_meta": {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "n_sites": len(rows),
            "max_per_site_used": max_per_site,
        },
        "rows": rows,
        "pearson": {"r": round(pearson_r, 4), "p": round(pearson_p, 4)},
        "spearman": {"rho": round(spearman_r, 4), "p": round(spearman_p, 4)},
    }, indent=2))
    print(f"\n[4/4] wrote {out_corr}")

    # ── headline ─────────────────────────────────────────────────────
    print(f"\n{'═' * 76}")
    print(f"  PRIOR HEADLINE  (docs/empirical_findings.md):  n=7,  r=-0.57, p=0.18")
    print(f"  EXTENDED:                                       n={len(rows)},  "
          f"r={pearson_r:+.2f}, p={pearson_p:.2f}")
    print(f"  Spearman:                                       ρ={spearman_r:+.2f}, "
          f"p={spearman_p:.2f}")
    print(f"  Wall time: {time.time() - t0:.1f}s")
    print(f"{'═' * 76}")
    print()
    print(f"  next: paste the new numbers into docs/empirical_findings.md.")
    print(f"  if |r| or p materially changed, update the interpretation section.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-per-site", type=int, default=500,
                    help="cap rows per site (default 500)")
    p.add_argument("--min-rows", type=int, default=5,
                    help="minimum rows per site to include (default 5)")
    args = p.parse_args()
    _run(max_per_site=args.max_per_site, min_rows=args.min_rows)


if __name__ == "__main__":
    main()
