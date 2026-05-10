"""Precompute spectral signatures of known training sites.

Walks data/training/v7_bulk/*.jsonl, groups rows by source_id (the site),
loads each row's pre-computed spectrogram (.npy) and computes a 64-band
log-mel signature, then averages across all rows in the same site.

Output: data/known_sites.json — consumed by gemma.known_sites.compare_to_known_sites.

Usage:
    PYTHONPATH=src venv/bin/python scripts/precompute_signatures.py
    PYTHONPATH=src venv/bin/python scripts/precompute_signatures.py \
        --max-per-site 200  # cap rows per site for speed

Why we use the .npy specs (not raw audio):
- They're already 128-mel log-dB, matching the CNN's input space.
- One-shot disk read per row is much faster than re-decoding source WAVs.
- audio_features.signature_from_spec_array downsamples 128 → 64 to match
  the runtime signature shape used by Gemma.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ocean_sentinel.gemma.audio_features import signature_from_spec_array


def _iter_rows(jsonl_paths: list[Path]):
    for jp in jsonl_paths:
        with jp.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _site_meta(source_id: str) -> dict[str, Any]:
    """Best-effort per-site human metadata. Extend as new sites land."""
    table = {
        "mbari":          {"label": "MBARI MARS",       "lat": 36.71,  "lon": -122.19},
        "mbari-diverse":  {"label": "MBARI MARS",       "lat": 36.71,  "lon": -122.19},
        "ais-correlated-orcasound-lab":   {"label": "Orcasound Lab",      "lat": 48.56, "lon": -123.17},
        "ais-correlated-bush-point":      {"label": "Orcasound Bush Pt",  "lat": 48.05, "lon": -122.74},
        "ais-correlated-port-townsend":   {"label": "Port Townsend",      "lat": 48.11, "lon": -122.76},
        "ais-correlated-sunset-bay":      {"label": "Sunset Bay",         "lat": 47.92, "lon": -123.07},
        "ais-correlated-point-robinson":  {"label": "Point Robinson",     "lat": 47.39, "lon": -122.37},
        "ais-correlated-andrews-bay":     {"label": "Andrews Bay",        "lat": 47.55, "lon": -122.27},
        "ais-correlated-north-sjc":       {"label": "North San Juan",     "lat": 48.62, "lon": -123.16},
        "sb01":          {"label": "Stellwagen Bank",   "lat": 42.30,  "lon": -70.42},
        "oc01":          {"label": "Ocean City NJ",     "lat": 39.27,  "lon": -74.59},
    }
    return table.get(source_id, {"label": source_id, "lat": None, "lon": None})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*",
                        default=["data/training/v7_bulk", "data/training"],
                        help="Directories to scan for *.jsonl training files.")
    parser.add_argument("--max-per-site", type=int, default=200,
                        help="Cap number of rows averaged per site (speed).")
    parser.add_argument("--out", default="data/known_sites.json")
    parser.add_argument("--n-bands", type=int, default=64)
    args = parser.parse_args()

    jsonl_paths: list[Path] = []
    for d in args.inputs:
        p = Path(d)
        if not p.exists():
            continue
        if p.is_file() and p.suffix == ".jsonl":
            jsonl_paths.append(p)
        else:
            jsonl_paths.extend(sorted(p.glob("*.jsonl")))
    if not jsonl_paths:
        raise SystemExit("no .jsonl found in --inputs")

    print(f"[precompute] {len(jsonl_paths)} jsonl files")

    # Group .npy paths by site
    by_site: dict[str, list[Path]] = defaultdict(list)
    for row in _iter_rows(jsonl_paths):
        spec_path = row.get("spectrogram_path")
        if not spec_path:
            continue
        prov = row.get("provenance") or {}
        source_id = (prov.get("source_id") or
                     prov.get("hydrophone") or
                     "unknown")
        if Path(spec_path).exists():
            by_site[source_id].append(Path(spec_path))

    print(f"[precompute] {len(by_site)} unique sites")

    sites_out: list[dict[str, Any]] = []
    for source_id, paths in sorted(by_site.items(), key=lambda kv: -len(kv[1])):
        sample = paths[: args.max_per_site]
        sigs = []
        for sp in sample:
            try:
                spec = np.load(sp)
                sig = signature_from_spec_array(spec, n_bands=args.n_bands)
                sigs.append(sig)
            except Exception as e:
                print(f"  ! failed {sp.name}: {type(e).__name__}: {e}")
        if not sigs:
            continue
        mean_sig = np.mean(np.array(sigs, dtype=np.float32), axis=0)
        meta = _site_meta(source_id)
        sites_out.append({
            "id":         source_id,
            "label":      meta["label"],
            "lat":        meta["lat"],
            "lon":        meta["lon"],
            "n_samples":  len(sigs),
            "signature":  [round(float(x), 3) for x in mean_sig.tolist()],
        })
        print(f"  ✓ {source_id:38s}  n={len(sigs):4d}  "
              f"median PSD {float(np.median(mean_sig)):.1f} dB")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"sites": sites_out}, indent=2))
    print(f"[precompute] wrote {len(sites_out)} sites → {out_path}")


if __name__ == "__main__":
    main()
