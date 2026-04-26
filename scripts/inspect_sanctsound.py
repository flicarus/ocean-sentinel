"""Sanctsound deep-dive: which samples does v6 miss, and why?

10/60 sanctsound val samples are misclassified — all not_ship → ship
(false alarms, would needlessly wake Gemma but never miss a real ship).
One sits at 0.967 confidence on `oc01`.

This script:
  1. Reproduces the v6 val split
  2. Filters to sanctsound entries
  3. Groups by site (the SanctSound prefix in the spectrogram filename:
     oc01, gr01, sb03, etc.)
  4. Dumps acoustic features (engine_band_ratio, peak_freq, flatness)
     for misses vs correct — looks for separating patterns
  5. Saves PNG renders of all misses + a control group of correct samples
     to data/diagnostic/sanctsound/ for visual inspection

The output answers: is sanctsound failure (a) labeling noise like
andrews-bay, (b) a specific site/condition the CNN can't handle, or
(c) borderline acoustic content that genuinely looks ship-like.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_cnn import LABEL_MAP, SpecDataset, _split_session_indices
from ocean_sentinel.services.cnn_classifier import CNNClassifier


OUT_DIR = Path("data/diagnostic/sanctsound")


def _site_from_event_id(event_id: str) -> str:
    """SanctSound IDs look like sanctsound_<site>_<file>_<offset>.

    Site code (oc01, gr01, sb03, ...) is the second segment.
    """
    parts = event_id.split("_")
    return parts[1] if len(parts) >= 2 else "unknown"


def _save_spec_png(spec_path: str, out_path: Path, title: str) -> None:
    spec = np.load(spec_path)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel bin")
    plt.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    miss_dir = OUT_DIR / "misses"
    correct_dir = OUT_DIR / "correct_control"
    miss_dir.mkdir(exist_ok=True)
    correct_dir.mkdir(exist_ok=True)

    ds = SpecDataset(
        "data/training/gemma_labels.v4.jsonl",
        representation_version="abs_db_v1",
        augment=False,
    )
    _, val_idx = _split_session_indices(ds.entries)

    clf = CNNClassifier("data/models/cnn_v6.pt")

    sanctsound = []
    for i in val_idx:
        e = ds.entries[i]
        if e["provenance"]["source_id"] != "sanctsound":
            continue
        spec = np.load(e["spectrogram_path"]).astype(np.float32)
        out = clf.predict(spec, source_id="sanctsound")
        sanctsound.append({
            "event_id": e["event_id"],
            "site": _site_from_event_id(e["event_id"]),
            "truth": e["label"],
            "pred": out["label"],
            "correct": e["label"] == out["label"],
            "confidence": out["confidence"],
            "p_ship": out["probabilities"]["ship"],
            "p_not_ship": out["probabilities"]["not_ship"],
            "spec_path": e["spectrogram_path"],
            "features": e.get("features", {}),
        })

    n_total = len(sanctsound)
    misses = [s for s in sanctsound if not s["correct"]]
    correct = [s for s in sanctsound if s["correct"]]
    print()
    print(f"=== Sanctsound: {len(misses)}/{n_total} miss "
          f"({len(misses) / n_total:.1%}) ===")
    print()

    # 1. Per-site breakdown
    print("--- Per-site miss rate ---")
    print(f"{'site':<10s} {'misses':>7s}/{'total':<7s} {'rate':>6s}  "
          f"{'avg_p_ship_on_miss':>20s}")
    print("-" * 60)
    by_site: dict[str, list] = defaultdict(list)
    for s in sanctsound:
        by_site[s["site"]].append(s)
    for site, items in sorted(by_site.items()):
        miss = [s for s in items if not s["correct"]]
        avg_p = (
            sum(s["p_ship"] for s in miss) / len(miss)
            if miss else float("nan")
        )
        rate_str = f"{len(miss) / len(items):.1%}"
        avg_str = f"{avg_p:.3f}" if miss else "-"
        print(f"{site:<10s} {len(miss):>7d}/{len(items):<7d} "
              f"{rate_str:>6s}  {avg_str:>20s}")
    print()

    # 2. Feature comparison — misses vs correct
    print("--- Acoustic features: misses vs correct ---")
    feature_keys = (
        "engine_band_energy_db",
        "engine_band_ratio",
        "peak_frequency_hz",
        "spectral_flatness",
        "rms_energy",
    )
    print(f"{'feature':<25s} {'miss_mean':>12s} {'miss_std':>10s} "
          f"{'correct_mean':>14s} {'correct_std':>12s}")
    print("-" * 80)
    for k in feature_keys:
        m_vals = [s["features"].get(k, float("nan")) for s in misses
                  if k in s["features"]]
        c_vals = [s["features"].get(k, float("nan")) for s in correct
                  if k in s["features"]]
        if not m_vals or not c_vals:
            continue
        m_arr, c_arr = np.array(m_vals), np.array(c_vals)
        print(f"{k:<25s} {m_arr.mean():>12.3f} {m_arr.std():>10.3f} "
              f"{c_arr.mean():>14.3f} {c_arr.std():>12.3f}")
    print()

    # 3. List all misses sorted by confidence (scariest first)
    print("--- All misses, sorted by confidence ---")
    print(f"{'event_id':<55s} {'truth→pred':<22s} "
          f"{'p_ship':>8s} {'site':>8s}")
    print("-" * 100)
    for s in sorted(misses, key=lambda r: -r["p_ship"]):
        print(
            f"{s['event_id']:<55s} "
            f"{s['truth']+'→'+s['pred']:<22s} "
            f"{s['p_ship']:>7.3f}  {s['site']:>8s}"
        )
    print()

    # 4. Save PNGs of every miss
    print(f"--- Saving miss spectrograms to {miss_dir} ---")
    for s in misses:
        title = (
            f"MISS  {s['event_id']}\n"
            f"truth={s['truth']}  pred={s['pred']}  "
            f"p_ship={s['p_ship']:.3f}  site={s['site']}"
        )
        out_path = miss_dir / f"{s['event_id']}.png"
        _save_spec_png(s["spec_path"], out_path, title)
    print(f"  saved {len(misses)} miss PNGs")

    # 5. Save a control group: 5 correct samples per site (or fewer)
    print(f"--- Saving control correct samples to {correct_dir} ---")
    saved = 0
    for site, items in by_site.items():
        c = [s for s in items if s["correct"]]
        for s in c[:5]:
            title = (
                f"CORRECT  {s['event_id']}\n"
                f"truth={s['truth']}  pred={s['pred']}  "
                f"p_ship={s['p_ship']:.3f}  site={s['site']}"
            )
            out_path = correct_dir / f"{s['event_id']}.png"
            _save_spec_png(s["spec_path"], out_path, title)
            saved += 1
    print(f"  saved {saved} control PNGs")


if __name__ == "__main__":
    main()
