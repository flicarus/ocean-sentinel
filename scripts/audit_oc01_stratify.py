"""Phase 1 of deeper oc01 audit: stratify by source_file + visualize
the 2D feature distribution.

Goal: understand WHY the 240 "unchanged ambients" look louder/more
tonal than the 560 "corrections". Three angles:

  1. Per-source-file pairing: are both groups drawn from the same
     recordings, just different time chunks? Or from different
     recordings entirely?
  2. 2D scatter (blade_band_db vs spectral_flatness): clusters or
     continuum?
  3. Peak frequency distribution: does ambient peak at characteristic
     biological/geological bands (whale calls, storms)?

Output:
  - data/audit/oc01_stratify.json
  - data/audit/oc01_2d_features.png (scatter)
  - data/audit/oc01_peak_freq_hist.png (histogram)
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter, defaultdict

import librosa
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/training/sanctsound_corrected.jsonl"
OUT_DIR = ROOT / "data/audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MEL_FREQS = librosa.mel_frequencies(n_mels=128, fmax=1000.0)
BLADE = (MEL_FREQS >= 5) & (MEL_FREQS < 50)
ENGINE = (MEL_FREQS >= 50) & (MEL_FREQS < 500)
HF = (MEL_FREQS >= 500) & (MEL_FREQS < 1000)


def features(spec: np.ndarray) -> dict:
    mdb = spec.mean(axis=1)
    power = 10 ** (mdb / 10)
    flat = float(np.exp(np.log(power + 1e-30).mean()) / (power.mean() + 1e-30))
    peak_bin = int(np.argmax(mdb))
    return dict(
        blade=float(mdb[BLADE].mean()),
        engine=float(mdb[ENGINE].mean()),
        hf=float(mdb[HF].mean()),
        flat=flat,
        tonality=float((mdb[BLADE].mean() + mdb[ENGINE].mean()) / 2 - mdb[HF].mean()),
        peak_freq=float(MEL_FREQS[peak_bin]),
        peak_bin=peak_bin,
    )


def main() -> None:
    rows = []
    with open(CORPUS) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            prov = d.get("provenance", {}).get("source_id", "")
            if "oc01" not in prov.lower():
                continue
            spec = np.load(d["spectrogram_path"])
            feats = features(spec)
            feats["label"] = d["label"]
            feats["source_file"] = d.get("provenance", {}).get("source_file", "")
            feats["event_id"] = d["event_id"]
            rows.append(feats)

    corr = [r for r in rows if r["label"] == "ship"]
    amb = [r for r in rows if r["label"] == "not_ship"]
    print(f"corrected={len(corr)}  ambient={len(amb)}")

    # === 1. Per-source-file pairing ===
    files_corr = Counter(r["source_file"] for r in corr)
    files_amb = Counter(r["source_file"] for r in amb)
    common = set(files_corr) & set(files_amb)
    corr_only = set(files_corr) - set(files_amb)
    amb_only = set(files_amb) - set(files_corr)
    print()
    print(f"distinct source files — corrected: {len(files_corr)}, ambient: {len(files_amb)}")
    print(f"shared files: {len(common)}")
    print(f"corrected-only files: {len(corr_only)}")
    print(f"ambient-only files: {len(amb_only)}")
    if amb_only:
        print(f"  ambient-only files (first 5):")
        for f in list(amb_only)[:5]:
            print(f"    {f}  (n={files_amb[f]})")
    if corr_only:
        print(f"  corrected-only files (first 5):")
        for f in list(corr_only)[:5]:
            print(f"    {f}  (n={files_corr[f]})")

    # Within-shared-file comparison
    within_file_stats = []
    for sf in common:
        c_in = [r for r in corr if r["source_file"] == sf]
        a_in = [r for r in amb if r["source_file"] == sf]
        if len(c_in) >= 2 and len(a_in) >= 2:
            c_blade = np.mean([r["blade"] for r in c_in])
            a_blade = np.mean([r["blade"] for r in a_in])
            within_file_stats.append({
                "file": sf,
                "n_corr": len(c_in),
                "n_amb": len(a_in),
                "blade_corr": c_blade,
                "blade_amb": a_blade,
                "blade_diff": c_blade - a_blade,
            })
    print()
    print(f"files with both groups (≥2 each): {len(within_file_stats)}")
    if within_file_stats:
        diffs = np.array([w["blade_diff"] for w in within_file_stats])
        print(f"  within-file blade_db diff (corr - amb): mean={diffs.mean():.2f}, median={np.median(diffs):.2f}")
        print(f"  files where corr LOUDER than amb: {(diffs > 0).sum()}/{len(diffs)}")
        print(f"  files where corr QUIETER than amb: {(diffs < 0).sum()}/{len(diffs)}")

    # === 2. 2D scatter ===
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    ax.scatter([r["blade"] for r in amb], [r["flat"] for r in amb],
               s=12, alpha=0.5, c="#9CA3AF", label=f"ambient (n={len(amb)})", edgecolors="none")
    ax.scatter([r["blade"] for r in corr], [r["flat"] for r in corr],
               s=12, alpha=0.5, c="#EF4444", label=f"corrected (n={len(corr)})", edgecolors="none")
    # Mark "unambiguous vessel" quadrant
    ax.axvline(-20, ls="--", c="#00D4C8", alpha=0.6, lw=1)
    ax.axhline(0.05, ls="--", c="#00D4C8", alpha=0.6, lw=1)
    ax.text(-15, 0.02, "unambiguous\nvessel zone", fontsize=9, c="#00D4C8")
    ax.set_xlabel("blade-band energy (dB, 5-50 Hz)")
    ax.set_ylabel("spectral flatness (0=tonal, 1=broadband)")
    ax.set_title("OC01 — 2D acoustic feature distribution by current label")
    ax.legend(loc="upper right")
    ax.set_xlim(-80, 0)
    ax.set_ylim(0, 1)
    out_scatter = OUT_DIR / "oc01_2d_features.png"
    fig.savefig(out_scatter, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_scatter}")

    # === 3. Peak frequency histogram ===
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    bins = np.linspace(0, 1000, 50)
    ax.hist([r["peak_freq"] for r in amb], bins=bins, alpha=0.55, color="#9CA3AF", label=f"ambient (n={len(amb)})")
    ax.hist([r["peak_freq"] for r in corr], bins=bins, alpha=0.55, color="#EF4444", label=f"corrected (n={len(corr)})")
    # Annotate known bands
    for f, name, c in [(40, "blade-rate", "#00D4C8"), (200, "humpback", "#A78BFA"), (700, "delphinid", "#F59E0B")]:
        ax.axvline(f, ls=":", c=c, alpha=0.6)
        ax.text(f + 5, ax.get_ylim()[1] * 0.85, name, fontsize=8, c=c)
    ax.set_xlabel("peak frequency (Hz)")
    ax.set_ylabel("count")
    ax.set_title("OC01 — peak frequency distribution")
    ax.legend()
    out_hist = OUT_DIR / "oc01_peak_freq_hist.png"
    fig.savefig(out_hist, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_hist}")

    # Peak freq counts in bands
    def band_count(records, lo, hi):
        return sum(1 for r in records if lo <= r["peak_freq"] < hi)
    print()
    print("peak frequency band counts:")
    print(f"  {'band (Hz)':<20s} {'corrected':>10s} {'ambient':>10s}")
    for lo, hi, name in [(0, 50, "blade-rate"), (50, 200, "engine"), (200, 500, "biological mid"), (500, 1000, "biological hi")]:
        print(f"  {name:<20s} {band_count(corr, lo, hi):>10d} {band_count(amb, lo, hi):>10d}")

    # Save JSON summary
    OUT_JSON = OUT_DIR / "oc01_stratify.json"
    OUT_JSON.write_text(json.dumps({
        "n_corr": len(corr),
        "n_amb": len(amb),
        "n_files_corr": len(files_corr),
        "n_files_amb": len(files_amb),
        "n_shared_files": len(common),
        "n_corr_only_files": len(corr_only),
        "n_amb_only_files": len(amb_only),
        "within_file_pairs": within_file_stats[:50],
        "amb_only_file_examples": [{"file": f, "n": files_amb[f]} for f in list(amb_only)[:20]],
        "corr_only_file_examples": [{"file": f, "n": files_corr[f]} for f in list(corr_only)[:20]],
    }, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
