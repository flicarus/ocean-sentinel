"""Independent acoustic audit of OC01 labels — NO neural network.

Computes per-sample acoustic features directly from the cached mel
spectrograms (128 bins, fmax=1000 Hz):

  - peak_freq_hz: frequency of max mean energy across time
  - spectral_flatness: geometric/arithmetic mean ratio (tonal vs broadband)
  - blade_band_db: mean dB in 5-50 Hz (cargo blade-rate harmonics)
  - engine_band_db: mean dB in 50-500 Hz (engine harmonics)
  - hf_band_db: mean dB in 500-1000 Hz (high-freq broadband ref)
  - tonality_ratio: low-band excess vs high-band background (vessel proxy)

Then compares the 560 ambient→ship corrected samples vs the 240
unchanged ambients on those features. If the corrections genuinely
contain vessel acoustic signatures, we expect statistically significant
separation in tonality features — independent of any NN.

Reports Mann-Whitney U + Cohen's d, plus per-bucket means/stds and the
fraction of corrections that exceed an ambient-derived threshold on
each feature.

Output: data/audit/oc01_acoustic_features.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/training/sanctsound_corrected.jsonl"
OUT_JSON = ROOT / "data/audit/oc01_acoustic_features.json"
OUT_MD = ROOT / "data/audit/oc01_acoustic_features.md"

# Mel bin → Hz mapping (matches cnn_v7_classifier config)
N_MELS = 128
FMAX = 1000.0
MEL_FREQS = librosa.mel_frequencies(n_mels=N_MELS, fmax=FMAX)

BLADE_LO, BLADE_HI = 5.0, 50.0      # cargo blade-rate harmonics
ENGINE_LO, ENGINE_HI = 50.0, 500.0  # diesel engine harmonics
HF_LO, HF_HI = 500.0, 1000.0        # high-freq broadband reference

BLADE_MASK  = (MEL_FREQS >= BLADE_LO) & (MEL_FREQS < BLADE_HI)
ENGINE_MASK = (MEL_FREQS >= ENGINE_LO) & (MEL_FREQS < ENGINE_HI)
HF_MASK     = (MEL_FREQS >= HF_LO) & (MEL_FREQS < HF_HI)


def features_from_spec(spec_db: np.ndarray) -> dict:
    """spec_db: (128, T) absolute-dB mel spectrogram."""
    # Time-averaged dB per mel bin
    mean_db_per_bin = spec_db.mean(axis=1)  # (128,)

    # Peak frequency
    peak_bin = int(np.argmax(mean_db_per_bin))
    peak_freq = float(MEL_FREQS[peak_bin])

    # Convert dB → power for flatness (flatness on dB values is wrong)
    power = 10.0 ** (mean_db_per_bin / 10.0)
    # Spectral flatness = geometric mean / arithmetic mean of power
    log_power = np.log(power + 1e-30)
    flatness = float(np.exp(log_power.mean()) / (power.mean() + 1e-30))

    blade_band_db  = float(mean_db_per_bin[BLADE_MASK].mean())
    engine_band_db = float(mean_db_per_bin[ENGINE_MASK].mean())
    hf_band_db     = float(mean_db_per_bin[HF_MASK].mean())

    # Tonality proxy: how much do low frequencies exceed high frequencies?
    # Vessels have characteristic low-freq tonals above broadband background.
    tonality_ratio = float((blade_band_db + engine_band_db) / 2.0 - hf_band_db)

    return {
        "peak_freq_hz":      peak_freq,
        "spectral_flatness": flatness,
        "blade_band_db":     blade_band_db,
        "engine_band_db":    engine_band_db,
        "hf_band_db":        hf_band_db,
        "tonality_ratio":    tonality_ratio,
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD effect size. Sign positive when a > b on average."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2) + 1e-12)
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


def main() -> int:
    # Load oc01 samples
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

    rows = []
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        if i % 200 == 0:
            print(f"  {i}/{len(samples)}")
        path = ROOT / s["spectrogram_path"]
        try:
            spec = np.load(path)
        except Exception:
            continue
        feats = features_from_spec(spec)
        feats["current_label"] = s["label"]
        feats["original_taxonomy"] = s.get("taxonomy", {}).get("category", "")
        feats["event_id"] = s["event_id"]
        feats["source_file"] = s.get("provenance", {}).get("source_file", "")
        rows.append(feats)
    print(f"done in {time.perf_counter() - t0:.1f}s; n={len(rows)}")

    # Two buckets, by current label (corrections vs unchanged ambient)
    corrected = [r for r in rows if r["current_label"] == "ship"]
    ambient = [r for r in rows if r["current_label"] == "not_ship"]
    print(f"corrected (label=ship): n={len(corrected)}")
    print(f"unchanged ambient:      n={len(ambient)}")

    feature_names = [
        "peak_freq_hz",
        "spectral_flatness",
        "blade_band_db",
        "engine_band_db",
        "hf_band_db",
        "tonality_ratio",
    ]

    stats: dict = {}
    for fn in feature_names:
        a = np.array([r[fn] for r in corrected])
        b = np.array([r[fn] for r in ambient])
        # Mann-Whitney U is non-parametric; works for skewed dists.
        u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
        d = cohens_d(a, b)
        # Direction interpretation
        if fn == "spectral_flatness":
            # Lower flatness = more tonal = more vessel-like
            interpretation = "lower in vessels (more tonal)" if d < 0 else "higher in vessels"
        elif fn == "peak_freq_hz":
            interpretation = "lower in vessels (engine/blade band)" if d < 0 else "higher in vessels"
        elif fn == "hf_band_db":
            interpretation = "lower in vessels" if d < 0 else "higher in vessels"
        else:
            interpretation = "higher in vessels (more energy)" if d > 0 else "lower in vessels"
        stats[fn] = {
            "corrected_mean": float(a.mean()),
            "corrected_std":  float(a.std(ddof=1)),
            "ambient_mean":   float(b.mean()),
            "ambient_std":    float(b.std(ddof=1)),
            "cohens_d":       d,
            "mann_whitney_u": float(u_stat),
            "p_value":        float(p_val),
            "interpretation": interpretation,
        }

    # Pick an ambient-derived threshold for each feature and report
    # what fraction of "corrected" samples are on the vessel side.
    # We use the 95th percentile of the ambient distribution as the
    # cutoff (or 5th, depending on direction).
    thresholded = {}
    for fn in feature_names:
        b = np.array([r[fn] for r in ambient])
        a = np.array([r[fn] for r in corrected])
        # Direction: vessel-side is higher for engine/blade/tonality,
        # lower for flatness/peak_freq/hf.
        if fn in ("blade_band_db", "engine_band_db", "tonality_ratio"):
            cutoff = float(np.percentile(b, 95))
            frac_vessel = float((a > cutoff).mean())
        else:
            cutoff = float(np.percentile(b, 5))
            frac_vessel = float((a < cutoff).mean())
        thresholded[fn] = {
            "ambient_percentile_cutoff": cutoff,
            "frac_corrected_on_vessel_side": frac_vessel,
        }

    # Composite vessel score: z-score each feature on the ambient
    # distribution (so direction is "how unlike ambient"), sum across
    # features. Higher = more unlike ambient = more vessel-like.
    def z_score_vs_ambient(values: np.ndarray, amb_vals: np.ndarray, vessel_dir: int) -> np.ndarray:
        m, s = amb_vals.mean(), amb_vals.std(ddof=1) + 1e-12
        return ((values - m) / s) * vessel_dir

    directions = {
        "peak_freq_hz":      -1,
        "spectral_flatness": -1,
        "blade_band_db":     +1,
        "engine_band_db":    +1,
        "hf_band_db":        -1,
        "tonality_ratio":    +1,
    }

    amb_features = {fn: np.array([r[fn] for r in ambient]) for fn in feature_names}
    corr_features = {fn: np.array([r[fn] for r in corrected]) for fn in feature_names}

    amb_composite = np.zeros(len(ambient))
    corr_composite = np.zeros(len(corrected))
    for fn in feature_names:
        amb_composite  += z_score_vs_ambient(amb_features[fn],  amb_features[fn], directions[fn])
        corr_composite += z_score_vs_ambient(corr_features[fn], amb_features[fn], directions[fn])

    composite_cutoff = float(np.percentile(amb_composite, 95))
    frac_corrected_vessel_like = float((corr_composite > composite_cutoff).mean())
    composite_d = cohens_d(corr_composite, amb_composite)

    # Top-confidence corrected examples
    order = np.argsort(-corr_composite)
    top_corrected = [
        {
            "composite_score": float(corr_composite[idx]),
            "event_id": corrected[idx]["event_id"],
            "source_file": corrected[idx]["source_file"],
            "blade_band_db": corrected[idx]["blade_band_db"],
            "engine_band_db": corrected[idx]["engine_band_db"],
            "tonality_ratio": corrected[idx]["tonality_ratio"],
            "spectral_flatness": corrected[idx]["spectral_flatness"],
        }
        for idx in order[:10]
    ]
    # And the most ambient-like "corrected" (potential false correction)
    order_low = np.argsort(corr_composite)
    weak_corrected = [
        {
            "composite_score": float(corr_composite[idx]),
            "event_id": corrected[idx]["event_id"],
            "source_file": corrected[idx]["source_file"],
        }
        for idx in order_low[:5]
    ]

    summary = {
        "method": "independent acoustic features (no neural network)",
        "site": "oc01",
        "feature_definitions": {
            "peak_freq_hz":      "Hz of mel bin with max time-averaged energy",
            "spectral_flatness": "geometric/arithmetic mean ratio of power; low=tonal, high=broadband",
            "blade_band_db":     "mean dB in 5-50 Hz (cargo blade-rate harmonics)",
            "engine_band_db":    "mean dB in 50-500 Hz (diesel engine harmonics)",
            "hf_band_db":        "mean dB in 500-1000 Hz (broadband reference)",
            "tonality_ratio":    "(blade+engine)/2 - hf, in dB; vessel proxy",
        },
        "samples": {
            "corrected_ship": len(corrected),
            "unchanged_ambient": len(ambient),
        },
        "per_feature": stats,
        "thresholded": thresholded,
        "composite": {
            "method": "z-score each feature vs ambient distribution (vessel-direction signed), sum across features",
            "ambient_95th_pctile_cutoff": composite_cutoff,
            "frac_corrected_above_cutoff": frac_corrected_vessel_like,
            "cohens_d_corrected_vs_ambient": composite_d,
        },
        "top_corrected_examples": top_corrected,
        "weakest_corrected_examples": weak_corrected,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_JSON}")

    # Markdown summary
    md = ["# OC01 acoustic feature audit — independent of any NN\n\n"]
    md.append(f"- corrected (label=ship): {len(corrected)}\n")
    md.append(f"- unchanged ambient:      {len(ambient)}\n\n")
    md.append("## Per-feature separation\n\n")
    md.append("| feature | corrected mean ± SD | ambient mean ± SD | Cohen's d | p-value | dir |\n")
    md.append("|---------|---------------------|--------------------|-----------|---------|-----|\n")
    for fn in feature_names:
        s = stats[fn]
        md.append(
            f"| {fn} | {s['corrected_mean']:.3f} ± {s['corrected_std']:.3f} | "
            f"{s['ambient_mean']:.3f} ± {s['ambient_std']:.3f} | "
            f"{s['cohens_d']:.2f} | {s['p_value']:.2e} | {s['interpretation']} |\n"
        )
    md.append("\n## Composite vessel-likeness score\n\n")
    md.append(f"- Cohen's d (corrected vs ambient): {composite_d:.2f}\n")
    md.append(f"- Fraction of corrections above 95th-pctile ambient cutoff: {frac_corrected_vessel_like:.1%}\n")
    md.append(f"- Mann-Whitney p-values for each feature: see JSON.\n")
    OUT_MD.write_text("".join(md))
    print(f"wrote {OUT_MD}")

    # Terminal bottom line
    print()
    print(f"composite separation (Cohen's d): {composite_d:.2f}")
    print(f"frac corrected on vessel side (>95th pctile ambient): {frac_corrected_vessel_like:.1%}")
    for fn in feature_names:
        s = stats[fn]
        print(f"  {fn:20s}  d={s['cohens_d']:+.2f}  p={s['p_value']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
