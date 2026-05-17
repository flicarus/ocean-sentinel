"""Rigorous v7.4 benchmark on all available held-out DeepShip clips.

What this answers
-----------------
The earlier `evaluate_unseen.py` used only 5 clips per class × 2 offsets
(n=24). That's a sanity check, not a benchmark — the binomial 95% CI on
24/24 is ~86%-100%, too wide to claim "above day-9 LOHO 90% baseline".

This script runs:
  - EVERY DeepShip clip whose (class, number) tuple is NOT in any
    training JSONL under data/training/ (held-out by construction)
  - At THREE offsets per clip (30 s, 90 s, 150 s where the recording is
    long enough) — captures within-clip variability that production
    `os monitor` will see
  - PLUS a held-out ambient floor (synthetic, deterministic) so we can
    sanity-check false-alarm rate at the same time

It also runs an UNCERTAIN_MAX threshold sweep over the resulting
predictions, so we can tune the abstention threshold based on the
observed uncertainty distribution rather than a guessed value. The
current cnn_inference.py uses 0.20, which slices the day-9 unc_mean
distribution (0.16-0.22) approximately in half.

Outputs
-------
  data/eval/benchmark_v7_4.json   — full per-clip predictions + agg
  data/eval/threshold_sweep.json  — UNCERTAIN_MAX threshold candidates
                                     and their accuracy/confident-rate

Honest reporting
----------------
This eval still under-counts the realistic deployment surface:
  - DeepShip clips are technically in-distribution by SITE (we trained
    on the dataset) but UNSEEN per-clip. So this measures whether the
    model memorises specific recordings.
  - Day-9 LOHO at n=4011 across 14 sites is the canonical metric and
    should remain the headline number.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier


CKPT = "data/models/cnn_v7_4.pt"
SR = 16_000
CLIP_SECONDS = 60
OFFSETS = [30.0, 90.0, 150.0]
THRESHOLD_CANDIDATES = [0.18, 0.20, 0.22, 0.25, 0.27, 0.30]


# ── unseen clip discovery ──────────────────────────────────────────────


def _trained_deepship_ids() -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for jsonl in Path("data/training").rglob("*.jsonl"):
        try:
            for line in jsonl.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                eid = (row.get("event_id") or "").lower()
                if "deepship" in eid:
                    parts = eid.split("_")
                    if len(parts) >= 3:
                        out.add((parts[1], parts[2]))
        except Exception:
            pass
    return out


def _all_unseen_clips() -> list[tuple[str, Path, float]]:
    """Every (label, path, offset) triple we can evaluate. Vessel label
    is 'ship'. Multiple offsets per clip if duration allows."""
    trained = _trained_deepship_ids()
    out: list[tuple[str, Path, float]] = []
    for cls in ("Cargo", "Passengership", "Tanker", "Tug"):
        folder = Path(f"data/deepship/{cls}")
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.wav"),
                        key=lambda x: int(x.stem) if x.stem.isdigit() else 9999):
            if (cls.lower(), p.stem) in trained:
                continue
            try:
                duration = librosa.get_duration(path=str(p))
            except Exception:
                continue
            if duration < CLIP_SECONDS:
                continue
            for off in OFFSETS:
                if off + CLIP_SECONDS <= duration:
                    out.append(("ship", p, off))
    return out


# ── synthetic ambient (sanity floor) ───────────────────────────────────


def _synth_ambient(out: Path, seed: int) -> Path:
    rng = np.random.RandomState(seed)
    n = SR * CLIP_SECONDS
    y = (0.1 * rng.randn(n)).astype(np.float32)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    y = np.fft.irfft(
        Y * (1.0 / (1.0 + freqs / 100.0) ** 0.5).astype("float32"), n=n,
    ).astype(np.float32)
    y = y / max(1.0, float(np.max(np.abs(y))) * 1.05) * 0.4
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y, SR)
    return out


# ── inference ──────────────────────────────────────────────────────────


def _predict(clf: CNNV7Classifier, clip: Path, offset: float = 0.0) -> dict:
    y, sr = librosa.load(
        str(clip), sr=SR, mono=True,
        offset=offset, duration=CLIP_SECONDS,
    )
    if y.size < sr * 5:
        return {"ok": False, "error": "too short"}
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=1000)
    spec = librosa.power_to_db(mel, ref=1.0)
    pred = clf.predict(spec, source_id="benchmark")
    p = pred["confidence"] if pred["label"] == "ship" else 1.0 - pred["confidence"]
    return {
        "ok": True,
        "ship_prob": float(p),
        "raw_label": pred["label"],
        "uncertainty": float(pred["uncertainty"]),
    }


# ── threshold sweep ────────────────────────────────────────────────────


def _decide_with_threshold(
    ship_prob: float, uncertainty: float, conformal_threshold: float,
    unc_max: float, label_truth: str,
) -> dict[str, Any]:
    """Mirror _decide_tier in cnn_inference.py but parameterise unc_max."""
    if uncertainty > unc_max:
        tier = "UNCERTAIN"
    elif ship_prob < conformal_threshold:
        tier = "AMBIENT"
    elif ship_prob >= 0.85:
        tier = "DARK_VESSEL"
    else:
        tier = "ACOUSTIC_ONLY_LOW"

    truth_to_ship = {"ship": True, "not_ship": False}
    pred_to_ship = {
        "DARK_VESSEL": True, "ACOUSTIC_ONLY_LOW": True, "CONFIRMED_VESSEL": True,
        "AMBIENT": False, "UNCERTAIN": None,
    }
    pred = pred_to_ship.get(tier)
    truth = truth_to_ship.get(label_truth)
    return {
        "tier": tier,
        "is_confident": pred is not None,
        "is_correct_when_confident": pred == truth if pred is not None else None,
    }


def _binom_ci(success: int, total: int) -> tuple[float, float]:
    """Wilson 95% CI — better than normal approx at extremes (0/N or N/N)."""
    if total == 0:
        return 0.0, 0.0
    p = success / total
    z = 1.96
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ── runner ─────────────────────────────────────────────────────────────


def main() -> None:
    t0 = time.time()
    print("=" * 76)
    print("RIGOROUS v7.4 BENCHMARK ON UNSEEN DEEPSHIP + SYNTHETIC AMBIENT")
    print("=" * 76)

    vessel_set = _all_unseen_clips()
    print(f"\nUnseen vessel evaluations: {len(vessel_set)} "
          f"(every held-out clip × up to {len(OFFSETS)} offsets)")

    # Synthetic ambient floor
    amb_dir = Path("data/synthetic")
    amb_dir.mkdir(parents=True, exist_ok=True)
    amb_clips = []
    for i, seed in enumerate([101, 202, 303, 404, 505, 606]):
        path = amb_dir / f"benchmark_amb_{seed}.wav"
        _synth_ambient(path, seed=seed)
        amb_clips.append(("not_ship", path, 0.0))
    print(f"Synthetic ambient evaluations: {len(amb_clips)} "
          f"(deterministic seeds for reproducibility)")

    eval_set = vessel_set + amb_clips

    # Inference
    print("\n[1/3] running inference ...")
    clf = CNNV7Classifier(CKPT)
    clf.set_site_adapter(None)

    # Pick a sensible global conformal threshold (the same one the
    # production decision_engine uses, from the v7.4 calibration).
    conformal_path = Path("data/calibration/conformal_v7_4.json")
    conformal_threshold = float(json.loads(conformal_path.read_text())["threshold"])

    rows = []
    for i, (label, clip, off) in enumerate(eval_set):
        pred = _predict(clf, clip, offset=off)
        if not pred.get("ok"):
            continue
        rows.append({
            "label":      label,
            "clip":       str(clip),
            "offset_s":   off,
            "class":      clip.parent.name if "deepship" in str(clip) else "ambient",
            "ship_prob":  pred["ship_prob"],
            "uncertainty": pred["uncertainty"],
            "raw_label":  pred["raw_label"],
            "raw_correct": pred["raw_label"] == label,
        })
        if (i + 1) % 25 == 0:
            print(f"      {i + 1}/{len(eval_set)} ...")

    n = len(rows)
    n_vessel = sum(1 for r in rows if r["label"] == "ship")
    n_amb = sum(1 for r in rows if r["label"] == "not_ship")
    raw_correct = sum(r["raw_correct"] for r in rows)
    raw_lo, raw_hi = _binom_ci(raw_correct, n)
    print(f"\n      n total: {n}  (vessel {n_vessel} + ambient {n_amb})")
    print(f"      raw CNN accuracy: {raw_correct}/{n} = "
          f"{raw_correct / n * 100:.1f}%  (Wilson 95% CI "
          f"{raw_lo * 100:.1f}-{raw_hi * 100:.1f}%)")

    # ── threshold sweep ──────────────────────────────────────────────
    print(f"\n[2/3] UNCERTAIN_MAX threshold sweep ...")
    print(f"  conformal_threshold (global v7.4): {conformal_threshold:.3f}")
    print()
    print(f"  {'unc_max':>8} {'confident':>10} {'conf_rate':>10} "
          f"{'acc_when_confident':>20} {'overall':>10}")

    sweep_results = []
    for thr in THRESHOLD_CANDIDATES:
        n_confident = 0
        n_correct_confident = 0
        for r in rows:
            d = _decide_with_threshold(
                ship_prob=r["ship_prob"],
                uncertainty=r["uncertainty"],
                conformal_threshold=conformal_threshold,
                unc_max=thr,
                label_truth=r["label"],
            )
            if d["is_confident"]:
                n_confident += 1
                if d["is_correct_when_confident"]:
                    n_correct_confident += 1
        conf_rate = n_confident / n
        acc_conf = n_correct_confident / n_confident if n_confident else 0
        # "overall" = raw_correct using r["raw_correct"] but only for
        # confident decisions; uncertain are treated as not-counted.
        # We report acc_when_confident × conf_rate as a single combined metric.
        combined = acc_conf * conf_rate
        sweep_results.append({
            "unc_max":       thr,
            "n_confident":   n_confident,
            "confident_rate": round(conf_rate, 4),
            "acc_when_confident": round(acc_conf, 4),
            "combined":      round(combined, 4),
        })
        print(f"  {thr:>8.2f} {n_confident:>10} {conf_rate * 100:>9.1f}% "
              f"{acc_conf * 100:>19.1f}% {combined * 100:>9.1f}%")

    print(f"\n  combined = confident_rate × accuracy_when_confident")
    best = max(sweep_results, key=lambda r: r["combined"])
    print(f"  best by combined: unc_max={best['unc_max']}  "
          f"({best['confident_rate']*100:.0f}% confident · "
          f"{best['acc_when_confident']*100:.1f}% acc · "
          f"combined {best['combined']*100:.1f}%)")

    # ── per-class breakdown (vessel only) ─────────────────────────────
    print(f"\n[3/3] per-vessel-class accuracy (raw)")
    by_class = defaultdict(lambda: {"n": 0, "ok": 0})
    for r in rows:
        if r["label"] != "ship":
            continue
        c = r["class"]
        by_class[c]["n"] += 1
        by_class[c]["ok"] += int(r["raw_correct"])
    for c, m in sorted(by_class.items()):
        if m["n"] == 0:
            continue
        lo, hi = _binom_ci(m["ok"], m["n"])
        print(f"  {c:<18}  {m['ok']:>3}/{m['n']:<3} = "
              f"{m['ok'] / m['n'] * 100:.1f}%  "
              f"(95% CI {lo * 100:.1f}-{hi * 100:.1f}%)")

    # ── persist ───────────────────────────────────────────────────────
    out_full = Path("data/eval/benchmark_v7_4.json")
    out_full.parent.mkdir(parents=True, exist_ok=True)
    out_full.write_text(json.dumps({
        "_meta": {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "n_total":     n,
            "n_vessel":    n_vessel,
            "n_ambient":   n_amb,
            "conformal_threshold": conformal_threshold,
            "offsets_per_clip":    OFFSETS,
            "raw_accuracy":        raw_correct / n,
            "raw_accuracy_ci":     [round(raw_lo, 4), round(raw_hi, 4)],
        },
        "rows": rows,
    }, indent=2))
    out_sweep = Path("data/eval/threshold_sweep.json")
    out_sweep.write_text(json.dumps({"sweep": sweep_results, "best": best}, indent=2))
    print(f"\n  wrote {out_full}")
    print(f"  wrote {out_sweep}")

    print(f"\n{'═' * 76}")
    print(f"  HEADLINE")
    print(f"  raw CNN on n={n} ({n_vessel} unseen vessels + {n_amb} "
          f"synthetic ambient): {raw_correct}/{n} = {raw_correct / n * 100:.1f}%")
    print(f"  Wilson 95% CI: {raw_lo * 100:.1f}% - {raw_hi * 100:.1f}%")
    print(f"  recommended unc_max threshold: {best['unc_max']:.2f}")
    print(f"  → confident decisions: {best['confident_rate']*100:.0f}% (was 46% at unc_max=0.20)")
    print(f"{'═' * 76}")
    print(f"  wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
