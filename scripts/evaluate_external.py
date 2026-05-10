"""Run v7.4 on NOAA SanctSound audio from stations NOT in training.

Run *after* `scripts/download_external_eval.sh`. Walks every audio file
in data/external_eval/, samples 5 windows per file (spaced through the
recording), and reports per-window predictions + per-station summary.

What this answers
-----------------
"Does v7.4 generalise to hydrophones recorded by entirely different
campaigns (different sensor deployments, different ambient regimes,
different vessel traffic patterns) than anything in its training set?"

This is the missing piece from `scripts/benchmark_v7_4.py`, which only
tested unseen DeepShip clips — same dataset family as training. Here
the *station identity* is held out, not just the clip number.

Honest interpretation
---------------------
We don't have per-window ground truth — these are long ambient
recordings that may or may not contain transit vessels at any moment.
What we report:

  - Distribution of ship_prob across windows: median, p95
  - % of windows above the global conformal threshold (0.613) — these
    are positives the system would alert on
  - % of windows where the evidential head abstains (UNCERTAIN)

A *high* alert rate on these recordings doesn't mean the model is
broken — busy MPAs see real vessels, especially at peak hours. But a
median ship_prob > 0.7 on a station where no vessels were nearby would
be informative.

For verified ground truth we'd need NOAA's per-deployment AIS-correlated
detection products (out of scope for this hackathon).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier


CKPT = "data/models/cnn_v7_4.pt"
EXTERNAL_DIR = Path("data/external_eval")
SR = 16_000
WINDOW_S = 60
WINDOWS_PER_FILE = 5            # 5 evenly-spaced 60-second windows
GLOBAL_CONFORMAL = 0.613


def _station_from_filename(name: str) -> str:
    """data/external_eval/hi05_SanctSound_..._.flac → 'hi05'."""
    return name.split("_", 1)[0].lower()


def _list_external() -> list[Path]:
    if not EXTERNAL_DIR.exists():
        return []
    return sorted(EXTERNAL_DIR.glob("*.flac")) + sorted(EXTERNAL_DIR.glob("*.wav"))


def _windows_for_file(path: Path) -> list[float]:
    """Pick N evenly-spaced offsets (in seconds) inside `path`."""
    try:
        duration = librosa.get_duration(path=str(path))
    except Exception:
        return []
    usable = duration - WINDOW_S
    if usable <= 0:
        return []
    if WINDOWS_PER_FILE == 1:
        return [usable / 2.0]
    return [
        round((i + 1) * usable / (WINDOWS_PER_FILE + 1), 1)
        for i in range(WINDOWS_PER_FILE)
    ]


def _predict_window(clf: CNNV7Classifier, path: Path, offset: float) -> dict | None:
    try:
        y, sr = librosa.load(
            str(path), sr=SR, mono=True,
            offset=offset, duration=WINDOW_S,
        )
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if y.size < sr * 5:
        return None
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=1000)
    spec = librosa.power_to_db(mel, ref=1.0)
    pred = clf.predict(spec, source_id=_station_from_filename(path.name))
    p = pred["confidence"] if pred["label"] == "ship" else 1.0 - pred["confidence"]
    return {
        "ok":          True,
        "ship_prob":   float(p),
        "uncertainty": float(pred["uncertainty"]),
        "raw_label":   pred["label"],
    }


def main() -> None:
    files = _list_external()
    if not files:
        print(f"No audio in {EXTERNAL_DIR}/. Run scripts/download_external_eval.sh first.")
        return

    print("=" * 76)
    print("v7.4 ON NOAA SANCTSOUND — STATIONS NOT IN TRAINING")
    print("=" * 76)
    print(f"  files:     {len(files)} in {EXTERNAL_DIR}/")
    print(f"  windows:   {WINDOWS_PER_FILE} × {WINDOW_S}s evenly spaced per file")
    print(f"  threshold: global conformal = {GLOBAL_CONFORMAL:.3f}, "
          f"UNCERTAIN_MAX = 0.25")

    t0 = time.time()
    clf = CNNV7Classifier(CKPT)
    clf.set_site_adapter(None)

    rows: list[dict[str, Any]] = []
    by_station: dict[str, list[dict]] = {}

    for f in files:
        station = _station_from_filename(f.name)
        offsets = _windows_for_file(f)
        if not offsets:
            print(f"  [skip] {f.name} — too short")
            continue
        print(f"\n  [run]  {station:<6}  {f.name}  ({len(offsets)} windows)")
        for off in offsets:
            pred = _predict_window(clf, f, off)
            if pred is None or not pred.get("ok"):
                print(f"      offset {off:>5.0f}s  ERROR: {pred.get('error') if pred else 'too short'}")
                continue
            row = {
                "station":    station,
                "file":       f.name,
                "offset_s":   off,
                "ship_prob":  pred["ship_prob"],
                "uncertainty": pred["uncertainty"],
                "raw_label":  pred["raw_label"],
                "fires_global_threshold":
                    pred["ship_prob"] >= GLOBAL_CONFORMAL,
                "would_be_uncertain": pred["uncertainty"] > 0.25,
            }
            rows.append(row)
            by_station.setdefault(station, []).append(row)
            tag = "FIRE" if row["fires_global_threshold"] else "----"
            unc = " UNCERTAIN" if row["would_be_uncertain"] else ""
            print(f"      offset {off:>5.0f}s  ship_prob={pred['ship_prob']:.3f} "
                  f"unc={pred['uncertainty']:.2f}  {tag}{unc}")

    if not rows:
        print("\n  No usable windows.")
        return

    # ── per-station summary ────────────────────────────────────────────
    print(f"\n{'─' * 76}")
    print(f"Per-station summary")
    print(f"{'─' * 76}")
    print(f"  {'station':<8} {'n':>3} {'med_p':>7} {'p95_p':>7} "
          f"{'fire%':>6} {'unc%':>6}")
    for station, station_rows in sorted(by_station.items()):
        probs = [r["ship_prob"] for r in station_rows]
        med = float(np.median(probs))
        p95 = float(np.percentile(probs, 95)) if len(probs) > 1 else med
        fire_pct = 100.0 * sum(r["fires_global_threshold"] for r in station_rows) / len(station_rows)
        unc_pct = 100.0 * sum(r["would_be_uncertain"] for r in station_rows) / len(station_rows)
        print(f"  {station:<8} {len(station_rows):>3} {med:>7.3f} {p95:>7.3f} "
              f"{fire_pct:>5.0f}% {unc_pct:>5.0f}%")

    # ── overall ────────────────────────────────────────────────────────
    print(f"\n{'═' * 76}")
    n = len(rows)
    n_fire = sum(r["fires_global_threshold"] for r in rows)
    n_unc = sum(r["would_be_uncertain"] for r in rows)
    median_overall = float(np.median([r["ship_prob"] for r in rows]))
    print(f"  TOTAL  n={n} windows across {len(by_station)} untrained stations")
    print(f"  median ship_prob:               {median_overall:.3f}")
    print(f"  windows ≥ global threshold:     {n_fire} ({100*n_fire/n:.0f}%)")
    print(f"  windows abstained (UNCERTAIN):  {n_unc} ({100*n_unc/n:.0f}%)")
    print(f"  wall time:                      {time.time() - t0:.0f}s")
    print(f"{'═' * 76}")

    # ── persist ────────────────────────────────────────────────────────
    out = Path("data/eval/external_sanctsound.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "_meta": {
            "computed_at":         datetime.now(timezone.utc).isoformat(),
            "n_windows":           n,
            "n_stations":          len(by_station),
            "windows_per_file":    WINDOWS_PER_FILE,
            "window_seconds":      WINDOW_S,
            "global_threshold":    GLOBAL_CONFORMAL,
        },
        "rows": rows,
    }, indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
