"""Render two oc01 mel spectrograms for the case-study page:
one confidently-flagged correction (vessel) + one confident ambient.

Saves PNG to oceansentinelfrontend/public/case-study/.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FRONT = Path("/Users/jakub/oceansentinelfrontend/public/case-study")
FRONT.mkdir(parents=True, exist_ok=True)

from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier

clf = CNNV7Classifier(str(ROOT / "data/models/cnn_v7_6.pt"))
clf.set_site_thresholds(str(ROOT / "data/calibration/per_site_thresholds_v7_6.json"))

# Collect oc01 samples
samples = []
with open(ROOT / "data/training/sanctsound_corrected.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        prov = d.get("provenance", {}).get("source_id", "")
        if "oc01" not in prov.lower():
            continue
        samples.append(d)

# Score all
scored = []
for s in samples:
    p = ROOT / s["spectrogram_path"]
    try:
        spec = np.load(p)
    except Exception:
        continue
    out = clf.predict(spec, source_id="oc01")
    scored.append((float(out["probabilities"]["ship"]), s, spec))

ship_samples = sorted(
    [x for x in scored if x[1]["label"] == "ship"],
    key=lambda x: -x[0],
)
amb_samples = sorted(
    [x for x in scored if x[1]["label"] == "not_ship"],
    key=lambda x: x[0],
)

picks = [
    ("vessel.png", "Mislabeled by NOAA, flagged by Ocean Sentinel", ship_samples[5][2], ship_samples[5][0], "ship"),
    ("ambient.png", "Correctly labeled ambient", amb_samples[5][2], amb_samples[5][0], "ambient"),
]

for filename, title, spec, prob, kind in picks:
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=160)
    spec_db = spec.copy()
    if spec_db.shape[1] > 200:
        spec_db = spec_db[:, :200]
    im = ax.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=np.percentile(spec_db, 5),
        vmax=np.percentile(spec_db, 99),
    )
    ax.set_xlabel("time (frames)", fontsize=9, color="#9CA3AF")
    ax.set_ylabel("mel bin (low → high freq)", fontsize=9, color="#9CA3AF")
    ax.set_title(f"{title}  ·  v7.6 ship_prob = {prob:.3f}", fontsize=10, color="#111")
    ax.tick_params(colors="#9CA3AF", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#E5E7EB")
    fig.tight_layout()
    out_path = FRONT / filename
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}  (ship_prob={prob:.3f})")
