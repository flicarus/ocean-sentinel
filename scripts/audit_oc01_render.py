"""Render the case-study Part II visuals from the MarineCadastre data.

Visuals:
  1. JOSCO HUIZHOU track on March 9 (distance over time + 12:39 CPA point)
  2. Per-recording 24h vessel-presence stacked bars by CPA band
  3. CPA distribution histogram across all 800 chunks
  4. The "no chunk is ambient" plot — chunk timestamp vs nearest-vessel-distance

Output: oceansentinelfrontend/public/case-study/*.png
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MC = ROOT / "data/audit/oc01_mc_track.json"
FRONT = Path("/Users/jakub/oceansentinelfrontend/public/case-study")
FRONT.mkdir(parents=True, exist_ok=True)
OUT_DATA = ROOT / "data/audit"

TEAL = "#00D4C8"
RED = "#EF4444"
GRAY = "#9CA3AF"


def main():
    mc = json.loads(MC.read_text())
    chunks = mc["per_chunk"]
    cpa_per = mc["vessel_cpa"]

    # 1. JOSCO HUIZHOU track — need raw points. Re-parse the CSV briefly.
    import csv
    josco_pts = []
    for csv_path in [
        ROOT / "data/audit/marine_cadastre/AIS_2019_03_09.csv",
    ]:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("MMSI") == "477133400":
                    try:
                        ts = datetime.strptime(row["BaseDateTime"], "%Y-%m-%dT%H:%M:%S")
                        lat = float(row["LAT"]); lon = float(row["LON"])
                    except Exception:
                        continue
                    R = 6371.0
                    a, b = radians(48.4), radians(-124.7)
                    c, d = radians(lat), radians(lon)
                    h = sin((c-a)/2)**2 + cos(a)*cos(c)*sin((d-b)/2)**2
                    dist = 2*R*asin(sqrt(h))
                    josco_pts.append((ts, dist, lat, lon))
    josco_pts.sort()
    print(f"JOSCO HUIZHOU points: {len(josco_pts)}")
    if josco_pts:
        cpa_pt = min(josco_pts, key=lambda x: x[1])
        print(f"  CPA: {cpa_pt[1]:.2f}km at {cpa_pt[0]}")

    # ---- Figure 1: JOSCO track on 03-09 ----
    if josco_pts:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
        times = [p[0] for p in josco_pts]
        dists = [p[1] for p in josco_pts]
        ax.plot(times, dists, color=RED, lw=1.0)
        ax.fill_between(times, dists, max(dists), color=RED, alpha=0.08)
        # Mark CPA
        ax.scatter([cpa_pt[0]], [cpa_pt[1]], color=RED, s=80, zorder=5,
                   edgecolors="white", linewidths=2)
        ax.annotate(
            f"CPA {cpa_pt[1]:.2f} km\n{cpa_pt[0].strftime('%H:%M:%S')} UTC",
            xy=(cpa_pt[0], cpa_pt[1]),
            xytext=(20, 35), textcoords="offset points",
            fontsize=10, fontweight=600, color="#111",
            arrowprops=dict(arrowstyle="->", color="#111", lw=1),
        )
        # Mark Part I window
        ax.axvspan(
            datetime(2019, 3, 9, 12, 14),
            datetime(2019, 3, 9, 12, 44),
            color=TEAL, alpha=0.15,
            label="Part I CNN flag window (12:14–12:44 UTC)",
        )
        ax.axhline(10, ls="--", color=GRAY, lw=0.8, alpha=0.7)
        ax.text(times[0], 10.5, "10 km audibility band", fontsize=9, color=GRAY)
        ax.set_xlabel("UTC")
        ax.set_ylabel("distance to OC01 hydrophone (km)")
        ax.set_title("JOSCO HUIZHOU · 2019-03-09 · MarineCadastre AIS")
        ax.set_ylim(0, max(50, max(dists) + 5))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.legend(loc="upper right", fontsize=9)
        for s in ax.spines.values():
            s.set_color("#E5E7EB")
        ax.grid(alpha=0.15)
        out = FRONT / "josco_track.png"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out}")

    # ---- Figure 2: per-chunk nearest-vessel distance over time ----
    # Each chunk has chunk_start_utc + nearest_in_window.distance_km + current_label
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=160)
    for lbl, color, marker in [("ship", RED, "."), ("not_ship", TEAL, ".")]:
        xs, ys = [], []
        for c in chunks:
            if c["current_label"] != lbl:
                continue
            nv = c.get("nearest_in_window")
            if not nv:
                continue
            ts = datetime.fromisoformat(c["chunk_start_utc"])
            xs.append(ts)
            ys.append(nv["distance_km"])
        ax.scatter(xs, ys, color=color, s=8, alpha=0.6, label=f"label={lbl}")
    ax.axhline(7,  ls="--", color="#111",  lw=0.8, alpha=0.6)
    ax.axhline(10, ls=":",  color=GRAY,    lw=0.8, alpha=0.5)
    ax.text(datetime(2019, 3, 8, 19, 30), 7.2, "7 km", fontsize=9, color="#111")
    ax.text(datetime(2019, 3, 8, 19, 30), 10.2, "10 km", fontsize=9, color=GRAY)
    ax.set_xlabel("UTC (chunk timestamp)")
    ax.set_ylabel("distance to nearest AIS vessel (km)")
    ax.set_title("OC01 — every chunk has a vessel within 10 km · MarineCadastre AIS ±2 min")
    ax.set_ylim(0, 20)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="upper right", fontsize=9)
    for s in ax.spines.values():
        s.set_color("#E5E7EB")
    ax.grid(alpha=0.15)
    out = FRONT / "chunk_distances.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")

    # ---- Figure 3: CPA distance histogram ----
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    ship_dists = [c["nearest_in_window"]["distance_km"]
                  for c in chunks if c["current_label"] == "ship" and c.get("nearest_in_window")]
    amb_dists = [c["nearest_in_window"]["distance_km"]
                 for c in chunks if c["current_label"] == "not_ship" and c.get("nearest_in_window")]
    bins = np.linspace(0, 20, 41)
    ax.hist(amb_dists, bins=bins, alpha=0.55, color=TEAL, label=f"label=ambient (n={len(amb_dists)})")
    ax.hist(ship_dists, bins=bins, alpha=0.55, color=RED, label=f"label=ship (n={len(ship_dists)})")
    ax.axvline(7, ls="--", color="#111", lw=0.8, alpha=0.6)
    ax.axvline(10, ls=":", color=GRAY, lw=0.8, alpha=0.5)
    ax.set_xlabel("distance to nearest AIS vessel (km)")
    ax.set_ylabel("chunks")
    ax.set_title("OC01 — per-chunk nearest-vessel CPA · MarineCadastre AIS")
    ax.legend()
    for s in ax.spines.values():
        s.set_color("#E5E7EB")
    ax.grid(alpha=0.15)
    out = FRONT / "cpa_histogram.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")

    # ---- Save a small summary JSON the frontend can statically import ----
    summary = {
        "data_source": "MarineCadastre.gov AIS archive (NOAA Office for Coastal Management)",
        "n_chunks_total": len(chunks),
        "n_chunks_label_ship":     sum(1 for c in chunks if c["current_label"] == "ship"),
        "n_chunks_label_ambient":  sum(1 for c in chunks if c["current_label"] == "not_ship"),
        "ais_pings_in_bbox_2days": 464413,
        "unique_mmsi_in_bbox_2days": 809,
        "josco_huizhou_cpa_km": round(cpa_pt[1], 2) if josco_pts else None,
        "josco_huizhou_cpa_ts_utc": cpa_pt[0].isoformat() if josco_pts else None,
        "buckets_label_x_cpa": mc.get("by_label_band", {}),
        "top_named_vessels_in_bbox": mc.get("vessel_cpa", [])[:20],
    }
    (OUT_DATA / "oc01_case_study_summary.json").write_text(json.dumps(summary, indent=2))
    (FRONT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote summary.json")


if __name__ == "__main__":
    main()
