"""OC01 systematic pattern — PSD across multiple AIS-positive vs AIS-negative hours.

Stronger argument than a single negative control: take EVERY hour where
audio + GFW data overlap, classify by closest-vessel distance, extract
5-min PSDs, look for the 28-37 Hz blade-rate signature.

If AIS-positive (vessel <=10 km) chunks consistently show a peak at
~30 Hz and AIS-negative chunks don't, the pattern is systematic — not
a one-off coincidence we got lucky on.

Method:
  1. Per-hour GFW query at 10 km radius for 2019-03-08 + 2019-03-09.
  2. Cross-reference with our 23h audio coverage from 4 FLAC files.
  3. Group hours into:
       - close (>=1 vessel within 10 km)
       - far  (vessels only at 10-50 km)
       - empty (zero vessels in 50 km)
  4. For each group, pick representative hours, extract 5-min audio,
     compute Welch's PSD on each.
  5. Plot all PSDs together with the 28-37 Hz blade-rate band shaded.
  6. Numerically: peak-detect in 5-50 Hz for each chunk, report the
     dominant frequency and its dB excess.

Output: data/diagnostic/sanctsound/audio/psd_pattern_multi.png
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from math import cos, radians
from pathlib import Path

import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import welch

sys.path.insert(0, "src")
from ocean_sentinel.config import Settings


OC01_LAT, OC01_LON = 48.400, -124.700
GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"

FLAC_DIR = Path("data/sanctsound/oc01")
FLAC_FILES = {
    datetime(2019, 3, 8, 19, 0, 0, tzinfo=timezone.utc):
        FLAC_DIR / "SanctSound_OC01_01_671399974_20190308T190000Z.flac",
    datetime(2019, 3, 8, 23, 59, 55, tzinfo=timezone.utc):
        FLAC_DIR / "SanctSound_OC01_01_671399974_20190308T235955Z.flac",
    datetime(2019, 3, 9, 5, 59, 52, tzinfo=timezone.utc):
        FLAC_DIR / "SanctSound_OC01_01_671399974_20190309T055952Z.flac",
    datetime(2019, 3, 9, 11, 59, 49, tzinfo=timezone.utc):
        FLAC_DIR / "SanctSound_OC01_01_671399974_20190309T115949Z.flac",
}

OUT_PNG = Path("data/diagnostic/sanctsound/audio/psd_pattern_multi.png")


def bbox(lat: float, lon: float, radius_km: float) -> dict:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * cos(radians(lat)))
    return {
        "geojson": {
            "type": "Polygon",
            "coordinates": [[
                [lon - dlon, lat - dlat],
                [lon + dlon, lat - dlat],
                [lon + dlon, lat + dlat],
                [lon - dlon, lat + dlat],
                [lon - dlon, lat - dlat],
            ]],
        }
    }


async def query_hourly(client, token, day, radius_km):
    d0 = day.strftime("%Y-%m-%d")
    d1 = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        f"{GFW_BASE}/4wings/report"
        f"?datasets[0]=public-global-presence:latest"
        f"&date-range={d0},{d1}"
        f"&temporal-resolution=HOURLY"
        f"&spatial-resolution=HIGH"
        f"&group-by=VESSEL_ID"
        f"&format=JSON"
    )
    resp = await client.post(
        url, json=bbox(OC01_LAT, OC01_LON, radius_km),
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    rows = []
    for entry in resp.json().get("entries", []):
        if isinstance(entry, dict):
            for k, v in entry.items():
                if k.startswith("public-") and isinstance(v, list):
                    rows.extend(v)
    return rows


def hour_of(row):
    ts = row.get("entryTimestamp") or row.get("date")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).hour
    except Exception:
        return None


def find_audio_offset(target):
    candidates = []
    for start, path in FLAC_FILES.items():
        delta_s = (target - start).total_seconds()
        if delta_s < 0:
            continue
        if delta_s + 600 > 6 * 3600:
            continue
        candidates.append((path, delta_s))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1])
    return candidates[0]


def welch_psd(audio, sr):
    nperseg = 1 << 14
    f, p = welch(audio, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    return f, 10 * np.log10(p + 1e-20)


def find_peak(freqs, psd_db, lo=5, hi=50):
    mask = (freqs >= lo) & (freqs <= hi)
    f_band = freqs[mask]
    p_band = psd_db[mask]
    # Robust baseline: median across the broad band
    baseline = np.median(p_band)
    excess = p_band - baseline
    idx = int(np.argmax(excess))
    return f_band[idx], excess[idx], p_band[idx], baseline


async def main():
    settings = Settings()
    if not settings.gfw_api_token:
        sys.exit("OS_GFW_API_TOKEN missing")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    print("== Step 1: GFW hourly queries at 10 km AND 50 km ==")
    days = (datetime(2019, 3, 8, tzinfo=timezone.utc),
            datetime(2019, 3, 9, tzinfo=timezone.utc))
    hours_10: dict[datetime, set[str]] = {}
    hours_50: dict[datetime, set[str]] = {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        for day in days:
            r10 = await query_hourly(client, settings.gfw_api_token, day, 10)
            r50 = await query_hourly(client, settings.gfw_api_token, day, 50)
            for r in r10:
                h = hour_of(r)
                if h is None:
                    continue
                vid = r.get("vesselId") or r.get("mmsi") or "?"
                hours_10.setdefault(day.replace(hour=h), set()).add(vid)
            for r in r50:
                h = hour_of(r)
                if h is None:
                    continue
                vid = r.get("vesselId") or r.get("mmsi") or "?"
                hours_50.setdefault(day.replace(hour=h), set()).add(vid)

            # explicit zeros for absent hours
            for h in range(24):
                k = day.replace(hour=h)
                hours_10.setdefault(k, set())
                hours_50.setdefault(k, set())

    print("\n== Step 2: classify each covered hour ==")
    print(f"  {'hour UTC':<22s}  {'≤10km':>5s}  {'≤50km':>5s}  group")
    groups = {"close": [], "far": [], "empty": []}
    for h in sorted(hours_50):
        if find_audio_offset(h) is None:
            continue
        c10 = len(hours_10[h])
        c50 = len(hours_50[h])
        if c10 > 0:
            grp = "close"
        elif c50 == 0:
            grp = "empty"
        else:
            grp = "far"
        groups[grp].append(h)
        marker = "*" if grp != "far" else " "
        print(f"  {h.isoformat():<22s}  {c10:>5d}  {c50:>5d}  {grp}{marker}")

    print(f"\n  close (≤10km vessel):  {len(groups['close'])} hours")
    print(f"  far   (only 10-50km):  {len(groups['far'])} hours")
    print(f"  empty (no vessels):    {len(groups['empty'])} hours")

    # Pick representatives — up to 4 close, up to 2 empty.
    # The disputed event is Mar 9 12:00 — keep that one if available.
    disputed = datetime(2019, 3, 9, 12, tzinfo=timezone.utc)
    close_hours = groups["close"][:]
    if disputed in close_hours:
        close_hours.remove(disputed)
        chosen_close = [disputed] + close_hours[:3]
    else:
        chosen_close = close_hours[:4]

    chosen_empty = groups["empty"][:2]
    if not chosen_empty:
        # fallback: pick "far" hours with smallest 50km count
        chosen_empty = sorted(groups["far"], key=lambda h: len(hours_50[h]))[:2]

    print(f"\n  selected close hours: {[h.strftime('%m-%d %H:00') for h in chosen_close]}")
    print(f"  selected empty hours: {[h.strftime('%m-%d %H:00') for h in chosen_empty]}")

    print("\n== Step 3: extract 5-min audio + Welch's PSD per hour ==")
    win_seconds = 5 * 60

    def extract_psd(hour, label):
        match = find_audio_offset(hour)
        path, offset = match
        info = sf.info(str(path))
        sr = info.samplerate
        # use middle of hour for stability
        offset = min(offset + 60, info.frames / sr - win_seconds - 1)
        audio, _ = sf.read(
            str(path), start=int(offset * sr),
            frames=int(win_seconds * sr), dtype="float32",
        )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        f, p_db = welch_psd(audio, sr)
        peak_f, peak_excess, peak_db, baseline = find_peak(f, p_db, 5, 50)
        print(
            f"  {label:<28s}  peak at {peak_f:5.1f} Hz  "
            f"excess {peak_excess:+5.2f} dB  (baseline {baseline:.1f} dB)"
        )
        return f, p_db, peak_f, peak_excess

    print("\n  CLOSE-RANGE (≤10km vessel, AIS-positive):")
    close_data = []
    for h in chosen_close:
        v_count = len(hours_10[h])
        label = f"close {h.strftime('%m-%d %H:00')} (n={v_count})"
        close_data.append((h, label, *extract_psd(h, label)))

    print("\n  EMPTY (no AIS in 50km):")
    empty_data = []
    for h in chosen_empty:
        label = f"empty {h.strftime('%m-%d %H:00')}"
        empty_data.append((h, label, *extract_psd(h, label)))

    print("\n== Step 4: render multi-panel comparison ==")
    n_close = len(close_data)
    n_empty = len(empty_data)
    n_rows = max(n_close, n_empty)

    fig, axes = plt.subplots(
        n_rows + 1, 2, figsize=(16, 3.0 * (n_rows + 1)),
        sharex=True, sharey=False,
    )

    # Row 0: overlay all
    ax_l, ax_r = axes[0]
    plot_mask = None  # set below
    ax_l.set_title("CLOSE-RANGE (AIS-positive ≤10 km) — PSD overlay",
                   fontsize=11, fontweight="bold")
    ax_r.set_title("EMPTY (no AIS in 50 km) — PSD overlay",
                   fontsize=11, fontweight="bold")
    for h, label, f, p_db, peak_f, _ in close_data:
        m = (f >= 1) & (f <= 1000)
        plot_mask = m
        ax_l.semilogx(f[m], p_db[m], linewidth=1.0, alpha=0.85, label=label)
    for h, label, f, p_db, peak_f, _ in empty_data:
        m = (f >= 1) & (f <= 1000)
        ax_r.semilogx(f[m], p_db[m], linewidth=1.0, alpha=0.85, label=label)
    for ax in (ax_l, ax_r):
        ax.axvspan(28, 37, alpha=0.15, color="orange",
                   label="cargo blade-rate 28-37 Hz")
        ax.axvspan(5, 50, alpha=0.05, color="orange")
        ax.set_ylabel("PSD (dB / Hz)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(which="both", alpha=0.3)

    # Below: per-hour plots, one chunk per row, peak annotated
    # left col: close, right col: empty
    for i in range(n_rows):
        ax_l = axes[i + 1, 0]
        ax_r = axes[i + 1, 1]
        if i < len(close_data):
            h, label, f, p_db, peak_f, peak_excess = close_data[i]
            m = (f >= 1) & (f <= 1000)
            ax_l.semilogx(f[m], p_db[m], color="crimson", linewidth=1.0)
            ax_l.axvspan(28, 37, alpha=0.15, color="orange")
            ax_l.axvline(peak_f, color="orange", linestyle="--", linewidth=1.2,
                         label=f"peak @ {peak_f:.1f} Hz (+{peak_excess:.1f} dB)")
            ax_l.set_title(label, fontsize=9)
            ax_l.legend(loc="upper right", fontsize=7)
            ax_l.set_ylabel("PSD (dB/Hz)")
            ax_l.grid(which="both", alpha=0.3)
        else:
            ax_l.set_visible(False)

        if i < len(empty_data):
            h, label, f, p_db, peak_f, peak_excess = empty_data[i]
            m = (f >= 1) & (f <= 1000)
            ax_r.semilogx(f[m], p_db[m], color="steelblue", linewidth=1.0)
            ax_r.axvspan(28, 37, alpha=0.15, color="orange")
            ax_r.axvline(peak_f, color="steelblue", linestyle="--", linewidth=1.2,
                         label=f"peak @ {peak_f:.1f} Hz (+{peak_excess:.1f} dB)")
            ax_r.set_title(label, fontsize=9)
            ax_r.legend(loc="upper right", fontsize=7)
            ax_r.set_ylabel("PSD (dB/Hz)")
            ax_r.grid(which="both", alpha=0.3)
        else:
            ax_r.set_visible(False)

    axes[-1, 0].set_xlabel("Frequency (Hz, log scale)")
    axes[-1, 1].set_xlabel("Frequency (Hz, log scale)")

    fig.suptitle(
        "OC01 — does the cargo blade-rate signature appear systematically?\n"
        "Orange band = expected cargo blade-rate (28-37 Hz). "
        "Dashed vertical = where peak landed in 5-50 Hz.",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT_PNG}")

    print("\n== Summary table ==")
    print(f"{'group':<8s}  {'hour':<14s}  {'peak Hz':>8s}  {'excess dB':>10s}  in band?")
    for h, label, f, p_db, peak_f, peak_excess in close_data:
        in_band = "YES" if 28 <= peak_f <= 37 else "no"
        print(f"close     {h.strftime('%m-%d %H:00 UTC'):<14s}  "
              f"{peak_f:>8.1f}  {peak_excess:>+10.2f}   {in_band}")
    for h, label, f, p_db, peak_f, peak_excess in empty_data:
        in_band = "YES" if 28 <= peak_f <= 37 else "no"
        print(f"empty     {h.strftime('%m-%d %H:00 UTC'):<14s}  "
              f"{peak_f:>8.1f}  {peak_excess:>+10.2f}   {in_band}")


if __name__ == "__main__":
    asyncio.run(main())
