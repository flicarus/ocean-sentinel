"""OC01 negative control — proves PSD pipeline doesn't generate phantom peaks.

The Mar-9 12:00 UTC suspect window shows a +13 dB excess at 28-37 Hz
versus the same-file control window. The reviewer might ask: "is your
analysis method producing artifacts? maybe Welch's delta always shows
peaks?". This script answers that.

Method:
  1. Query GFW (hourly) for vessel presence within 50 km of OC01 on
     Mar 8 and Mar 9 2019.
  2. Find an hour with ZERO broadcasting vessels — that's the negative
     control hour.
  3. From the FLAC covering that hour, extract a 5-min "suspect" window
     and a 5-min "control" window (different offsets in same file).
  4. Run the same Welch's PSD analysis as `scripts/oc01_psd.py`.
  5. Render a side-by-side comparison: disputed-vessel chunk on top,
     confirmed-empty chunk on bottom. If method is honest, the empty
     panel should show flat delta, no peak at 28-37 Hz.

Output:  data/diagnostic/sanctsound/audio/psd_negative_control.png
         data/diagnostic/sanctsound/audio/psd_side_by_side.png
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
RADIUS_KM = 50

# 4 FLAC files, each ~6h. UTC start times -> file path.
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

OUT_DIR = Path("data/diagnostic/sanctsound/audio")
OUT_NEG_PNG = OUT_DIR / "psd_negative_control.png"
OUT_SIDE_PNG = OUT_DIR / "psd_side_by_side.png"


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


async def query_hourly(client: httpx.AsyncClient, token: str, day: datetime) -> list[dict]:
    """Hourly vessel presence rows around OC01 for the given UTC day."""
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
        url, json=bbox(OC01_LAT, OC01_LON, RADIUS_KM),
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    rows: list[dict] = []
    for entry in resp.json().get("entries", []):
        if isinstance(entry, dict):
            for k, v in entry.items():
                if k.startswith("public-") and isinstance(v, list):
                    rows.extend(v)
    return rows


def hour_of(row: dict) -> int | None:
    ts = row.get("entryTimestamp") or row.get("date")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).hour
    except Exception:
        return None


def hours_with_vessels(rows: list[dict]) -> dict[int, set[str]]:
    """Map UTC hour -> set of distinct vessel IDs present that hour."""
    out: dict[int, set[str]] = {}
    for r in rows:
        h = hour_of(r)
        if h is None:
            continue
        vid = r.get("vesselId") or r.get("mmsi") or "unknown"
        out.setdefault(h, set()).add(vid)
    return out


def find_audio_offset(target: datetime) -> tuple[Path, float] | None:
    """Map a target UTC time to (flac_path, seconds_offset_into_file)."""
    candidates = []
    for start, path in FLAC_FILES.items():
        delta_s = (target - start).total_seconds()
        if delta_s < 0:
            continue
        # FLAC files are ~6h = 21600s. Suspect file (Mar9 12:00) ran for >5h.
        # Keep generous so we can pull windows up to 5 min in.
        if delta_s + 600 > 6 * 3600:
            continue
        candidates.append((path, delta_s))
    if not candidates:
        return None
    # Prefer the candidate whose start is closest to the target (smallest offset).
    candidates.sort(key=lambda x: x[1])
    return candidates[0]


def welch_psd(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    nperseg = 1 << 14
    f, p = welch(audio, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    return f, 10 * np.log10(p + 1e-20)


def render_psd(
    ax_top, ax_bot, suspect_db, control_db, freqs, title_top, title_bot,
):
    plot_mask = (freqs >= 1) & (freqs <= 1000)
    diff = suspect_db - control_db

    ax_top.semilogx(freqs[plot_mask], suspect_db[plot_mask],
                    color="crimson", linewidth=1.0,
                    label="Window A (suspect)")
    ax_top.semilogx(freqs[plot_mask], control_db[plot_mask],
                    color="steelblue", linewidth=1.0,
                    label="Window B (control)")
    ax_top.axvspan(5, 50, alpha=0.08, color="orange",
                   label="Cargo blade-rate band 5-50 Hz")
    ax_top.axvspan(50, 500, alpha=0.05, color="red",
                   label="Engine harmonic band 50-500 Hz")
    ax_top.set_title(title_top, fontsize=11)
    ax_top.set_xlabel("Frequency (Hz, log scale)")
    ax_top.set_ylabel("PSD (dB / Hz)")
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(which="both", alpha=0.3)

    ax_bot.semilogx(freqs[plot_mask], diff[plot_mask],
                    color="purple", linewidth=1.0)
    ax_bot.axhline(0, color="black", linewidth=0.5)
    ax_bot.axvspan(5, 50, alpha=0.08, color="orange")
    ax_bot.axvspan(50, 500, alpha=0.05, color="red")
    ax_bot.set_title(title_bot, fontsize=11)
    ax_bot.set_xlabel("Frequency (Hz, log scale)")
    ax_bot.set_ylabel("Δ PSD (dB)")
    ax_bot.set_ylim(-20, 20)
    ax_bot.grid(which="both", alpha=0.3)


async def main() -> None:
    settings = Settings()
    if not settings.gfw_api_token:
        sys.exit("OS_GFW_API_TOKEN missing")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("== Step 1: query GFW hourly for Mar 8 + Mar 9 2019 ==")
    print(f"OC01 hydrophone:  {OC01_LAT}°N, {OC01_LON}°W   (50 km radius)")

    async with httpx.AsyncClient(timeout=60.0) as client:
        all_hours: dict[datetime, set[str]] = {}
        for day in (datetime(2019, 3, 8, tzinfo=timezone.utc),
                    datetime(2019, 3, 9, tzinfo=timezone.utc)):
            rows = await query_hourly(client, settings.gfw_api_token, day)
            buckets = hours_with_vessels(rows)
            print(f"\n  {day.strftime('%Y-%m-%d')}: "
                  f"{len(rows)} hourly rows total")
            for h in sorted(buckets):
                vessels = buckets[h]
                print(f"    {h:02d}:00 UTC  -> {len(vessels):2d} distinct vessels")
                all_hours[day.replace(hour=h)] = vessels
            for h in range(24):
                k = day.replace(hour=h)
                if k not in all_hours:
                    all_hours[k] = set()  # explicit zero-vessel hour

    print("\n== Step 2: find candidate empty hour ==")
    # Restrict to hours where we have audio coverage.
    audio_covered = []
    for h, vessels in sorted(all_hours.items()):
        match = find_audio_offset(h)
        if match is None:
            continue
        audio_covered.append((h, vessels, match))

    print(f"  {len(audio_covered)} hours have audio coverage")
    print(f"  hours with ZERO vessels in 50km:")
    empty = [(h, m) for h, vs, m in audio_covered if len(vs) == 0]
    if not empty:
        print("    NONE — every hour has at least one broadcasting vessel")
        print("    Falling back to QUIETEST hour")
        # Pick min-vessel hour among covered.
        audio_covered.sort(key=lambda x: (len(x[1]), x[0]))
        chosen_h, chosen_vessels, chosen_match = audio_covered[0]
        print(f"    quietest covered hour: {chosen_h.isoformat()} "
              f"({len(chosen_vessels)} vessels)")
        chosen_status = f"quietest ({len(chosen_vessels)} vessels in 50km)"
    else:
        for h, m in empty[:5]:
            print(f"    {h.isoformat()}  (file={m[0].name}, offset={m[1]:.0f}s)")
        # Pick the hour furthest from the disputed window (Mar 9 12:00).
        disputed = datetime(2019, 3, 9, 12, tzinfo=timezone.utc)
        empty.sort(key=lambda x: -abs((x[0] - disputed).total_seconds()))
        chosen_h, chosen_match = empty[0]
        chosen_vessels = set()
        chosen_status = "ZERO vessels in 50km"

    chosen_path, chosen_offset = chosen_match
    print(f"\n  -> Chosen negative control: {chosen_h.isoformat()}")
    print(f"     status: {chosen_status}")
    print(f"     audio:  {chosen_path.name} @ {chosen_offset:.0f}s")

    print("\n== Step 3: extract audio + run Welch's PSD ==")
    info = sf.info(str(chosen_path))
    sr = info.samplerate
    duration = info.frames / sr
    print(f"  FLAC: sr={sr} Hz, duration={duration:.0f}s")

    # Two non-overlapping 5-min windows from the negative control hour.
    win_seconds = 5 * 60
    if chosen_offset + 2 * win_seconds > duration:
        # Not enough room; back off
        chosen_offset = max(0, duration - 2 * win_seconds - 60)
    win_a_start = int(chosen_offset * sr)
    win_b_start = int((chosen_offset + win_seconds) * sr)
    win_len = int(win_seconds * sr)

    audio_a, _ = sf.read(str(chosen_path), start=win_a_start,
                         frames=win_len, dtype="float32")
    audio_b, _ = sf.read(str(chosen_path), start=win_b_start,
                         frames=win_len, dtype="float32")
    if audio_a.ndim > 1:
        audio_a = audio_a.mean(axis=1)
    if audio_b.ndim > 1:
        audio_b = audio_b.mean(axis=1)

    f_a, p_a_db = welch_psd(audio_a, sr)
    f_b, p_b_db = welch_psd(audio_b, sr)
    p_b_interp = np.interp(f_a, f_b, p_b_db)
    diff = p_a_db - p_b_interp

    band_mask = (f_a >= 5) & (f_a <= 50)
    blade_mean = diff[band_mask].mean()
    band_mask = (f_a >= 50) & (f_a <= 500)
    engine_mean = diff[band_mask].mean()
    band_mask = (f_a >= 500) & (f_a <= 1000)
    cav_mean = diff[band_mask].mean()
    print(f"  Mean Δ in 5-50 Hz blade-rate band:  {blade_mean:+5.2f} dB")
    print(f"  Mean Δ in 50-500 Hz engine band:    {engine_mean:+5.2f} dB")
    print(f"  Mean Δ in 500-1000 Hz cavitation:   {cav_mean:+5.2f} dB")

    # Render single negative-control plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fmt_h = chosen_h.strftime("%Y-%m-%d %H:00 UTC")
    render_psd(
        axes[0], axes[1], p_a_db, p_b_interp, f_a,
        title_top=(
            f"OC01 NEGATIVE CONTROL — {fmt_h} ({chosen_status})\n"
            f"Window A (mins 0-5) vs Window B (mins 5-10), same file"
        ),
        title_bot=(
            f"Δ PSD (Window A − Window B):  blade-rate {blade_mean:+.2f} dB,  "
            f"engine {engine_mean:+.2f} dB,  cavitation {cav_mean:+.2f} dB"
        ),
    )
    fig.tight_layout()
    fig.savefig(OUT_NEG_PNG, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT_NEG_PNG}")

    print("\n== Step 4: side-by-side w/ disputed window ==")
    suspect_path = OUT_DIR / "suspect_window_12min.wav"
    control_path = OUT_DIR / "control_5min.wav"
    suspect_audio, sr_s = sf.read(str(suspect_path), dtype="float32")
    control_audio, sr_c = sf.read(str(control_path), dtype="float32")
    if suspect_audio.ndim > 1:
        suspect_audio = suspect_audio.mean(axis=1)
    if control_audio.ndim > 1:
        control_audio = control_audio.mean(axis=1)
    f_s, p_s_db = welch_psd(suspect_audio, sr_s)
    f_c, p_c_db = welch_psd(control_audio, sr_c)
    p_c_interp = np.interp(f_s, f_c, p_c_db)
    diff_disputed = p_s_db - p_c_interp

    bm = (f_s >= 5) & (f_s <= 50)
    blade_disp = diff_disputed[bm].mean()
    bm = (f_s >= 50) & (f_s <= 500)
    engine_disp = diff_disputed[bm].mean()
    bm = (f_s >= 500) & (f_s <= 1000)
    cav_disp = diff_disputed[bm].mean()

    fig, axes = plt.subplots(2, 2, figsize=(18, 9))
    # Left col: disputed
    render_psd(
        axes[0, 0], axes[1, 0], p_s_db, p_c_interp, f_s,
        title_top="DISPUTED — Mar 9 2019 12:00 UTC (CNN: ship 0.91, AIS: JOSCO HUIZHOU)",
        title_bot=(
            f"Δ:  blade-rate {blade_disp:+.2f} dB,  engine {engine_disp:+.2f} dB,  "
            f"cavitation {cav_disp:+.2f} dB"
        ),
    )
    # Right col: negative control
    render_psd(
        axes[0, 1], axes[1, 1], p_a_db, p_b_interp, f_a,
        title_top=(
            f"NEGATIVE CONTROL — {fmt_h} ({chosen_status})"
        ),
        title_bot=(
            f"Δ:  blade-rate {blade_mean:+.2f} dB,  engine {engine_mean:+.2f} dB,  "
            f"cavitation {cav_mean:+.2f} dB"
        ),
    )
    fig.suptitle(
        "OC01 PSD analysis — disputed (vessel) vs negative control (no vessel)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_SIDE_PNG, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT_SIDE_PNG}")

    print("\n== Summary ==")
    print(f"  Disputed Δ blade-rate:   {blade_disp:+.2f} dB  (vessel signature)")
    print(f"  Negative ctrl Δ blade-rate: {blade_mean:+.2f} dB  (should be ~0)")
    print(f"  Difference: {blade_disp - blade_mean:+.2f} dB")
    print()


if __name__ == "__main__":
    asyncio.run(main())
