"""Re-extract SanctSound chunks at 60s windows from already-downloaded FLACs.

Reads the FLACs we already have in data/sanctsound/{site}/, slices them into
60s windows (vs the existing 5s chunks), AIS-cross-validates each window,
saves new spectrograms to data/spectrograms_60s/, and writes a fresh
manifest at data/training/sanctsound_60s.jsonl.

This gives v8 longer-context training samples without re-downloading audio.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from math import cos, radians
from pathlib import Path

import httpx
import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, "src")
from ocean_sentinel.config import Settings


SITE_LOCATIONS: dict[str, tuple[float, float]] = {
    "oc01": (48.400, -124.700),
    "sb01": (42.460, -70.500),
    "sb02": (42.320, -70.600),
    "fk01": (24.650, -81.200),
    "hi01": (21.500, -157.800),
    "mb01": (36.800, -121.800),
    "gr01": (31.400, -80.850),
}

AUDIO_DIR = Path("data/sanctsound")
SPEC_DIR = Path("data/spectrograms_60s")
OUTPUT_JSONL = Path("data/training/sanctsound_60s.jsonl")
TARGET_SR = 16000
CHUNK_SECONDS = 60
N_MELS = 128
F_MAX = 1000.0
GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"
FILENAME_TS_RE = re.compile(r"_(\d{8}T\d{6})Z(?:_[A-Za-z\-]+)?\.flac")


def _bbox(lat: float, lon: float, radius_km: float) -> dict:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * cos(radians(lat)))
    return {"geojson": {"type": "Polygon", "coordinates": [[
        [lon - dlon, lat - dlat], [lon + dlon, lat - dlat],
        [lon + dlon, lat + dlat], [lon - dlon, lat + dlat],
        [lon - dlon, lat - dlat],
    ]]}}


async def fetch_hourly(client: httpx.AsyncClient, token: str, lat: float,
                       lon: float, day: datetime, radius_km: float,
                       max_retries: int = 6) -> dict[int, list[dict]]:
    d0 = day.strftime("%Y-%m-%d")
    d1 = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (f"{GFW_BASE}/4wings/report"
           f"?datasets[0]=public-global-presence:latest"
           f"&date-range={d0},{d1}&temporal-resolution=HOURLY"
           f"&spatial-resolution=HIGH&group-by=VESSEL_ID&format=JSON")
    delay = 5.0
    for _ in range(max_retries):
        resp = await client.post(
            url, json=_bbox(lat, lon, radius_km),
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code == 429:
            await asyncio.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        out: dict[int, list[dict]] = {h: [] for h in range(24)}
        for entry in resp.json().get("entries", []):
            if isinstance(entry, dict):
                for k, v in entry.items():
                    if k.startswith("public-") and isinstance(v, list):
                        for r in v:
                            ts = r.get("entryTimestamp") or r.get("date")
                            if not ts:
                                continue
                            try:
                                h = datetime.fromisoformat(
                                    ts.replace("Z", "+00:00")
                                ).hour
                            except Exception:
                                continue
                            out[h].append(r)
        return out
    raise RuntimeError("rate limit exhausted")


def make_mel(samples: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=samples, sr=sr, n_mels=N_MELS, fmax=F_MAX,
    )
    return librosa.power_to_db(mel, ref=1.0)


async def process_flac(flac_path: Path, site: str, lat: float, lon: float,
                       gfw_client: httpx.AsyncClient, gfw_token: str,
                       written_event_ids: set[str]) -> dict[str, int]:
    counters = {"ship": 0, "ship_distant": 0, "ambient": 0,
                "skipped_tail": 0, "skipped_no_ts": 0}
    m = FILENAME_TS_RE.search(flac_path.name)
    if not m:
        counters["skipped_no_ts"] += 1
        return counters
    capture_start = datetime.strptime(
        m.group(1), "%Y%m%dT%H%M%S"
    ).replace(tzinfo=timezone.utc)
    day = capture_start.replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"  capture_start={capture_start.isoformat()}")

    h10 = await fetch_hourly(gfw_client, gfw_token, lat, lon, day, 10.0)
    await asyncio.sleep(1.0)
    h50 = await fetch_hourly(gfw_client, gfw_token, lat, lon, day, 50.0)
    print(f"  vessels in 10km/50km: "
          f"{sum(1 for v in h10.values() if v)}/24, "
          f"{sum(1 for v in h50.values() if v)}/24")

    info = sf.info(str(flac_path))
    sr_native = info.samplerate
    duration_s = info.frames / sr_native
    chunk_native = int(CHUNK_SECONDS * sr_native)
    n_chunks = int(duration_s // CHUNK_SECONDS)
    print(f"  duration={duration_s/3600:.2f}h  → {n_chunks} chunks at 60s")

    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    for chunk_idx in range(n_chunks):
        chunk_ts = capture_start + timedelta(seconds=chunk_idx * CHUNK_SECONDS)
        h = chunk_ts.hour
        close = bool(h10.get(h))
        wide = bool(h50.get(h))
        prior_wide = bool(h50.get((h - 1) % 24))
        if close:
            label, distance = "ship", "close"
        elif wide:
            label, distance = "ship_distant", "medium"
        elif prior_wide:
            counters["skipped_tail"] += 1
            continue
        else:
            label, distance = "ambient", "none"

        start_frame = chunk_idx * chunk_native
        if start_frame + chunk_native > info.frames:
            break
        samples, _ = sf.read(
            str(flac_path), start=start_frame, frames=chunk_native, dtype="float32",
        )
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sr_native != TARGET_SR:
            samples = librosa.resample(samples, orig_sr=sr_native, target_sr=TARGET_SR)
        mel = make_mel(samples, TARGET_SR)

        event_id = f"sanctsound60s_{site}_{flac_path.stem}_{chunk_idx}"
        if event_id in written_event_ids:
            continue
        written_event_ids.add(event_id)
        spec_path = SPEC_DIR / f"{event_id}.npy"
        np.save(spec_path, mel.astype(np.float32))

        entry = {
            "event_id": event_id,
            "spectrogram_path": str(spec_path),
            "label": "ship" if label == "ship" else "not_ship",
            "sanctsound_ais_label": label,
            "distance_bucket": distance,
            "audio_capture_start": chunk_ts.isoformat(),
            "duration_s": float(CHUNK_SECONDS),
            "audio": {"duration_s": float(CHUNK_SECONDS), "sample_rate": TARGET_SR},
            "provenance": {
                "source_id": f"sanctsound60s-{site}",
                "source_file": f"{site}/{flac_path.name}",
                "is_60s_pull": True,
            },
        }
        OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_JSONL.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        counters[label] += 1

        if (chunk_idx + 1) % 50 == 0:
            print(f"    [{chunk_idx+1}/{n_chunks}] "
                  f"ship={counters['ship']} dist={counters['ship_distant']} "
                  f"amb={counters['ambient']}")
    return counters


async def main() -> None:
    settings = Settings()
    if not settings.gfw_api_token:
        sys.exit("OS_GFW_API_TOKEN missing")

    written: set[str] = set()
    if OUTPUT_JSONL.exists():
        with OUTPUT_JSONL.open() as f:
            for line in f:
                try:
                    written.add(json.loads(line)["event_id"])
                except Exception:
                    continue
        print(f"Resuming — {len(written)} chunks already written")

    aggregate: dict[str, int] = {}
    async with httpx.AsyncClient(timeout=60.0) as gfw_client:
        for site_dir in sorted(AUDIO_DIR.iterdir()):
            site = site_dir.name
            if site not in SITE_LOCATIONS:
                continue
            lat, lon = SITE_LOCATIONS[site]
            for flac in sorted(site_dir.glob("*.flac")):
                print(f"\n=== {site}: {flac.name} ===")
                try:
                    counters = await process_flac(
                        flac, site, lat, lon,
                        gfw_client, settings.gfw_api_token, written,
                    )
                except Exception as e:
                    print(f"  ! failed: {e}")
                    continue
                print(f"  -> {counters}")
                for k, v in counters.items():
                    aggregate[k] = aggregate.get(k, 0) + v

    print("\n=== Aggregate ===")
    for k, v in sorted(aggregate.items(), key=lambda x: -x[1]):
        print(f"  {k:<20s} {v:>6d}")
    total = sum(v for k, v in aggregate.items() if not k.startswith("skipped"))
    print(f"\nTotal emitted: {total}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    asyncio.run(main())
