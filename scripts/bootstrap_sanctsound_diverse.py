"""Download + AIS-label SanctSound data from new sites for v7 retraining.

Pulls one FLAC file per target deployment, chunks it into 5s windows,
computes mel spectrograms, and AIS-cross-validates each window's hour to
attach a binary ship/ambient label.

Sites picked for geographic diversity:
  - sb02 (Stellwagen Bank, MA)
  - fk01 (Florida Keys, FL — warm Atlantic, very different acoustics)
  - hi01 (Hawaiian Islands NMS — central Pacific, deep water)
  - mb01 (Monterey Bay, CA)
  - gr01 (Gray's Reef, GA — Atlantic continental shelf)

Coordinates are sanctuary-center approximations; deployments within a
sanctuary cluster within ~10-50 km, well inside our AIS bbox queries.

Output:
  - audio FLACs in data/sanctsound/<site>/
  - spectrograms in data/spectrograms/sanctsound_<site>_*.npy
  - labels in data/training/sanctsound_diverse.jsonl
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
import structlog

sys.path.insert(0, "src")
from ocean_sentinel.config import Settings

log = structlog.get_logger()


# (site, deployment_num) — deployment_num is 2-digit (01/02/03/04). Each
# site has multiple deployments; we take the first FLAC from each chosen one.
SITES_TO_PULL: list[tuple[str, str]] = [
    ("sb02", "01"),
    ("fk01", "01"),
    ("hi01", "01"),
    ("mb01", "01"),
    ("gr01", "01"),
]

# Approximate sanctuary-center coordinates. For AIS queries with 50 km bbox
# this is fine — deployments cluster within the sanctuary and are well inside
# our bbox.
SITE_LOCATIONS: dict[str, tuple[float, float]] = {
    "sb02": (42.32, -70.60),     # Stellwagen Bank, MA
    "fk01": (24.65, -81.20),     # Florida Keys, FL
    "hi01": (21.50, -157.80),    # Hawaiian Islands HW NMS
    "mb01": (36.80, -121.80),    # Monterey Bay, CA
    "gr01": (31.40, -80.85),     # Gray's Reef, GA
}

GCS_BASE = "https://storage.googleapis.com/noaa-passive-bioacoustic"
GCS_API = "https://storage.googleapis.com/storage/v1/b/noaa-passive-bioacoustic/o"
AUDIO_DIR = Path("data/sanctsound")
SPEC_DIR = Path("data/spectrograms")
OUTPUT_JSONL = Path("data/training/sanctsound_diverse.jsonl")
TARGET_SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
N_MELS = 128
F_MAX = 1000.0

GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"

# SanctSound filenames vary: some are "..._<ts>Z.flac", others append a
# deployment-status suffix like "..._<ts>Z_Post-Deployment.flac".
FILENAME_TS_RE = re.compile(r"_(\d{8}T\d{6})Z(?:_[A-Za-z\-]+)?\.flac")


def _bbox(lat: float, lon: float, radius_km: float) -> dict:
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


async def fetch_hourly(
    client: httpx.AsyncClient, token: str,
    lat: float, lon: float, day: datetime, radius_km: float,
    max_retries: int = 6,
) -> dict[int, list[dict]]:
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
    delay = 5.0
    for attempt in range(max_retries):
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


async def list_first_flac(client: httpx.AsyncClient, site: str, deployment: str) -> str | None:
    """Return the GCS object name of the first FLAC under
    sanctsound_<site>_<deployment>/audio/."""
    prefix = f"sanctsound/audio/{site}/sanctsound_{site}_{deployment}/audio/"
    resp = await client.get(GCS_API, params={"prefix": prefix, "maxResults": "5"})
    resp.raise_for_status()
    items = resp.json().get("items", [])
    for it in items:
        if it["name"].endswith(".flac"):
            return it["name"]
    return None


async def download_flac(client: httpx.AsyncClient, gcs_name: str, local_path: Path) -> None:
    if local_path.exists() and local_path.stat().st_size > 1_000_000:
        log.info("flac_already_present", path=str(local_path),
                 size_mb=round(local_path.stat().st_size / 1e6, 1))
        return
    url = f"{GCS_BASE}/{gcs_name}"
    log.info("downloading_flac", url=url)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    async with client.stream("GET", url, timeout=600.0) as resp:
        resp.raise_for_status()
        total = 0
        with local_path.open("wb") as f:
            async for chunk in resp.aiter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                total += len(chunk)
    log.info("flac_downloaded", path=str(local_path),
             size_mb=round(total / 1e6, 1))


def make_mel(samples: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=samples, sr=sr, n_mels=N_MELS, fmax=F_MAX,
    )
    return librosa.power_to_db(mel, ref=1.0)


async def process_flac(
    flac_path: Path, site: str, deployment: str, lat: float, lon: float,
    gfw_client: httpx.AsyncClient, gfw_token: str,
    written_event_ids: set[str],
) -> dict[str, int]:
    """Chunk one FLAC into 5s windows, AIS-label each, append to JSONL."""
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
    print(f"  capture_start: {capture_start.isoformat()}")

    print(f"  fetching hourly AIS for {day.date()} ...")
    hr10 = await fetch_hourly(gfw_client, gfw_token, lat, lon, day, 10.0)
    await asyncio.sleep(1.0)
    hr50 = await fetch_hourly(gfw_client, gfw_token, lat, lon, day, 50.0)
    n10 = sum(1 for v in hr10.values() if v)
    n50 = sum(1 for v in hr50.values() if v)
    print(f"  vessels in 10km: {n10}/24 hours,  in 50km: {n50}/24 hours")

    info = sf.info(str(flac_path))
    sr_native = info.samplerate
    n_frames_native = info.frames
    duration_s = n_frames_native / sr_native
    print(f"  flac duration: {duration_s/3600:.2f}h, sr={sr_native}")

    # Iterate 5s chunks. Resample to TARGET_SAMPLE_RATE for spec generation.
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    chunk_native = int(CHUNK_SECONDS * sr_native)
    n_chunks = int(duration_s // CHUNK_SECONDS)

    for chunk_idx in range(n_chunks):
        chunk_ts = capture_start + timedelta(seconds=chunk_idx * CHUNK_SECONDS)
        h = chunk_ts.hour
        close = bool(hr10.get(h))
        wide = bool(hr50.get(h))
        prior_wide = bool(hr50.get((h - 1) % 24))

        if close:
            label = "ship"
            distance_bucket = "close"
        elif wide:
            label = "ship_distant"
            distance_bucket = "medium"
        elif prior_wide:
            counters["skipped_tail"] += 1
            continue
        else:
            label = "ambient"
            distance_bucket = "none"

        # Read just this 5s chunk + resample
        start_frame = chunk_idx * chunk_native
        if start_frame + chunk_native > n_frames_native:
            break
        samples, _ = sf.read(
            str(flac_path), start=start_frame, frames=chunk_native, dtype="float32",
        )
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sr_native != TARGET_SAMPLE_RATE:
            samples = librosa.resample(
                samples, orig_sr=sr_native, target_sr=TARGET_SAMPLE_RATE,
            )

        mel = make_mel(samples, TARGET_SAMPLE_RATE)
        event_id = (
            f"sanctsound_{site}_{flac_path.stem}_{chunk_idx}"
        )
        if event_id in written_event_ids:
            continue
        written_event_ids.add(event_id)

        spec_path = SPEC_DIR / f"{event_id}.npy"
        np.save(spec_path, mel.astype(np.float32))

        binary_label = "ship" if label == "ship" else "not_ship"
        entry = {
            "event_id": event_id,
            "spectrogram_path": str(spec_path),
            "label": binary_label,
            "sanctsound_ais_label": label,
            "distance_bucket": distance_bucket,
            "audio_capture_start": chunk_ts.isoformat(),
            "audio": {
                "duration_s": float(CHUNK_SECONDS),
                "sample_rate": TARGET_SAMPLE_RATE,
            },
            "provenance": {
                "source_id": f"sanctsound-diverse-{site}",
                "source_file": f"{site}/{flac_path.name}",
                "deployment": f"sanctsound_{site}_{deployment}",
                "is_diverse_pull": True,
            },
        }
        OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_JSONL.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        counters[label] += 1

        if (chunk_idx + 1) % 200 == 0:
            print(f"    [{chunk_idx+1}/{n_chunks}] "
                  f"ship={counters['ship']} dist={counters['ship_distant']} "
                  f"amb={counters['ambient']} tail={counters['skipped_tail']}")
    return counters


async def main() -> None:
    settings = Settings()
    if not settings.gfw_api_token:
        sys.exit("OS_GFW_API_TOKEN missing")

    written_event_ids: set[str] = set()
    if OUTPUT_JSONL.exists():
        with OUTPUT_JSONL.open() as f:
            for line in f:
                try:
                    written_event_ids.add(json.loads(line)["event_id"])
                except Exception:
                    continue
        print(f"Resuming — {len(written_event_ids)} chunks already in {OUTPUT_JSONL}")

    aggregate: dict[str, int] = {}

    async with httpx.AsyncClient(timeout=600.0) as gcs_client, \
               httpx.AsyncClient(timeout=60.0) as gfw_client:
        for site, deployment in SITES_TO_PULL:
            print(f"\n=== {site} (deployment {deployment}) ===")
            if site not in SITE_LOCATIONS:
                print(f"  ! no coords for {site}, skipping")
                continue
            lat, lon = SITE_LOCATIONS[site]
            print(f"  location: {lat:.3f}, {lon:.3f}")
            gcs_name = await list_first_flac(gcs_client, site, deployment)
            if not gcs_name:
                print(f"  ! no FLAC found, skipping")
                continue
            local = AUDIO_DIR / site / Path(gcs_name).name
            try:
                await download_flac(gcs_client, gcs_name, local)
            except Exception as e:
                print(f"  ! download failed: {e}")
                continue
            try:
                counters = await process_flac(
                    local, site, deployment, lat, lon,
                    gfw_client, settings.gfw_api_token,
                    written_event_ids,
                )
            except Exception as e:
                print(f"  ! processing failed: {e}")
                continue
            print(f"  -> counters: {counters}")
            for k, v in counters.items():
                aggregate[k] = aggregate.get(k, 0) + v

    print("\n=== Aggregate across all sites ===")
    for k, v in sorted(aggregate.items(), key=lambda x: -x[1]):
        print(f"  {k:<20s} {v:>6d}")
    total_emitted = sum(v for k, v in aggregate.items() if not k.startswith("skipped"))
    print(f"\nTotal emitted training rows: {total_emitted}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    asyncio.run(main())
