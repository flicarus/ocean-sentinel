"""Pull NEW SanctSound sites at 60s windows for v8/v9 training.

Like bootstrap_sanctsound_diverse.py, but:
  - chunks at 60s instead of 5s (matches v8 target_frames=1876)
  - targets sites we DON'T already have downloaded
  - writes to data/training/sanctsound_more_60s.jsonl
  - spectrograms saved to data/spectrograms_60s/

New sites picked for offshore / quiet ambient diversity (ours are mostly
shipping-busy). These should bump the ambient class which is currently
underrepresented in our v8 dataset.
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


SITES_TO_PULL: list[tuple[str, str]] = [
    ("hi03", "02"),
    ("hi03", "03"),
    ("hi04", "01"),
    ("hi04", "02"),
    ("hi06", "01"),
    ("hi07", "01"),
    ("oc02", "01"),
    ("oc02", "02"),
    ("oc03", "01"),
    ("oc04", "01"),
    ("ci02", "01"),
    ("ci03", "01"),
    ("ci04", "01"),
    ("ci05", "01"),
    ("fk02", "01"),
    ("fk03", "01"),
    ("sb03", "01"),
]

SITE_LOCATIONS: dict[str, tuple[float, float]] = {
    "hi03": (20.80, -156.50),
    "hi04": (21.30, -157.20),
    "hi06": (19.90, -156.10),
    "hi07": (22.10, -159.50),
    "oc02": (48.10, -124.70),
    "oc03": (48.40, -124.95),
    "oc04": (48.55, -125.20),
    "ci02": (33.95, -119.65),
    "ci03": (34.05, -120.05),
    "ci04": (34.10, -119.45),
    "ci05": (33.65, -119.55),
    "fk02": (24.55, -81.50),
    "fk03": (24.75, -80.80),
    "sb03": (42.40, -70.50),
}

GCS_BASE = "https://storage.googleapis.com/noaa-passive-bioacoustic"
GCS_API = "https://storage.googleapis.com/storage/v1/b/noaa-passive-bioacoustic/o"
AUDIO_DIR = Path("data/sanctsound")
SPEC_DIR = Path("data/spectrograms_60s")
OUTPUT_JSONL = Path("data/training/sanctsound_more_60s.jsonl")
TARGET_SR = 16000
CHUNK_SECONDS = 60
N_MELS = 128
F_MAX = 1000.0

GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"

FILENAME_TS_RE = re.compile(r"_(\d{8}T\d{6})Z(?:_[A-Za-z\-]+)?\.flac")
FILENAME_TS_RE_ALT = re.compile(r"_(\d{12})\.flac")


def parse_capture_start(filename: str) -> datetime | None:
    m = FILENAME_TS_RE.search(filename)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    m = FILENAME_TS_RE_ALT.search(filename)
    if m:
        ts = m.group(1)
        try:
            return datetime.strptime("20" + ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


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


async def list_first_flac(client: httpx.AsyncClient, site: str,
                          deployment: str) -> str | None:
    prefix = f"sanctsound/audio/{site}/sanctsound_{site}_{deployment}/audio/"
    resp = await client.get(GCS_API, params={"prefix": prefix, "maxResults": "5"})
    resp.raise_for_status()
    for it in resp.json().get("items", []):
        if it["name"].endswith(".flac"):
            return it["name"]
    return None


async def download_flac(client: httpx.AsyncClient, gcs_name: str,
                        local_path: Path) -> None:
    if local_path.exists() and local_path.stat().st_size > 1_000_000:
        log.info("flac_already_present", path=str(local_path))
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


async def process_flac(flac_path: Path, site: str, deployment: str,
                       lat: float, lon: float,
                       gfw_client: httpx.AsyncClient, gfw_token: str,
                       written_event_ids: set[str]) -> dict[str, int]:
    counters = {"ship": 0, "ship_distant": 0, "ambient": 0,
                "skipped_tail": 0, "skipped_no_ts": 0}
    capture_start = parse_capture_start(flac_path.name)
    if capture_start is None:
        counters["skipped_no_ts"] += 1
        return counters
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
                "source_id": f"sanctsound-more-60s-{site}",
                "source_file": f"{site}/{flac_path.name}",
                "deployment": f"sanctsound_{site}_{deployment}",
                "is_60s_pull": True,
                "is_more_pull": True,
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
    async with httpx.AsyncClient(timeout=600.0) as gcs_client, \
               httpx.AsyncClient(timeout=60.0) as gfw_client:
        for site, deployment in SITES_TO_PULL:
            print(f"\n=== {site} (deployment {deployment}) ===")
            if site not in SITE_LOCATIONS:
                continue
            lat, lon = SITE_LOCATIONS[site]
            try:
                gcs_name = await list_first_flac(gcs_client, site, deployment)
            except Exception as e:
                print(f"  ! list failed: {e}")
                continue
            if not gcs_name:
                print(f"  ! no FLAC found")
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
                    gfw_client, settings.gfw_api_token, written,
                )
            except Exception as e:
                print(f"  ! processing failed: {e}")
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
