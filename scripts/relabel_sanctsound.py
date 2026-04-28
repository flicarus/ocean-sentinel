"""Auto-relabel SanctSound clips using AIS cross-reference.

Methodology (validated on OC01 in case-oc01-mislabel-discovery):
  1. For each clip, derive its (site, capture_time) from the SanctSound
     filename and chunk index.
  2. Query GFW vessel-presence at HOURLY resolution within RADIUS_KM of
     the hydrophone for the clip's recording date(s).
  3. If at least one vessel was broadcasting AIS within RADIUS_KM during
     the clip's hour, relabel as `ship`. Otherwise keep `not_ship`.
  4. Emit gemma_labels.v6.jsonl: identical to v5 except sanctsound rows
     have updated labels and a `provenance.relabeled_from` field for
     auditability.

Output also includes a per-site flip rate and the list of recording
files where ANY chunks flipped — these are candidate "missed vessel
passages" for inclusion in the submission story.
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from math import cos, radians
from pathlib import Path

import httpx

from ocean_sentinel.config import Settings


# Hydrophone coordinates (verified from NOAA / IQOE deployment metadata).
SITE_COORDS: dict[str, tuple[float, float]] = {
    "oc01": (48.400, -124.700),       # Olympic Coast NMS
    "sb01": (42.43668, -70.546655),   # Stellwagen Bank NMS
}

RADIUS_KM = 10.0       # Audit-grade radius validated by OC01 case.
GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"
SOURCE_JSONL = Path("data/training/gemma_labels.v5.jsonl")
TARGET_JSONL = Path("data/training/gemma_labels.v6.jsonl")


def parse_event_id(event_id: str) -> tuple[str, str, int] | None:
    """`sanctsound_<site>_SanctSound_<...>_<timestamp>Z_<chunk>` → (site, ts, chunk).

    Returns None if event_id doesn't match the SanctSound pattern.
    """
    parts = event_id.split("_")
    if len(parts) < 4 or parts[0] != "sanctsound":
        return None
    site = parts[1]
    chunk_str = parts[-1]
    try:
        chunk = int(chunk_str)
    except ValueError:
        return None
    # Timestamp segment is the one matching YYYYMMDDTHHMMSSZ
    timestamp_part = next(
        (p for p in parts if p.endswith("Z") and len(p) == 16 and "T" in p),
        None,
    )
    if not timestamp_part:
        return None
    return site, timestamp_part, chunk


def derive_capture_time(timestamp_str: str, chunk_idx: int) -> datetime:
    """SanctSound bootstrap chunks every 60s starting from filename ts."""
    ts = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc,
    )
    return ts + timedelta(seconds=chunk_idx * 60)


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


async def query_day(
    client: httpx.AsyncClient, token: str,
    site: str, day: datetime,
) -> set[int]:
    """Return the set of UTC hours during which at least one AIS-broadcasting
    vessel was within RADIUS_KM of `site` on `day`.
    """
    lat, lon = SITE_COORDS[site]
    day_str = day.strftime("%Y-%m-%d")
    next_day_str = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        f"{GFW_BASE}/4wings/report"
        f"?datasets[0]=public-global-presence:latest"
        f"&date-range={day_str},{next_day_str}"
        f"&temporal-resolution=HOURLY"
        f"&spatial-resolution=HIGH"
        f"&group-by=VESSEL_ID"
        f"&format=JSON"
    )
    resp = await client.post(
        url, json=bbox(lat, lon, RADIUS_KM),
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    data = resp.json()
    rows: list[dict] = []
    for entry in data.get("entries", []):
        if isinstance(entry, dict):
            for k, v in entry.items():
                if k.startswith("public-") and isinstance(v, list):
                    rows.extend(v)

    hours_with_vessel: set[int] = set()
    for r in rows:
        ts_str = r.get("entryTimestamp") or r.get("date")
        if not ts_str:
            continue
        try:
            t = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if t.date() == day.date():
                hours_with_vessel.add(t.hour)
        except Exception:
            continue
    return hours_with_vessel


async def main() -> None:
    settings = Settings()
    if not settings.gfw_api_token:
        print("OS_GFW_API_TOKEN missing — abort")
        sys.exit(1)

    # Load all entries; collect sanctsound clips for relabeling.
    all_entries: list[dict] = []
    sanctsound_clips: list[dict] = []
    with SOURCE_JSONL.open() as f:
        for line in f:
            e = json.loads(line)
            all_entries.append(e)
            if e.get("provenance", {}).get("source_id") == "sanctsound":
                sanctsound_clips.append(e)

    print(f"Loaded {len(all_entries)} entries, "
          f"{len(sanctsound_clips)} sanctsound clips")

    # Plan GFW queries: unique (site, recording-day) pairs.
    plan: dict[tuple[str, datetime], list[dict]] = {}
    parse_failures = 0
    for e in sanctsound_clips:
        parsed = parse_event_id(e["event_id"])
        if not parsed:
            parse_failures += 1
            continue
        site, ts_str, chunk = parsed
        if site not in SITE_COORDS:
            continue
        capture = derive_capture_time(ts_str, chunk)
        day = capture.replace(hour=0, minute=0, second=0, microsecond=0)
        key = (site, day)
        plan.setdefault(key, []).append({"entry": e, "capture": capture})

    print(f"Unique (site, day) pairs to query: {len(plan)}")
    print(f"Parse failures: {parse_failures}")
    if parse_failures:
        sys.exit(f"abort: {parse_failures} unparseable event_ids — fix parser")

    # Run all GFW queries with caching by (site, day).
    cache: dict[tuple[str, datetime], set[int]] = {}
    print("\nQuerying GFW...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        for (site, day), clips in plan.items():
            print(f"  {site} {day.date().isoformat()} "
                  f"({len(clips)} clips)... ", end="", flush=True)
            try:
                hours = await query_day(client, settings.gfw_api_token, site, day)
                cache[(site, day)] = hours
                print(f"vessel-active hours: {sorted(hours)}")
            except httpx.HTTPStatusError as ex:
                print(f"HTTP {ex.response.status_code}: {ex.response.text[:120]}")
                cache[(site, day)] = set()
            except Exception as ex:
                print(f"failed: {ex}")
                cache[(site, day)] = set()

    # Relabel pass.
    flips = Counter()                # (site, original→new) → count
    flipped_files: set[str] = set()  # recording filenames with ≥ 1 flip
    new_sanctsound_count = Counter()
    relabeled_clips: list[dict] = []
    for e in sanctsound_clips:
        parsed = parse_event_id(e["event_id"])
        if not parsed:
            relabeled_clips.append(e)
            continue
        site, ts_str, chunk = parsed
        capture = derive_capture_time(ts_str, chunk)
        day = capture.replace(hour=0, minute=0, second=0, microsecond=0)
        active = cache.get((site, day), set())
        new_label = "ship" if capture.hour in active else "not_ship"
        original = e["label"]
        if new_label != original:
            flips[(site, f"{original}→{new_label}")] += 1
            flipped_files.add(e["provenance"].get("source_file", "?"))
            new_e = json.loads(json.dumps(e))  # deep copy
            new_e["label"] = new_label
            new_e["provenance"]["original_label"] = original
            new_e["provenance"]["relabeled_from"] = "sanctsound_ais_crossref_v1"
            new_e["provenance"]["relabel_radius_km"] = RADIUS_KM
            relabeled_clips.append(new_e)
        else:
            relabeled_clips.append(e)
        new_sanctsound_count[new_label] += 1

    # Write v6 jsonl: non-sanctsound entries unchanged, sanctsound replaced.
    sanctsound_event_ids = {e["event_id"] for e in sanctsound_clips}
    output: list[dict] = []
    relabeled_by_id = {e["event_id"]: e for e in relabeled_clips}
    for e in all_entries:
        if e["event_id"] in sanctsound_event_ids:
            output.append(relabeled_by_id[e["event_id"]])
        else:
            output.append(e)

    TARGET_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with TARGET_JSONL.open("w") as f:
        for e in output:
            f.write(json.dumps(e) + "\n")

    print(f"\n=== Relabel summary ===")
    print(f"Wrote {len(output)} entries → {TARGET_JSONL}")
    print(f"Sanctsound new label distribution: {dict(new_sanctsound_count)}")
    print(f"Total flips: {sum(flips.values())}")
    if flips:
        print("Per (site, direction):")
        for (site, direction), n in sorted(flips.items()):
            print(f"  {site:<8s} {direction:<25s} {n:>4d}")
        print(f"\nRecording files with ≥1 flip ({len(flipped_files)}):")
        for f in sorted(flipped_files):
            print(f"  {f}")
    else:
        print("No flips — sanctsound labels already match AIS evidence.")


if __name__ == "__main__":
    asyncio.run(main())
