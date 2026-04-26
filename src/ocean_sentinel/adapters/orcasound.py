"""Orcasound adapter v2 — discovers nodes and streams via the Orcasite JSON API.

v1 relied on anonymous S3 ListBucket, which the public bucket blocks (403).
v2 uses the Orcasite public API instead — no listing needed:

    https://live.orcasound.net/api/json/feeds          (all nodes)
    https://live.orcasound.net/api/json/feed_streams   (streams per feed)

Each adapter instance is pinned to one node. `fetch_at_offset(dt, ...)` ignores
`dt` — Orcasound streams are timestamp-keyed, not datetime-keyed. The returned
AudioSegment carries the real capture time in `time_window`, which is what
downstream correlation + persistence actually use.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import structlog

from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import AudioSegment, GeoPoint, TimeWindow

log = structlog.get_logger()

ORCASITE_API = "https://live.orcasound.net/api/json"

# From /api/json/feeds — 8 nodes, all in Salish Sea / Puget Sound.
# Keyed by the API's node_name value (needed to build the S3 path).
ORCASOUND_LOCATIONS: dict[str, GeoPoint] = {
    "rpi_mast_center":    GeoPoint(lat=47.34922,   lon=-122.32512),
    "rpi_sunset_bay":     GeoPoint(lat=47.864973,  lon=-122.333936),
    "rpi_point_robinson": GeoPoint(lat=47.388383,  lon=-122.37267),
    "rpi_andrews_bay":    GeoPoint(lat=48.546653,  lon=-123.166408),
    "rpi_port_townsend":  GeoPoint(lat=48.135743,  lon=-122.760614),
    "rpi_orcasound_lab":  GeoPoint(lat=48.5583362, lon=-123.1735774),
    "rpi_north_sjc":      GeoPoint(lat=48.591294,  lon=-123.058779),
    "rpi_bush_point":     GeoPoint(lat=48.0336664, lon=-122.6040035),
}


class OrcasoundAdapter:
    """Pulls HLS segments from one Orcasound node via the Orcasite JSON API."""

    def __init__(self, node_name: str, settings: Settings) -> None:
        del settings  # accepted for adapter-constructor symmetry; unused here
        if node_name not in ORCASOUND_LOCATIONS:
            raise ValueError(
                f"Unknown Orcasound node '{node_name}'. "
                f"Known: {list(ORCASOUND_LOCATIONS)}"
            )
        self.node_name = node_name
        self.location = ORCASOUND_LOCATIONS[node_name]
        self.source_id = f"orcasound_{node_name.removeprefix('rpi_')}"
        self._client = httpx.AsyncClient(timeout=30.0)
        self._target_sample_rate = 16000
        self._bucket: str | None = None
        self._stream_ts: int | None = None
        self._segments: list[str] | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def _fetch_node_feed_id(self) -> tuple[str, str]:
        """Return (feed_id, bucket) for this adapter's node from /api/json/feeds."""
        resp = await self._client.get(f"{ORCASITE_API}/feeds")
        resp.raise_for_status()
        for feed in resp.json()["data"]:
            if feed["attributes"]["node_name"] == self.node_name:
                return feed["id"], feed["attributes"]["bucket"]
        raise RuntimeError(
            f"Orcasound node '{self.node_name}' not found in feeds API"
        )

    async def _find_viable_stream(
        self, feed_id: str, bucket: str,
    ) -> tuple[int, list[str]]:
        """Newest stream with >= 3 segments. Returns (playlist_timestamp, segments)."""
        resp = await self._client.get(
            f"{ORCASITE_API}/feed_streams",
            params={"filter[feed_id]": feed_id, "page[limit]": 20},
        )
        resp.raise_for_status()
        for stream in resp.json()["data"]:
            ts = int(stream["attributes"]["playlist_timestamp"])
            m3u8_url = (
                f"https://{bucket}.s3.amazonaws.com/"
                f"{self.node_name}/hls/{ts}/live.m3u8"
            )
            try:
                m3u8_resp = await self._client.get(m3u8_url)
                m3u8_resp.raise_for_status()
            except httpx.HTTPError:
                continue
            segments = [
                line.strip()
                for line in m3u8_resp.text.splitlines()
                if line.strip() and not line.startswith("#")
            ]
            if len(segments) >= 3:
                return ts, segments
        raise RuntimeError(
            f"No viable stream found for node '{self.node_name}'"
        )

    async def _ensure_stream(self) -> None:
        if self._segments is not None:
            return
        feed_id, bucket = await self._fetch_node_feed_id()
        ts, segments = await self._find_viable_stream(feed_id, bucket)
        self._stream_ts = ts
        self._segments = segments
        self._bucket = bucket
        log.info(
            "orcasound_stream_selected",
            node=self.node_name,
            stream_ts=ts,
            segments=len(segments),
            bucket=bucket,
        )

    async def fetch_at_offset(
        self,
        dt: datetime,
        offset_seconds: int,
        duration_seconds: int = 60,
    ) -> AudioSegment:
        """dt is ignored; offset_seconds selects which cached segment to fetch."""
        del dt
        await self._ensure_stream()
        assert self._segments is not None and self._stream_ts is not None
        assert self._bucket is not None

        seg_idx = (offset_seconds // 10) % len(self._segments)
        seg_name = self._segments[seg_idx]
        seg_url = (
            f"https://{self._bucket}.s3.amazonaws.com/"
            f"{self.node_name}/hls/{self._stream_ts}/{seg_name}"
        )

        resp = await self._client.get(seg_url)
        resp.raise_for_status()
        samples = await _decode_ts_to_pcm(resp.content, self._target_sample_rate)

        capture_start = datetime.fromtimestamp(
            self._stream_ts, tz=timezone.utc,
        ) + timedelta(seconds=seg_idx * 10)
        capture_end = capture_start + timedelta(seconds=duration_seconds)

        log.info(
            "orcasound_segment_fetched",
            node=self.node_name,
            segment=seg_name,
            seg_idx=seg_idx,
            samples=len(samples),
            capture_start=capture_start.isoformat(),
        )

        return AudioSegment(
            source_file=seg_url,
            location=self.location,
            time_window=TimeWindow(start=capture_start, end=capture_end),
            sample_rate=self._target_sample_rate,
            samples=samples,
            source_id=self.source_id,
        )


async def _decode_ts_to_pcm(ts_bytes: bytes, target_sample_rate: int) -> np.ndarray:
    """Pipe .ts bytes through ffmpeg; return mono float32 samples at target rate."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(target_sample_rate),
        "-loglevel", "error",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    pcm, err = await proc.communicate(input=ts_bytes)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed: {err.decode(errors='replace').strip()}"
        )
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
