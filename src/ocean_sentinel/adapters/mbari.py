import struct
from datetime import timezone
from datetime import datetime, timezone

import httpx
import librosa
import numpy as np
import structlog

from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import AudioSegment, GeoPoint, TimeWindow
from ocean_sentinel.exceptions import MBARIError

log = structlog.get_logger()

# MBARI MARS hydrophone — fixed location in Monterey Canyon
MONTEREY_CANYON = GeoPoint(lat=36.7128, lon=-122.186)


class MBARIAdapter:
    """Fetches hydrophone audio from MBARI Pacific Sound S3 bucket."""

    def __init__(self, settings: Settings) -> None:
        self._bucket = settings.mbari_bucket
        self._sample_rate = settings.mbari_sample_rate
        self._segment_seconds = settings.mbari_segment_seconds
        self._client = httpx.AsyncClient(timeout=30.0)
        self.location = MONTEREY_CANYON
        self.source_id = "mbari"

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_audio(
        self, location: GeoPoint, time_window: TimeWindow
    ) -> list[AudioSegment]:
        """Download audio segments covering the time window."""
        segments: list[AudioSegment] = []

        # Generate one S3 URL per day in the window
        current = time_window.start
        while current <= time_window.end:
            url = self._build_url(current)
            try:
                raw_audio = await self._download_chunk(url, self._segment_seconds)
                samples = self._decode_wav(raw_audio)

                segment = AudioSegment(
                    source_file=url,
                    location=MONTEREY_CANYON,
                    time_window=TimeWindow(
                        start=current,
                        end=current,
                    ),
                    sample_rate=self._sample_rate,
                    samples=samples,
                    source_id=self.source_id,
                )
                segments.append(segment)
                log.info(
                    "audio_fetched",
                    source="mbari",
                    url=url,
                    samples=len(samples),
                )

            except httpx.HTTPError as e:
                log.error("audio_fetch_failed", source="mbari", url=url, error=str(e))
                raise MBARIError(
                    code="mbari_download_failed",
                    message=f"Failed to download {url}",
                    details={"url": url, "error": str(e)},
                ) from e

            # Move to next day
            current = current.replace(day=current.day + 1)

        return segments

    def _build_url(self, dt: datetime) -> str:
        """S3 URL for a given date. Format: MARS-YYYYMMDDTHHMMSSz-16kHz.wav"""
        filename = dt.strftime("MARS-%Y%m%dT%H%M%SZ") + f"-{self._sample_rate // 1000}kHz.wav"
        year = dt.strftime("%Y")
        month = dt.strftime("%m")
        return f"https://{self._bucket}.s3-us-west-2.amazonaws.com/{year}/{month}/{filename}"

    async def fetch_at_offset(
        self, dt: datetime, offset_seconds: int, duration_seconds: int = 60,
    ) -> AudioSegment:
        """Download a chunk from a specific time offset within a day's file.

        offset_seconds: how far into the file to skip (e.g. 3600 = 1 hour in)
        duration_seconds: how many seconds to grab
        """
        url = self._build_url(dt)
        bytes_per_second = self._sample_rate * 2  # 16-bit PCM
        start_byte = 44 + (offset_seconds * bytes_per_second)
        end_byte = start_byte + (duration_seconds * bytes_per_second) - 1

        response = await self._client.get(
            url, headers={"Range": f"bytes={start_byte}-{end_byte}"}
        )
        response.raise_for_status()

        # Raw PCM — no WAV header since we skipped past it
        raw = np.frombuffer(response.content, dtype=np.int16).astype(np.float32)
        raw /= 32768.0  # normalize to [-1, 1]

        from datetime import timedelta
        seg_start = dt + timedelta(seconds=offset_seconds)
        seg_end = seg_start + timedelta(seconds=duration_seconds)

        return AudioSegment(
            source_file=f"{url}#offset={offset_seconds}s",
            location=MONTEREY_CANYON,
            time_window=TimeWindow(start=seg_start, end=seg_end),
            sample_rate=self._sample_rate,
            samples=raw,
            source_id=self.source_id,
        )

    async def _download_chunk(self, url: str, seconds: int) -> bytes:
        """HTTP range request — download only first N seconds of audio."""
        byte_count = 44 + (seconds * self._sample_rate * 2)  # WAV header + PCM data
        response = await self._client.get(
            url, headers={"Range": f"bytes=0-{byte_count - 1}"}
        )
        response.raise_for_status()
        return response.content

    def _decode_wav(self, data: bytes) -> np.ndarray:
        """Fix truncated WAV header and decode to numpy array."""
        if data[:4] != b"RIFF":
            raise MBARIError(
                code="mbari_invalid_format",
                message="Expected WAV file, got unknown format",
            )

        # Fix RIFF and data chunk sizes for our truncated file
        fixed = bytearray(data)
        struct.pack_into("<I", fixed, 4, len(fixed) - 8)
        struct.pack_into("<I", fixed, 40, len(fixed) - 44)

        # librosa reads from file path — write to temp buffer
        import io
        import soundfile as sf

        buffer = io.BytesIO(bytes(fixed))
        samples, _ = sf.read(buffer, dtype="float32")
        return samples
