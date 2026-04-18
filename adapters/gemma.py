from __future__ import annotations

import base64
import io
import json

import httpx
import numpy as np
import structlog
from matplotlib import pyplot as plt

from ocean_sentinel.config import Settings
from ocean_sentinel.domain.enums import ThreatLevel
from ocean_sentinel.domain.models import (
    AISGapEvent,
    AudioSegment,
    ClassificationResult,
    OceanConditions,
)
from ocean_sentinel.exceptions import ClassificationError

log = structlog.get_logger()


SYSTEM_PROMPT = """\
You are Ocean Sentinel, a marine surveillance AI. You analyze hydrophone
spectrograms alongside vessel tracking and ocean data to detect illegal
fishing and maritime threats.

You will receive:
1. A mel spectrogram image from an underwater hydrophone
2. AIS gap events (vessels that stopped transmitting position)
3. Ocean conditions (currents, temperature)
4. Acoustic features extracted from the audio

Classify the threat level as one of: CRITICAL, HIGH, MEDIUM, LOW, NONE.
Respond ONLY with valid JSON matching this schema:
{
    "threat_level": "CRITICAL|HIGH|MEDIUM|LOW|NONE",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of your classification",
    "vessel_type": "trawler|cargo|fishing|recreational|unknown|none",
    "recommended_action": "alert_coast_guard|monitor|log_only|none"
}
"""

class GemmaAdapter:
    """Multimodal threat classifier using Gemma 4 via Google AI or Ollama."""

    def __init__(self, settings: Settings) -> None:
        self._google_api_key = settings.google_ai_api_key
        self._ollama_url = settings.ollama_base_url
        self._model = settings.gemma_model
        self._client = httpx.AsyncClient(timeout=120.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def classify(
        self,
        audio: AudioSegment,
        ais_gaps: list[AISGapEvent],
        ocean: OceanConditions | None,
        features: dict | None = None,
    ) -> ClassificationResult:
        """Send spectrogram + context to Gemma, parse verdict."""

        image_b64 = self._spectrogram_to_base64(audio.spectrogram)
        text_context = self._build_context(audio, ais_gaps, ocean, features)

        try:
            if self._google_api_key:
                raw = await self._call_google_ai(image_b64, text_context)
            else:
                raw = await self._call_ollama(image_b64, text_context)
        except httpx.HTTPError as e:
            raise ClassificationError(
                code="model_request_failed",
                message="Failed to reach Gemma model",
                details={"error": str(e)},
            ) from e

        return self._parse_response(raw)

    def _spectrogram_to_base64(self, spectrogram: np.ndarray | None) -> str:
        """Convert numpy spectrogram to base64 PNG for the vision model."""
        if spectrogram is None:
            raise ClassificationError(
                code="missing_spectrogram",
                message="AudioSegment has no spectrogram - run AudioAnalyzer first",
            )

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency (mel)")
        ax.set_title("Mel Spectrogram")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _build_context(
        self,
        audio: AudioSegment,
        ais_gaps: list[AISGapEvent],
        ocean: OceanConditions | None,
        features: dict | None,
    ) -> str:
        """Build the text part of the prompt."""
        parts = []

        parts.append(f"Hydrophone: {audio.source_file}")
        parts.append(f"Location: {audio.location.lat}°N {audio.location.lon}°W")
        parts.append(f"Time: {audio.time_window.start} to {audio.time_window.end}")

        if features:
            parts.append("\nAcoustic features:")
            parts.append(f"  Engine band energy: {features['engine_band_energy_db']} dB")
            parts.append(f"  Engine band ratio: {features['engine_band_ratio']}")
            parts.append(f"  Peak frequency: {features['peak_frequency_hz']} Hz")
            parts.append(f"  Spectral flatness: {features['spectral_flatness']}")
            parts.append(f"  Engine band dominant: {features['is_engine_band_dominant']}")

        if ais_gaps:
            parts.append(f"\nAIS gaps detected: {len(ais_gaps)} vessels went dark")
            for gap in ais_gaps:
                parts.append(f"  - {gap.vessel_name or gap.vessel_id}: "
                           f"dark since {gap.gap_start}, "
                           f"{gap.gap_duration_hours:.1f}h, "
                           f"flag: {gap.flag_state or 'unknown'}, "
                           f"in MPA: {gap.in_mpa}")
        else:
            parts.append("\nNo AIS gaps detected in this area/timeframe.")

        if ocean:
            parts.append("\nOcean conditions:")
            parts.append(f"  SST: {ocean.sea_surface_temp_c}°C")
            parts.append(f"  Current: {ocean.current_speed_ms} m/s, {ocean.current_direction_deg}°")

        return "\n".join(parts)

    async def _call_google_ai(self, image_b64: str, text: str) -> dict:
        """Call Gemma via Google AI Studio API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent"

        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                    {"text": text},
                ],
            }],
            "generation_config": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        }

        resp = await self._client.post(
            url,
            json=payload,
            params={"key": self._google_api_key},
        )
        resp.raise_for_status()

        data = resp.json()
        text_out = data["candidates"][0]["content"]["parts"][0]["text"]

        log.info("gemma_response", source="google_ai", model=self._model)
        return json.loads(text_out)

    async def _call_ollama(self, image_b64: str, text: str) -> dict:
        """Call Gemma via local Ollama instance."""
        url = f"{self._ollama_url}/api/chat"

        payload = {
            "model": self._model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": text,
                    "images": [image_b64],
                },
            ],
            "options": {"temperature": 0.1},
        }

        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()

        data = resp.json()
        text_out = data["message"]["content"]

        log.info("gemma_response", source="ollama", model=self._model)
        return json.loads(text_out)

    def _parse_response(self, raw: dict) -> ClassificationResult:
        """Parse model JSON into domain object."""
        try:
            threat = ThreatLevel(raw["threat_level"].upper())
        except (KeyError, ValueError):
            threat = ThreatLevel.MEDIUM
            log.warning("gemma_unknown_threat_level", raw_value=raw.get("threat_level"))

        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return ClassificationResult(
            threat_level=threat,
            confidence=confidence,
            reasoning=raw.get("reasoning", "No reasoning provided"),
            raw_output=raw,
        )
