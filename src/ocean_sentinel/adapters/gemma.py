from __future__ import annotations

import base64
import io
import json
import uuid
from datetime import datetime, timezone

import httpx
import numpy as np
import structlog
from matplotlib import pyplot as plt

from ocean_sentinel.config import Settings
from ocean_sentinel.domain.enums import ThreatLevel
from ocean_sentinel.domain.models import (
    AcousticEntry,
    AcousticFeatures,
    AISGapEvent,
    AudioSegment,
    ClassificationResult,
    OceanConditions,
    SimilarMatch,
)
from ocean_sentinel.domain.protocols import AcousticMemory
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
5. Prior observations — acoustically similar sounds from past scans with
   their classifications. Use these as calibration: if similar signatures
   were previously classified, weigh that evidence in your assessment.

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
    """Multimodal threat classifier using Gemma 4 via Google AI or Ollama.

    When initialised with an AcousticMemory, each classification:
    1. Queries similar past detections and injects them as RAG context.
    2. Stores the new (features → verdict) pair back into memory.

    This creates a self-improving loop — every scan enriches future scans.
    """

    def __init__(
        self,
        settings: Settings,
        memory: AcousticMemory | None = None,
    ) -> None:
        self._google_api_key = settings.google_ai_api_key
        self._ollama_url = settings.ollama_base_url
        self._model = settings.gemma_model
        self._client = httpx.AsyncClient(timeout=300.0)
        self._memory = memory

    async def close(self) -> None:
        await self._client.aclose()
        # Memory is NOT closed here — it's a shared resource managed
        # by the app lifespan, not owned by this adapter.

    async def classify(
        self,
        audio: AudioSegment,
        ais_gaps: list[AISGapEvent],
        ocean: OceanConditions | None,
        features: dict | None = None,
    ) -> ClassificationResult:
        """Send spectrogram + context + RAG history to Gemma, parse verdict,
        then store the result back into acoustic memory."""

        image_b64 = self._spectrogram_to_base64(audio.spectrogram)
        text_context = self._build_context(audio, ais_gaps, ocean, features)

        # --- RAG retrieval: enrich context with similar past detections ---
        similar_matches: list[SimilarMatch] = []
        if self._memory is not None and features is not None:
            acoustic_features = AcousticFeatures.from_analyzer_dict(features)
            similar_matches = await self._memory.query_similar(acoustic_features, n=3)

            if similar_matches:
                rag_section = self._build_rag_context(similar_matches)
                text_context = f"{text_context}\n{rag_section}"
                log.info(
                    "rag_context_injected",
                    n_matches=len(similar_matches),
                    closest_score=similar_matches[0].score,
                )

        # --- Model inference ---
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

        result = self._parse_response(raw)

        # --- Store this classification back into memory for future RAG ---
        if self._memory is not None and features is not None:
            await self._store_to_memory(
                audio=audio,
                features=features,
                context_text=text_context,
                result=result,
                raw=raw,
            )

        return result

    async def _store_to_memory(
        self,
        audio: AudioSegment,
        features: dict,
        context_text: str,
        result: ClassificationResult,
        raw: dict,
    ) -> None:
        """Persist this classification as a new acoustic memory entry."""
        entry = AcousticEntry(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            location=audio.location,
            features=AcousticFeatures.from_analyzer_dict(features),
            context_text=context_text,
            threat_level=result.threat_level,
            confidence=result.confidence,
            reasoning=result.reasoning,
            vessel_type=raw.get("vessel_type"),
            recommended_action=raw.get("recommended_action"),
        )
        await self._memory.store(entry)

        log.info(
            "classification_stored_to_memory",
            event_id=entry.event_id,
            threat_level=result.threat_level.value,
            memory_size=await self._memory.count(),
        )

    def _build_rag_context(self, matches: list[SimilarMatch]) -> str:
        """Format similar past detections as a prompt section for Gemma."""
        lines = ["\nPrior similar acoustic signatures:"]

        for i, match in enumerate(matches, 1):
            entry = match.entry
            similarity = max(0.0, 1.0 - match.score)  # distance → similarity

            lines.append(
                f"{i}. [{entry.timestamp:%Y-%m-%d %H:%M UTC}, "
                f"{entry.location.lat:.1f}°N {entry.location.lon:.1f}°W] "
                f"→ {entry.threat_level.value} (confidence: {entry.confidence:.2f})"
            )
            lines.append(f'   "{entry.reasoning}"')
            lines.append(f"   Acoustic similarity: {similarity:.2f}")

        return "\n".join(lines)

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
        text_out = data["message"]["content"].strip()

        # Gemma sometimes wraps JSON in markdown code blocks — strip them
        if text_out.startswith("```"):
            text_out = text_out.split("\n", 1)[-1]  # remove ```json line
        if text_out.endswith("```"):
            text_out = text_out[: text_out.rfind("```")].strip()

        log.info("gemma_response", source="ollama", model=self._model, raw_length=len(text_out))

        if not text_out:
            log.warning("gemma_empty_response", source="ollama")
            return {
                "threat_level": "NONE",
                "confidence": 0.0,
                "reasoning": "Model returned empty response",
                "vessel_type": "none",
                "recommended_action": "none",
            }

        try:
            return json.loads(text_out)
        except json.JSONDecodeError:
            log.warning("gemma_invalid_json", source="ollama", raw=text_out[:200])
            return {
                "threat_level": "NONE",
                "confidence": 0.0,
                "reasoning": f"Model returned invalid JSON: {text_out[:100]}",
                "vessel_type": "none",
                "recommended_action": "none",
            }

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
