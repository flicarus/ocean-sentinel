from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import httpx
import structlog

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
from ocean_sentinel.services.agent_tools import Toolbox

log = structlog.get_logger()


SYSTEM_PROMPT = """\
You are Ocean Sentinel, a marine surveillance threat assessor.

A specialized acoustic CNN has already analyzed the underwater audio and
decided whether a vessel is present. Your job is to assign a THREAT LEVEL
by reasoning over the CNN verdict and the corroborating evidence.

You receive text only (no audio, no images):
1. CNN verdict — binary ship / not_ship plus a calibrated confidence
   (Expected Calibration Error ≈ 2%, so a stated 0.90 means real
   accuracy ≈ 90% — trust this number)
2. Acoustic features (engine band energy, peak frequency, spectral
   flatness) — corroborating signal when CNN confidence is borderline
3. AIS gap events nearby — vessels that stopped transmitting AIS, with
   `intentional_disabling` and `in_mpa` flags from Global Fishing Watch
4. Ocean conditions — currents and sea surface temperature
5. Prior similar detections — acoustically similar past observations
   retrieved from a CNN-embedding RAG store, with their threat levels.
   Use these as calibration anchors.

You also have access to tools to gather more evidence — call them when
relevant BEFORE issuing the final verdict:
  • check_mpa_proximity(lat, lon) — nearest MPA + protection level.
    ALWAYS call this when CNN says ship and a location is given.
  • lookup_vessel_registry(mmsi) — vessel name, flag, type, IUU history.
    ALWAYS call this when an AIS gap event has a vessel_id.
  • query_recent_detections(lat, lon, radius_km, hours) — pattern of
    activity near the detection. Call when CNN confidence is borderline
    or AIS gap evidence is ambiguous.

Skip tools entirely when CNN says not_ship — they add no value on quiet
chunks.

Decision rubric (use the SHORTEST applicable rule):

NONE     CNN says not_ship, OR ship at confidence < 0.5 with no
         supporting AIS evidence.

LOW      CNN says ship with reasonable confidence, no AIS gap nearby,
         no MPA involvement. Routine traffic.

MEDIUM   Ship detected + EXACTLY ONE moderate anomaly:
         - brief AIS gap (< 1h, not intentional, not inside MPA), OR
         - vessel near (not inside) an MPA, OR
         - conflicting signals between CNN confidence and acoustic
           features.

HIGH     Ship detected + ANY ONE of these serious anomalies is sufficient:
         - AIS gap > 1h, OR
         - vessel inside an MPA (in_mpa = true OR check_mpa_proximity
           returned inside=true), OR
         - intentional_disabling = true, OR
         - vessel registry shows ≥1 prior IUU event.
         Even a single one of these qualifies — do not downgrade to MEDIUM.
         Likely non-compliant fishing; coast guard should be notified.

CRITICAL Ship detected + TWO OR MORE of the HIGH anomalies stacked
         (e.g. AIS gap > 1h AND in_mpa, recidivist vessel inside no-take
         MPA, persistent_activity AND in_mpa). Strong evidence of illegal
         fishing in a protected area — immediate response.

Weight prior similar detections from RAG context: if acoustically
similar past observations were classified HIGH, that is calibration
data, not noise.

Cite the specific evidence that drove your decision in `reasoning`
(1-2 sentences), including any tool results. Respond ONLY with valid JSON:
{
    "threat_level": "CRITICAL|HIGH|MEDIUM|LOW|NONE",
    "confidence": 0.0-1.0,
    "reasoning": "evidence-grounded justification",
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

    MAX_TOOL_ITERATIONS = 4

    def __init__(
        self,
        settings: Settings,
        memory: AcousticMemory | None = None,
        toolbox: Toolbox | None = None,
    ) -> None:
        self._google_api_key = settings.google_ai_api_key
        self._ollama_url = settings.ollama_base_url
        self._model = settings.gemma_model
        self._client = httpx.AsyncClient(timeout=300.0)
        self._memory = memory
        self._toolbox = toolbox

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
        cnn_verdict: dict | None = None,
    ) -> ClassificationResult:
        """Send spectrogram + context + RAG history to Gemma, parse verdict,
        then store the result back into acoustic memory."""

        text_context = self._build_context(audio, ais_gaps, ocean, features, cnn_verdict)

        # --- RAG retrieval: enrich context with similar past detections ---
        similar_matches: list[SimilarMatch] = []
        if self._memory is not None:
            cnn_embedding = (
                cnn_verdict.get("embedding") if cnn_verdict else None
            )
            if cnn_embedding is not None:
                similar_matches = await self._memory.query_by_embedding(
                    cnn_embedding, n=3,
                )
                rag_path = "cnn_embedding"
            elif features is not None:
                acoustic_features = AcousticFeatures.from_analyzer_dict(features)
                similar_matches = await self._memory.query_similar(
                    acoustic_features, n=3,
                )
                rag_path = "acoustic_features"
            else:
                rag_path = None

            if similar_matches:
                rag_section = self._build_rag_context(similar_matches)
                text_context = f"{text_context}\n{rag_section}"
                log.info(
                    "rag_context_injected",
                    path=rag_path,
                    n_matches=len(similar_matches),
                    closest_score=similar_matches[0].score,
                )

        # --- Model inference ---
        try:
            if self._google_api_key:
                raw = await self._call_google_ai(text_context)
            else:
                raw = await self._call_ollama(text_context)
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
                cnn_verdict=cnn_verdict,
            )

        return result

    async def _store_to_memory(
        self,
        audio: AudioSegment,
        features: dict,
        context_text: str,
        result: ClassificationResult,
        raw: dict,
        cnn_verdict: dict | None = None,
    ) -> None:
        """Persist this classification as a new acoustic memory entry."""
        embedding = cnn_verdict.get("embedding") if cnn_verdict else None
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
            embedding=embedding,
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

    def _build_context(
        self,
        audio: AudioSegment,
        ais_gaps: list[AISGapEvent],
        ocean: OceanConditions | None,
        features: dict | None,
        cnn_verdict: dict | None = None,
    ) -> str:
        """Build the text part of the prompt."""
        parts = []

        parts.append(f"Hydrophone: {audio.source_file}")
        parts.append(f"Location: {audio.location.lat}°N {audio.location.lon}°W")
        parts.append(f"Time: {audio.time_window.start} to {audio.time_window.end}")

        if cnn_verdict:
            probs = cnn_verdict.get("probabilities", {})
            parts.append("\nCNN tier-1 verdict (binary ship classifier):")
            parts.append(f"  Label: {cnn_verdict['label']}")
            parts.append(f"  Confidence: {cnn_verdict['confidence']:.1%}")
            for cls in ("not_ship", "ship"):
                if cls in probs:
                    parts.append(f"  P({cls}) = {probs[cls]:.1%}")

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
                name = gap.vessel_name or "unnamed"
                parts.append(
                    f"  - {name} (MMSI: {gap.vessel_id}): "
                    f"dark since {gap.gap_start}, "
                    f"{gap.gap_duration_hours:.1f}h, "
                    f"flag: {gap.flag_state or 'unknown'}, "
                    f"in_mpa: {gap.in_mpa}, "
                    f"intentional_disabling: {gap.intentional_disabling}"
                )
        else:
            parts.append("\nNo AIS gaps detected in this area/timeframe.")

        if ocean:
            parts.append("\nOcean conditions:")
            parts.append(f"  SST: {ocean.sea_surface_temp_c}°C")
            parts.append(f"  Current: {ocean.current_speed_ms} m/s, {ocean.current_direction_deg}°")

        return "\n".join(parts)

    async def _call_google_ai(self, text: str) -> dict:
        """Call Gemma via Google AI Studio API (text-only threat assessor)."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent"

        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": text}]}],
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

    async def _call_ollama(self, text: str) -> dict:
        """Call Gemma via local Ollama, with optional agent tool loop.

        Loop semantics:
          1. POST messages (+ tool schemas if Toolbox configured).
          2. If response has tool_calls → dispatch each tool, append the
             assistant turn AND tool result messages, re-POST.
          3. If response has no tool_calls → that's the final JSON verdict.
          4. Bail at MAX_TOOL_ITERATIONS to prevent runaway loops.
        """
        url = f"{self._ollama_url}/api/chat"

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        payload: dict = {
            "model": self._model,
            "stream": False,
            "format": "json",
            "messages": messages,
            "options": {"temperature": 0.1},
        }
        if self._toolbox is not None:
            payload["tools"] = self._toolbox.schemas

        for iteration in range(self.MAX_TOOL_ITERATIONS):
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            msg = data["message"]
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                # Final answer — parse JSON verdict.
                text_out = (msg.get("content") or "").strip()
                if text_out.startswith("```"):
                    text_out = text_out.split("\n", 1)[-1]
                if text_out.endswith("```"):
                    text_out = text_out[: text_out.rfind("```")].strip()

                log.info(
                    "gemma_response",
                    source="ollama",
                    model=self._model,
                    raw_length=len(text_out),
                    tool_iterations=iteration,
                )

                if not text_out:
                    log.warning("gemma_empty_response", source="ollama")
                    return self._fallback_response("Model returned empty response")

                try:
                    return json.loads(text_out)
                except json.JSONDecodeError:
                    log.warning("gemma_invalid_json", source="ollama", raw=text_out[:200])
                    return self._fallback_response(f"Invalid JSON: {text_out[:100]}")

            # Tool calls requested — run each, append results, loop.
            messages.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            })

            for call in tool_calls:
                fn = call["function"]
                name = fn["name"]
                args = fn["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                result_json = await self._toolbox.dispatch(name, args)
                messages.append({"role": "tool", "name": name, "content": result_json})

            payload["messages"] = messages

        log.warning("gemma_tool_loop_exhausted", iterations=self.MAX_TOOL_ITERATIONS)
        return self._fallback_response(
            f"Tool loop exhausted at {self.MAX_TOOL_ITERATIONS} iterations"
        )

    def _fallback_response(self, reason: str) -> dict:
        """Synthesize a NONE-verdict dict when the model can't produce one."""
        return {
            "threat_level": "NONE",
            "confidence": 0.0,
            "reasoning": reason,
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
