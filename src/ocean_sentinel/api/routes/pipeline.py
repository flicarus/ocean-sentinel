import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ocean_sentinel.adapters.alerts.sendgrid import SendGridAdapter
from ocean_sentinel.domain.enums import AlertChannel, AlertStatus, ThreatLevel
from ocean_sentinel.domain.models import Alert
from ocean_sentinel.exceptions import AlertDeliveryError

router = APIRouter()


# Sources the dashboard may request. Keep in sync with Dashboard.jsx source dropdown.
SUPPORTED_SOURCES = (
    "mbari",
    "orcasound_lab",
    "port_townsend",
    "bush_point",
    "sunset_bay",
    "mast_center",
    "point_robinson",
    "andrews_bay",
    "north_sjc",
)


class ScanRequest(BaseModel):
    source: str = "mbari"                 # which hydrophone to scan
    date: str = "2024-02-01"               # date — ignored by Orcasound sources
    step: int = 300                        # seconds between chunks
    sample_every: int = 5                  # classify every Nth chunk


class ScanChunkResult(BaseModel):
    time: str
    offset: int
    engine_band_energy_db: float
    engine_band_ratio: float
    peak_frequency_hz: float
    spectral_flatness: float
    rms_energy: float
    is_engine_band_dominant: bool
    is_hit: bool


class ClassificationResult(BaseModel):
    time: str
    threat_level: str
    confidence: float
    reasoning: str
    vessel_type: str | None = None
    recommended_action: str | None = None
    ais_gaps: int = 0
    has_ocean_data: bool = False
    lat: float
    lon: float


class ScanResponse(BaseModel):
    date: str
    source: str
    chunks_scanned: int
    chunks_classified: int
    chunks_failed: int
    chunks: list[ScanChunkResult]
    classifications: list[ClassificationResult]


def _build_source(source_id: str, settings):
    """Factory: map dashboard source name → HydrophoneSource instance."""
    from ocean_sentinel.adapters.mbari import MBARIAdapter
    from ocean_sentinel.adapters.orcasound import OrcasoundAdapter

    if source_id == "mbari":
        return MBARIAdapter(settings)
    if source_id in SUPPORTED_SOURCES:
        return OrcasoundAdapter(f"rpi_{source_id}", settings)
    raise HTTPException(
        status_code=400,
        detail=f"Unknown source '{source_id}'. Known: {list(SUPPORTED_SOURCES)}",
    )


async def _send_pipeline_alert(event, settings, store) -> None:
    """Fire a SendGrid email and persist the alert record for HIGH/CRITICAL events."""
    sendgrid = SendGridAdapter(settings)
    alert = Alert(
        id=str(uuid.uuid4()),
        event_id=event.id,
        channel=AlertChannel.EMAIL,
        recipient=settings.sendgrid_from_email,
        sent_at=None,
        status=AlertStatus.PENDING,
    )
    try:
        await sendgrid.send(alert, event)
        alert = Alert(
            id=alert.id,
            event_id=alert.event_id,
            channel=alert.channel,
            recipient=alert.recipient,
            sent_at=datetime.now(timezone.utc),
            status=AlertStatus.SENT,
        )
    except AlertDeliveryError as exc:
        alert = Alert(
            id=alert.id,
            event_id=alert.event_id,
            channel=alert.channel,
            recipient=alert.recipient,
            sent_at=None,
            status=AlertStatus.FAILED,
            failure_reason=str(exc),
        )
    await store.save_alert(alert)


@router.post("/scan", response_model=ScanResponse)
async def run_scan(req: ScanRequest, request: Request):
    """Run the full pipeline scan on a given date + hydrophone source."""
    from ocean_sentinel.services.audio_analyzer import AudioAnalyzer
    from ocean_sentinel.services.classifier import ThreatClassifierService

    from ocean_sentinel.domain.models import AcousticFeatures, DetectionEvent

    settings = request.app.state.settings
    training_logger = request.app.state.training_logger
    store = request.app.state.store

    source = _build_source(req.source, settings)
    analyzer = AudioAnalyzer(settings)
    classifier = ThreatClassifierService(
        analyzer=analyzer,
        cnn=request.app.state.cnn,
        engine=request.app.state.decision_engine,
    )

    dt = datetime.strptime(req.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    chunks = []
    classifications = []
    scanned = 0
    failed = 0
    classified = 0

    spec_dir = Path("data/spectrograms")
    spec_dir.mkdir(parents=True, exist_ok=True)

    for offset in range(0, 86400, req.step):
        hours = offset // 3600
        minutes = (offset % 3600) // 60
        time_str = f"{hours:02d}:{minutes:02d} UTC"

        try:
            segment = await source.fetch_at_offset(dt, offset, 60)
            analyzed, features = analyzer.analyze(segment)
            scanned += 1

            event_id = f"{source.source_id}_{req.date}_{offset}"
            np.save(spec_dir / f"{event_id}.npy", analyzed.spectrogram)

            chunks.append(ScanChunkResult(
                time=time_str,
                offset=offset,
                engine_band_energy_db=features["engine_band_energy_db"],
                engine_band_ratio=features["engine_band_ratio"],
                peak_frequency_hz=features["peak_frequency_hz"],
                spectral_flatness=features["spectral_flatness"],
                rms_energy=features["rms_energy"],
                is_engine_band_dominant=features["is_engine_band_dominant"],
                is_hit=features["is_engine_band_dominant"],
            ))

            if scanned % req.sample_every == 0:
                # AIS lookup is now done inside DecisionEngine when the
                # tier decision actually needs it (strong-ship branch).
                # No outer correlator hook required.
                result = await classifier.classify(audio=analyzed)
                classified += 1

                # Surface engine evidence onto the response payload.
                windowed = result.raw_output.get("windowed", {}) or {}
                ais = (
                    result.raw_output.get("provenance", {})
                    .get("evidence", {})
                    .get("ais", {})
                )

                classifications.append(ClassificationResult(
                    time=time_str,
                    threat_level=result.threat_level.value,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    vessel_type=windowed.get("vessel_type"),
                    recommended_action=None,
                    ais_gaps=ais.get("n_vessels", 0),
                    has_ocean_data=False,
                    lat=analyzed.location.lat,
                    lon=analyzed.location.lon,
                ))

                spec_path = str(spec_dir / f"{event_id}.npy")
                await training_logger.log(
                    event_id=event_id,
                    spectrogram_path=spec_path,
                    features=AcousticFeatures.from_analyzer_dict(features),
                    context_text=f"{source.source_id} | {analyzed.time_window.start.isoformat()}",
                    gemma_verdict=result.raw_output,
                    source_id=source.source_id,
                )

                event = DetectionEvent(
                    id=str(uuid.uuid4()),
                    timestamp=analyzed.time_window.start,
                    location=analyzed.location,
                    threat_level=result.threat_level,
                    confidence=result.confidence,
                    classification_reasoning=result.reasoning,
                    audio_segment=None,
                    ais_gaps=[],
                    ocean_conditions=None,
                    raw_model_output=result.raw_output,
                )
                await store.save_event(event)

                if event.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
                    await _send_pipeline_alert(event, settings, store)

        except Exception as e:
            import traceback
            traceback.print_exc()
            failed += 1

    await source.close()

    return ScanResponse(
        date=req.date,
        source=req.source,
        chunks_scanned=scanned,
        chunks_classified=classified,
        chunks_failed=failed,
        chunks=chunks,
        classifications=classifications,
    )
