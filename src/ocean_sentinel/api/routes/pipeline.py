from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from datetime import datetime, timezone

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


@router.post("/scan", response_model=ScanResponse)
async def run_scan(req: ScanRequest, request: Request):
    """Run the full pipeline scan on a given date + hydrophone source."""
    from ocean_sentinel.adapters.gemma import GemmaAdapter
    from ocean_sentinel.services.audio_analyzer import AudioAnalyzer
    from ocean_sentinel.services.correlation import CorrelationService
    from ocean_sentinel.services.classifier import ThreatClassifierService

    import uuid
    from ocean_sentinel.domain.models import AcousticFeatures, DetectionEvent

    settings = request.app.state.settings
    memory = request.app.state.memory
    training_logger = request.app.state.training_logger
    store = request.app.state.store

    source = _build_source(req.source, settings)
    analyzer = AudioAnalyzer(settings)
    gemma = GemmaAdapter(settings, memory=memory)
    correlation = CorrelationService(
        request.app.state.gfw,
        request.app.state.copernicus,
        settings,
    )
    classifier = ThreatClassifierService(analyzer, gemma, cnn=request.app.state.cnn)

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
                # Lazy correlation: only fetch AIS / ocean if the classifier
                # escalates to Gemma. CNN fast-path skips this entirely
                # (saves ~13% of GFW calls on val).
                correlation_box: dict = {"ais_gaps": [], "ocean": None}

                async def _correlate():
                    gaps, oc = await correlation.correlate(analyzed, features)
                    correlation_box["ais_gaps"] = gaps
                    correlation_box["ocean"] = oc
                    return gaps, oc

                result = await classifier.classify(
                    audio=analyzed, correlator=_correlate,
                )
                ais_gaps = correlation_box["ais_gaps"]
                ocean = correlation_box["ocean"]
                classified += 1

                classifications.append(ClassificationResult(
                    time=time_str,
                    threat_level=result.threat_level.value,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    vessel_type=result.raw_output.get("vessel_type"),
                    recommended_action=result.raw_output.get("recommended_action"),
                    ais_gaps=len(ais_gaps),
                    has_ocean_data=ocean is not None,
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
                )

                event = DetectionEvent(
                    id=str(uuid.uuid4()),
                    timestamp=analyzed.time_window.start,
                    location=analyzed.location,
                    threat_level=result.threat_level,
                    confidence=result.confidence,
                    classification_reasoning=result.reasoning,
                    audio_segment=None,
                    ais_gaps=ais_gaps,
                    ocean_conditions=ocean,
                    raw_model_output=result.raw_output,
                )
                await store.save_event(event)

        except Exception as e:
            import traceback
            traceback.print_exc()
            failed += 1

    await source.close()
    await gemma.close()

    return ScanResponse(
        date=req.date,
        source=req.source,
        chunks_scanned=scanned,
        chunks_classified=classified,
        chunks_failed=failed,
        chunks=chunks,
        classifications=classifications,
    )
