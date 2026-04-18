import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI needed — render to buffer only
import matplotlib.pyplot as plt
import numpy as np

from fastapi import APIRouter, Request, Query, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from ocean_sentinel.api.schemas import (
    DetectionEventSchema, DetectionEventListResponse,
    GeoPointSchema, AISGapSchema, OceanConditionsSchema,
)
from ocean_sentinel.domain.enums import ThreatLevel
from ocean_sentinel.domain.models import DetectionEvent

router = APIRouter()


def _to_schema(event: DetectionEvent) -> DetectionEventSchema:
    return DetectionEventSchema(
        id=event.id,
        timestamp=event.timestamp,
        location=GeoPointSchema(lat=event.location.lat, lon=event.location.lon),
        threat_level=event.threat_level,
        confidence=event.confidence,
        classification_reasoning=event.classification_reasoning,
        ais_gaps=[
            AISGapSchema(
                vessel_id=g.vessel_id,
                vessel_name=g.vessel_name,
                flag_state=g.flag_state,
                last_known_position=GeoPointSchema(
                    lat=g.last_known_position.lat,
                    lon=g.last_known_position.lon,
                ),
                gap_start=g.gap_start,
                gap_end=g.gap_end,
                gap_duration_hours=g.gap_duration_hours,
                intentional_disabling=g.intentional_disabling,
                in_mpa=g.in_mpa,
            )
            for g in event.ais_gaps
        ],
        ocean_conditions=OceanConditionsSchema(
            sea_surface_temp_c=event.ocean_conditions.sea_surface_temp_c,
            current_speed_ms=event.ocean_conditions.current_speed_ms,
            current_direction_deg=event.ocean_conditions.current_direction_deg,
        ) if event.ocean_conditions else None,
    )


@router.get("/", response_model=DetectionEventListResponse)
async def list_events(
    request: Request,
    threat_level: ThreatLevel | None = Query(default=None),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0),
):
    store = request.app.state.store
    events = await store.list_events(min_threat=threat_level)
    paginated = events[offset:offset + limit]

    return DetectionEventListResponse(
        events=[_to_schema(e) for e in paginated],
        total=len(events),
    )


@router.get("/{event_id}", response_model=DetectionEventSchema)
async def get_event(event_id: str, request: Request):
    store = request.app.state.store
    event = await store.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return _to_schema(event)


class FeedbackRequest(BaseModel):
    correct: bool
    corrected_threat_level: str | None = None


@router.post("/{event_id}/feedback")
async def submit_feedback(event_id: str, req: FeedbackRequest, request: Request):
    if req.corrected_threat_level is not None:
        try:
            ThreatLevel(req.corrected_threat_level)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid threat level: {req.corrected_threat_level}. "
                       f"Must be one of {[t.value for t in ThreatLevel]}",
            )
    store = request.app.state.store
    fb_id = await store.save_feedback(event_id, req.correct, req.corrected_threat_level)
    return {"id": fb_id, "event_id": event_id, "correct": req.correct}


@router.get("/{event_id}/feedback")
async def list_event_feedback(event_id: str, request: Request):
    return await request.app.state.store.list_feedback(event_id=event_id)


@router.get("/{event_id}/spectrogram")
async def event_spectrogram(event_id: str):
    npy_path = Path(f"data/spectrograms/{event_id}.npy")
    if not npy_path.exists():
        raise HTTPException(status_code=404, detail="Spectrogram not found for this event")

    data = np.load(npy_path)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(data, aspect="auto", origin="lower", cmap="magma")
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")