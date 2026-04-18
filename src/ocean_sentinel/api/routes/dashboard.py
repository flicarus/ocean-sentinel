from fastapi import APIRouter, Request
from ocean_sentinel.api.schemas import GeoJSONResponse, GeoJSONFeature

router = APIRouter()


@router.get("/geojson", response_model=GeoJSONResponse)
async def get_geojson(request: Request):
    """Returns all detection events as GeoJSON for Leaflet map."""

    from ocean_sentinel.api.routes.events import _mock_events
    events = _mock_events()

    features = []
    for event in events:
        feature = GeoJSONFeature(
            geometry={
                "type": "Point",
                "coordinates": [event.location.lon, event.location.lat]
            },
            properties={
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "threat_level": event.threat_level.value,
                "confidence": event.confidence,
                "reasoning": event.classification_reasoning,
                "vessel_name": event.ais_gaps[0].vessel_name if event.ais_gaps else None,
                "flag_state": event.ais_gaps[0].flag_state if event.ais_gaps else None,
                "intentional_disabling": event.ais_gaps[0].intentional_disabling if event.ais_gaps else None,
                "in_mpa": event.ais_gaps[0].in_mpa if event.ais_gaps else None,
                "gap_duration_hours": event.ais_gaps[0].gap_duration_hours if event.ais_gaps else None,
            }
        )
        features.append(feature)

    return GeoJSONResponse(features=features)