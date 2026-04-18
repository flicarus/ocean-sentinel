from pydantic import BaseModel
from datetime import datetime
from ocean_sentinel.domain.enums import ThreatLevel, AlertChannel, AlertStatus


class GeoPointSchema(BaseModel):
    lat: float
    lon: float


class AISGapSchema(BaseModel):
    vessel_id: str
    vessel_name: str | None
    flag_state: str | None
    last_known_position: GeoPointSchema
    gap_start: datetime
    gap_end: datetime | None
    gap_duration_hours: float
    intentional_disabling: bool | None
    in_mpa: bool | None


class OceanConditionsSchema(BaseModel):
    current_speed_ms: float | None
    current_direction_deg: float | None
    sea_surface_temp_c: float | None


class DetectionEventSchema(BaseModel):
    id: str
    timestamp: datetime
    location: GeoPointSchema
    threat_level: ThreatLevel
    confidence: float
    classification_reasoning: str
    ais_gaps: list[AISGapSchema]
    ocean_conditions: OceanConditionsSchema | None


class DetectionEventListResponse(BaseModel):
    events: list[DetectionEventSchema]
    total: int


class AlertSchema(BaseModel):
    id: str
    event_id: str
    channel: AlertChannel
    recipient: str
    sent_at: datetime | None
    status: AlertStatus
    failure_reason: str | None


class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: dict
    properties: dict


class GeoJSONResponse(BaseModel):
    type: str = "FeatureCollection"
    features: list[GeoJSONFeature]