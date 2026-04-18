import json
import uuid
import aiosqlite
from datetime import datetime, timezone
from typing import Any
from ocean_sentinel.domain.models import (
    Alert, AISGapEvent, DetectionEvent, GeoPoint,
    OceanConditions, TimeWindow,
)
from ocean_sentinel.domain.enums import AlertChannel, AlertStatus, ThreatLevel

DB_PATH = "data/ocean_sentinel.db"

DDL = """
CREATE TABLE IF NOT EXISTS detection_events (
    id                       TEXT PRIMARY KEY,
    timestamp                TEXT NOT NULL,
    lat                      REAL NOT NULL,
    lon                      REAL NOT NULL,
    threat_level             TEXT NOT NULL,
    confidence               REAL NOT NULL,
    classification_reasoning TEXT NOT NULL,
    ais_gaps                 TEXT NOT NULL,
    ocean_conditions         TEXT,
    raw_model_output         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id             TEXT PRIMARY KEY,
    event_id       TEXT NOT NULL,
    channel        TEXT NOT NULL,
    recipient      TEXT NOT NULL,
    sent_at        TEXT,
    status         TEXT NOT NULL,
    failure_reason TEXT,
    FOREIGN KEY (event_id) REFERENCES detection_events(id)
);

CREATE TABLE IF NOT EXISTS feedback (
    id                     TEXT PRIMARY KEY,
    event_id               TEXT NOT NULL,
    correct                BOOLEAN NOT NULL,
    corrected_threat_level TEXT,
    created_at             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_event ON feedback(event_id);
"""

# ── serialisation helpers ──────────────────────────────────────────────────

def _dt(ts: datetime) -> str:
    return ts.isoformat()

def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)

def _ais_gaps_to_json(gaps: list[AISGapEvent]) -> str:
    return json.dumps([
        {
            "vessel_id": g.vessel_id,
            "vessel_name": g.vessel_name,
            "flag_state": g.flag_state,
            "lat": g.last_known_position.lat,
            "lon": g.last_known_position.lon,
            "gap_start": _dt(g.gap_start),
            "gap_end": _dt(g.gap_end) if g.gap_end else None,
            "gap_duration_hours": g.gap_duration_hours,
            "intentional_disabling": g.intentional_disabling,
            "in_mpa": g.in_mpa,
        }
        for g in gaps
    ])

def _ais_gaps_from_json(raw: str) -> list[AISGapEvent]:
    return [
        AISGapEvent(
            vessel_id=g["vessel_id"],
            vessel_name=g["vessel_name"],
            flag_state=g["flag_state"],
            last_known_position=GeoPoint(lat=g["lat"], lon=g["lon"]),
            gap_start=_parse_dt(g["gap_start"]),
            gap_end=_parse_dt(g["gap_end"]) if g["gap_end"] else None,
            gap_duration_hours=g["gap_duration_hours"],
            intentional_disabling=g["intentional_disabling"],
            in_mpa=g["in_mpa"],
        )
        for g in json.loads(raw)
    ]

def _ocean_to_json(oc: OceanConditions | None) -> str | None:
    if oc is None:
        return None
    return json.dumps({
        "lat": oc.location.lat,
        "lon": oc.location.lon,
        "timestamp": _dt(oc.timestamp),
        "sea_surface_temp_c": oc.sea_surface_temp_c,
        "current_speed_ms": oc.current_speed_ms,
        "current_direction_deg": oc.current_direction_deg,
    })

def _ocean_from_json(raw: str | None) -> OceanConditions | None:
    if raw is None:
        return None
    d = json.loads(raw)
    return OceanConditions(
        location=GeoPoint(lat=d["lat"], lon=d["lon"]),
        timestamp=_parse_dt(d["timestamp"]),
        sea_surface_temp_c=d["sea_surface_temp_c"],
        current_speed_ms=d["current_speed_ms"],
        current_direction_deg=d["current_direction_deg"],
    )

def _row_to_event(row: aiosqlite.Row) -> DetectionEvent:
    return DetectionEvent(
        id=row["id"],
        timestamp=_parse_dt(row["timestamp"]),
        location=GeoPoint(lat=row["lat"], lon=row["lon"]),
        threat_level=ThreatLevel(row["threat_level"]),
        confidence=row["confidence"],
        classification_reasoning=row["classification_reasoning"],
        ais_gaps=_ais_gaps_from_json(row["ais_gaps"]),
        ocean_conditions=_ocean_from_json(row["ocean_conditions"]),
        raw_model_output=json.loads(row["raw_model_output"]),
        audio_segment=None,  # not persisted — too large
    )

def _row_to_alert(row: aiosqlite.Row) -> Alert:
    return Alert(
        id=row["id"],
        event_id=row["event_id"],
        channel=AlertChannel(row["channel"]),
        recipient=row["recipient"],
        sent_at=_parse_dt(row["sent_at"]) if row["sent_at"] else None,
        status=AlertStatus(row["status"]),
        failure_reason=row["failure_reason"],
    )


# ── EventStore ─────────────────────────────────────────────────────────────

class SQLiteEventStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(DDL)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def save_event(self, event: DetectionEvent) -> None:
        await self._db.execute(
            """
            INSERT OR REPLACE INTO detection_events
                (id, timestamp, lat, lon, threat_level, confidence,
                 classification_reasoning, ais_gaps, ocean_conditions,
                 raw_model_output)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event.id,
                _dt(event.timestamp),
                event.location.lat,
                event.location.lon,
                event.threat_level.value,
                event.confidence,
                event.classification_reasoning,
                _ais_gaps_to_json(event.ais_gaps),
                _ocean_to_json(event.ocean_conditions),
                json.dumps(event.raw_model_output),
            ),
        )
        await self._db.commit()

    async def get_event(self, event_id: str) -> DetectionEvent | None:
        cursor = await self._db.execute(
            "SELECT * FROM detection_events WHERE id = ?", (event_id,)
        )
        row = await cursor.fetchone()
        return _row_to_event(row) if row else None

    async def list_events(
        self,
        time_window: TimeWindow | None = None,
        min_threat: ThreatLevel | None = None,
    ) -> list[DetectionEvent]:
        threat_order = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

        query = "SELECT * FROM detection_events WHERE 1=1"
        params: list[Any] = []

        if time_window:
            query += " AND timestamp >= ? AND timestamp <= ?"
            params += [_dt(time_window.start), _dt(time_window.end)]

        if min_threat:
            placeholders = ",".join(
                f"'{t}'" for t in threat_order[threat_order.index(min_threat.value):]
            )
            query += f" AND threat_level IN ({placeholders})"

        query += " ORDER BY timestamp DESC"

        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_event(r) for r in rows]

    async def save_alert(self, alert: Alert) -> None:
        await self._db.execute(
            """
            INSERT OR REPLACE INTO alerts
                (id, event_id, channel, recipient, sent_at, status, failure_reason)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                alert.id,
                alert.event_id,
                alert.channel.value,
                alert.recipient,
                _dt(alert.sent_at) if alert.sent_at else None,
                alert.status.value,
                alert.failure_reason,
            ),
        )
        await self._db.commit()

    async def list_alerts(self, event_id: str | None = None) -> list[Alert]:
        if event_id:
            cursor = await self._db.execute(
                "SELECT * FROM alerts WHERE event_id = ? ORDER BY sent_at DESC",
                (event_id,)
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM alerts ORDER BY sent_at DESC"
            )
        rows = await cursor.fetchall()
        return [_row_to_alert(r) for r in rows]

    async def save_feedback(
        self, event_id: str, correct: bool, corrected_threat_level: str | None
    ) -> str:
        fb_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT INTO feedback (id, event_id, correct, corrected_threat_level, created_at)
            VALUES (?,?,?,?,?)
            """,
            (fb_id, event_id, correct, corrected_threat_level, created_at),
        )
        await self._db.commit()
        return fb_id

    async def list_feedback(self, event_id: str | None = None) -> list[dict]:
        if event_id:
            cursor = await self._db.execute(
                "SELECT * FROM feedback WHERE event_id = ? ORDER BY created_at DESC",
                (event_id,)
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC"
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]