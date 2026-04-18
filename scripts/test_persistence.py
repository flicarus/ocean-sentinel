import asyncio
from datetime import datetime, timezone
from ocean_sentinel.adapters.persistence import SQLiteEventStore
from ocean_sentinel.domain.models import (
    DetectionEvent, Alert, GeoPoint, AISGapEvent, OceanConditions
)
from ocean_sentinel.domain.enums import ThreatLevel, AlertChannel, AlertStatus
import uuid

async def test():
    store = SQLiteEventStore("data/test_ocean_sentinel.db")
    await store.init()
    print("✅ DB initialized")

    # Create a fake detection event
    event = DetectionEvent(
        id=str(uuid.uuid4()),
        timestamp=datetime(2024, 2, 1, 12, 0, tzinfo=timezone.utc),
        location=GeoPoint(lat=48.0, lon=-5.0),
        threat_level=ThreatLevel.HIGH,
        confidence=0.87,
        classification_reasoning="Engine noise detected, vessel went dark in MPA",
        audio_segment=None,
        ais_gaps=[
            AISGapEvent(
                vessel_id="abc123",
                vessel_name="DARK TRAWLER",
                flag_state="CHN",
                last_known_position=GeoPoint(lat=48.1, lon=-5.1),
                gap_start=datetime(2024, 2, 1, 10, 0, tzinfo=timezone.utc),
                gap_end=None,
                gap_duration_hours=14.5,
                intentional_disabling=True,
                in_mpa=True,
            )
        ],
        ocean_conditions=OceanConditions(
            location=GeoPoint(lat=48.0, lon=-5.0),
            timestamp=datetime(2024, 2, 1, 12, 0, tzinfo=timezone.utc),
            sea_surface_temp_c=11.72,
            current_speed_ms=0.014,
            current_direction_deg=339.6,
        ),
        raw_model_output={"model": "gemma4", "tokens": 512},
    )

    # Test save + get
    await store.save_event(event)
    print("✅ Event saved")

    fetched = await store.get_event(event.id)
    assert fetched is not None
    assert fetched.threat_level == ThreatLevel.HIGH
    assert fetched.confidence == 0.87
    assert len(fetched.ais_gaps) == 1
    assert fetched.ocean_conditions.sea_surface_temp_c == 11.72
    print("✅ Event fetched and verified")

    # Test list_events
    events = await store.list_events()
    assert len(events) >= 1
    print(f"✅ list_events returned {len(events)} event(s)")

    # Test min_threat filter
    high_events = await store.list_events(min_threat=ThreatLevel.HIGH)
    assert all(e.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL) for e in high_events)
    print(f"✅ min_threat filter works — {len(high_events)} HIGH+ event(s)")

    # Test save_alert
    alert = Alert(
        id=str(uuid.uuid4()),
        event_id=event.id,
        channel=AlertChannel.EMAIL,
        recipient="rychlewskibusiness@gmail.com",
        sent_at=datetime(2024, 2, 1, 12, 1, tzinfo=timezone.utc),
        status=AlertStatus.SENT,
        failure_reason=None,
    )
    await store.save_alert(alert)
    print("✅ Alert saved")

    alerts = await store.list_alerts(event_id=event.id)
    assert len(alerts) == 1
    assert alerts[0].status == AlertStatus.SENT
    print("✅ Alert fetched and verified")

    await store.close()
    print("\n🎉 All tests passed!")

asyncio.run(test())