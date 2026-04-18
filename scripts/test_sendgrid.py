import asyncio
from dotenv import load_dotenv
load_dotenv()

from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import Alert, DetectionEvent, GeoPoint
from ocean_sentinel.domain.enums import ThreatLevel, AlertChannel, AlertStatus
from ocean_sentinel.adapters.alerts.sendgrid import SendGridAdapter
from datetime import datetime, timezone
import uuid

async def test():
    settings = Settings()
    adapter = SendGridAdapter(settings)

    event = DetectionEvent(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        location=GeoPoint(lat=36.7, lon=-122.0),
        threat_level=ThreatLevel.HIGH,
        confidence=0.91,
        classification_reasoning="Test alert from Ocean Sentinel — acoustic anomaly + AIS gap detected in Monterey Bay.",
        ais_gaps=[],
        audio_segment=None,
        ocean_conditions=None,
        raw_model_output="test",
    )

    alert = Alert(
        id=str(uuid.uuid4()),
        event_id=event.id,
        channel=AlertChannel.EMAIL,
        recipient="rychlewskibusiness@gmail.com",
        status=AlertStatus.PENDING,
        sent_at=None,
    )

    result = await adapter.send(alert, event)
    print("Done:", result)

asyncio.run(test())