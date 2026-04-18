import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Request, HTTPException
from ocean_sentinel.api.schemas import AlertSchema
from ocean_sentinel.domain.models import Alert, DetectionEvent
from ocean_sentinel.domain.enums import AlertChannel, AlertStatus, ThreatLevel
from ocean_sentinel.adapters.alerts.sendgrid import SendGridAdapter
from ocean_sentinel.exceptions import AlertDeliveryError
import structlog

log = structlog.get_logger(__name__)

router = APIRouter()


async def _send_alert_for_event(event: DetectionEvent, request: Request) -> None:
    """Send email alert and persist it for a given detection event."""
    settings = request.app.state.settings
    store = request.app.state.store

    alert = Alert(
        id=str(uuid.uuid4()),
        event_id=event.id,
        channel=AlertChannel.EMAIL,
        recipient=settings.sendgrid_from_email,
        sent_at=None,
        status=AlertStatus.PENDING,
    )

    sendgrid = SendGridAdapter(settings)

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
        log.info("alert_sent", event_id=event.id, threat=event.threat_level.value)
    except AlertDeliveryError as e:
        alert = Alert(
            id=alert.id,
            event_id=alert.event_id,
            channel=alert.channel,
            recipient=alert.recipient,
            sent_at=None,
            status=AlertStatus.FAILED,
            failure_reason=str(e),
        )
        log.error("alert_failed", event_id=event.id, reason=str(e))

    await store.save_alert(alert)


@router.get("/", response_model=dict)
async def list_alerts(request: Request):
    store = request.app.state.store
    alerts = await store.list_alerts()
    return {
        "alerts": [
            AlertSchema(
                id=a.id,
                event_id=a.event_id,
                channel=a.channel,
                recipient=a.recipient,
                sent_at=a.sent_at,
                status=a.status,
                failure_reason=a.failure_reason,
            )
            for a in alerts
        ],
        "total": len(alerts),
    }


@router.post("/trigger/{event_id}", response_model=AlertSchema)
async def trigger_alert(event_id: str, request: Request):
    """Manually trigger an alert for a specific event."""
    store = request.app.state.store
    event = await store.get_event(event_id)

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    await _send_alert_for_event(event, request)

    alerts = await store.list_alerts(event_id=event_id)
    if not alerts:
        raise HTTPException(status_code=500, detail="Alert not saved")

    latest = alerts[0]
    return AlertSchema(
        id=latest.id,
        event_id=latest.event_id,
        channel=latest.channel,
        recipient=latest.recipient,
        sent_at=latest.sent_at,
        status=latest.status,
        failure_reason=latest.failure_reason,
    )