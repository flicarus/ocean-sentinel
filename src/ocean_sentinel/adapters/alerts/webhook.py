import httpx
import structlog
from ocean_sentinel.domain.models import Alert, DetectionEvent
from ocean_sentinel.config import Settings
from ocean_sentinel.exceptions import AlertDeliveryError

log = structlog.get_logger(__name__)


class WebhookAdapter:

    def __init__(self, settings: Settings):
        self.settings = settings

    async def send(self, alert: Alert, event: DetectionEvent) -> Alert:
        if not alert.recipient:
            log.warning("webhook_no_url")
            return alert

        payload = {
            "event_id": event.id,
            "timestamp": event.timestamp.isoformat(),
            "location": {
                "lat": event.location.lat,
                "lon": event.location.lon,
            },
            "threat_level": event.threat_level.value,
            "confidence": event.confidence,
            "reasoning": event.classification_reasoning,
            "intentional_disabling": any(
                g.intentional_disabling for g in event.ais_gaps
            ),
            "in_mpa": any(g.in_mpa for g in event.ais_gaps),
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(alert.recipient, json=payload)
                response.raise_for_status()
                log.info("webhook_sent", url=alert.recipient, status=response.status_code)
        except httpx.HTTPStatusError as e:
            log.error("webhook_http_error", status=e.response.status_code, url=alert.recipient)
            raise AlertDeliveryError(
                code="webhook_error",
                message=f"Webhook failed: {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            log.error("webhook_request_error", url=alert.recipient, detail=str(e))
            raise AlertDeliveryError(
                code="webhook_request_error",
                message=f"Webhook request failed: {e}"
            ) from e

        return alert