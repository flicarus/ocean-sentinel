import httpx
import structlog
from ocean_sentinel.domain.models import Alert, DetectionEvent
from ocean_sentinel.config import Settings
from ocean_sentinel.exceptions import AlertDeliveryError

log = structlog.get_logger(__name__)


class SendGridAdapter:
    BASE_URL = "https://api.sendgrid.com/v3/mail/send"

    def __init__(self, settings: Settings):
        self.api_key = settings.sendgrid_api_key
        self.from_email = settings.sendgrid_from_email
        self.from_name = settings.sendgrid_from_name

    async def send(self, alert: Alert, event: DetectionEvent) -> Alert:
        if not self.api_key:
            log.warning("sendgrid_not_configured")
            return alert

        subject = f"🌊 Ocean Sentinel Alert — {event.threat_level.value.upper()} threat detected"

        body = f"""
Ocean Sentinel Detection Event
===============================
Time:       {event.timestamp.strftime("%Y-%m-%d %H:%M UTC")}
Location:   {event.location.lat:.4f}°, {event.location.lon:.4f}°
Threat:     {event.threat_level.value.upper()}
Confidence: {event.confidence * 100:.0f}%

Reasoning:
{event.classification_reasoning}

---
Ocean Sentinel — Multimodal Ocean Intelligence
        """.strip()

        payload = {
            "personalizations": [
                {"to": [{"email": alert.recipient}]}
            ],
            "from": {"email": self.from_email, "name": self.from_name},
            "subject": subject,
            "content": [{"type": "text/plain", "value": body}]
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BASE_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }
                )
                response.raise_for_status()
                log.info("alert_email_sent", recipient=alert.recipient)
        except httpx.HTTPStatusError as e:
            log.error("sendgrid_error", status=e.response.status_code, detail=e.response.text)
            raise AlertDeliveryError(
                code="sendgrid_error",
                message=f"SendGrid failed: {e.response.status_code}"
            ) from e

        return alert