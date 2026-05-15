"""Twilio SMS adapter — sends short threat-level alerts to ranger phones.

Mirrors SendGridAdapter's contract. Constructor takes Settings, .send()
takes Alert + DetectionEvent and posts to Twilio's Messages API.

Auth: HTTP basic with account_sid as username, auth_token as password.
SMS body capped to ~160 chars so the alert fits one segment — rangers
on field radios sometimes pay per segment.
"""
from __future__ import annotations

import base64
import httpx
import structlog

from ocean_sentinel.domain.models import Alert, DetectionEvent
from ocean_sentinel.config import Settings
from ocean_sentinel.exceptions import AlertDeliveryError

log = structlog.get_logger(__name__)


class TwilioAdapter:
    BASE_URL = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"

    def __init__(self, settings: Settings) -> None:
        self.account_sid = settings.twilio_account_sid
        self.auth_token = settings.twilio_auth_token
        self.from_number = settings.twilio_from_number

    async def send(self, alert: Alert, event: DetectionEvent) -> Alert:
        if not (self.account_sid and self.auth_token and self.from_number):
            log.warning("twilio_not_configured")
            return alert
        if not alert.recipient:
            log.warning("twilio_no_recipient")
            return alert

        # Short body — Twilio segments at 160 chars. We aim for one segment.
        ts = event.timestamp.strftime("%Y-%m-%d %H:%MZ")
        body = (
            f"OCEAN SENTINEL {event.threat_level.value} · {ts}"
            f"\n{event.location.lat:.3f},{event.location.lon:.3f}"
            f"\nconf {event.confidence*100:.0f}% · {event.classification_reasoning[:60]}"
        )
        body = body[:320]  # hard cap (~2 segments)

        url = self.BASE_URL.format(sid=self.account_sid)
        creds = base64.b64encode(
            f"{self.account_sid}:{self.auth_token}".encode()
        ).decode()

        data = {
            "From": self.from_number,
            "To": alert.recipient,
            "Body": body,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    url,
                    data=data,
                    headers={"Authorization": f"Basic {creds}"},
                )
                response.raise_for_status()
                log.info(
                    "alert_sms_sent",
                    recipient=alert.recipient,
                    sid=response.json().get("sid"),
                )
        except httpx.HTTPStatusError as e:
            log.error("twilio_error", status=e.response.status_code, detail=e.response.text[:200])
            raise AlertDeliveryError(
                code="twilio_error",
                message=f"Twilio failed: {e.response.status_code}: {e.response.text[:200]}",
            ) from e
        except httpx.RequestError as e:
            log.error("twilio_request_error", detail=str(e))
            raise AlertDeliveryError(
                code="twilio_request_error",
                message=f"Twilio request failed: {e}",
            ) from e

        return alert
