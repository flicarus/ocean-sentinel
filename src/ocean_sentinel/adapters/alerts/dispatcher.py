"""Multi-channel alert dispatcher.

Given a DetectionEvent + site config (with channels + recipients + minimum
threat level), routes the alert through every configured channel:

  - EMAIL via SendGrid
  - SMS via Twilio
  - WEBHOOK (generic POST)

Skips channels for which credentials are missing. Records each dispatch
attempt to the store (if available). Returns a list of Alert records
with their final status.

This is the only place that knows about "channels"; surfaces (CLI, FastAPI,
mock-stream demo) call dispatch_alerts(event, site_cfg) and receive the
list back.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

from ocean_sentinel.config import Settings
from ocean_sentinel.domain.enums import AlertChannel, AlertStatus, ThreatLevel
from ocean_sentinel.domain.models import Alert, DetectionEvent
from ocean_sentinel.exceptions import AlertDeliveryError

from .sendgrid import SendGridAdapter
from .twilio import TwilioAdapter
from .webhook import WebhookAdapter

log = structlog.get_logger(__name__)


# Ordered tier list — used to compare "is event severe enough"
_LADDER = [
    ThreatLevel.NONE, ThreatLevel.LOW, ThreatLevel.MEDIUM,
    ThreatLevel.HIGH, ThreatLevel.CRITICAL,
]


def _meets_min(level: ThreatLevel, minimum: ThreatLevel) -> bool:
    return _LADDER.index(level) >= _LADDER.index(minimum)


@dataclass
class ChannelConfig:
    channel: AlertChannel
    recipient: str
    min_threat: ThreatLevel = ThreatLevel.LOW


@dataclass
class SiteAlertPolicy:
    """Per-site alert routing: which channels, who, at what threat level."""
    site_id: str
    channels: list[ChannelConfig]


async def dispatch_alerts(
    event: DetectionEvent,
    policy: SiteAlertPolicy,
    settings: Settings,
    *,
    store=None,
) -> list[Alert]:
    """Route the event through every configured channel meeting min_threat.

    Returns the list of Alert records (with their final status). If a
    channel's credentials are missing, the corresponding Alert is marked
    FAILED with a clear reason — never silently dropped.
    """
    sent: list[Alert] = []
    for ch in policy.channels:
        if not _meets_min(event.threat_level, ch.min_threat):
            log.info(
                "alert_skipped_below_threshold",
                channel=ch.channel.value,
                event_threat=event.threat_level.value,
                channel_min=ch.min_threat.value,
            )
            continue

        alert = Alert(
            id=str(uuid.uuid4()),
            event_id=event.id,
            channel=ch.channel,
            recipient=ch.recipient,
            sent_at=None,
            status=AlertStatus.PENDING,
        )

        try:
            # Credential pre-check — adapters silently no-op without creds,
            # so we raise here so the dispatcher path below records FAILED
            # with a clear reason rather than misreporting SENT.
            if ch.channel == AlertChannel.EMAIL and not settings.sendgrid_api_key:
                raise AlertDeliveryError(
                    code="sendgrid_not_configured",
                    message="SENDGRID_API_KEY not set",
                )
            if ch.channel == AlertChannel.SMS and not (
                settings.twilio_account_sid
                and settings.twilio_auth_token
                and settings.twilio_from_number
            ):
                raise AlertDeliveryError(
                    code="twilio_not_configured",
                    message="Twilio creds (sid/token/from_number) not set",
                )

            if ch.channel == AlertChannel.EMAIL:
                await SendGridAdapter(settings).send(alert, event)
            elif ch.channel == AlertChannel.SMS:
                await TwilioAdapter(settings).send(alert, event)
            elif ch.channel == AlertChannel.WEBHOOK:
                await WebhookAdapter(settings).send(alert, event)
            else:
                raise AlertDeliveryError(
                    code="unknown_channel",
                    message=f"no adapter for channel {ch.channel}",
                )
            alert = Alert(
                id=alert.id,
                event_id=alert.event_id,
                channel=alert.channel,
                recipient=alert.recipient,
                sent_at=datetime.now(timezone.utc),
                status=AlertStatus.SENT,
            )
            log.info("alert_sent", channel=ch.channel.value, recipient=ch.recipient)
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
            log.error(
                "alert_failed",
                channel=ch.channel.value,
                recipient=ch.recipient,
                reason=str(e),
            )

        sent.append(alert)
        if store is not None:
            try:
                await store.save_alert(alert)
            except Exception as e:
                log.warning("alert_persist_failed", reason=str(e))

    return sent
