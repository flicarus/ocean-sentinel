from enum import Enum


class ThreatLevel(str, Enum):
    """Classification output from Gemma — how serious is this detection."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertChannel(str, Enum):
    """How we deliver the alert."""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"


class AlertStatus(str, Enum):
    """Lifecycle of an alert."""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"