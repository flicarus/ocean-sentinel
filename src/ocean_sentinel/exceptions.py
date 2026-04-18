from typing import Any

class OceanSentinelError(Exception):
    """Base for all project-specific exceptions. API middleware catches this."""

    def __init__(
            self, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        self.code = code
        self.message = message
        self.details = details 
        super().__init__(message)


class DataSourceError(OceanSentinelError):
    """Any upstream API / S3 failure."""


class MBARIError(DataSourceError):
    """MBARI S3 bucket access or audio download failure."""


class GFWError(DataSourceError):
    """Global Fishing Watch API failure."""


class CopernicusError(DataSourceError):
    """Copernicus Marine API failure."""


# --- Processing ---

class ProcessingError(OceanSentinelError):
    """Audio analysis, correlation, or classification failure."""


class SpectrogramError(ProcessingError):
    """Failed to generate spectrogram from audio."""


class CorrelationError(ProcessingError):
    """Failed to correlate data sources."""


class ClassificationError(ProcessingError):
    """Gemma returned invalid or unparseable output."""


# --- Alerts ---

class AlertDeliveryError(OceanSentinelError):
    """Alert sending failure."""


class EmailDeliveryError(AlertDeliveryError):
    """SendGrid failure."""


class SMSDeliveryError(AlertDeliveryError):
    """Twilio failure."""