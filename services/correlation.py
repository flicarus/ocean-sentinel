from __future__ import annotations

from datetime import timedelta

import structlog

from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import (
    AudioSegment,
    OceanConditions,
    TimeWindow,
    AISGapEvent,
)
from ocean_sentinel.domain.protocols import VesselTracker, OceanDataSource
from ocean_sentinel.exceptions import CorrelationError

log = structlog.get_logger()


class CorrelationService:
    """Correlates acoustic anomalies with vessel tracking and ocean data.

    Connects the dots between 'I hear an engine' and
    'a vessel disappeared from radar nearby'.
    """

    def __init__(
        self,
        vessel_tracker: VesselTracker,
        ocean_source: OceanDataSource,
        settings: Settings,
    ) -> None:
        self._vessels = vessel_tracker
        self._ocean = ocean_source
        self._radius_km = settings.correlation_radius_km
        self._time_hours = settings.correlation_time_window_hours

    async def correlate(
        self,
        audio: AudioSegment,
        features: dict,
    ) -> tuple[list[AISGapEvent], OceanConditions | None]:
        """Find AIS gaps and ocean conditions matching an acoustic anomaly.

        Returns correlated evidence — what was happening around this
        hydrophone at the time it picked up suspicious audio.
        """
        location = audio.location
        midpoint = audio.time_window.start + (
            audio.time_window.end - audio.time_window.start
        ) / 2

        search_window = TimeWindow(
            start=midpoint - timedelta(hours=self._time_hours),
            end=midpoint + timedelta(hours=self._time_hours),
        )

        log.info(
            "correlation_started",
            lat=location.lat,
            lon=location.lon,
            radius_km=self._radius_km,
            window_hours=self._time_hours * 2,
            engine_dominant=features.get("is_engine_band_dominant"),
        )

        ais_gaps: list[AISGapEvent] = []
        ocean: OceanConditions | None = None

        try:
            ais_gaps = await self._vessels.get_ais_gaps(
                region=location,
                radius_km=self._radius_km,
                time_window=search_window,
            )
        except Exception as e:
            log.error("correlation_ais_failed", error=str(e))
            raise CorrelationError(
                code="ais_correlation_failed",
                message="Failed to fetch AIS gaps for correlation",
                details={"error": str(e)},
            ) from e

        try:
            ocean = await self._ocean.get_conditions(
                location=location,
                timestamp=midpoint,
            )
        except Exception as e:
            log.warning("correlation_ocean_failed", error=str(e))

        log.info(
            "correlation_complete",
            ais_gaps_found=len(ais_gaps),
            has_ocean_data=ocean is not None,
        )

        return ais_gaps, ocean
