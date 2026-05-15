from __future__ import annotations

from ocean_sentinel.domain.enums import ThreatLevel
from ocean_sentinel.domain.models import AISGapEvent


def compute_threat_level(
    base_level: ThreatLevel, ais_gaps: list[AISGapEvent]
) -> ThreatLevel:
    """Escalate to CRITICAL when any correlated AIS gap vessel was inside an MPA.

    A no-detection (NONE) is never escalated — there must be an acoustic signal
    before MPA context can push the threat higher.
    """
    if base_level is ThreatLevel.NONE:
        return ThreatLevel.NONE
    if any(gap.in_mpa for gap in ais_gaps):
        return ThreatLevel.CRITICAL
    return base_level
