"""Decision-tier classification — combines acoustic CNN output with AIS context.

A `tier` describes WHAT the detection event IS, before any threat-level
escalation. Tier requires both modalities (acoustic + AIS) to disambiguate:

  - DARK_VESSEL: acoustic strong AND no AIS in radius (true illegal-fishing
    pattern — vessel either has no AIS or deliberately turned it off)

  - GONE_DARK_VESSEL: acoustic strong AND a vessel was broadcasting AIS
    in the area within the last N minutes but stopped (classic AIS-off
    playbook: turn off AIS to enter MPA, do illegal fishing, turn back on)

  - CONFIRMED_VESSEL: acoustic strong AND AIS-broadcasting vessel present
    (vessel is lawfully advertising its position)

  - ACOUSTIC_ONLY_LOW: acoustic borderline, AIS state unclear

  - AMBIENT: no acoustic vessel signal

  - UNCERTAIN: model abstained (high evidential uncertainty)

This is the only place that decides tier; threat_scoring.py escalates
the tier to a ThreatLevel using MPA + sensitivity + CPA + nighttime context.
"""
from __future__ import annotations

from dataclasses import dataclass


# Probability thresholds — calibrated empirically on v7.6
SHIP_PROB_STRONG = 0.85
SHIP_PROB_MEDIUM = 0.65
UNCERTAINTY_MAX = 0.25


@dataclass(frozen=True)
class AISContext:
    """AIS state for the chunk's location + time window.

    `ais_vessels_in_radius`: count of AIS-broadcasting vessels within the
        configured radius (usually 10 km) during the chunk's hour.
    `recently_gone_dark_count`: number of vessels that were broadcasting
        AIS within `gone_dark_lookback_minutes` ago but stopped. Each
        such vessel is a potential illegal-fishing trigger.
    `nearest_vessel_cpa_km`: CPA distance (km) of the nearest currently
        broadcasting vessel, if any.
    """
    ais_vessels_in_radius: int = 0
    recently_gone_dark_count: int = 0
    nearest_vessel_cpa_km: float | None = None
    gone_dark_lookback_minutes: int = 30


def decide_tier(
    *,
    ship_prob: float,
    uncertainty: float,
    site_threshold: float,
    ais: AISContext,
) -> tuple[str, str]:
    """Map (CNN output + AIS state) → (decision_tier, severity).

    Returns:
        (tier, severity) where tier is one of:
            DARK_VESSEL, GONE_DARK_VESSEL, CONFIRMED_VESSEL,
            ACOUSTIC_ONLY_LOW, AMBIENT, UNCERTAIN
        and severity is one of NONE, LOW, MEDIUM, HIGH.
    """
    # Honest abstention first
    if uncertainty > UNCERTAINTY_MAX:
        return "UNCERTAIN", "MEDIUM"

    # No acoustic signal — easy bucket
    if ship_prob < site_threshold:
        return "AMBIENT", "NONE"

    # Strong acoustic — branch on AIS state
    if ship_prob >= SHIP_PROB_STRONG:
        # Strong acoustic + a vessel just went dark = highest-confidence
        # illegal-fishing pattern. We surface this as a distinct tier so
        # threat_scoring can escalate it differently than vanilla
        # DARK_VESSEL (which could be a small craft without AIS-mandate).
        if ais.recently_gone_dark_count >= 1:
            return "GONE_DARK_VESSEL", "HIGH"
        if ais.ais_vessels_in_radius == 0:
            return "DARK_VESSEL", "HIGH"
        return "CONFIRMED_VESSEL", "LOW"

    # Medium acoustic — split on AIS too
    if ship_prob >= SHIP_PROB_MEDIUM:
        if ais.recently_gone_dark_count >= 1:
            # Medium acoustic + gone-dark AIS = still suspicious enough to flag
            return "GONE_DARK_VESSEL", "MEDIUM"
        if ais.ais_vessels_in_radius >= 1:
            return "CONFIRMED_VESSEL", "LOW"
        return "ACOUSTIC_ONLY_LOW", "MEDIUM"

    # Below SHIP_PROB_MEDIUM but above site threshold — low-confidence acoustic
    return "ACOUSTIC_ONLY_LOW", "MEDIUM"


def gone_dark_alone_tier(
    *,
    in_mpa: bool,
    recently_gone_dark_count: int,
) -> str | None:
    """A vessel going dark AND in / near an MPA is itself an alert
    trigger, even without an acoustic detection. Returns a tier string
    if the AIS gap alone justifies an alert, else None.

    This catches: vessel turns AIS off just before entering MPA, hydrophone
    is too distant or vessel too quiet to give us an acoustic positive,
    but the AIS gap pattern is the signal.
    """
    if recently_gone_dark_count == 0:
        return None
    if in_mpa:
        return "AIS_GAP_IN_MPA"
    return "AIS_GAP_NEAR_AREA"
