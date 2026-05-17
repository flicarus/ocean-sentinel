"""Context-aware threat-level scoring for detection events.

Takes a CNN decision (tier + severity) plus site context (MPA status,
sensitivity policy, AIS proximity) and produces a ThreatLevel that
drives alert routing and operator-visible severity.

Rules are deliberate and rule-based — no LLM in the loop here. Gemma
*reads* threat levels to compose briefs and ask "why?" via
explain_decision, but the scoring itself is reproducible and auditable.

Escalation policy:

  - DARK_VESSEL inside MPA           → CRITICAL
  - DARK_VESSEL outside MPA          → HIGH
  - CONFIRMED_VESSEL inside MPA      → HIGH    (legal vessel in protected waters)
  - CONFIRMED_VESSEL outside MPA     → LOW
  - ACOUSTIC_ONLY_LOW (uncertain)    → MEDIUM
  - AMBIENT                          → NONE
  - UNCERTAIN (model abstained)      → LOW + flag_for_review

Adjusters layered on top:
  - sensitivity=high                 → bump by one tier (LOW→MEDIUM, MEDIUM→HIGH, HIGH→CRITICAL)
  - sensitivity=low                  → demote by one tier
  - CPA ≤ 1 km                       → bump by one tier
  - mpa_buffer_km set + CPA < buffer → bump by one tier (vessel near MPA boundary)
"""
from __future__ import annotations

from dataclasses import dataclass

from ocean_sentinel.domain.enums import ThreatLevel


_TIER_BASE = {
    # Acoustic + AIS-state combined tiers
    "DARK_VESSEL":       ThreatLevel.HIGH,    # acoustic + no AIS at all
    "GONE_DARK_VESSEL":  ThreatLevel.HIGH,    # acoustic + AIS-broadcasting vessel just disappeared
    "CONFIRMED_VESSEL":  ThreatLevel.LOW,     # acoustic + active AIS — legal
    "ACOUSTIC_ONLY_LOW": ThreatLevel.MEDIUM,  # weak acoustic, AIS state unclear
    "AMBIENT":           ThreatLevel.NONE,
    "UNCERTAIN":         ThreatLevel.LOW,
    # AIS-gap-alone alerts (no acoustic)
    "AIS_GAP_IN_MPA":    ThreatLevel.HIGH,    # vessel turned AIS off inside MPA
    "AIS_GAP_NEAR_AREA": ThreatLevel.MEDIUM,  # vessel turned AIS off outside MPA but near site
}

_LADDER = [
    ThreatLevel.NONE,
    ThreatLevel.LOW,
    ThreatLevel.MEDIUM,
    ThreatLevel.HIGH,
    ThreatLevel.CRITICAL,
]


def _bump(level: ThreatLevel, by: int = 1) -> ThreatLevel:
    idx = _LADDER.index(level)
    new_idx = max(0, min(len(_LADDER) - 1, idx + by))
    return _LADDER[new_idx]


@dataclass(frozen=True)
class SiteContext:
    """Subset of site config needed for threat scoring."""
    site_id: str
    in_mpa: bool = False
    mpa_name: str | None = None
    mpa_buffer_km: float | None = None
    sensitivity: str = "medium"  # 'low' | 'medium' | 'high'


@dataclass(frozen=True)
class ThreatAssessment:
    threat_level: ThreatLevel
    base_from_tier: ThreatLevel
    escalated_by: tuple[str, ...]
    reasoning: str


def compute_threat_level(
    decision_tier: str,
    site: SiteContext,
    *,
    cpa_km: float | None = None,
    nighttime: bool = False,
) -> ThreatAssessment:
    """Score the threat for a single detection event.

    Args:
        decision_tier: one of {DARK_VESSEL, CONFIRMED_VESSEL,
            ACOUSTIC_ONLY_LOW, AMBIENT, UNCERTAIN}.
        site: SiteContext with MPA + sensitivity policy.
        cpa_km: Distance to nearest AIS-broadcasting vessel (km), if known.
        nighttime: True if detection occurred outside daylight hours
            (operators often weight night detections higher for fishing
            enforcement).
    """
    base = _TIER_BASE.get(decision_tier, ThreatLevel.LOW)
    level = base
    reasons: list[str] = []

    # MPA escalation — acoustic-based tiers
    if site.in_mpa:
        if decision_tier in ("DARK_VESSEL", "GONE_DARK_VESSEL"):
            level = ThreatLevel.CRITICAL
            tag = "DARK_VESSEL" if decision_tier == "DARK_VESSEL" else "vessel went dark"
            reasons.append(f"{tag} inside MPA ({site.mpa_name or site.site_id})")
        elif decision_tier in ("CONFIRMED_VESSEL", "ACOUSTIC_ONLY_LOW"):
            level = _bump(level, 1)
            reasons.append(f"vessel detected inside MPA ({site.mpa_name or site.site_id})")
        # AIS-gap-alone (no acoustic) inside MPA — these are already
        # base=HIGH/MEDIUM via _TIER_BASE; reflect that in reasoning
        elif decision_tier == "AIS_GAP_IN_MPA":
            level = _bump(level, 1)
            reasons.append(f"AIS gap event inside MPA ({site.mpa_name or site.site_id})")
    elif site.mpa_buffer_km is not None and cpa_km is not None and cpa_km < site.mpa_buffer_km:
        if decision_tier in (
            "DARK_VESSEL", "GONE_DARK_VESSEL", "CONFIRMED_VESSEL",
            "ACOUSTIC_ONLY_LOW", "AIS_GAP_NEAR_AREA",
        ):
            level = _bump(level, 1)
            reasons.append(
                f"event within MPA buffer ({cpa_km:.1f} km < {site.mpa_buffer_km} km)"
            )

    # Sensitivity policy
    if site.sensitivity == "high" and level != ThreatLevel.NONE:
        prev = level
        level = _bump(level, 1)
        if level != prev:
            reasons.append("site sensitivity=high")
    elif site.sensitivity == "low" and level != ThreatLevel.NONE:
        prev = level
        level = _bump(level, -1)
        if level != prev:
            reasons.append("site sensitivity=low")

    # Close CPA escalation
    if cpa_km is not None and cpa_km <= 1.0:
        if decision_tier != "AMBIENT":
            prev = level
            level = _bump(level, 1)
            if level != prev:
                reasons.append(f"vessel within 1 km (CPA {cpa_km:.2f} km)")

    # Night-time weighting for fishing enforcement
    if nighttime and decision_tier in (
        "DARK_VESSEL", "GONE_DARK_VESSEL", "ACOUSTIC_ONLY_LOW",
        "AIS_GAP_IN_MPA", "AIS_GAP_NEAR_AREA",
    ):
        prev = level
        level = _bump(level, 1)
        if level != prev:
            reasons.append("nighttime detection (fishing enforcement weighting)")

    reasoning = (
        f"base={base.value} ({decision_tier}) → escalated to {level.value} "
        f"via: {', '.join(reasons) if reasons else 'no adjustments'}"
    )

    return ThreatAssessment(
        threat_level=level,
        base_from_tier=base,
        escalated_by=tuple(reasons),
        reasoning=reasoning,
    )
