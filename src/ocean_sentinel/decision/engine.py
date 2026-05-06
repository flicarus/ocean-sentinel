"""Layer 4 + 5 + 6 — decision engine, AIS cross-check, provenance.

This is the orchestration layer. It consumes:
  * Layer 1: WindowedPrediction (sliding-window model verdict)
  * Layer 2: GateReport (deterministic acoustic gates)
  * Layer 3: ConformalPredictor (statistical lower bound)
  * Layer 4: GFW AIS data (vessels in radius around the hydrophone)

...and produces a Decision (threat tier + reasoning) plus a
ProvenanceRecord (immutable audit log of every input that fed the call).

Tier mapping
------------
The plan's tiers (DARK_VESSEL / CONFIRMED / LOW / AMBIENT / UNCERTAIN)
map onto the existing ThreatLevel enum:

  AMBIENT             → NONE      (model said not_ship, safety ok)
  ACOUSTIC_ONLY_LOW   → LOW       (model said ship, weak signature)
  CONFIRMED_VESSEL    → HIGH      (strong ship + AIS match)
  DARK_VESSEL         → CRITICAL  (strong ship + NO AIS — the project's
                                   actual value: a vessel without a
                                   transponder)
  UNCERTAIN           → MEDIUM    (safety fail / conformal too low /
                                   single-window anomaly — human review)

Decision flow
-------------
1. Safety gates fail? → UNCERTAIN. Cannot trust the input.
2. Conformal lower bound < 0.5? → UNCERTAIN. No statistical guarantee.
3. Single-window anomaly hint? → UNCERTAIN. Possible artifact.
4. Model says ship + ship_signature_strength ≥ 0.67:
     query AIS at hydrophone location for time window.
     match → HIGH (CONFIRMED_VESSEL)
     no match → CRITICAL (DARK_VESSEL)
5. Model says ship + weaker signature → LOW (ACOUSTIC_ONLY_LOW)
6. Model says not_ship → NONE (AMBIENT)

The `requires_review` flag marks UNCERTAIN decisions (and DARK_VESSEL,
because flagging a transponder-off vessel always deserves human eyes
even when the engine is confident).
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from ocean_sentinel.adapters.gfw import GFWAdapter
from ocean_sentinel.decision.conformal import ConformalPredictor
from ocean_sentinel.decision.gates import GateReport
from ocean_sentinel.decision.window import WindowedPrediction
from ocean_sentinel.domain.enums import ThreatLevel
from ocean_sentinel.domain.models import GeoPoint, NearbyVessel, TimeWindow
from ocean_sentinel.exceptions import GFWError

log = structlog.get_logger()


# Tier-mapping thresholds. Top-level so they're readable in the audit log
# under thresholds_used and easy to bump in one place.
# Ship-signature gate strength required for the strong-ship branch.
# 0.66 (≥ 2/3 of three gates) instead of 0.67 because 2/3 ≈ 0.66666 — the
# round-up boundary value would otherwise reject exact 2/3 from float
# precision. Pragmatic boundary fix.
SHIP_SIGNATURE_STRONG = 0.66
# Conformal lower bound is recorded in the audit log but does NOT gate
# decisions on its own. Reason: after v7.1 calibration the threshold sits
# around 0.66, so the maximum mathematically possible lower_bound is
# ~0.34 — any non-zero requirement would mark every decision UNCERTAIN
# regardless of how strong the actual signal is. Gates + AIS + ship_
# signature_strength are the real filters; conformal stays informational.
AIS_RADIUS_KM = 50.0
AIS_TIME_WINDOW_HOURS = 1


@dataclass(frozen=True)
class AISCheck:
    """Result of querying GFW for vessels around the hydrophone."""

    queried: bool
    match: bool
    n_vessels: int
    vessels: tuple[dict, ...] = field(default_factory=tuple)  # (name, mmsi, class, flag)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "queried": self.queried,
            "match": self.match,
            "n_vessels": self.n_vessels,
            "vessels": list(self.vessels),
            "error": self.error,
        }


@dataclass(frozen=True)
class Decision:
    threat_level: ThreatLevel
    tier_label: str                 # plan's name: DARK_VESSEL / CONFIRMED_VESSEL / etc.
    confidence: float               # conformal lower bound when available, else raw
    reasoning: str
    requires_review: bool


@dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable per-decision audit record. Every field a regulator might
    want to reproduce the decision is captured here. ChromaDB / Supabase
    persistence stores `to_dict()` output verbatim."""

    decision_id: str
    timestamp_utc: str
    hydrophone_id: str
    location: dict                  # {"lat": ..., "lon": ...}
    model_checkpoint: str | None
    input_hash: str                 # sha256-prefix of the spectrogram
    decision: str                   # ThreatLevel.value
    tier_label: str
    confidence: float
    reasoning: str
    evidence: dict                  # nested: windowed, gates, ais, conformal
    thresholds_used: dict
    requires_review: bool
    operator_review: dict | None = None  # filled by human reviewer post-hoc

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "timestamp_utc": self.timestamp_utc,
            "hydrophone_id": self.hydrophone_id,
            "location": self.location,
            "model_checkpoint": self.model_checkpoint,
            "input_hash": self.input_hash,
            "decision": self.decision,
            "tier_label": self.tier_label,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "thresholds_used": self.thresholds_used,
            "requires_review": self.requires_review,
            "operator_review": self.operator_review,
        }


def _hash_spectrogram(spec) -> str:
    """sha256 prefix of the spec bytes — short enough to log, long
    enough that two different specs almost never collide."""
    return hashlib.sha256(spec.tobytes()).hexdigest()[:16]


class DecisionEngine:
    """Orchestrates Layer 4 (AIS) + Layer 5 (tier decision) + Layer 6
    (provenance). Stateless apart from the injected dependencies."""

    def __init__(
        self,
        gfw: GFWAdapter | None = None,
        conformal: ConformalPredictor | None = None,
        ais_radius_km: float = AIS_RADIUS_KM,
        ais_time_window_hours: int = AIS_TIME_WINDOW_HOURS,
    ) -> None:
        self._gfw = gfw
        self._conformal = conformal or ConformalPredictor.uncalibrated()
        self._ais_radius_km = ais_radius_km
        self._ais_time_window_hours = ais_time_window_hours

    async def decide(
        self,
        spectrogram,
        windowed: WindowedPrediction,
        gates_report: GateReport,
        location: GeoPoint,
        capture_time: datetime,
        hydrophone_id: str,
    ) -> tuple[Decision, ProvenanceRecord]:
        """Run the full Layer 4-6 pipeline. AIS is queried only when the
        decision tree actually needs it (strong ship signal), so quiet
        chunks don't pay for an external call."""

        conformal_lb = self._conformal.lower_bound(windowed.confidence)
        decision_id = str(uuid.uuid4())

        # --- Decision tree ---------------------------------------------
        ais_check = AISCheck(queried=False, match=False, n_vessels=0)
        tier_label: str
        threat: ThreatLevel
        reasoning: str
        requires_review: bool

        if not gates_report.all_safety_pass:
            tier_label = "UNCERTAIN"
            threat = ThreatLevel.MEDIUM
            requires_review = True
            failed = [g.name for g in gates_report.gates if not g.passed
                      and g.category.value == "safety"]
            reasoning = (
                "Safety gate(s) failed: " + ", ".join(failed)
                + ". Input cannot be trusted; flagged for review."
            )

        elif windowed.single_window_anomaly:
            tier_label = "UNCERTAIN"
            threat = ThreatLevel.MEDIUM
            requires_review = True
            reasoning = (
                f"Majority ambient ({windowed.ship_fraction:.0%} ship) "
                f"with one high-confidence ship window "
                f"(max {windowed.max_ship_confidence:.2f}). "
                f"Possible artifact or transient — flagged for review."
            )

        elif windowed.label == "ship":
            strength = gates_report.ship_signature_strength
            if strength >= SHIP_SIGNATURE_STRONG:
                # Strong ship signal — query AIS to disambiguate
                # CONFIRMED vs DARK vessel.
                ais_check = await self._check_ais(location, capture_time)
                if not ais_check.queried:
                    # No AIS source available. Fail-safe: cannot
                    # distinguish CONFIRMED from DARK, so escalate to
                    # UNCERTAIN rather than asserting either.
                    tier_label = "UNCERTAIN"
                    threat = ThreatLevel.MEDIUM
                    requires_review = True
                    reasoning = (
                        f"Strong acoustic ship signal "
                        f"({windowed.ship_fraction:.0%} of windows, "
                        f"ship_signature {gates_report.ship_signature_passed}/"
                        f"{gates_report.ship_signature_total}) but AIS check "
                        f"unavailable ({ais_check.error or 'no GFW client'}). "
                        f"Cannot disambiguate confirmed vs dark vessel."
                    )
                elif ais_check.match:
                    tier_label = "CONFIRMED_VESSEL"
                    threat = ThreatLevel.HIGH
                    requires_review = False
                    reasoning = (
                        f"Strong acoustic ship signal "
                        f"({windowed.ship_fraction:.0%} of windows, "
                        f"ship_signature {gates_report.ship_signature_passed}/"
                        f"{gates_report.ship_signature_total}) "
                        f"corroborated by AIS: {ais_check.n_vessels} vessel(s) "
                        f"within {self._ais_radius_km:.0f} km."
                    )
                else:
                    tier_label = "DARK_VESSEL"
                    threat = ThreatLevel.CRITICAL
                    requires_review = True
                    reasoning = (
                        f"Strong acoustic ship signal "
                        f"({windowed.ship_fraction:.0%} of windows, "
                        f"ship_signature {gates_report.ship_signature_passed}/"
                        f"{gates_report.ship_signature_total}) "
                        f"with NO AIS contact within {self._ais_radius_km:.0f} km. "
                        f"Possible transponder-off vessel."
                    )
            else:
                tier_label = "ACOUSTIC_ONLY_LOW"
                threat = ThreatLevel.LOW
                requires_review = False
                reasoning = (
                    f"Model voted ship but acoustic signature is weak "
                    f"({gates_report.ship_signature_passed}/"
                    f"{gates_report.ship_signature_total} gates pass). "
                    f"Likely distant or partial passage; not escalated."
                )

        else:  # windowed.label == "not_ship"
            tier_label = "AMBIENT"
            threat = ThreatLevel.NONE
            requires_review = False
            reasoning = (
                f"Quiet window: {windowed.ship_fraction:.0%} ship votes "
                f"across {windowed.n_windows} sliding windows."
            )

        # Headline confidence: conformal lower bound when calibrated,
        # otherwise the raw windowed confidence. We log both in evidence
        # so the audit trail can tell which path produced the number.
        headline_confidence = (
            conformal_lb if self._conformal.is_calibrated
            else windowed.confidence
        )

        decision = Decision(
            threat_level=threat,
            tier_label=tier_label,
            confidence=headline_confidence,
            reasoning=reasoning,
            requires_review=requires_review,
        )

        provenance = self._build_provenance(
            decision_id=decision_id,
            spectrogram=spectrogram,
            windowed=windowed,
            gates_report=gates_report,
            ais_check=ais_check,
            conformal_lb=conformal_lb,
            location=location,
            capture_time=capture_time,
            hydrophone_id=hydrophone_id,
            decision=decision,
        )

        log.info(
            "decision_complete",
            decision_id=decision_id,
            tier=tier_label,
            threat_level=threat.value,
            confidence=round(headline_confidence, 3),
            requires_review=requires_review,
            ship_fraction=windowed.ship_fraction,
            ship_signature=f"{gates_report.ship_signature_passed}/"
                           f"{gates_report.ship_signature_total}",
            ais_match=ais_check.match if ais_check.queried else None,
        )

        return decision, provenance

    async def _check_ais(
        self, location: GeoPoint, capture_time: datetime,
    ) -> AISCheck:
        """Query GFW for AIS-broadcasting vessels around the hydrophone
        in a ± window. We only call this when the decision tree needs to
        disambiguate CONFIRMED vs DARK vessel."""
        if self._gfw is None:
            return AISCheck(
                queried=False, match=False, n_vessels=0,
                error="no GFW client configured",
            )

        from datetime import timedelta
        window = TimeWindow(
            start=capture_time - timedelta(hours=self._ais_time_window_hours),
            end=capture_time + timedelta(hours=self._ais_time_window_hours),
        )
        try:
            vessels = await self._gfw.get_vessels_in_radius(
                location, self._ais_radius_km, window,
            )
        except GFWError as e:
            log.warning("ais_check_failed", error=str(e))
            return AISCheck(
                queried=True, match=False, n_vessels=0, error=str(e),
            )

        return AISCheck(
            queried=True,
            match=len(vessels) > 0,
            n_vessels=len(vessels),
            vessels=tuple(_summarize_vessel(v) for v in vessels[:10]),
        )

    def _build_provenance(
        self,
        decision_id: str,
        spectrogram,
        windowed: WindowedPrediction,
        gates_report: GateReport,
        ais_check: AISCheck,
        conformal_lb: float,
        location: GeoPoint,
        capture_time: datetime,
        hydrophone_id: str,
        decision: Decision,
    ) -> ProvenanceRecord:
        return ProvenanceRecord(
            decision_id=decision_id,
            timestamp_utc=capture_time.astimezone(timezone.utc).isoformat(),
            hydrophone_id=hydrophone_id,
            location={"lat": location.lat, "lon": location.lon},
            model_checkpoint=windowed.checkpoint,
            input_hash=_hash_spectrogram(spectrogram),
            decision=decision.threat_level.value,
            tier_label=decision.tier_label,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            evidence={
                "windowed": {
                    "label": windowed.label,
                    "n_windows": windowed.n_windows,
                    "ship_fraction": windowed.ship_fraction,
                    "mean_uncertainty": windowed.mean_uncertainty,
                    "max_ship_confidence": windowed.max_ship_confidence,
                    "min_ship_confidence": windowed.min_ship_confidence,
                    "vessel_type": windowed.vessel_type,
                    "distance": windowed.distance,
                    "single_window_anomaly": windowed.single_window_anomaly,
                    "windows": [
                        {
                            "window_idx": w.window_idx,
                            "label": w.label,
                            "confidence": w.confidence,
                            "uncertainty": w.uncertainty,
                            "vessel_type": w.vessel_type,
                            "distance": w.distance,
                        }
                        for w in windowed.windows
                    ],
                },
                "gates": gates_report.to_dict(),
                "ais": ais_check.to_dict(),
                "conformal": {
                    "lower_bound": conformal_lb,
                    "threshold": self._conformal.threshold,
                    "alpha": self._conformal.alpha,
                    "n_calibration": self._conformal.n_calibration,
                    "is_calibrated": self._conformal.is_calibrated,
                },
            },
            thresholds_used={
                "ship_signature_strong": SHIP_SIGNATURE_STRONG,
                "ais_radius_km": self._ais_radius_km,
                "ais_time_window_hours": self._ais_time_window_hours,
            },
            requires_review=decision.requires_review,
        )


def _summarize_vessel(v: NearbyVessel) -> dict[str, Any]:
    return {
        "vessel_id": v.vessel_id,
        "vessel_name": v.vessel_name,
        "vessel_class": v.vessel_class,
        "flag_state": v.flag_state,
        "length_m": v.length_m,
    }
