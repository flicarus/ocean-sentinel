#!/usr/bin/env python3
"""Verify that compute_threat_level() escalates to CRITICAL for DARK_VESSEL + MPA."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timezone

from ocean_sentinel.domain.enums import ThreatLevel
from ocean_sentinel.domain.models import AISGapEvent, GeoPoint
from ocean_sentinel.services.threat_level import compute_threat_level


def _mpa_gap(in_mpa: bool | None = True) -> AISGapEvent:
    return AISGapEvent(
        vessel_id="TEST-DV-001",
        vessel_name="Shadow Trawler",
        flag_state="XX",
        last_known_position=GeoPoint(lat=36.8, lon=-121.9),
        gap_start=datetime(2024, 3, 1, 8, 0, tzinfo=timezone.utc),
        gap_end=None,
        gap_duration_hours=6.5,
        intentional_disabling=True,
        in_mpa=in_mpa,
    )


def test_dark_vessel_in_mpa_is_critical():
    # DARK_VESSEL detection (mapped to HIGH by engine before MPA check) + in_mpa → CRITICAL
    level = compute_threat_level(ThreatLevel.HIGH, [_mpa_gap(in_mpa=True)])
    assert level is ThreatLevel.CRITICAL, f"Expected CRITICAL, got {level}"
    print(f"  DARK_VESSEL + MPA → {level.value}")


def test_none_never_escalates():
    # No acoustic signal — MPA context must not manufacture a false alert
    level = compute_threat_level(ThreatLevel.NONE, [_mpa_gap(in_mpa=True)])
    assert level is ThreatLevel.NONE, f"Expected NONE, got {level}"
    print(f"  NONE + MPA → {level.value}  (no acoustic signal, no alert)")


def test_outside_mpa_keeps_original_level():
    level = compute_threat_level(ThreatLevel.HIGH, [_mpa_gap(in_mpa=False)])
    assert level is ThreatLevel.HIGH, f"Expected HIGH, got {level}"
    print(f"  HIGH outside MPA → {level.value}  (no escalation)")


def test_unknown_mpa_does_not_escalate():
    # in_mpa=None means data unavailable — should not escalate
    level = compute_threat_level(ThreatLevel.HIGH, [_mpa_gap(in_mpa=None)])
    assert level is ThreatLevel.HIGH, f"Expected HIGH, got {level}"
    print(f"  HIGH + in_mpa=None → {level.value}  (unknown MPA, no escalation)")


def test_critical_stays_critical():
    level = compute_threat_level(ThreatLevel.CRITICAL, [_mpa_gap(in_mpa=True)])
    assert level is ThreatLevel.CRITICAL
    print(f"  CRITICAL + MPA → {level.value}  (already at max)")


if __name__ == "__main__":
    tests = [
        test_dark_vessel_in_mpa_is_critical,
        test_none_never_escalates,
        test_outside_mpa_keeps_original_level,
        test_unknown_mpa_does_not_escalate,
        test_critical_stays_critical,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)
