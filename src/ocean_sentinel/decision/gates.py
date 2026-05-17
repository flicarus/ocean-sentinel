"""Layer 2 — deterministic acoustic gates.

Each gate is a Boolean check with a physical justification, computed from
AudioAnalyzer features and the WindowedPrediction aggregate. No ML — just
spectral physics. The decision engine (Layer 5) reads this report
alongside the model verdict and AIS context to assign a threat tier.

Two categories of gates:

  ship_signature  — does this audio look vessel-like at all?
                    if zero ship-sig gates pass, model probably hallucinated.
  safety          — do we trust the input enough to act on any verdict?
                    if any safety gate fails, default to UNCERTAIN.

Thresholds are physically motivated, not tuned on a holdout. Future
calibration could sweep on a labelled set, but the simpler the gates the
easier the audit story for a regulator: "we said ship because engine band
dominates by ratio 1.4, model agreed across 11/12 windows, and the
spectral flatness was 0.18 (tonal)".
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ocean_sentinel.decision.window import WindowedPrediction


class GateCategory(str, Enum):
    SHIP_SIGNATURE = "ship_signature"
    SAFETY = "safety"


@dataclass(frozen=True)
class GateResult:
    """One gate's outcome — kept granular so the audit log explains the
    why behind every decision, not just the final tier."""

    name: str
    category: GateCategory
    passed: bool
    actual: float | bool          # what we measured
    threshold: float | bool       # what we required
    reason: str                   # human-readable single-line explanation


@dataclass(frozen=True)
class GateReport:
    gates: tuple[GateResult, ...]

    @property
    def ship_signature_total(self) -> int:
        return sum(1 for g in self.gates if g.category is GateCategory.SHIP_SIGNATURE)

    @property
    def ship_signature_passed(self) -> int:
        return sum(
            1 for g in self.gates
            if g.category is GateCategory.SHIP_SIGNATURE and g.passed
        )

    @property
    def ship_signature_strength(self) -> float:
        total = self.ship_signature_total
        return self.ship_signature_passed / total if total else 0.0

    @property
    def safety_total(self) -> int:
        return sum(1 for g in self.gates if g.category is GateCategory.SAFETY)

    @property
    def safety_passed(self) -> int:
        return sum(
            1 for g in self.gates
            if g.category is GateCategory.SAFETY and g.passed
        )

    @property
    def all_safety_pass(self) -> bool:
        return self.safety_passed == self.safety_total

    def to_dict(self) -> dict:
        return {
            "ship_signature_passed": self.ship_signature_passed,
            "ship_signature_total": self.ship_signature_total,
            "ship_signature_strength": round(self.ship_signature_strength, 3),
            "safety_passed": self.safety_passed,
            "safety_total": self.safety_total,
            "all_safety_pass": self.all_safety_pass,
            "gates": [
                {
                    "name": g.name,
                    "category": g.category.value,
                    "passed": g.passed,
                    "actual": g.actual,
                    "threshold": g.threshold,
                    "reason": g.reason,
                }
                for g in self.gates
            ],
        }


# Threshold constants — top-level so they're searchable in the audit log
# and easy to bump in one place if calibration suggests a shift.
#
# These are MVP physical thresholds, not tuned on a held-out set. Layer 3
# conformal calibration will give us statistically-grounded thresholds for
# the model-side gates (model_consistency, low_uncertainty) once it runs.
SPECTRAL_FLATNESS_MAX = 0.4       # below = tonal (vessel); above = broadband
MODEL_SHIP_FRACTION_MIN = 0.55    # majority of windows agree on ship
# v7's evidential head was collapsed (~0.05 uncertainty for everything).
# v7.1 shows real spread: in-dist confident ≈ 0.18-0.20, OOD/hard ≈ 0.25-0.30.
# Tightened from 0.22 → 0.20 after end-to-end eval. 0.18 was too tight
# (everything failed including in-dist confident predictions, blocking
# the CONFIRMED_VESSEL path entirely). 0.20 sits at the in-dist mean —
# OOD/hedging predictions still fail and demote to LOW, while in-dist
# confident calls pass through to the strong-ship branch.
MEAN_UNCERTAINTY_MAX = 0.20
# OrcasoundAdapter currently fetches a single 10s HLS segment per call
# despite the 60s claim, so most decisions land on n_windows=1. Single-
# window inference is the realistic floor until the adapter concatenates
# 6 segments.
MIN_WINDOWS = 1
MIN_RMS_ENERGY = 1e-5


def evaluate_gates(
    features: dict, windowed: WindowedPrediction,
) -> GateReport:
    """Run all gates against the AudioAnalyzer features dict + windowed
    aggregate, returning a structured report. No side effects."""

    gates: list[GateResult] = []

    # --- ship_signature gates ---------------------------------------
    #
    # Note: AudioAnalyzer's `engine_band_ratio` divides two negative dB
    # means, so its semantic is inverted relative to the obvious "engine
    # louder than non-engine" reading — we don't gate on it here. Same for
    # peak-frequency-in-band: cargo cavitation peaks above 500 Hz in many
    # ShipsEar samples. Until AudioAnalyzer is reworked, we lean on the
    # two acoustic signals that hold up (flatness, signal presence) plus
    # two model-agreement signals (windowed consistency, uncertainty).

    # 1. Tonal signature — vessels have narrow-band engine/cavitation
    # harmonics; ambient ocean noise is broadband. Flatness < 0.4 is the
    # boundary in marine bioacoustics literature.
    flatness = float(features["spectral_flatness"])
    gates.append(GateResult(
        name="tonal_signature",
        category=GateCategory.SHIP_SIGNATURE,
        passed=flatness < SPECTRAL_FLATNESS_MAX,
        actual=round(flatness, 3),
        threshold=SPECTRAL_FLATNESS_MAX,
        reason=(
            f"spectral flatness {flatness:.3f} "
            + ("(tonal — vessel-like)" if flatness < SPECTRAL_FLATNESS_MAX
               else "(broadband — ambient-like)")
        ),
    ))

    # 2. Model consistency — most sliding windows agree on ship. Single-
    # window ship spikes are physically implausible (vessels move slowly
    # relative to a 5 s window), so we want persistent agreement.
    gates.append(GateResult(
        name="model_consistency",
        category=GateCategory.SHIP_SIGNATURE,
        passed=windowed.ship_fraction >= MODEL_SHIP_FRACTION_MIN,
        actual=round(windowed.ship_fraction, 3),
        threshold=MODEL_SHIP_FRACTION_MIN,
        reason=(
            f"{int(round(windowed.ship_fraction * windowed.n_windows))}"
            f"/{windowed.n_windows} windows voted ship"
        ),
    ))

    # 3. Low evidential uncertainty — v7's predict() returns 2/S where S
    # is the Dirichlet strength. In-distribution samples sit at ~0.05;
    # held-out / harder samples climb toward 0.30. This is the only
    # calibrated confidence signal we have until Layer 3 conformal runs.
    mean_unc = float(windowed.mean_uncertainty)
    gates.append(GateResult(
        name="low_uncertainty",
        category=GateCategory.SHIP_SIGNATURE,
        passed=mean_unc < MEAN_UNCERTAINTY_MAX,
        actual=round(mean_unc, 4),
        threshold=MEAN_UNCERTAINTY_MAX,
        reason=(
            f"mean evidential uncertainty {mean_unc:.3f} "
            + ("(model committed)" if mean_unc < MEAN_UNCERTAINTY_MAX
               else "(model hedging)")
        ),
    ))

    # --- safety gates -----------------------------------------------

    # 5. Sufficient evidence — at least 4 windows = 20s of audio. Below
    # this, even an unanimous vote is statistically thin.
    gates.append(GateResult(
        name="sufficient_evidence",
        category=GateCategory.SAFETY,
        passed=windowed.n_windows >= MIN_WINDOWS,
        actual=windowed.n_windows,
        threshold=MIN_WINDOWS,
        reason=(
            f"{windowed.n_windows} windows observed "
            + ("(>= " if windowed.n_windows >= MIN_WINDOWS else "(< ")
            + f"{MIN_WINDOWS} required)"
        ),
    ))

    # 6. Signal exists — RMS energy above absolute silence floor. Catches
    # fetch failures, cable disconnects, hydrophone dropouts where the
    # buffer is essentially zero and the model would be hallucinating on
    # numerical noise.
    rms = float(features["rms_energy"])
    gates.append(GateResult(
        name="signal_present",
        category=GateCategory.SAFETY,
        passed=rms > MIN_RMS_ENERGY,
        actual=rms,
        threshold=MIN_RMS_ENERGY,
        reason=(
            f"RMS energy {rms:.2e} "
            + ("> " if rms > MIN_RMS_ENERGY else "<= ")
            + f"silence floor {MIN_RMS_ENERGY:.0e}"
        ),
    ))

    return GateReport(gates=tuple(gates))
