"""Layered decision pipeline.

Each layer is a separate module:
  window     — Layer 1: sliding-window inference, per-window verdicts +
               aggregate stats consumed by every layer below.
  gates      — Layer 2: deterministic acoustic gates (TODO).
  conformal  — Layer 3: conformal calibration (TODO).
  engine     — Layer 5/6: threat-tier decision + provenance (TODO).

Layer 4 (AIS cross-check) reuses adapters/gfw, no module here.
"""
from ocean_sentinel.decision.window import (
    WindowPrediction,
    WindowedPrediction,
    predict_windowed,
)
from ocean_sentinel.decision.gates import (
    GateCategory,
    GateResult,
    GateReport,
    evaluate_gates,
)
from ocean_sentinel.decision.conformal import ConformalPredictor
from ocean_sentinel.decision.engine import (
    AISCheck,
    Decision,
    DecisionEngine,
    ProvenanceRecord,
)

__all__ = [
    "WindowPrediction", "WindowedPrediction", "predict_windowed",
    "GateCategory", "GateResult", "GateReport", "evaluate_gates",
    "ConformalPredictor",
    "AISCheck", "Decision", "DecisionEngine", "ProvenanceRecord",
]
