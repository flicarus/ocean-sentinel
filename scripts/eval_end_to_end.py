"""End-to-end Decision pipeline eval.

Raw CNN accuracy is the WRONG metric for Ocean Sentinel. The system's job
is to assign a deterministic threat tier (CONFIRMED / DARK / LOW /
AMBIENT / UNCERTAIN), not to make a perfect ship-vs-not-ship call. Errors
of the CNN are supposed to be caught by Layer 2-5 (gates, conformal,
AIS) and routed to UNCERTAIN — that's the regulatory value prop.

This script measures the *real* deployment KPI: what fraction of decisions
are CORRECT, ABSTAIN, or WRONG, broken down per site.

Two AIS modes:

  --ais-mode none
      No AIS lookup. Strong-ship decisions become UNCERTAIN ("we'd need
      AIS to commit"). Pure acoustic + abstention measurement.

  --ais-mode oracle
      Stub AIS that returns ground-truth — simulates a working AIS infra
      where ship rows have a vessel match and ambient rows don't. Measures
      the upper bound: end-to-end correctness when AIS is available.

Outcome buckets (per row):

  truth=ship + tier in {CONFIRMED_VESSEL, DARK_VESSEL}  → correct_strong
  truth=ship + tier == ACOUSTIC_ONLY_LOW                → correct_weak
  truth=ship + tier == AMBIENT                          → MISS (FN)
  truth=ship + tier == UNCERTAIN                        → abstain
  truth=not_ship + tier == AMBIENT                      → correct
  truth=not_ship + tier in {CONFIRMED, DARK}            → FALSE_ALARM
  truth=not_ship + tier == ACOUSTIC_ONLY_LOW            → FALSE_ALARM (weak)
  truth=not_ship + tier == UNCERTAIN                    → abstain

Usage:
    PYTHONPATH=src venv/bin/python scripts/eval_end_to_end.py \\
        --model data/models/cnn_v7_1.pt \\
        --conformal data/calibration/conformal.json \\
        --ais-mode oracle \\
        --out data/eval/end_to_end_v7_1.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import structlog

sys.path.insert(0, "src")

from ocean_sentinel.decision.conformal import ConformalPredictor
from ocean_sentinel.decision.engine import DecisionEngine
from ocean_sentinel.decision.gates import evaluate_gates
from ocean_sentinel.decision.window import predict_windowed
from ocean_sentinel.domain.models import GeoPoint, NearbyVessel, TimeWindow
from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier

log = structlog.get_logger()


DEFAULT_FILES = [
    "data/training/sanctsound_corrected.jsonl",
    "data/training/sanctsound_diverse.jsonl",
    "data/training/gemma_labels.v7.jsonl",
]


# Visual-spec params (must match AudioAnalyzer config in production)
_VIS_N_MELS = 128
_VIS_FMAX = 1000
_ENGINE_LOW = 50
_ENGINE_HIGH = 500


def features_from_spec(spec: np.ndarray) -> dict:
    """Recompute spectral_flatness + rms_energy from a saved mel spec when
    the JSONL row didn't store them. The saved spec is in dB (ref=1.0) at
    128 mels / fmax=1000 Hz. This is a slightly different basis from the
    original AudioAnalyzer (which used 256 mels / 8 kHz) but the threshold
    semantics for the gate (flatness < 0.4) hold."""
    power = librosa.db_to_power(spec, ref=1.0)  # back to linear
    freqs = librosa.mel_frequencies(n_mels=_VIS_N_MELS, fmax=_VIS_FMAX)
    engine_mask = (freqs >= _ENGINE_LOW) & (freqs <= _ENGINE_HIGH)
    if engine_mask.any():
        flatness = float(np.mean(
            librosa.feature.spectral_flatness(S=power[engine_mask])
        ))
    else:
        flatness = 0.5
    # RMS proxy: sqrt of mean linear power. Correlates strongly with raw-
    # audio RMS for the silence-floor gate (1e-5).
    rms = float(np.sqrt(np.mean(power)))
    return {
        "spectral_flatness": flatness,
        "rms_energy": rms,
    }


def site_key(row: dict) -> str:
    src = (row.get("provenance") or {}).get("source_id", "unknown")
    if "sanctsound" in src and "-" in src:
        return src.rsplit("-", 1)[-1]
    return src


class OracleGFWAdapter:
    """Eval-only AIS stub. Returns a single fake vessel iff the row's
    ground-truth label is 'ship'. Production uses the real GFWAdapter."""

    def __init__(self) -> None:
        self.current_truth: str | None = None

    async def get_vessels_in_radius(
        self, location: GeoPoint, radius_km: float, window: TimeWindow,
    ) -> list[NearbyVessel]:
        if self.current_truth == "ship":
            return [NearbyVessel(
                vessel_id="ORACLE-1",
                vessel_name="oracle_truth",
                vessel_class="cargo",
                flag_state="--",
                position=location,
                distance_km=0.0,
                length_m=100.0,
            )]
        return []


# Outcome bucket mapping
OUTCOME_CORRECT_STRONG = "correct_strong"
OUTCOME_CORRECT_WEAK = "correct_weak"
OUTCOME_CORRECT_AMBIENT = "correct_ambient"
OUTCOME_MISS = "miss"
OUTCOME_FALSE_ALARM = "false_alarm"
OUTCOME_FALSE_ALARM_WEAK = "false_alarm_weak"
OUTCOME_ABSTAIN = "abstain"

ALL_OUTCOMES = [
    OUTCOME_CORRECT_STRONG, OUTCOME_CORRECT_WEAK, OUTCOME_CORRECT_AMBIENT,
    OUTCOME_MISS, OUTCOME_FALSE_ALARM, OUTCOME_FALSE_ALARM_WEAK,
    OUTCOME_ABSTAIN,
]

CORRECT_OUTCOMES = {
    OUTCOME_CORRECT_STRONG, OUTCOME_CORRECT_WEAK, OUTCOME_CORRECT_AMBIENT,
}
WRONG_OUTCOMES = {OUTCOME_MISS, OUTCOME_FALSE_ALARM, OUTCOME_FALSE_ALARM_WEAK}


def classify_outcome(truth: str, tier: str) -> str:
    """Map (truth, decided tier) → outcome bucket."""
    if truth == "ship":
        if tier in ("CONFIRMED_VESSEL", "DARK_VESSEL"):
            return OUTCOME_CORRECT_STRONG
        if tier == "ACOUSTIC_ONLY_LOW":
            return OUTCOME_CORRECT_WEAK
        if tier == "UNCERTAIN":
            return OUTCOME_ABSTAIN
        return OUTCOME_MISS  # AMBIENT for a true ship
    # truth == not_ship
    if tier == "AMBIENT":
        return OUTCOME_CORRECT_AMBIENT
    if tier == "UNCERTAIN":
        return OUTCOME_ABSTAIN
    if tier == "ACOUSTIC_ONLY_LOW":
        return OUTCOME_FALSE_ALARM_WEAK
    return OUTCOME_FALSE_ALARM  # CONFIRMED / DARK on ambient = bad


async def evaluate_site(
    cnn: CNNV7Classifier,
    engine: DecisionEngine,
    oracle: OracleGFWAdapter | None,
    rows: list[dict],
) -> dict:
    """Run the full pipeline on every row in this site, count outcomes."""
    counts: dict[str, int] = defaultdict(int)
    tier_counts: dict[str, int] = defaultdict(int)
    n_processed = 0

    # Default location/time when row doesn't carry them. AIS oracle ignores
    # location so this is fine; without oracle, the gfw adapter just won't
    # be queried (real GFW would need real coords).
    default_loc = GeoPoint(lat=0.0, lon=0.0)
    default_time = datetime.now(timezone.utc)

    for r in rows:
        spec_path = Path(r["spectrogram_path"])
        truth = r.get("label")
        if not spec_path.exists() or truth not in ("ship", "not_ship"):
            continue

        try:
            spec = np.load(spec_path)
        except Exception:
            continue

        if oracle is not None:
            oracle.current_truth = truth

        try:
            windowed = predict_windowed(cnn, spec, source_id=site_key(r))
            features = r.get("features") or {}
            if ("spectral_flatness" not in features
                    or "rms_energy" not in features):
                features = {**features, **features_from_spec(spec)}

            gates_report = evaluate_gates(features, windowed)

            decision, _ = await engine.decide(
                spectrogram=spec,
                windowed=windowed,
                gates_report=gates_report,
                location=default_loc,
                capture_time=default_time,
                hydrophone_id=site_key(r),
            )
        except Exception as e:
            log.warning("row_failed", error=str(e), event_id=r.get("event_id"))
            continue

        outcome = classify_outcome(truth, decision.tier_label)
        counts[outcome] += 1
        tier_counts[decision.tier_label] += 1
        n_processed += 1

    n_committed = sum(counts[o] for o in ALL_OUTCOMES if o != OUTCOME_ABSTAIN)
    n_correct = sum(counts[o] for o in CORRECT_OUTCOMES)
    n_wrong = sum(counts[o] for o in WRONG_OUTCOMES)
    n_abstain = counts[OUTCOME_ABSTAIN]

    return {
        "n": n_processed,
        # Headline: of decisions where we COMMITTED (not abstained), how
        # many were right? This is the regulator's question.
        "correct_when_committed": (
            round(n_correct / n_committed, 4) if n_committed else None
        ),
        # Of all rows, how many did we just route to UNCERTAIN.
        "abstain_rate": (
            round(n_abstain / n_processed, 4) if n_processed else None
        ),
        # Of all rows, how often did we make a WRONG call (not abstain).
        "wrong_rate": (
            round(n_wrong / n_processed, 4) if n_processed else None
        ),
        "outcomes": {o: counts[o] for o in ALL_OUTCOMES},
        "tier_counts": dict(tier_counts),
    }


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/cnn_v7_1.pt")
    p.add_argument("--conformal", default="data/calibration/conformal.json")
    p.add_argument(
        "--ais-mode", choices=["none", "oracle"], default="oracle",
        help="'none' = no AIS (strong ship → UNCERTAIN). "
             "'oracle' = stub AIS w/ ground truth (best-case end-to-end).",
    )
    p.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    p.add_argument("--out", default="data/eval/end_to_end.json")
    p.add_argument("--limit-per-site", type=int, default=300)
    args = p.parse_args()

    cnn = CNNV7Classifier(args.model)

    conformal_path = Path(args.conformal)
    if conformal_path.exists():
        conformal = ConformalPredictor.load(conformal_path)
    else:
        conformal = ConformalPredictor.uncalibrated(model_checkpoint=args.model)

    oracle: OracleGFWAdapter | None = None
    if args.ais_mode == "oracle":
        oracle = OracleGFWAdapter()
        engine = DecisionEngine(gfw=oracle, conformal=conformal)
    else:
        engine = DecisionEngine(gfw=None, conformal=conformal)

    # Group rows per site
    rows_by_site: dict[str, list[dict]] = defaultdict(list)
    for fp in args.files:
        path = Path(fp)
        if not path.exists():
            log.warning("file_missing", path=fp)
            continue
        with path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows_by_site[site_key(r)].append(r)

    if args.limit_per_site > 0:
        rng = np.random.default_rng(42)
        for site, rows in rows_by_site.items():
            if len(rows) > args.limit_per_site:
                idx = rng.choice(len(rows), args.limit_per_site, replace=False)
                rows_by_site[site] = [rows[i] for i in idx]

    results: dict[str, dict] = {}
    for site in sorted(rows_by_site):
        log.info("site_eval_start", site=site, n=len(rows_by_site[site]))
        results[site] = await evaluate_site(
            cnn, engine, oracle, rows_by_site[site],
        )

    # Aggregate
    total = {o: 0 for o in ALL_OUTCOMES}
    total_n = 0
    for r in results.values():
        total_n += r["n"]
        for o, c in r["outcomes"].items():
            total[o] += c

    n_committed = sum(total[o] for o in ALL_OUTCOMES if o != OUTCOME_ABSTAIN)
    n_correct = sum(total[o] for o in CORRECT_OUTCOMES)
    n_wrong = sum(total[o] for o in WRONG_OUTCOMES)
    n_abstain = total[OUTCOME_ABSTAIN]

    overall = {
        "n": total_n,
        "correct_when_committed": (
            round(n_correct / n_committed, 4) if n_committed else None
        ),
        "abstain_rate": (
            round(n_abstain / total_n, 4) if total_n else None
        ),
        "wrong_rate": (
            round(n_wrong / total_n, 4) if total_n else None
        ),
        "outcomes": total,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model,
        "conformal": str(conformal_path) if conformal_path.exists() else None,
        "ais_mode": args.ais_mode,
        "overall": overall,
        "per_site": results,
    }, indent=2))

    # Pretty print
    print(f"\n=== End-to-end Decision eval (ais_mode={args.ais_mode}) ===\n")
    print(f"{'site':<30} {'n':>5} {'commit_acc':>11} {'abstain':>9} {'wrong':>8}")
    print("-" * 70)
    for site, r in sorted(results.items(), key=lambda x: -x[1]["n"]):
        ca = f"{r['correct_when_committed']:.1%}" if r['correct_when_committed'] is not None else "  -  "
        ab = f"{r['abstain_rate']:.1%}" if r['abstain_rate'] is not None else "  -  "
        wr = f"{r['wrong_rate']:.1%}" if r['wrong_rate'] is not None else "  -  "
        print(f"{site:<30} {r['n']:>5} {ca:>11} {ab:>9} {wr:>8}")

    print("-" * 70)
    ca = f"{overall['correct_when_committed']:.1%}" if overall['correct_when_committed'] is not None else "  -  "
    ab = f"{overall['abstain_rate']:.1%}" if overall['abstain_rate'] is not None else "  -  "
    wr = f"{overall['wrong_rate']:.1%}" if overall['wrong_rate'] is not None else "  -  "
    print(f"{'OVERALL':<30} {total_n:>5} {ca:>11} {ab:>9} {wr:>8}")

    print(f"\nOutcome breakdown (overall):")
    for o in ALL_OUTCOMES:
        print(f"  {o:<22} {total[o]:>5}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
