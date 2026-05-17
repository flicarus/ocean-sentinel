"""Real fine-tune end-to-end on the synthetic reef.

1. Train per-site adapter on the user's ambient with held-out vessel
   recall floor. Reports val_acc and holdout_recall before/after.
2. Recalibrate conformal threshold on the ADAPTED distribution.
3. Print the new threshold so we can plug it into measure_safeguard_impact.

Use this script after generating the reef audio:
    PYTHONPATH=src venv/bin/python scripts/test_drastically_different_site.py
    PYTHONPATH=src venv/bin/python scripts/run_real_finetune.py
    PYTHONPATH=src venv/bin/python scripts/measure_safeguard_impact.py
"""
from __future__ import annotations

from pathlib import Path

from ocean_sentinel.gemma.site_adapter import train_site_adapter
from ocean_sentinel.gemma.conformal import calibrate_conformal_real

SITE_ID = "reef-test-measure"
AMBIENT = "data/synthetic/tropical_reef_3min.wav"


def main() -> None:
    if not Path(AMBIENT).exists():
        print(f"Missing {AMBIENT}. Run test_drastically_different_site.py first.")
        return

    print("=" * 72)
    print("REAL ADAPTER FINE-TUNE ON SYNTHETIC REEF")
    print("=" * 72)
    print(f"  ambient: {AMBIENT}")
    print(f"  site_id: {SITE_ID}")

    print("\n[step 1] training per-site adapter (12 epochs, holdout floor=0.85)...\n")
    train_out = train_site_adapter(site_id=SITE_ID, ambient_source=AMBIENT)
    if not train_out.get("ok"):
        print(f"  TRAINING FAILED: {train_out.get('error')}")
        return

    print(f"\n[training result]")
    print(f"  adapter_path:               {train_out['adapter_path']}")
    print(f"  epochs_trained:             {train_out['epochs_trained']}")
    print(f"  early_stopped:              {train_out['early_stopped']}")
    if train_out.get("early_stop_reason"):
        print(f"  early_stop_reason:          {train_out['early_stop_reason']}")
    print(f"\n  ── ambient (user) ──")
    print(f"  val_acc_before:             {train_out['val_acc_before']:.4f}  "
          f"(% of ambient correctly classified as not_ship by base CNN)")
    print(f"  val_acc_after:              {train_out['val_acc_after']:.4f}  "
          f"(after adapter)")
    print(f"  median_ship_prob_before:    {train_out['median_ship_prob_before']:.4f}")
    print(f"  median_ship_prob_after:     {train_out['median_ship_prob_after']:.4f}")
    print(f"\n  ── held-out training-distribution vessels (DeepShip) ──")
    print(f"  holdout_recall_before:      {train_out['holdout_recall_before']:.4f}")
    print(f"  holdout_recall_after:       {train_out['holdout_recall_after']:.4f}")

    print("\n[step 2] recalibrating conformal on the adapted distribution...\n")
    cal = calibrate_conformal_real(
        site_id=SITE_ID, ambient_source=AMBIENT, alpha=0.05,
    )
    if not cal.get("ok"):
        print(f"  CALIBRATION FAILED: {cal.get('error')}")
        return
    print(f"  per-site threshold (post-adapter):  {cal['threshold_p']}")
    print(f"  median ship_prob (adapted ambient): {cal['median_ambient_ship_prob']}")
    print(f"  empirical coverage:                 {cal['coverage']}")
    if cal.get("contract_warning"):
        print(f"  ⚠ contract_warning:                 {cal['contract_warning']}")
    print(f"  summary:                            {cal['summary']}")


if __name__ == "__main__":
    main()
