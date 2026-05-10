"""End-to-end programmatic check after Plan B + Tier 1 wiring.

What this verifies (without Ollama):
- Step 4 finetune_adapter tool returns the shape Step4 extractor reads
- Step 5 calibrate_conformal handles the post-adapter distribution
- Step 7 simulate_detection persists a decision and explain_decision
  finds it
- All extractors (_extract_step4, _extract_step5, _extract_step7)
  populate the OnboardingContext correctly
- The final ctx has every field site_registered() reads

If any field is None where the CLI's site_registered table expects a
value, the real `os onboard` will look broken at the end. This script
catches that before a juror does.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ocean_sentinel.gemma.agent import ToolCallResult
from ocean_sentinel.gemma.onboarding.context import OnboardingContext
from ocean_sentinel.gemma.onboarding.steps import (
    _extract_step4,
    _extract_step5,
    _extract_step7,
)
from ocean_sentinel.gemma.tools import dispatch


SITE_ID = "e2e-check"
AMBIENT = "data/synthetic/tropical_reef_3min.wav"
TEST_CLIP = "data/deepship/Tug/49.wav"


def main() -> int:
    if not Path(AMBIENT).exists() or not Path(TEST_CLIP).exists():
        print(f"FAIL — missing input audio: ambient={Path(AMBIENT).exists()}, "
              f"clip={Path(TEST_CLIP).exists()}")
        return 1

    # Fresh per-site state
    site_dir = Path("data/sites") / SITE_ID
    if site_dir.exists():
        shutil.rmtree(site_dir)

    ctx = OnboardingContext(site_id=SITE_ID, ambient_source=AMBIENT)
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        status = "OK" if condition else "FAIL"
        print(f"  [{status}] {name}{(': ' + detail) if detail else ''}")
        if not condition:
            failures.append(name)

    print("=" * 72)
    print("E2E PROGRAMMATIC CHECK — POST PLAN B + TIER 1")
    print("=" * 72)

    # ── Step 4: finetune_adapter ──────────────────────────────────────
    print("\n[Step 4] finetune_adapter")
    r4 = dispatch("finetune_adapter", {
        "site_id": SITE_ID,
        "ambient_source": AMBIENT,
    })
    print(f"  result keys: {sorted(r4.keys())}")
    check("step4 ok=True", r4.get("ok") is True, str(r4.get("error", "")))
    check("final_val_acc present", r4.get("final_val_acc") is not None)
    check("assessment present", r4.get("assessment") is not None,
          str(r4.get("assessment")))
    check("recommendation present", r4.get("recommendation") is not None)
    check("adapter_path present", r4.get("adapter_path") is not None)

    # Run through the actual extractor used by the live flow
    _extract_step4(ToolCallResult(name="finetune_adapter", result=r4), ctx)
    check("ctx.adapter_val_acc populated", ctx.adapter_val_acc is not None,
          str(ctx.adapter_val_acc))
    check("ctx.adapter_assessment populated", ctx.adapter_assessment is not None,
          str(ctx.adapter_assessment))
    check("ctx.adapter_recommendation populated", ctx.adapter_recommendation is not None)

    # ── Step 5: calibrate_conformal (uses adapter from step 4) ─────────
    print("\n[Step 5] calibrate_conformal — uses adapter from step 4")
    r5 = dispatch("calibrate_conformal", {
        "site_id": SITE_ID, "ambient_source": AMBIENT, "alpha": 0.05,
    })
    print(f"  result keys: {sorted(r5.keys())}")
    check("step5 ok=True", r5.get("ok") is True, str(r5.get("error", "")))
    check("threshold_p present", r5.get("threshold_p") is not None,
          str(r5.get("threshold_p")))

    _extract_step5(ToolCallResult(name="calibrate_conformal", result=r5), ctx)
    check("ctx.conformal_threshold_p populated",
          ctx.conformal_threshold_p is not None,
          str(ctx.conformal_threshold_p))

    # ── Step 7: simulate_detection (persists decision) ─────────────────
    print("\n[Step 7] simulate_detection on real Tug clip")
    r7a = dispatch("simulate_detection", {
        "site_id": SITE_ID, "clip": TEST_CLIP,
    })
    print(f"  decision_tier: {r7a.get('decision_tier')}  "
          f"cnn_p: {r7a.get('cnn_confidence')}  "
          f"thresh: {r7a.get('conformal_threshold')}")
    check("step7 simulate ok=True", r7a.get("ok") is True,
          str(r7a.get("error", "")))
    check("decision_id present", r7a.get("decision_id") is not None,
          str(r7a.get("decision_id")))

    decision_id = r7a.get("decision_id")
    decision_record = Path("data/decisions") / f"{decision_id}.json"
    check("decision persisted to disk", decision_record.exists(),
          str(decision_record))

    # explain_decision with real Gemma multimodal disabled (no Ollama needed)
    print("\n[Step 7+] explain_decision (multimodal disabled for this check)")
    import os
    os.environ["OS_DISABLE_MULTIMODAL"] = "1"
    r7b = dispatch("explain_decision", {
        "decision_id": decision_id, "modality": "spectrogram+text",
    })
    print(f"  result keys: {sorted(r7b.keys())}")
    check("step7 explain ok=True", r7b.get("ok") is True,
          str(r7b.get("error", "")))
    check("spectrogram_path present", r7b.get("spectrogram_path") is not None)
    check("explanation present", r7b.get("explanation") is not None)
    check("trace has real features",
          all(k in (r7b.get("trace") or {}) for k in (
              "peak_frequency_hz", "spectral_centroid_hz",
              "low_band_energy_fraction_below_200hz",
          )))
    check("trace lacks fabricated features",
          all(k not in (r7b.get("trace") or {}) for k in (
              "blade_rate_hz", "harmonics",
          )))

    # extractor coverage
    _extract_step7(ToolCallResult(name="simulate_detection", result=r7a), ctx)
    _extract_step7(ToolCallResult(name="explain_decision", result=r7b), ctx)
    check("ctx.test_decision_id populated", ctx.test_decision_id is not None)
    check("ctx.test_decision_tier populated", ctx.test_decision_tier is not None,
          str(ctx.test_decision_tier))
    check("ctx.test_decision_severity populated",
          ctx.test_decision_severity is not None,
          str(ctx.test_decision_severity))
    check("ctx.test_explain_summary populated",
          ctx.test_explain_summary is not None)

    # ── final ctx state matches what site_registered() displays ────────
    print("\n[final ctx → site_registered] check fields the CLI table reads")
    fields_for_table = {
        "site_id":                 ctx.site_id,
        "ambient_source":          ctx.ambient_source,
        "adapter_val_acc":         ctx.adapter_val_acc,
        "conformal_threshold_p":   ctx.conformal_threshold_p,
        "test_decision_tier":      ctx.test_decision_tier,
    }
    for name, val in fields_for_table.items():
        check(f"ctx.{name} populated", val is not None, str(val))

    # ── verdict ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    if not failures:
        print("  ALL CHECKS PASSED ✓")
        print("=" * 72)
        return 0
    print(f"  {len(failures)} FAILURE(S):")
    for f in failures:
        print(f"   - {f}")
    print("=" * 72)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
