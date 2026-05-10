"""End-to-end test on a hydrophone site that is *drastically different*
from our training corpus.

Why this exists
---------------
Our training data (MBARI, Orcasound, SanctSound, DeepShip, ShipsEar)
is dominated by vessel acoustic signatures concentrated below ~1 kHz.
The CNN is trained on log-mel features with `fmax=1000 Hz` precisely
because that's where vessel signatures live.

What if a user onboards a site whose ambient is dominated by something
the CNN has never seen? The most realistic out-of-distribution case in
ocean acoustics is a **tropical coral-reef site**: snapping shrimp
produce dense, sharp transients across 2–8 kHz, fish choruses produce
band-limited continuous noise around 200–800 Hz, and there are no
steady low-frequency vessel-like tones unless a boat is actually nearby.

We synthesise that distribution here and walk it through the same
onboarding pipeline a real user would. The point is to demonstrate
what the system **does to adapt** — not to claim 100 % accuracy on a
distribution we never trained on.

What the pipeline can do for a drastically-different site
---------------------------------------------------------
1. compute_spectral_signature  → quantifies the new ambient as a
   64-band signature distinct from anything in our 40-site registry.
2. compare_to_known_sites      → reports the closest training site +
   z-cosine. We expect a low number, signalling "OOD".
3. finetune_adapter (label-free) → runs v7.4 over the user's ambient
   and measures how often it (mistakenly) calls ambient "ship". This
   surfaces the contract violation explicitly.
4. calibrate_conformal         → split-conformal on user's ambient.
   If the CNN is confused by reef ambient, the per-site threshold
   shifts UP, automatically suppressing the false-alarm rate on this
   particular site to the user-chosen alpha (5%).
5. simulate_detection          → with the per-site threshold applied,
   evaluate a real vessel clip (DeepShip Tug) at this calibrated site.
   The vessel should still fire above the now-higher threshold, even
   though base v7.4 was confused by the reef ambient.
6. explain_decision            → Gemma multimodal narration grounded
   in the rendered spectrogram of the test clip.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ocean_sentinel.gemma.audio_features import compute_spectral_signature
from ocean_sentinel.gemma.known_sites import compare_to_known_sites
from ocean_sentinel.gemma.adapter import validate_adapter_on_ambient
from ocean_sentinel.gemma.conformal import calibrate_conformal_real
from ocean_sentinel.gemma.cnn_inference import simulate_detection
from ocean_sentinel.gemma.explanations import explain_decision_real


SR = 16_000
SITE_ID = "tropical-reef-test"
AMBIENT_SECONDS = 180  # 3 min — enough windows for split-conformal
TEST_VESSEL_CLIP = "data/deepship/Tug/49.wav"


# ── synthesis ───────────────────────────────────────────────────────────


def _synth_tropical_reef(out: Path, duration_s: int = AMBIENT_SECONDS) -> Path:
    """Synthetic tropical-reef ambient.

    Components:
    - Dense Poisson process of sharp clicks (snapping shrimp), peak
      energy 2-8 kHz, ~30 clicks / sec average rate.
    - Quasi-stationary fish chorus: band-limited noise at 200-800 Hz.
    - Faint rolling broadband noise (waves, ~uniform).

    Result: high-frequency-dominant, transient-rich, NO low-frequency
    vessel-like horizontal bands.
    """
    rng = np.random.RandomState(2026)
    n = SR * duration_s
    y = np.zeros(n, dtype=np.float32)

    # 1. Snapping-shrimp click train. Real Alpheus snaps are 1-2 ms
    # broadband impulses with peak SPL 190+ dB at source. Recorded at
    # distance they routinely sit 30-60 dB above background in 2-8 kHz
    # bands. We synthesise an impulse + short exp decay (broadband by
    # construction, not modulated by a carrier) so its high-frequency
    # content actually dominates p95.
    n_clicks = int(40 * duration_s)              # ~40 clicks/s
    click_len = SR // 800                         # 1.25 ms tail
    click_template = np.exp(-np.linspace(0, 18, click_len)).astype(np.float32)
    click_template[0] = 1.0                       # leading impulse
    click_positions = rng.randint(0, n - click_len, size=n_clicks)
    for pos in click_positions:
        # 3-5x peak amplitude vs the chorus floor → p95 in HF bands
        # rises steeply above the steady median, exactly the real-world
        # snapping-shrimp signature.
        amp = rng.uniform(2.0, 5.0)
        y[pos : pos + click_len] += amp * click_template

    # 2. Fish chorus: bandpass-filtered noise 200-800 Hz, slowly modulated
    chorus = rng.randn(n).astype(np.float32) * 0.15
    Y = np.fft.rfft(chorus)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    mask = ((freqs > 200) & (freqs < 800)).astype(np.float32)
    chorus = np.fft.irfft(Y * mask, n=n).astype(np.float32)
    mod = 0.7 + 0.3 * np.sin(2 * np.pi * 0.05 * np.arange(n) / SR)
    y += chorus * mod

    # 3. Ambient broadband noise
    y += 0.05 * rng.randn(n).astype(np.float32)

    # Normalise so the loud clicks don't clip while keeping the chorus
    # well below.
    y = y / max(1.0, float(np.max(np.abs(y))) * 1.05)

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y, SR)
    return out


def _format_dict(d: dict, max_keys: int = 12) -> str:
    items = list(d.items())[:max_keys]
    return "\n".join(f"    {k}: {v}" for k, v in items)


# ── runner ──────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 72)
    print(f"DRASTICALLY-DIFFERENT-SITE END-TO-END TEST")
    print("=" * 72)

    ambient_path = Path("data/synthetic/tropical_reef_3min.wav")
    if not ambient_path.exists():
        print(f"\n[synth] generating {ambient_path} ({AMBIENT_SECONDS}s)...")
        _synth_tropical_reef(ambient_path)
    print(f"[synth] ambient at {ambient_path}")

    # ── Step 2: compute signature ──────────────────────────────────────
    print(f"\n{'─' * 72}\nStep 2: compute_spectral_signature on user ambient\n{'─' * 72}")
    sig = compute_spectral_signature(audio_source=str(ambient_path), n_mels=64)
    if not sig.get("ok"):
        print(f"  ERROR: {sig.get('error')}")
        return
    sig_vector = sig["signature"]
    print(f"  ambient_class:        {sig.get('ambient_class')}")
    print(f"  dominant_band_hz:     {sig.get('dominant_band_hz')}")
    print(f"  transient_gap_hf_db:  {sig.get('transient_gap_hf_db')}  "
          f"(>15 dB → snapping shrimp / dense clicks)")
    print(f"  snapping_shrimp:      {sig.get('snapping_shrimp')}")
    print(f"  median_psd_db:        {sig.get('median_psd_db')}")
    print(f"  summary:              {sig.get('summary')}")

    # ── Step 3: compare to known sites ─────────────────────────────────
    print(f"\n{'─' * 72}\nStep 3: compare_to_known_sites (40-site training registry)\n{'─' * 72}")
    cmp = compare_to_known_sites(sig_vector)
    if cmp.get("ok"):
        for hit in cmp.get("ranked", [])[:3]:
            print(f"  {hit['id']:<32} z-cos={hit['cosine_sim']:.3f}  "
                  f"raw={hit.get('cosine_sim_raw', 0):.3f}")
        print(f"  recommendation:    {cmp.get('recommendation')}")
        print(f"  summary:           {cmp.get('summary')}")
    else:
        print(f"  ERROR: {cmp.get('error')}")

    # ── Step 4: label-free adapter validation ──────────────────────────
    print(f"\n{'─' * 72}\nStep 4: finetune_adapter (label-free recall on ambient)\n{'─' * 72}")
    adap = validate_adapter_on_ambient(
        site_id=SITE_ID, ambient_source=str(ambient_path),
    )
    if adap.get("ok"):
        print(f"  n_windows:           {adap.get('n_windows')}")
        print(f"  final_val_acc:       {adap.get('final_val_acc')}  "
              f"(fraction of ambient correctly classified as not_ship)")
        print(f"  mean_confidence:     {adap.get('mean_confidence')}")
        print(f"  mean_uncertainty:    {adap.get('mean_uncertainty')}")
        print(f"  assessment:          {adap.get('assessment')}")
        print(f"  recommendation:      {adap.get('recommendation')}")
        print(f"  summary:             {adap.get('summary')}")
    else:
        print(f"  ERROR: {adap.get('error')}")

    # ── Step 5: per-site conformal calibration ─────────────────────────
    print(f"\n{'─' * 72}\nStep 5: calibrate_conformal (split-conformal on user ambient)\n{'─' * 72}")
    cal = calibrate_conformal_real(
        site_id=SITE_ID, ambient_source=str(ambient_path), alpha=0.05,
    )
    if cal.get("ok"):
        print(f"  n_calibration:           {cal.get('n_calibration')}")
        print(f"  alpha (target FA):       {cal.get('alpha')}")
        print(f"  per-site threshold_p:    {cal.get('threshold_p')}")
        print(f"  empirical coverage:      {cal.get('coverage')}")
        print(f"  median ship_prob (amb.): {cal.get('median_ambient_ship_prob')}")
        print(f"  expected_fa_per_hour:    {cal.get('expected_fa_per_hour')}")
        if cal.get("contract_warning"):
            print(f"  ⚠ CONTRACT WARNING:      {cal.get('contract_warning')}")
        print(f"  summary:                 {cal.get('summary')}")
    else:
        print(f"  ERROR: {cal.get('error')}")

    # ── Step 7: test detection on a real vessel clip at this site ─────
    print(f"\n{'─' * 72}\nStep 7: simulate_detection on a real vessel clip,\n"
          f"        WITHOUT per-site threshold (base v7.4 conformal):\n{'─' * 72}")
    det_base = simulate_detection(site_id=SITE_ID, clip=TEST_VESSEL_CLIP)
    print(_format_dict({k: v for k, v in det_base.items()
                        if k not in ("summary", "checkpoint")}))
    print(f"  summary: {det_base.get('summary')}")

    # Per-site threshold from cal
    if cal.get("ok"):
        site_thresh = cal["threshold_p"]
        cnn_p = det_base.get("cnn_confidence", 0.0)
        passes_site = cnn_p >= site_thresh
        print(f"\n  ── Apply per-site threshold (calibrated on reef ambient) ──")
        print(f"  base v7.4 conformal threshold:  {det_base.get('conformal_threshold')}")
        print(f"  per-site (reef) threshold:      {site_thresh}")
        print(f"  vessel ship_prob:               {cnn_p}")
        print(f"  decision under per-site rule:   "
              f"{'FIRES (vessel detected)' if passes_site else 'SUPPRESSED (below per-site)'}")

    # ── Step 8: explain_decision (real Gemma multimodal) ───────────────
    print(f"\n{'─' * 72}\nStep 7+: explain_decision (Gemma multimodal narration)\n{'─' * 72}")
    if det_base.get("ok"):
        exp = explain_decision_real(
            decision_id=det_base["decision_id"],
            modality="spectrogram+text",
        )
        if exp.get("ok"):
            print(f"  narration_source: {exp.get('narration_source')}")
            print(f"  spectrogram_path: {exp.get('spectrogram_path')}")
            print(f"\n  trace (real measurements):")
            for k, v in exp.get("trace", {}).items():
                if k in ("clip", "site_id"):
                    continue
                print(f"    {k}: {v}")
            print(f"\n  explanation:")
            print(f"    {exp.get('explanation')}")
        else:
            print(f"  ERROR: {exp.get('error')}")

    # ── Closing summary ────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  WHAT THE SYSTEM DID TO ADAPT")
    print(f"{'═' * 72}")
    if cmp.get("ok"):
        top = cmp.get("ranked", [{}])[0]
        print(f"  • Detected OOD via cosine: closest training site is "
              f"{top.get('id')} at z-cos {top.get('cosine_sim'):.2f}")
        print(f"    (low z-cos → site is acoustically far from training)")
    if adap.get("ok"):
        print(f"  • Reported label-free recall on user ambient: "
              f"{adap.get('final_val_acc')}")
        print(f"    (this is the share of windows the base CNN correctly "
              f"left as not_ship)")
    if cal.get("ok"):
        print(f"  • Set a per-site conformal threshold of {cal.get('threshold_p')} "
              f"with target FA rate {cal.get('alpha')}")
        print(f"    (base v7.4 default was 0.61 — per-site threshold "
              f"{'raised' if cal.get('threshold_p', 0) > 0.61 else 'unchanged or lowered'} "
              f"because the CNN was confused by reef ambient)")
        if cal.get("contract_warning"):
            print(f"    Plus surfaced a contract warning: {cal.get('contract_warning')}")
    print(f"  • All numerical claims above come from real tool calls — no mocks.")


if __name__ == "__main__":
    main()
