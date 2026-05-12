# Limitations & honest scope

This document inventories what Ocean Sentinel **has** been measured on,
what it **has not**, and which gaps are deferred to pilot / post-pilot
versus genuinely out of scope.

The goal is to make every claim auditable: a reader of the Kaggle
write-up can come here, find the underlying eval data file in
`data/eval/`, and trace exactly how the headline number was produced.

---

## What was measured

### Quality

| Metric | Value | n | Wilson 95 % CI | Source |
|---|---|---|---|---|
| **v7.6 + per-site calibration, held-out test split (PRIMARY)** | **96.4 %** | 4044 | — | `data/calibration/per_site_thresholds_v7_6_honest.json` |
| **v7.6 + per-site calibration, OOD (8 sites × fresh dates)** | **96.0 %** | 4414 | — | `scripts/eval_ood.py` (fresh pull 2026-05-12) |
| v7.6 vanilla (threshold 0.5) | 89.3 % | 8082 | — | `data/eval/per_site_v7_6.json` |
| v7.5 baseline (27 sites) | 87.0 % | 6349 | — | `data/eval/per_site_v7_5.json` |
| Point-robinson recall, v7.5 → v7.6+cal on OOD date | 13.4 % → **100 %** | 679 (fresh) | — | `scripts/eval_ood.py` 2024-08-15 pull |
| MBARI ambient recovery, vanilla → calibrated | 0.3 % → **100 %** | 300 | — | threshold 0.88 |
| Day-9 LOHO accuracy (raw v7.4) | 89.98 % | 4011 | — | `data/eval/per_site_v7_4.json` |
| Day-9 LOHO end-to-end (full pipeline) | 97.2 % | 4011 | — | `data/eval/per_site_v7_4.json` |
| Held-out memorisation probe (raw, vessels only) | 100 % | 29 | 88.0 – 100 % | `data/eval/benchmark_v7_4.json` |
| Held-out probe (full pipeline) | 100 % | 35 | 90.0 – 100 % | `data/eval/benchmark_v7_4.json` |
| OOD synthetic-reef + per-site adapter (FA reduction) | 90.3 % → 3.2 % | 31 windows | — | `scripts/measure_safeguard_impact.py` |
| Cross-station eval (NOAA SanctSound, untrained sites) | median ship_prob 0.849, fire 80 %, abstain 0 % | 20 windows × 4 sites | — | `data/eval/external_sanctsound.json` |
| Cosine-similarity correlation with accuracy (n=14, post-extension) | r = −0.23 (p=0.41), ρ = −0.09 (p=0.75) | 14 | — | `data/eval/loho_correlation_extended.json` |

Both the Pearson and Spearman correlations are weak and not
statistically significant (p > 0.4). This is itself a finding: with
n=14 sites the data does not support a usable predictive relationship
between cosine similarity to the training distribution and per-site
accuracy. The earlier n=7 result (r = −0.57, p = 0.18) was a
small-sample artefact that did not survive doubling the evaluation set.
See [`docs/empirical_findings.md`](empirical_findings.md).

### Operations

| Metric | Value | Source |
|---|---|---|
| End-to-end inference latency (median) | **19 ms** | `data/eval/inference_perf.json` |
| End-to-end inference latency (p95) | 22 ms | `data/eval/inference_perf.json` |
| Real-time factor | **3210 ×** | (60 s clip / 19 ms) |
| Sustained throughput | 52 clips / sec | back-to-back, MPS, single process |
| Cold start (model load) | 69 ms | M5 Pro MPS |
| Resident memory (model) | 18 MB | RSS delta after `CNNV7Classifier(...)` |
| Per-site adapter size | ~33 k params | residual MLP, init-as-identity |
| Conformal calibration provable bound | FA ≤ α with finite-sample correction | Lei 2018 |

### Pipeline contract

| Claim | Verified by |
|---|---|
| Gemma 4 actually reads spectrograms (not just text trace) | `scripts/verify_gemma_actually_reads.py` — adversarial test, white noise mislabelled DARK_VESSEL → flagged inconsistent |
| Decision tier maps correctly onto ground-truth labels | `scripts/e2e_post_planb_check.py` |
| `os monitor` events reach `/api/events/feed` | end-to-end live demo with `--replay` |
| Per-site adapter actually changes inference path | measured in `scripts/measure_safeguard_impact.py` (ship_prob shifts 0.91 → 0.75 after adapter loaded) |
| Per-site threshold actually applied at inference | `scripts/eval_ood.py` — point-robinson 62.7 % → 100 % on unseen 2024-08-15 date with thr=0.02 vs default 0.5 |

### Per-site threshold calibration — methodology

Day-15 finding: balanced-sampler training (1/n site_count weighting)
fixes class-rare-site regressions (point-robinson +48 pp) but
de-weights the dominant ambient class (MBARI: 67 % → 2.9 % effective
training). The model loses its "MBARI feature → ambient" prior and
hallucinates ships on MBARI ambient (97 pp regression). Industry-
standard fix: per-site decision threshold calibration.

Methodology (held-out cal/test split, seed=42):
1. For each site, split eval data 50/50 deterministically.
2. On the **calibration half only**, sweep threshold ∈ {0.02, 0.04, …, 0.98} ∪ {0.5}.
3. Pick threshold maximising calibration-half accuracy.
4. Report **test-half accuracy** as the deployment number.
5. **Safety guard**: skip thresholds where `test_acc_tuned < test_acc_default - 1pp`. One site (hi03) had a high-leverage cal threshold (0.08) that did not generalise → fall back to default 0.5.

Inference wiring: `data/calibration/per_site_thresholds_v7_6.json`
maps `source_id` → threshold. `CNNV7Classifier.predict(spec, source_id=...)`
applies the matched threshold; falls back to 0.5 when no match.
Auto-loaded by `cnn_inference.py` from the default path.

Saved thresholds (10 sites): MBARI 0.88, sanctsound 0.92, sb01 0.94,
ci02 0.86, oc01 0.14, mb02 0.06, sb03 0.08, point-robinson 0.02,
deepship 0.02, shipsear 0.52. All other sites use 0.5.

Trade-off honestly disclosed: MBARI threshold 0.88 means "only call
ship when very confident". Optimised for the MBARI hydrophone class
which is not part of the deployment fleet — MBARI was an ambient
data source, not a production site. For the deployed Orcasound /
SanctSound network the calibration uniformly recovers or improves
accuracy. See `data/eval/per_site_v7_6.json` for the per-site delta
breakdown vs v7.5.

---

## What was NOT measured

### Quality gaps

1. **Real ocean ambient (no vessels)**
   Reason: the only synthetic ambients in our test set are deterministic
   pink-tilt or band-limited noise. The held-out `data/diagnostic/sanctsound/audio/control_5min.wav`
   is named "control" but the CNN consistently returns ship_prob ≥ 0.79 on
   five offsets through it; without ground truth verification we cannot
   distinguish "model is wrong" from "control file actually contains
   vessels". Day-11 SanctSound audit found label issues in similar files.
   **Pilot deliverable**: 30+ verified ambient recordings from a single
   reserve, ground-truthed by the partner organisation.

2. **Cross-sensor evaluation with ground truth**
   Reason: we now have *unsupervised* cross-station evidence — 20
   windows across 4 NOAA SanctSound stations (Channel Islands, Gray's
   Reef, Hawaii, Monterey Bay) that the model has never seen. Pipeline
   runs cleanly: 0 % UNCERTAIN abstentions, latency holds at scale,
   median ship_prob 0.849, fire rate 80 %. But we lack per-window AIS
   ground truth — these are MPAs near shipping lanes where vessels are
   plausibly present, so a high fire rate is consistent with both
   "model is right" and "model over-fires". The mb03 (Monterey Bay)
   station produces near-constant 0.92 across all 5 widely-spaced
   windows (~0.0005 variance) even though the underlying audio is *not*
   flat (RMS 0.024–0.074, spectral centroid 86–216 Hz across the same
   windows). Two readings are consistent with this: (a) the model
   detects a persistent vessel-signature mel-spec pattern that the
   simple time-domain stats miss, plausible for a busy commercial port
   site; or (b) the model has saturated for this station's acoustic
   character. Conversely ci03 has near-flat audio statistics
   (RMS ~0.0075 across all windows) yet variable model output
   (0.45–0.91), which is the *expected* behaviour of a model that
   responds to spectral structure rather than total energy. Neither
   reading can be confirmed without AIS overlay.
   **Pilot deliverable**: deploy on at least one site whose hardware
   was not in the training corpus and measure FA over a 30-day window
   with NOAA's AIS-correlated detection products as ground truth.

3. **Independent vessel dataset**
   Every available vessel clip on disk (DeepShip, ShipsEar, MBARI
   ais-correlated subsets) was in training to some degree. ShipsEar:
   2223/2223 of the available .npy specs are in training. We bench on
   held-out clip *numbers* but not held-out *campaigns*.
   **Post-pilot**: add VTUAD (Vancouver Turning Underwater Acoustic
   Dataset) or NOAA Pacific Marine Mammal Lab vessel cuts as a fully
   held-out validation set.

4. **Long-term drift**
   Reason: 7-day hackathon timeline. Production sites deployed for 6+
   months will see seasonal acoustic changes, biological migrations,
   and hardware aging.
   **Continual-learning loop** (`os refresh` + warm-start adapter)
   is the architectural answer; *measuring* its effectiveness over
   time requires a real deployment.

5. **Adversarial audio**
   Reason: limited research value at this maturity. Acoustic spoofing
   in the wild is rare; vessel operators trying to fool a hydrophone
   network would change vessels, not signal characteristics.
   **Out of scope for v1**.

### Operations gaps

1. **Throughput at scale across many sites concurrently**
   Single-process throughput is 52 clips/sec on one MPS device. Multi-site
   deployments would benefit from process-level parallelism and a
   shared model service. Not measured.

2. **Network resilience** — if `/api/events/feed` drops connection
   mid-monitor, behaviour is "events still written to JSONL on disk,
   feed endpoint reads from disk on next poll". Verified by code
   inspection but not by chaos testing.

3. **GPU vs CPU vs Edge SoC**
   Tested only on M5 Pro MPS. CUDA path exists but unbenchmarked.
   Embedded ARM (Raspberry Pi class) would need quantisation work.

---

## Statistical caveats

Sample sizes for held-out probe are deliberately small because the
universe of *truly* held-out clips is bounded by what training never
touched. For per-class CIs:

```
Cargo          16 / 16 = 100 %    Wilson 95 % CI 80.6 – 100 %
Passengership   8 /  8 = 100 %    Wilson 95 % CI 67.6 – 100 %
Tanker          5 /  5 = 100 %    Wilson 95 % CI 56.6 – 100 %
```

The Tanker CI (~57 %) is the loosest. We do **not** claim per-class
performance from this probe — only that the model is consistent with
the day-9 LOHO 90 % overall figure on a small sanity set.

---

## What this means for the user

A reserve manager onboarding a hydrophone today should expect:

- **Latency**: alerts arrive within ~20 ms of clip availability.
  Bottleneck is audio capture / file I/O, not the model.
- **Accuracy on training-distribution sites**: ~90 % raw / ~97 % e2e
  per the day-9 LOHO benchmark.
- **Behaviour on out-of-distribution sites**: the per-site adapter
  drives false-alarm rate down to the user-chosen alpha (5 % default)
  even when base v7.4 is confused; demonstrated on a synthetic reef
  where base FA was 90.3 %.
- **Honest abstention**: the evidential head abstains (UNCERTAIN tier)
  on borderline-confidence clips rather than producing fake-confident
  decisions. Abstentions go to `flag_for_review` queue.
- **Continuous improvement**: `os refresh` re-fits the adapter and
  recalibrates threshold on accumulated ambient. Effectiveness over
  time is part of pilot deliverables, not measured here.

Anything that requires a guarantee — e.g. "this evidence is admissible
in a fishing-violation prosecution" — is **explicitly out of scope** of
this hackathon submission. Such use would require independent
verification, certified test sets, regulatory approval, and a longer
operational track record.
