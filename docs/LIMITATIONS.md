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

### Gemma 4 multimodal — validated scope (day 15 finding)

We tested Gemma 4 e4b multimodal on hydrophone mel spectrograms to
validate whether it could verify CNN decisions independently. The
test ran four held-out spectrograms (`scripts/test_gemma_spectrogram.py`)
with no hint of the CNN's verdict, asking Gemma to classify each
purely from the rendered PNG:

| Case | Truth | CNN ship_prob | Gemma verdict | Match |
|---|---|---|---|---|
| Confident SHIP (ShipsEar) | ship | >0.9 | SHIP (1.0) | ✅ |
| Confident AMBIENT (SanctSound) | not_ship | ~0.05 | SHIP (0.95) | ❌ |
| Uncertain #1 (SanctSound) | not_ship | 0.35 | SHIP (0.95) | ❌ |
| Uncertain #2 (ShipsEar) | not_ship | 0.35 | SHIP (1.0) | ❌ |

**Result: 1/4 accuracy, strong SHIP bias regardless of ground truth.**
Gemma's reasoning text is templatically plausible ("persistent
horizontal bands at low frequency…") but the verdict does not track
what is actually in the image. The model is **not** reliably reading
hydrophone mel spectrograms — this is a domain-shift failure (mel-
specs are not natural images Gemma was pre-trained on), not a model
size issue.

**Operational implications:**

- We do NOT claim Gemma multimodal verifies CNN acoustic decisions.
  The CNN's per-site calibrated decision is the auditable one.
- `os identify-vessel` scopes Gemma multimodal to natural-image
  inputs (vessel photographs), where the model demonstrably works.
- The existing `gemma/explanations.py` multimodal narration path
  remains in the codebase as a research artefact. In production it
  generates plausible-sounding text *aligned with the CNN decision*
  (the prompt instructs Gemma not to contradict), not an independent
  verification. We make that scope explicit in the module docstring
  rather than removing the feature.
- Gemma's primary role pivoted from "multimodal verifier" to
  "function-calling analytical synthesiser" — see `os brief`, which
  uses Gemma 4 as the agent that gathers data via tool calls and
  composes the intelligence-brief output.

Native-audio modality (Gemma 4 e4b supports it per the model card)
would likely solve the mel-spec problem outright, but Ollama does
not currently expose audio input — verified by sending the .wav as
`audio` / `audios` field, both ignored. Future work when the runtime
catches up.

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

---

## Model interpretability — what v7.6 actually attends to

Grad-CAM analysis on v7.6's final convolutional block (`backbone.block4`),
applied to OC01 chunks paired with MarineCadastre AIS ground truth,
shows the model attends to:

- **Cavitation broadband** (500-1000 Hz, mel bins 75-128) — primary attention
- **Lower engine band** (80-250 Hz, mel bins 25-50) — secondary attention
- **Blade-rate band** (5-50 Hz, mel bins 0-25) — **near-zero attention**

This last point is by design, not failure. The preprocessing applies
`_LOW_FREQ_MASK` (`src/ocean_sentinel/services/cnn_v7_classifier.py`):
all bins below the high-pass cutoff are replaced with the mean of the
unmasked bins **before the model sees the spectrogram**. The reason:
low-frequency content is dominated by hydrophone-specific noise
(mooring, surf, electrical hum) that does not generalise across
deployments. The mask forces the model to learn site-invariant
features.

The case-study Part I PSD finding of "+13 dB excess at 28-37 Hz
(blade-rate band)" was a Welch-PSD discovery on the raw audio
waveform — that representation has ~1 Hz resolution. The mel
spectrogram the CNN sees has 5–30 Hz per bin and starts above the
high-pass cutoff, so it cannot resolve narrow blade-rate tonals even
in principle.

What this means practically:

- The CNN and traditional PSD analysis use **complementary signal
  pathways**. They converge on the same vessels (verified on OC01)
  via different acoustic features.
- Operators interpreting Ocean Sentinel detections should not expect
  the CNN to "verify" the blade-rate signature an acoustician would
  highlight in a PSD plot. It uses different evidence.
- Cavitation broadband is a well-known vessel signature (Ross 1976,
  Urick 1983, Wales & Heitmeyer 2002) — the model is using a real
  physical feature, just one that doesn't show up as a sharp narrowband
  peak in PSD plots.
- 96.4% honest test accuracy + 96.0% OOD validation rule out shortcut
  learning. The model is not classifying based on recording-level
  texture or non-acoustic confounders.

Limitations of this finding:

- We have not tested whether the model fails on vessels with low
  cavitation but strong engine tonals (e.g. some passenger vessels,
  electric/hybrid hulls). Such cases would be the model's natural
  failure mode.
- Grad-CAM heatmaps on broadband classifiers are inherently diffuse;
  the heatmaps do not show a single "smoking gun" attention region.
  Visual interpretation requires careful framing — see
  `/case-study/audit` for the framing we use.
