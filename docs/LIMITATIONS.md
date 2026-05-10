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
| Day-9 LOHO accuracy (raw v7.4) | **89.98 %** | 4011 | — | `data/eval/per_site_v7_4.json` |
| Day-9 LOHO end-to-end (full pipeline) | **97.2 %** | 4011 | — | `data/eval/per_site_v7_4.json` |
| Held-out memorisation probe (raw, vessels only) | **100 %** | 29 | 88.0 – 100 % | `data/eval/benchmark_v7_4.json` |
| Held-out probe (full pipeline) | **100 %** | 35 | 90.0 – 100 % | `data/eval/benchmark_v7_4.json` |
| OOD synthetic-reef + per-site adapter (FA reduction) | **90.3 % → 3.2 %** | 31 windows | — | `scripts/measure_safeguard_impact.py` |
| Cosine-similarity correlation with accuracy (n=14, post-extension) | **r = +0.52, ρ = +0.12** | 14 | — | `data/eval/loho_correlation_extended.json` |

The Pearson/Spearman disagreement is itself a finding: the apparent
linear correlation collapses to almost zero when ranks are used,
indicating outlier-driven artefact. See
[`docs/empirical_findings.md`](empirical_findings.md).

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

2. **Cross-sensor evaluation**
   Reason: every clip we have was recorded with a hydrophone whose
   transfer function is consistent within its dataset (DeepShip,
   Orcasound, MBARI). We have not tested whether v7.4 generalises to a
   different hydrophone make / sample-rate / gain setting.
   **Pilot deliverable**: deploy on at least one site whose hardware
   was not in the training corpus and measure FA over a 30-day window.

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
