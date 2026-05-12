# Ocean Sentinel — v7.6 Decision Brief (for second-opinion AI review)

**Date:** 2026-05-11 23:00
**Status:** Last full retraining cycle before hackathon submission (deadline 2026-05-18)
**Asking:** Is the proposed plan optimal? Better alternatives?

---

## Project context (one paragraph)

Ocean Sentinel is a CNN-based vessel detection system for marine hydrophones,
built for Gemma 4 Good Hackathon. Three-layer pipeline: (1) `OceanSentinelV7`
CNN — 2.3M params, binary classification (ship / not_ship) on 60s mel-spectrograms
@ 16 kHz, 128 mels, fmax=1 kHz; (2) Gemma 4 multimodal reasoning layer for vessel
type + threat assessment; (3) AIS correlation (Global Fishing Watch API) for
ground truth at training time and dark-vessel detection at inference. Solo
project on M5 Pro 24GB. Inference 19 ms median end-to-end on MPS.

---

## Current state: v7.5

**Trained today (2026-05-11) on 147,520 samples**, MPS, 30 epochs, focal evidential
loss (gamma=2.0, kl_coef=0.05), cross_site_mix prob=0.5, session-split train/val,
hard-negative oversample ×5 from `hard_negatives_v4.jsonl` (3,189 clips v7.4
misclassified across all 84k training rows).

**Eval results — `data/eval/per_site_v7_5.json`:**

| metric | value |
|---|---|
| Overall accuracy (27 sites, n=6,349) | **87.0 %** |
| Best val_binary_acc during training | 0.5794 (epoch 26) |
| Conformal threshold (α=0.1, Lei 2018) | 0.4598 |
| Sites at 100 % | 8 (oc02, fk03, ci05, fk02, gr01, mbari, ci01, hi06) |
| Sites at 90-99 % | 9 |
| Sites at 80-89 % | 5 |
| Sites <60 % | 3 (point-robinson 13.4 %, sanctsound 58.7 %, sb03 59.6 %) |

**Comparison vs v7.4 (89.98 % on 14 sites, 71.3 % on same 27-site eval):**
- **+15.7pp** vs v7.4 on identical 27-site eval
- 8 previously-unseen sites went from 0-19 % → 83-100 %
- oc01 went 41 % → 93 % (validated hard-neg mining methodology)

---

## The problem: 2 catastrophic regressions

**ais-correlated-point-robinson: 100 % → 13.4 %** (-86.6pp, recall 0.134, precision 1.0)
**sanctsound (raw): 86 % → 59 %** (-27pp)

Two previously-perfect sites collapsed. Root cause **confirmed quantitatively**:

| | v7.4 era pool | v7.5 era pool |
|---|---|---|
| Total pool | 71,250 | 205,858 |
| point-robinson rows | 1,465 | 1,665 |
| point-robinson share | **2.06 %** | **0.81 %** (÷2.5) |
| MBARI rows | 47,559 | 170,842 |
| MBARI share | 66.8 % | **82.9 %** |

The bootstrap added 130k MBARI samples (24h continuous WAV → 3-6k chunks/run)
while non-MBARI sources (Orcasound HLS streams → 30-1000 chunks/run, SanctSound
FLAC → 50-200/deployment) couldn't keep pace structurally. point-robinson's
relative gradient signal dropped 2.5×, and `cross_site_mix` (50% of ship samples
get ambient mixed in from random other sites) amplified the dilution.

---

## Code/infra in place

- `scripts/train_cnn_v7_2.py` — v7.5 trainer (focal + KL low + cross-site-mix
  + session-split + hard-neg oversample)
- `scripts/train_cnn_v7_6.py` — fork of v7_2 with **new** `--balanced-sampler`
  flag that wraps DataLoader in `WeightedRandomSampler` with weight=1/site_count
  (normalizes source_id: e.g. `sanctsound-more-60s-ci01` → `ci01`,
  `ais-correlated-bush-point` → `bush-point`, all `mbari*` → `mbari`)
- `src/ocean_sentinel/gemma/site_adapter.py` — per-site adapter infrastructure
  (residual MLP, ~33k params, init-as-identity, applied at inference if site
  signature matches known_sites.json within cosine threshold). Used in
  `measure_safeguard_impact.py` to drop FA rate from 90.3 % → 3.2 % on
  synthetic reef test.
- `data/eval/per_site_v7_5.json` — per-site v7.5 numbers, source of truth
- `data/calibration/conformal_v7_5.json` — calibrated threshold 0.4598
- `data/training/v7_bulk/hard_negatives_v4.jsonl` — 3,189 misclassified clips
  (oversampled ×5 in v7.5 training, would re-mine for v7.6/v7.7)
- `src/ocean_sentinel/config.py` & `cnn_inference.py` — config now points to
  cnn_v7_5.pt + conformal_v7_5.json (updated today)
- Per-site signatures registry `data/known_sites.json` — 31 sites with 64-band
  log-mel signatures for cosine-similarity site detection at inference

---

## Time constraints

User has ONE more ~13-hour passive block:
- ~04:00 (after sleep) → ~10:00 (departure for work) — 6h
- ~10:00 → ~17:00 — at work, machine idle but accessible — 7h

After that, only minor tweaks possible. **No second retraining attempt.**

---

## Two proposed approaches

### Approach A: train v7.6 with WeightedRandomSampler

**Hypothesis:** balanced gradient updates per site will fix point-robinson +
sanctsound regressions without breaking other sites.

**Pros:**
- Addresses root cause directly (gradient signal proportional to site frequency)
- Improves entire model uniformly, not just 2 sites
- One-model story (clean for submission)

**Cons:**
- Untested interaction with `cross_site_mix` — could compound or cancel
- 4h training + 30 min eval
- If v7.6 < v7.5 → we ship v7.5 with regressions documented

**Cost:** ~5h compute, no manual intervention

### Approach B: per-site adapters for point-robinson + sanctsound, keep v7.5

**Hypothesis:** v7.5 base + 33k-param adapter per regressed site, trained on
site-specific data only with init-as-identity, will recover regressions.

**Pros:**
- Proven infrastructure (FA 90.3 → 3.2 % demo in `measure_safeguard_impact.py`)
- Cannot hurt base model — adapters only fire when site signature matches
- Fast: ~15 min per adapter × 2 = 30-40 min total
- Story-worthy: "diagnosed data imbalance, surgical fix via per-site
  calibration" mirrors medical-imaging deployment practice

**Cons:**
- Doesn't generalize to NEW sites without their own adapter
- Adds 33k params × N sites in deployment overhead (negligible)
- Some readers may see it as patching rather than fixing

**Cost:** ~40 min compute, 1h human review

### Approach C (proposed): do both, pick winner

1. **Now → ~05:00**: aggressive non-MBARI overnight pull (8 AIS-correlated
   chains, one per Orcasound node, offset start dates to avoid GFW 1-concurrent
   collision; +SanctSound expanded to 56 deployments). Target: MBARI share
   drops from 83 % to ~65 %.
2. **~05:00 → ~09:00**: auto-trigger v7.6 training with `--balanced-sampler`.
3. **~09:00 → ~09:30**: auto per_site eval + conformal calibrate.
4. **~17:00 (user returns)**: train per-site adapters for point-robinson +
   sanctsound on top of v7.5 (parallel safety net regardless of v7.6 outcome).
5. **Decision time**: compare `per_site_v7_5.json` (with optional adapters) vs
   `per_site_v7_6.json`. Ship whichever is better.

**Pros:** uses full passive block, two independent paths to success, decision
based on concrete numbers not hope.

**Cons:** complexity, must coordinate auto-pipeline overnight.

**Cost:** ~13h compute, 1h human work tomorrow evening.

---

## Specific implementation question

The overnight script (`scripts/overnight_v76_adaptive.sh`) launches 8 AIS-
correlated chains in parallel with offset start dates (each chain handles 40
dates serially). An adaptive controller checks 429 rate every 15 min:
- 429 in last 5 min > 30 → kill 2 chains (down to min 6)
- 429 in last 5 min < 5 AND procs < 12 → spawn 2 more chains

Then chains into v7.6 training when phase 1 finishes, then auto-eval.

**Specific concerns I want second opinion on:**

1. **Sampler × augmentation interaction.** `WeightedRandomSampler` upsamples
   under-represented sites (point-robinson sampled ~5× more often than a MBARI
   row). But cross_site_mix already mixes ambient from random sites into ship
   samples. Will the sampler effectively undo cross_site_mix's diversity
   benefit by re-concentrating gradient on a narrow set of ambient sources?
   Should I disable cross_site_mix in v7.6 (`--no-cross-site-mix`) to isolate
   the sampler effect?

2. **Hard-neg oversample × balanced sampler interaction.** Hard-negatives
   are oversampled ×5 (3,189 → 15,945 rows). They have their own site
   distribution (skewed toward oc01, sb02, oc02 — the v7.4-difficult sites).
   With site-balanced sampler on top, do oversampled hard-negs effectively get
   sampled at 25× normal rate (5× oversample × 5× site-rare boost for some
   sites)? Should I reduce hard-neg oversample to 2-3 in v7.6 to avoid
   compounding?

3. **Per-site adapter robustness.** The adapter is a residual MLP applied to
   the CNN's pre-classification embedding (33k params, init-as-identity so
   day-0 behavior matches base v7.5 exactly). Trained on ~1,500 point-robinson
   rows. Concern: 1,500 might be too few to learn a meaningful residual without
   overfitting. Is there a principled way to estimate min adapter training
   size, or should I fall back on early-stopping with held-out validation?

4. **Eval skew.** `eval_per_site.py` uses `clf.predict(spec)` which internally
   applies threshold 0.5 to ship_prob, NOT the conformal-calibrated 0.4598.
   So the 87.0 % v7.5 number under-reports actual deployment accuracy. Should
   I re-run eval with conformal threshold applied? Or is the 0.5-threshold
   number the more honest measurement (since real deployment uses a per-site
   adapter + conformal which I'd need to mock per site)?

5. **Cross-site mix probability.** Was 0.5 in v7.5 (50% of ship samples get
   ambient mixed in). Some authors suggest 0.2-0.3 for noise-augmentation
   tasks. Is 0.5 actually optimal for vessel detection, or is it part of why
   point-robinson regressed (over-aggressive mixing of point-robinson ship
   with new-site ambient until model can't distinguish)?

---

## Numbers summary for first-pass review

```
v7.4 (Day 13)         : 89.98 % overall on 14 sites, n=4011
v7.5 (today)          : 87.0  % overall on 27 sites, n=6349
                        — +13 new sites added at 0-100 % range (mostly 83-100 %)
                        — oc01 recovered 41 → 93 % (hard-neg mining win)
                        — point-robinson collapsed 100 → 13 %
                        — sanctsound collapsed 86 → 59 %
v7.6 (proposed)       : ??? — WeightedRandomSampler + cross-site-mix on
                        new pool (target 200-270k samples, ~65 % MBARI)
v7.5+adapter (proposed): ~91-93 % — surgical per-site fix
```

---

## What I'd love opinions on

1. Is Approach C (both paths, pick winner) actually worth the orchestration
   complexity, or should I commit to one approach to reduce failure modes?
2. Specific concerns 1-5 above.
3. Anything I'm missing about the regression that suggests a different fix
   entirely (e.g. is this really about cross_site_mix, not data imbalance)?
4. For submission narrative: which framing reads stronger to ML/conservation
   judges — "single retrained model" or "diagnosed + surgical fix with
   per-site calibration"?
