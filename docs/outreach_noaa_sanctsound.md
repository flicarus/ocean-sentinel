# NOAA SanctSound — outreach email (v2, post-MarineCadastre audit)

**To:**  sanctuaries@noaa.gov  *(primary)*
**Cc:**  Sofie Van Parijs (NOAA NEFSC PAM), Carrie Wall (NCEI archive), Leila Hatch (Stellwagen / SanctSound co-PI) — verify current addresses before sending
**Subject:**  OC01 label-quality methodology note — Ocean Sentinel

---

Dear SanctSound team,

I'm Jakub Kos, an independent ML/acoustic-monitoring engineer. Over the
last few weeks I built an open-source acoustic vessel-detection
pipeline ("Ocean Sentinel") trained partly on the SanctSound corpus.
During an audit of our own training labels for OC01, I ran into a
methodological finding that I think is useful for the next archive
revision, and I'd like to share it.

**Headline**

Using MarineCadastre.gov's AIS archive (NOAA Office for Coastal
Management) as ground truth, we verified that **every 60-second chunk
in our 800-chunk OC01 training corpus has at least one AIS-broadcasting
vessel within 10 km within ±2 minutes of the chunk timestamp**. The
test window covers 48 hours over 2019-03-08 and 2019-03-09, which
includes 464,413 AIS pings and 809 unique vessels in a 100×100 km box.
OC01 sits at the western mouth of Juan de Fuca strait — the busiest
shipping corridor on the US Pacific coast.

The implication: at OC01, *ambient* labels (in the sense of "no vessel
audible") are physically impossible. A 60-second window without a
vessel within 10 km is the empty set, not a sampling matter.

**Confirming the original Part I finding**

The case study we published earlier (`/case-study`) flagged ten OC01
chunks as mislabeled and named JOSCO HUIZHOU as the responsible vessel
within 10 km at 12:00 UTC on 2019-03-09. MarineCadastre confirms this
to the second: **JOSCO HUIZHOU (MMSI 477133400, HKG cargo) was at
6.81 km from OC01 at 2019-03-09T12:39:55Z**, with 698 AIS pings on
that day. That's inside the CNN flag window (12:14–12:44 UTC). The
original case study was right.

**What we attempted, what we found**

We started with a hypothesis that 560 of our 800 OC01 chunks were
mislabeled, based on agreement with our v7.6 acoustic CNN. After
catching ourselves in a circularity (the model was trained on those
exact corrected labels), and after pure acoustic-feature analysis hit
cross-recording calibration drift, we settled on MarineCadastre as
the un-arguable ground truth.

The honest finding is sharper:

- **144 chunks** have a named vessel at CPA 5–7 km within ±2 min of
  the chunk (PETER M., DUBLIN SEA, NRC CAPE FLATTERY, etc.).
- **416 chunks** have a vessel at CPA 7–10 km — likely audible.
- **20 chunks** *labeled ambient* have a vessel at CPA 5–7 km.
- **220 chunks** *labeled ambient* have a vessel at CPA 7–10 km.
- **0 chunks** in the entire 800-chunk corpus have nearest vessel
  further than ~10 km.

The binary *ship / ambient* label, applied at the recording level,
loses information at OC01.

**Recommendation**

At high-traffic sites (OC01, MB01 in Monterey Bay, SB02 in Stellwagen
Bank), we'd suggest labeling chunks with:

  1. CPA distance band (e.g. 0–3 km / 3–7 km / 7–15 km / >15 km),
  2. Dominant vessel class within the chunk window (cargo / fishing /
     tug / passenger / unknown),
  3. The named vessel(s) responsible, when AIS attribution is clean.

This is what a downstream acoustic-impact model actually needs, and
it's recoverable automatically from MarineCadastre + the chunk
timestamps. We have working code that does this for OC01 in under five
minutes; extending to the rest of the SanctSound US-waters deployments
is a config change.

**What we're offering**

  1. A JSON file with per-chunk CPA + named vessel attribution for
     the 800-chunk OC01 corpus, sourced from MarineCadastre.
  2. The reproducible audit script set
     (`scripts/audit_oc01_marine_cadastre.py`, `scripts/audit_oc01_cpa_bracket.py`,
     `scripts/audit_oc01_acoustic.py`) — MIT-licensed.
  3. The v7.6 model checkpoint (9.6 MB, 2.3M parameters, 96.4% honest
     test accuracy when calibrated per site).
  4. Time to discuss extending the audit to OC02/03/04 and any other
     US-waters SanctSound deployment.

We have no commercial interest in this and aren't asking for anything
in return. Ocean Sentinel is MIT-licensed:
<https://github.com/kubakos/ocean-sentinel>.

Happy to share the raw JSON, run the audit on a deployment of your
choice, or hop on a 30-minute call.

With thanks for the SanctSound archive — it's been a remarkable
resource to work with.

Best,
Jakub Kos
kubakos03@gmail.com
<https://oceansentinel.ai>

---

## Sending checklist

1. [ ] Verify `sanctuaries@noaa.gov` is still the right inbox
2. [ ] Look up Sofie / Carrie / Leila current email addresses
3. [ ] Push GitHub release `v1.0-oc01-audit` with all audit JSON + scripts + checkpoint
4. [ ] Ensure `https://oceansentinel.ai/case-study/audit` is live
5. [ ] Attach `data/audit/oc01_mc_track.json` and `data/audit/oc01_cpa_bracket.json`
6. [ ] Send Mon–Wed morning ET
