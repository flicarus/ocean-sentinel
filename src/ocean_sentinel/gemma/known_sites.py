"""Known-site signature registry — for compare_to_known_sites.

Loads pre-computed mean signatures of training sites from
``data/known_sites.json`` (built by scripts/precompute_signatures.py),
then ranks any incoming signature by *z-score normalized* cosine
similarity.

Why z-score normalization first?
Plain cosine on log-mel medians is dominated by absolute energy levels
— quiet sites cluster together regardless of what frequencies they're
quiet in. Z-score per signature removes that bias; what's left is the
shape of the spectral distribution.

Why we don't use a sliding-scale recommendation:
Empirical analysis (see scripts/derive_thresholds.py output) showed that
v7.4 LOHO accuracy does NOT correlate positively with z-cosine similarity
on this corpus. We have a small sample (n=7 with eval data), and Pearson
r is actually NEGATIVE (-0.57, p=0.18). Drawing fine-grained thresholds
from this would be honest-looking precision over a noisy signal.

Pragmatic choice: always recommend `finetune` for any new site whose
nearest match is below a high z-cosine bar. Fine-tune is cheap (~3 min)
and almost always helps. We surface the cosine number to Gemma as
*context*, not as a load-bearing decision.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

KNOWN_SITES_JSON = Path("data/known_sites.json")
CONFIG_YAML = Path("data/calibration/site_classification.yaml")


def _zscore(sig: list[float]) -> list[float]:
    """Center + scale to unit std. Empty / constant signatures fall through
    to plain centering."""
    n = len(sig)
    if n == 0:
        return []
    m = sum(sig) / n
    var = sum((x - m) ** 2 for x in sig) / n
    sd = math.sqrt(var)
    if sd == 0:
        return [x - m for x in sig]
    return [(x - m) / sd for x in sig]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _load_registry(path: Path = KNOWN_SITES_JSON) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        return raw.get("sites", [])
    return list(raw)


@lru_cache(maxsize=1)
def _load_thresholds() -> dict[str, float]:
    """Load empirically-derived adapter thresholds. Falls back to a
    conservative default that prefers `finetune` over `use_existing`."""
    if CONFIG_YAML.exists():
        cfg = yaml.safe_load(CONFIG_YAML.read_text())
        return cfg.get("adapter_strategy_thresholds", {})
    return {"use_existing_min_cos": 0.95, "finetune_min_cos": 0.50}


def _recommend(top_similarity: float) -> str:
    """Choose an adapter strategy. See module docstring for why these
    thresholds are conservative — small-sample empirical evidence does
    not support fine-grained cutoffs, so we err toward fine-tuning."""
    t = _load_thresholds()
    use_existing = float(t.get("use_existing_min_cos", 0.95))
    finetune     = float(t.get("finetune_min_cos", 0.50))
    if top_similarity > use_existing:
        return "use_existing"
    if top_similarity > finetune:
        return "finetune"
    return "full_calibration"


def compare_to_known_sites(
    signature: list[float],
    top_k: int = 3,
    registry_path: Path = KNOWN_SITES_JSON,
) -> dict[str, Any]:
    """Rank known sites by cosine similarity to the input signature."""
    registry = _load_registry(registry_path)
    if not registry:
        return {
            "ok": False,
            "error": (
                f"known-sites registry is empty at {registry_path}. "
                f"Run `python scripts/precompute_signatures.py` to bootstrap it."
            ),
        }

    if not signature:
        return {"ok": False, "error": "empty signature"}

    # Z-score the input signature once; we'll z-score each candidate too.
    z_input = _zscore(signature)

    scored = []
    for site in registry:
        sig = site.get("signature") or []
        if len(sig) != len(signature):
            continue
        z_candidate = _zscore(sig)
        sim_z = _cosine(z_input, z_candidate)   # primary: shape similarity
        sim_raw = _cosine(signature, sig)        # secondary: raw cosine
        scored.append({
            "id":         site.get("id"),
            "label":      site.get("label", site.get("id")),
            "lat":        site.get("lat"),
            "lon":        site.get("lon"),
            "n_samples":  site.get("n_samples"),
            "cosine_sim":      round(sim_z, 3),
            "cosine_sim_raw":  round(sim_raw, 3),
        })

    if not scored:
        return {
            "ok": False,
            "error": (
                f"no compatible sites in registry (expected sig dim "
                f"{len(signature)}, registry has different)"
            ),
        }

    scored.sort(key=lambda r: r["cosine_sim"], reverse=True)
    top = scored[:top_k]
    recommendation = _recommend(top[0]["cosine_sim"])

    return {
        "ok": True,
        "ranked": top,
        "recommendation": recommendation,
        "summary": (
            f"closest: {top[0]['label']} "
            f"(cos {top[0]['cosine_sim']:.2f}) → {recommendation}"
        ),
    }
