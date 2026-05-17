"""`os test <site_id>` — run the calibrated pipeline on bundled samples.

After onboarding a site, this command lets the user verify that their
adapter + threshold + CNN actually classify known-label audio correctly.

The samples are vendored under ``src/ocean_sentinel/test_samples/`` and
described in ``manifest.yaml``. Three real DeepShip vessel clips and two
synthesised ambients give us a small but defensible bench: not a
training-set leak (the adapter wasn't trained on them) and the labels
are guaranteed (vessels are vessels; the ambients are deterministically
synthesised so they cannot accidentally contain a vessel).

The output is two pieces:
- per-sample row showing expected vs actual decision tier and ship_prob
- an overall pass/fail count, plus a per-class breakdown

Failures are not necessarily bugs — on a drastically OOD site, the
calibrated pipeline can legitimately suppress a near-threshold vessel
to keep the false-alarm rate at spec. The output is informational, the
exit code reflects only "does the system run end-to-end".
"""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from ..gemma.cnn_inference import simulate_detection


def is_site_onboarded(site_id: str) -> bool:
    """A site is "onboarded" once its adapter checkpoint exists. The site
    YAML alone is not enough — the adapter is what makes inference
    site-specific. Without it the test would silently fall back to the
    global threshold and produce misleading numbers, which is exactly
    the failure mode `os test` exists to prevent."""
    return (Path("data/sites") / site_id / "adapter.pt").exists()


def _resource_dir() -> Path:
    """Path to the vendored test_samples directory.

    Works for both `pip install -e .` (path inside source tree) and
    later, packaged installs (via importlib.resources).
    """
    return Path(str(files("ocean_sentinel") / "test_samples"))


def _load_manifest() -> list[dict[str, Any]]:
    manifest_path = _resource_dir() / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.yaml missing — expected at {manifest_path}"
        )
    data = yaml.safe_load(manifest_path.read_text())
    return data.get("samples", [])


def _decision_matches(expected_label: str, decision_tier: str) -> bool:
    """Map decision_tier categories to ship/not_ship expectations.

    Vessel-positive tiers: DARK_VESSEL, CONFIRMED_VESSEL, ACOUSTIC_ONLY_LOW
    Non-vessel tiers:      AMBIENT
    Ambiguous:             UNCERTAIN — counted as fail with a clear note
    """
    ship_tiers = {"DARK_VESSEL", "CONFIRMED_VESSEL", "ACOUSTIC_ONLY_LOW"}
    if expected_label == "ship":
        return decision_tier in ship_tiers
    return decision_tier == "AMBIENT"


def run_tests(site_id: str) -> dict[str, Any]:
    """Run the bundled samples through `simulate_detection` for the
    given site_id (which must already be onboarded — the per-site
    adapter and conformal threshold are picked up automatically).

    Returns a dict with `ok`, per-sample `results`, and aggregate counts.
    """
    samples = _load_manifest()
    if not samples:
        return {"ok": False, "error": "manifest has no samples"}

    sample_dir = _resource_dir()
    rows: list[dict[str, Any]] = []
    n_correct = 0
    for s in samples:
        clip = sample_dir / s["path"]
        if not clip.exists():
            rows.append({
                **s, "error": f"clip missing: {clip}",
                "decision_tier": None, "ship_prob": None, "correct": False,
            })
            continue
        det = simulate_detection(site_id=site_id, clip=str(clip))
        if not det.get("ok"):
            rows.append({
                **s, "error": det.get("error"),
                "decision_tier": None, "ship_prob": None, "correct": False,
            })
            continue
        correct = _decision_matches(
            expected_label=s["expected_label"],
            decision_tier=det["decision_tier"],
        )
        n_correct += int(correct)
        rows.append({
            **s,
            "decision_tier":      det["decision_tier"],
            "severity":           det.get("severity"),
            "ship_prob":          det["cnn_confidence"],
            "conformal_threshold": det["conformal_threshold"],
            "conformal_pass":     det["conformal_pass"],
            "decision_id":        det.get("decision_id"),
            "correct":            correct,
        })

    return {
        "ok": True,
        "site_id": site_id,
        "n_samples": len(rows),
        "n_correct": n_correct,
        "results": rows,
    }
