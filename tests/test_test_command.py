"""Tests for `os test` — bundled-sample sanity check.

Covers:
- is_site_onboarded() correctly detects adapter presence
- _decision_matches() maps decision_tier → ship/not_ship correctly
- run_tests() handles missing manifest gracefully
- The bundled manifest + sample files are present and well-formed
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ocean_sentinel.cli import test_command


# ── is_site_onboarded ───────────────────────────────────────────────────


def test_is_site_onboarded_false_for_missing_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert test_command.is_site_onboarded("ghost") is False


def test_is_site_onboarded_false_when_yaml_present_but_no_adapter(tmp_path, monkeypatch):
    """A site that has only a YAML config but no trained adapter is NOT
    considered onboarded — adapter is what makes inference site-specific."""
    monkeypatch.chdir(tmp_path)
    site_id = "half-baked"
    Path("data/sites").mkdir(parents=True)
    (Path("data/sites") / f"{site_id}.yaml").write_text("site_id: half-baked\n")
    assert test_command.is_site_onboarded(site_id) is False


def test_is_site_onboarded_true_when_adapter_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    site_id = "real-site"
    site_dir = Path("data/sites") / site_id
    site_dir.mkdir(parents=True)
    # Anything at adapter.pt counts; we don't validate contents here.
    torch.save({}, str(site_dir / "adapter.pt"))
    assert test_command.is_site_onboarded(site_id) is True


# ── _decision_matches ──────────────────────────────────────────────────


def test_decision_matches_ship_label_passes_for_vessel_tiers():
    for tier in ("DARK_VESSEL", "CONFIRMED_VESSEL", "ACOUSTIC_ONLY_LOW"):
        assert test_command._decision_matches("ship", tier) is True


def test_decision_matches_ship_label_fails_for_ambient():
    assert test_command._decision_matches("ship", "AMBIENT") is False


def test_decision_matches_not_ship_label_passes_only_for_ambient():
    assert test_command._decision_matches("not_ship", "AMBIENT") is True
    for tier in ("DARK_VESSEL", "CONFIRMED_VESSEL", "ACOUSTIC_ONLY_LOW",
                 "UNCERTAIN"):
        assert test_command._decision_matches("not_ship", tier) is False


def test_decision_matches_uncertain_counts_as_fail_for_ship_label():
    """UNCERTAIN means the model abstained — we can't claim a 'ship'
    expectation was met, even though it wasn't called 'ambient'."""
    assert test_command._decision_matches("ship", "UNCERTAIN") is False


# ── manifest + bundled samples ─────────────────────────────────────────


def test_manifest_loads_and_has_expected_samples():
    samples = test_command._load_manifest()
    assert len(samples) == 5
    paths = {s["path"] for s in samples}
    # All vessel clips must be HELD-OUT — Tug got dropped because every
    # Tug clip on disk was in training. Cargo/Passenger/Tanker each
    # contribute one unseen clip.
    assert "vessel_cargo.wav" in paths
    assert "vessel_passenger.wav" in paths
    assert "vessel_tanker.wav" in paths
    assert "ambient_quiet.wav" in paths
    assert "ambient_busy.wav" in paths
    # Spot-check the held_out flag is set on vessel rows
    vessel_rows = [s for s in samples if s["expected_label"] == "ship"]
    assert all(s.get("held_out") for s in vessel_rows)


def test_bundled_samples_present_on_disk():
    """The .wav files must exist next to the manifest. If you forgot to
    rebuild after editing scripts/build_test_samples.py, this fires."""
    sample_dir = test_command._resource_dir()
    samples = test_command._load_manifest()
    for s in samples:
        clip = sample_dir / s["path"]
        assert clip.exists(), f"missing bundled sample: {clip}"
        # >100 KB sanity — anything tiny is a build mistake
        assert clip.stat().st_size > 100_000


def test_manifest_labels_are_plausible():
    """Every sample is labelled either ship or not_ship — no typos."""
    samples = test_command._load_manifest()
    for s in samples:
        assert s["expected_label"] in {"ship", "not_ship"}


# ── run_tests error handling ───────────────────────────────────────────


def test_run_tests_propagates_simulate_detection_errors(tmp_path, monkeypatch):
    """If simulate_detection fails for a sample (e.g. missing checkpoint),
    run_tests still returns ok=True with per-sample error rows so the
    user sees what went wrong instead of a stack trace."""
    monkeypatch.setattr(
        test_command, "simulate_detection",
        lambda **kwargs: {"ok": False, "error": "checkpoint not found"},
    )
    out = test_command.run_tests("anything")
    assert out["ok"] is True
    assert all(r.get("error") for r in out["results"])
    assert out["n_correct"] == 0
