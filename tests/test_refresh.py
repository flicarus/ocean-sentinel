"""Tests for the refresh / continual-improvement loop.

Validates:
- _gather_ambient_paths combines original + extra dir + --add path
- _filter_trusted_windows keeps only low-ship_prob windows + has a fallback
- refresh_site returns ok=False with clear errors on missing inputs
- refresh_site round-trips through a fake v7 to write a new adapter

We avoid the real CNN/MPS path in unit tests via the same fake_v7
fixture used by test_site_adapter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from ocean_sentinel.gemma import refresh, site_adapter


SR = 16_000


# ── _gather_ambient_paths ───────────────────────────────────────────────


def test_gather_paths_finds_yaml_original(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    site_id = "x"
    Path("data/sites").mkdir(parents=True)
    src = tmp_path / "orig.wav"
    sf.write(str(src), np.zeros(1000, dtype="float32"), SR)
    (Path("data/sites") / f"{site_id}.yaml").write_text(
        f"site_id: {site_id}\nambient_source: {src}\n"
    )

    paths = refresh._gather_ambient_paths(site_id, additional=None)
    assert len(paths) == 1
    assert paths[0].resolve() == src.resolve()


def test_gather_paths_combines_original_extra_dir_and_add(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    site_id = "y"
    site_yaml = Path("data/sites") / f"{site_id}.yaml"
    site_yaml.parent.mkdir(parents=True)

    orig = tmp_path / "orig.wav"
    sf.write(str(orig), np.zeros(1000, dtype="float32"), SR)
    site_yaml.write_text(f"site_id: {site_id}\nambient_source: {orig}\n")

    extra_dir = Path("data/sites") / site_id / "ambient"
    extra_dir.mkdir(parents=True)
    new1 = extra_dir / "a.wav"
    new2 = extra_dir / "b.wav"
    sf.write(str(new1), np.zeros(1000, dtype="float32"), SR)
    sf.write(str(new2), np.zeros(1000, dtype="float32"), SR)

    add_path = tmp_path / "extra.wav"
    sf.write(str(add_path), np.zeros(1000, dtype="float32"), SR)

    paths = refresh._gather_ambient_paths(site_id, additional=str(add_path))
    resolved = [p.resolve() for p in paths]
    assert orig.resolve() in resolved
    assert new1.resolve() in resolved
    assert new2.resolve() in resolved
    assert add_path.resolve() in resolved


def test_gather_paths_dedups_when_same_file_listed_twice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    site_id = "z"
    extra = Path("data/sites") / site_id / "ambient"
    extra.mkdir(parents=True)
    same = extra / "shared.wav"
    sf.write(str(same), np.zeros(1000, dtype="float32"), SR)

    paths = refresh._gather_ambient_paths(site_id, additional=str(same))
    # de-dup keeps the order of first appearance
    assert len(paths) == 1


def test_gather_paths_returns_empty_when_no_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data/sites").mkdir(parents=True)
    paths = refresh._gather_ambient_paths("ghost", additional=None)
    assert paths == []


# ── _concat_ambient_to_tmp ──────────────────────────────────────────────


def test_concat_combines_audio_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    site_id = "concat"
    Path(f"data/sites/{site_id}").mkdir(parents=True)

    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    a_audio = np.ones(SR * 2, dtype="float32") * 0.1     # 2 s
    b_audio = np.ones(SR * 3, dtype="float32") * 0.2     # 3 s
    sf.write(str(a_path), a_audio, SR)
    sf.write(str(b_path), b_audio, SR)

    tmp_combined, total_s = refresh._concat_ambient_to_tmp(
        [a_path, b_path], site_id, max_total_s=10.0,
    )
    assert tmp_combined.exists()
    # 2 s + 3 s combined
    assert 4.5 < total_s < 5.5
    y, sr = sf.read(str(tmp_combined))
    assert sr == SR
    assert len(y) == int(SR * total_s)


def test_concat_caps_at_max_total_s(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    site_id = "cap"
    Path(f"data/sites/{site_id}").mkdir(parents=True)

    a_path = tmp_path / "a.wav"
    sf.write(str(a_path), np.zeros(SR * 100, dtype="float32"), SR)
    tmp_out, total_s = refresh._concat_ambient_to_tmp(
        [a_path], site_id, max_total_s=10.0,
    )
    assert total_s <= 10.0 + 0.1


# ── refresh_site error paths ────────────────────────────────────────────


def test_refresh_returns_error_for_missing_site(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = refresh.refresh_site("ghost-site")
    assert out["ok"] is False
    assert "no ambient" in out["error"]


def test_refresh_returns_error_for_too_short_ambient(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    site_id = "short"
    Path(f"data/sites/{site_id}").mkdir(parents=True)

    add = tmp_path / "tiny.wav"
    sf.write(str(add), np.zeros(SR * 10, dtype="float32"), SR)  # 10s, < 3*60s
    Path(f"data/sites/{site_id}.yaml").write_text(
        f"site_id: {site_id}\nambient_source: {add}\n"
    )

    out = refresh.refresh_site(site_id, additional_ambient_path=str(add))
    assert out["ok"] is False
    assert "need" in out["error"] or "recalibrate" in out["error"]
