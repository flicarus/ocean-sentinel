"""Tests for `os monitor` — operational mode.

We mock `simulate_detection` so we don't need the CNN or any audio
backend; the goal is to exercise the watch / replay loops, the file
filtering, the JSONL persistence, and the failure paths.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ocean_sentinel.cli import monitor_command


SR = 16_000


def _make_wav(path: Path, duration_s: float = 1.0) -> Path:
    y = np.zeros(int(SR * duration_s), dtype=np.float32)
    sf.write(str(path), y, SR)
    return path


def _fake_detection(ok: bool = True, tier: str = "DARK_VESSEL", **extra):
    if not ok:
        return {"ok": False, "error": "fake error"}
    base = {
        "ok": True,
        "decision_id": "DET-FAKE",
        "site_id": "site",
        "clip": "fake.wav",
        "decision_tier": tier,
        "severity": "HIGH" if tier == "DARK_VESSEL" else "LOW",
        "cnn_confidence": 0.91,
        "cnn_uncertainty": 0.10,
        "conformal_threshold": 0.61,
        "conformal_pass": True,
        "ais_vessels_in_radius": 0,
        "summary": "test detection",
    }
    base.update(extra)
    return base


# ── filesystem helpers ────────────────────────────────────────────────


def test_list_audio_files_picks_only_wav(tmp_path):
    _make_wav(tmp_path / "a.wav")
    _make_wav(tmp_path / "b.WAV")
    (tmp_path / "c.txt").write_text("ignore me")
    (tmp_path / "d.mp4").write_bytes(b"\x00")
    out = monitor_command._list_audio_files(tmp_path)
    names = [p.name for p in out]
    assert "a.wav" in names
    assert "b.WAV" in names
    assert "c.txt" not in names
    assert "d.mp4" not in names


def test_list_audio_files_returns_empty_on_missing_dir(tmp_path):
    assert monitor_command._list_audio_files(tmp_path / "ghost") == []


def test_is_file_stable_after_settle(tmp_path):
    p = _make_wav(tmp_path / "x.wav")
    # File just written → not stable for any reasonable settle
    assert monitor_command.is_file_stable(p, settle_s=10.0) is False
    # ...but stable for a 0-second settle
    assert monitor_command.is_file_stable(p, settle_s=0.0) is True


# ── event persistence ─────────────────────────────────────────────────


def test_to_vessel_event_maps_severity_to_threat(monkeypatch):
    det = _fake_detection(severity="MEDIUM", decision_tier="ACOUSTIC_ONLY_LOW")
    ev = monitor_command._to_vessel_event(det, "test-site", {})
    assert ev["threat"] == "MEDIUM"
    assert ev["decision_tier"] == "ACOUSTIC_ONLY_LOW"
    assert ev["site_id"] == "test-site"


def test_to_vessel_event_uses_site_config_for_geolocation(monkeypatch):
    det = _fake_detection()
    config = {"lat": 36.7128, "lon": -121.9023, "nearest_mpa": "MBNMS",
              "mpa_distance_km": 8.2}
    ev = monitor_command._to_vessel_event(det, "test-site", config)
    assert ev["lat"] == pytest.approx(36.7128)
    assert ev["lng"] == pytest.approx(-121.9023)
    assert ev["mpa_name"] == "MBNMS"
    assert ev["mpa_distance_km"] == pytest.approx(8.2)


def test_append_event_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monitor_command.append_event("site", {"id": "DET-A", "ts": "2026-05-10T20:00Z"})
    monitor_command.append_event("site", {"id": "DET-B", "ts": "2026-05-10T20:01Z"})
    p = Path("data/sites/site/events.jsonl")
    assert p.exists()
    lines = p.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "DET-A"


def test_append_error_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monitor_command.append_error("site", Path("bad.wav"), "decode failed")
    p = Path("data/sites/site/errors.jsonl")
    assert p.exists()
    row = json.loads(p.read_text().strip())
    assert row["clip"] == "bad.wav"
    assert "decode failed" in row["error"]


# ── core processing ───────────────────────────────────────────────────


def test_process_one_writes_event_on_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        monitor_command, "simulate_detection",
        lambda **kwargs: _fake_detection(),
    )
    clip = _make_wav(tmp_path / "x.wav")
    out = monitor_command.process_one("site", clip)
    assert out is not None
    assert out["decision_tier"] == "DARK_VESSEL"
    events = Path("data/sites/site/events.jsonl").read_text().strip().split("\n")
    assert len(events) == 1


def test_process_one_logs_error_on_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        monitor_command, "simulate_detection",
        lambda **kwargs: _fake_detection(ok=False),
    )
    clip = _make_wav(tmp_path / "y.wav")
    out = monitor_command.process_one("site", clip)
    assert out is None
    err_lines = Path("data/sites/site/errors.jsonl").read_text().strip().split("\n")
    assert len(err_lines) == 1
    # No event row should have been appended
    assert not Path("data/sites/site/events.jsonl").exists()


def test_process_one_handles_simulate_detection_exception(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _raise(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(monitor_command, "simulate_detection", _raise)
    clip = _make_wav(tmp_path / "z.wav")
    out = monitor_command.process_one("site", clip)
    assert out is None
    err_lines = Path("data/sites/site/errors.jsonl").read_text().strip().split("\n")
    assert "RuntimeError: boom" in err_lines[0]


# ── replay mode ────────────────────────────────────────────────────────


def test_replay_processes_every_clip_in_mtime_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    counter = {"i": 0}

    def fake_det(**kwargs):
        counter["i"] += 1
        return _fake_detection(decision_id=f"DET-{counter['i']}")

    monkeypatch.setattr(monitor_command, "simulate_detection", fake_det)

    folder = tmp_path / "clips"
    folder.mkdir()
    a = _make_wav(folder / "a.wav")
    b = _make_wav(folder / "b.wav")
    # Make b strictly newer than a
    time.sleep(0.05)
    b.touch()
    events = monitor_command.replay("site", folder)
    assert len(events) == 2
    # mtime order: a first, b second
    assert events[0]["clip"] == str(a) or events[0]["id"] == "DET-1"


def test_replay_continues_past_failures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_det(*, site_id, clip):
        if "bad" in clip:
            return _fake_detection(ok=False)
        return _fake_detection()

    monkeypatch.setattr(monitor_command, "simulate_detection", fake_det)
    folder = tmp_path / "clips"
    folder.mkdir()
    _make_wav(folder / "ok.wav")
    _make_wav(folder / "bad.wav")
    events = monitor_command.replay("site", folder)
    # only 1 successful event should be written
    assert len(events) == 1


# ── watch mode (with stop predicate) ─────────────────────────────────────


def test_watch_picks_up_new_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        monitor_command, "simulate_detection",
        lambda **kwargs: _fake_detection(),
    )
    folder = tmp_path / "in"
    folder.mkdir()
    f = _make_wav(folder / "first.wav")
    # Force file to look ancient so the stable-mtime check passes
    import os
    os.utime(str(f), (time.time() - 60, time.time() - 60))

    iterations = {"n": 0}

    def stop_after_first():
        iterations["n"] += 1
        return iterations["n"] > 3

    events = list(monitor_command.watch(
        "site", folder,
        poll_interval_s=0.01, settle_s=0.1, stop=stop_after_first,
    ))
    assert len(events) == 1
    assert events[0]["decision_tier"] == "DARK_VESSEL"


def test_watch_skips_in_progress_writes(tmp_path, monkeypatch):
    """A file with a fresh mtime should NOT be processed yet."""
    monkeypatch.chdir(tmp_path)
    calls = {"n": 0}

    def fake_det(**kwargs):
        calls["n"] += 1
        return _fake_detection()

    monkeypatch.setattr(monitor_command, "simulate_detection", fake_det)
    folder = tmp_path / "in"
    folder.mkdir()
    _make_wav(folder / "still_writing.wav")  # fresh mtime, should be skipped

    iterations = {"n": 0}

    def stop_after_few():
        iterations["n"] += 1
        return iterations["n"] > 3

    list(monitor_command.watch(
        "site", folder,
        poll_interval_s=0.01,
        settle_s=10.0,   # require stable for 10s — far longer than the test
        stop=stop_after_few,
    ))
    assert calls["n"] == 0
