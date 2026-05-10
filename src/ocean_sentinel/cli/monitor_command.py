"""`os monitor` — operational mode for a deployed hydrophone site.

After onboarding, the user has a calibrated site sitting on disk. This
command is what they actually run day-to-day:

  os monitor <site> --watch ./incoming/   # continuous: process new .wav files
  os monitor <site> --replay ./clips/     # one-shot: process all files, exit

Each detection becomes one row in ``data/sites/{site}/events.jsonl`` —
the canonical operational event log. The /api/events/feed endpoint
reads that same JSONL so the dashboard sees every alert as it lands.

Implementation notes:

- Polling at 2 s is plenty for MPA scale (latency between sample and
  alert is bounded by clip length, not the watcher).
- A file is only processed once its ``mtime`` has been stable for 3 s.
  Without this, an in-progress write (cron dump, ffmpeg, USB sync) gets
  partially decoded and the CNN sees garbage.
- Per-file failures (corrupt audio, transient I/O error) are logged to
  ``data/sites/{site}/errors.jsonl`` and do not interrupt monitoring.
  Pipeline-level failures (missing checkpoint, model load error)
  propagate up and stop the loop.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from ..gemma.cnn_inference import simulate_detection


SUPPORTED_EXTS = {".wav", ".WAV"}
DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_SETTLE_S = 3.0


# ── filesystem ──────────────────────────────────────────────────────────


def _list_audio_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix in SUPPORTED_EXTS)


def is_file_stable(path: Path, settle_s: float) -> bool:
    """File is stable when its mtime is older than `settle_s` seconds —
    we assume nothing else is writing to it."""
    try:
        return time.time() - path.stat().st_mtime >= settle_s
    except FileNotFoundError:
        return False


# ── event persistence ──────────────────────────────────────────────────


def _site_dir(site_id: str) -> Path:
    return Path("data/sites") / site_id


def _site_config(site_id: str) -> dict[str, Any]:
    """Read the YAML config produced by Step 8 onboarding.

    Returns an empty dict if the file is missing — callers degrade
    gracefully (lat/lon/MPA fields will just be 0/empty in events).
    """
    p = Path("data/sites") / f"{site_id}.yaml"
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def _to_vessel_event(
    det: dict[str, Any],
    site_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Map the simulate_detection result + site config to a row that
    matches the VesselEvent shape used by the dashboard.

    Fields not directly available from a single inference (gemma_reasoning,
    long narrative, AIS context) get sensible defaults. The /api/events/feed
    endpoint recomputes time_ago / ais_offline_since at request time.
    """
    severity_to_threat = {"HIGH": "HIGH", "MEDIUM": "MEDIUM",
                          "LOW": "LOW", "NONE": "LOW"}
    threat = severity_to_threat.get(det.get("severity", "LOW"), "LOW")
    cnn_conf = float(det.get("cnn_confidence", 0.0))

    return {
        # internal/pipeline fields
        "ts":                   datetime.now(timezone.utc).isoformat(),
        "site_id":              site_id,
        "decision_tier":        det.get("decision_tier"),
        "severity":             det.get("severity"),
        "cnn_confidence":       cnn_conf,
        "cnn_uncertainty":      det.get("cnn_uncertainty"),
        "conformal_threshold":  det.get("conformal_threshold"),
        "conformal_pass":       det.get("conformal_pass"),
        "ais_vessels_in_radius": det.get("ais_vessels_in_radius", 0),
        "clip":                 det.get("clip"),
        "site_adapter":         det.get("site_adapter"),
        # VesselEvent display fields (frontend contract)
        "id":                   det.get("decision_id", "DET-?"),
        "vessel":               f"contact-{det.get('decision_id', '?')}",
        "lat":                  float(config.get("lat", 0.0) or 0.0),
        "lng":                  float(config.get("lon", 0.0) or 0.0),
        "threat":               threat,
        "confidence":           round(cnn_conf * 100.0, 1),
        "hydrophone":           site_id,
        "mpa_name":             str(config.get("nearest_mpa", "—")),
        "mpa_distance_km":      float(config.get("mpa_distance_km", 0.0) or 0.0),
        "gemma_reasoning":      det.get("summary", "—"),
        "vessel_window":        f"{int(det.get('ais_vessels_in_radius', 0) or 0)} AIS vessels in radius",
        "cnn_analysis":         (
            f"CNN ship_prob={cnn_conf:.2f}, "
            f"threshold={float(det.get('conformal_threshold', 0)):.2f}, "
            f"uncertainty={float(det.get('cnn_uncertainty', 0)):.2f}"
        ),
    }


def append_event(site_id: str, event: dict[str, Any]) -> Path:
    """Append a row to data/sites/{site_id}/events.jsonl. Idempotent on
    repeated calls; the dashboard reads this file directly."""
    site_dir = _site_dir(site_id)
    site_dir.mkdir(parents=True, exist_ok=True)
    events_path = site_dir / "events.jsonl"
    with events_path.open("a") as f:
        f.write(json.dumps(event) + "\n")
    return events_path


def append_error(site_id: str, clip: Path, error: str) -> None:
    site_dir = _site_dir(site_id)
    site_dir.mkdir(parents=True, exist_ok=True)
    err_path = site_dir / "errors.jsonl"
    with err_path.open("a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "clip": str(clip),
            "error": error,
        }) + "\n")


# ── core processing ─────────────────────────────────────────────────────


def process_one(site_id: str, clip: Path) -> dict[str, Any] | None:
    """Run one .wav through simulate_detection and persist the event.

    Returns the VesselEvent row that was appended to events.jsonl, or
    None if the clip was unprocessable (the failure is logged separately
    so the caller can keep going).
    """
    try:
        det = simulate_detection(site_id=site_id, clip=str(clip))
    except Exception as e:
        append_error(site_id, clip, f"{type(e).__name__}: {e}")
        return None

    if not det.get("ok"):
        append_error(site_id, clip, det.get("error", "unknown error"))
        return None

    event = _to_vessel_event(det, site_id, _site_config(site_id))
    append_event(site_id, event)
    return event


# ── modes ───────────────────────────────────────────────────────────────


def replay(
    site_id: str,
    folder: Path,
    *,
    on_event: Callable[[Path, dict[str, Any] | None], None] | None = None,
) -> list[dict[str, Any]]:
    """Process every .wav in `folder` once, in mtime order, then return.
    Used by `os monitor --replay` for demo / batch verification."""
    files = _list_audio_files(folder)
    files.sort(key=lambda p: p.stat().st_mtime)
    out: list[dict[str, Any]] = []
    for f in files:
        ev = process_one(site_id, f)
        if on_event is not None:
            on_event(f, ev)
        if ev is not None:
            out.append(ev)
    return out


def watch(
    site_id: str,
    folder: Path,
    *,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    settle_s: float = DEFAULT_SETTLE_S,
    on_event: Callable[[Path, dict[str, Any] | None], None] | None = None,
    stop: Callable[[], bool] | None = None,
) -> Iterable[dict[str, Any]]:
    """Generator: poll `folder` for new .wav files and process each as it
    becomes stable. Yields each persisted event.

    `stop` is an optional predicate the caller can use for clean
    shutdown from tests; in the CLI we just KeyboardInterrupt out.
    """
    seen: set[Path] = set()
    while True:
        if stop is not None and stop():
            return
        for f in _list_audio_files(folder):
            if f in seen:
                continue
            if not is_file_stable(f, settle_s):
                continue
            ev = process_one(site_id, f)
            seen.add(f)
            if on_event is not None:
                on_event(f, ev)
            if ev is not None:
                yield ev
        time.sleep(poll_interval_s)
