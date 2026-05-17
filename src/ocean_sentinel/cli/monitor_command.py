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


# Backwards-compat thin wrappers — the canonical logic lives in
# `services/event_persistence.py` so every command (monitor, detect, ...)
# stays in lockstep. Tests still import these names.

def _to_vessel_event(
    det: dict[str, Any], site_id: str, config: dict[str, Any],
) -> dict[str, Any]:
    from ..services.event_persistence import build_event
    return build_event(det, site_id, config)


def append_event(site_id: str, event: dict[str, Any]) -> Path:
    from ..services.event_persistence import append_local
    return append_local(site_id, event)


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

    Delegates to `services.event_persistence.persist_detection`, which
    handles both sinks (local jsonl + public ingest gateway). Returns
    the event row, or None on unprocessable input (those go to
    errors.jsonl so the watch loop keeps going).
    """
    try:
        det = simulate_detection(site_id=site_id, clip=str(clip))
    except Exception as e:
        append_error(site_id, clip, f"{type(e).__name__}: {e}")
        return None

    if not det.get("ok"):
        append_error(site_id, clip, det.get("error", "unknown error"))
        return None

    from ..services.event_persistence import persist_detection
    return persist_detection(det, site_id, clip, _site_config(site_id))


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
