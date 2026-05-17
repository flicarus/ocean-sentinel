"""`os detect <audio.wav>` — single-file detection.

The fastest way to verify the system works. Drop a .wav (or any audio
librosa can read) at this command and you get the full pipeline output:
ship_prob, decision tier, latency. Optionally take a `--site` flag to
apply the per-site calibrated threshold for that hydrophone.

Designed for two audiences:
- A new operator wanting to sanity-check after install ("does the model
  even load on my hardware?")
- A judge / reviewer wanting a single-command demo without spinning up
  the watch-folder or the dashboard.

Output is human-readable to stdout. --json flips to a machine-readable
single-line JSON for piping into other tools.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import typer


_TIER_COLOUR = {
    "DARK_VESSEL":       "bold red",
    "CONFIRMED_VESSEL":  "yellow",
    "ACOUSTIC_ONLY_LOW": "yellow",
    "AMBIENT":           "green",
    "UNCERTAIN":         "dim",
}

_TARGET_SR_HZ = 16_000
_DEFAULT_DURATION_S = 60.0
_N_MELS = 128
_FMAX_HZ = 1_000


def _complete_site(incomplete: str) -> list[str]:
    """Shell completion for the --site flag.

    Reads the loaded per-site thresholds JSON so the user can tab-complete
    from the exact sites the calibration knows about. Falls back to a
    static list of common hydrophones when the file is missing.
    """
    import json
    from pathlib import Path

    path = Path("data/calibration/per_site_thresholds_v7_6.json")
    sites: list[str] = []
    if path.exists():
        try:
            doc = json.loads(path.read_text())
            sites = list((doc.get("per_site_thresholds") or {}).keys())
            # Also offer the bare names (without ais-correlated- prefix)
            # since _threshold_for resolves them.
            for s in list(sites):
                if s.startswith("ais-correlated-"):
                    sites.append(s.removeprefix("ais-correlated-"))
        except Exception:
            pass
    if not sites:
        sites = [
            "point-robinson", "bush-point", "orcasound-lab", "north-sjc",
            "sunset-bay", "port-townsend", "mast-center", "andrews-bay",
            "mbari", "sanctsound",
        ]
    return [s for s in sorted(set(sites)) if s.startswith(incomplete)]


def _format_prob(p: float) -> str:
    bar_len = 20
    filled = int(round(p * bar_len))
    bar = "█" * filled + "░" * (bar_len - filled)
    return f"{bar} {p:.3f}"


def _decide_tier(
    ship_prob: float,
    threshold: float,
    uncertainty: float,
    ais_in_radius: int,
    recently_gone_dark_count: int = 0,
    nearest_vessel_cpa_km: float | None = None,
) -> tuple[str, str]:
    """Map raw outputs to (decision_tier, severity).

    Delegates to `domain.decision_tier.decide_tier` so monitor / detect /
    simulate_detection all share one tier policy (and `os detect` gains
    the GONE_DARK_VESSEL tier once AIS history is wired).
    """
    from ..domain.decision_tier import AISContext, decide_tier
    return decide_tier(
        ship_prob=ship_prob,
        uncertainty=uncertainty,
        site_threshold=threshold,
        ais=AISContext(
            ais_vessels_in_radius=ais_in_radius,
            recently_gone_dark_count=recently_gone_dark_count,
            nearest_vessel_cpa_km=nearest_vessel_cpa_km,
        ),
    )


def detect_command(
    audio: str = typer.Argument(..., help="Path to a .wav (or anything librosa loads)."),
    site: str | None = typer.Option(
        None, "--site", "-s",
        help="Hydrophone site_id for per-site threshold calibration. "
             "If omitted, the global default 0.5 is used.",
        autocompletion=_complete_site,
    ),
    ais_radius: int = typer.Option(
        0, "--ais",
        help="Number of AIS-registered vessels currently in radius. "
             "Drives DARK_VESSEL vs CONFIRMED_VESSEL classification.",
    ),
    as_json: bool = typer.Option(
        False, "--json",
        help="Emit a single-line JSON record instead of human-readable output.",
    ),
    memory: bool = typer.Option(
        False, "--memory",
        help="Query the ChromaDB acoustic memory (Tier-2 of the pipeline) "
             "for the top-3 most similar past events. Useful for triage.",
    ),
    no_push: bool = typer.Option(
        False, "--no-push",
        help="Skip pushing this detection to the central dashboard. "
             "By default `os detect --site X` mirrors `os monitor` "
             "behaviour and posts the event to the ingest gateway; pass "
             "this when you just want a local one-shot classification.",
    ),
) -> None:
    """Run the v7.6 + calibration pipeline on a single audio file.

    When --site is provided, the detection is also persisted to the
    project's central dashboard (same path as `os monitor`). Use
    --no-push to suppress that side effect — useful for ad-hoc
    sanity checks you don't want showing up as alerts.

    Examples:

        os detect my_clip.wav
        os detect my_clip.wav --site point-robinson --ais 0
        os detect my_clip.wav --site point-robinson --no-push  # local only
        os detect my_clip.wav --json | jq .ship_prob
    """
    from ..gemma.cnn_inference import _load_classifier, _DEFAULT_CHECKPOINT
    from .ui import console, fail, ok

    path = Path(audio)
    if not path.exists():
        fail(f"Audio file not found: {audio}")
        raise typer.Exit(code=2)

    try:
        duration = librosa.get_duration(path=str(path))
    except Exception as e:
        fail(f"Could not read audio file: {type(e).__name__}: {e}")
        raise typer.Exit(code=2)

    if duration < 1.0:
        fail(f"Clip is {duration:.2f}s; need at least 1s.")
        raise typer.Exit(code=2)

    # Load audio + make spectrogram (matches training preprocessing exactly).
    t0 = time.perf_counter()
    try:
        y, sr = librosa.load(
            str(path), sr=_TARGET_SR_HZ, mono=True,
            duration=_DEFAULT_DURATION_S,
        )
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=_N_MELS, fmax=_FMAX_HZ,
        )
        spec = librosa.power_to_db(mel, ref=1.0)
    except Exception as e:
        fail(f"Audio preprocessing failed: {type(e).__name__}: {e}")
        raise typer.Exit(code=1)

    # Load classifier (cached) — auto-loads per-site thresholds.
    classifier = _load_classifier(_DEFAULT_CHECKPOINT)

    # Predict with site context — this applies the per-site threshold
    # to the `label` decision. We still inspect ship_prob to surface
    # to the user.
    pred = classifier.predict(spec, source_id=site)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    ship_prob = float(pred["probabilities"]["ship"])
    uncertainty = float(pred["uncertainty"])

    # Active threshold for this detection — the per-site one if loaded,
    # else 0.5. Stay in sync with the classifier's internal lookup so
    # the tier we report matches its `label`.
    threshold = classifier._threshold_for(site)  # internal-but-stable
    tier, severity = _decide_tier(ship_prob, threshold, uncertainty, ais_radius)

    if as_json:
        out = {
            "audio": str(path),
            "duration_s": round(duration, 2),
            "site_id": site,
            "ais_vessels_in_radius": ais_radius,
            "ship_prob": round(ship_prob, 4),
            "label": pred["label"],
            "decision_tier": tier,
            "severity": severity,
            "uncertainty": round(uncertainty, 4),
            "active_threshold": round(threshold, 4),
            "threshold_source": "per-site" if threshold != 0.5 else "default",
            "pipeline_ms": round(elapsed_ms, 1),
            "model": _DEFAULT_CHECKPOINT,
        }
        print(json.dumps(out))
        return

    colour = _TIER_COLOUR.get(tier, "default")
    console.rule("[bold]Ocean Sentinel · detection")
    console.print(f"  file       : [bold]{path.name}[/bold]  ({duration:.1f}s)")
    if site:
        thr_note = "per-site calibration" if threshold != 0.5 else "no calibration for this site → default"
        console.print(f"  site       : {site}  ({thr_note})")
    else:
        console.print(f"  site       : (none) — default 0.5 threshold")
    console.print(f"  AIS in rng : {ais_radius}")
    console.print()
    console.print(f"  ship_prob  : {_format_prob(ship_prob)}")
    console.print(f"  threshold  : {threshold:.3f}  ({'PASS' if ship_prob >= threshold else 'fail'})")
    console.print(f"  uncertainty: {uncertainty:.3f}  (abstain if >0.25)")
    console.print()
    console.print(f"  → tier     : [{colour}]{tier}[/{colour}]  ({severity})")
    console.print(f"  → latency  : {elapsed_ms:.1f} ms end-to-end (load + mel + CNN)")
    console.print()

    if tier == "DARK_VESSEL":
        ok("No AIS vessel in radius but ship signature confirmed. Investigate.")
    elif tier == "CONFIRMED_VESSEL":
        console.print("[yellow]AIS reports a vessel in radius; CNN confirms the acoustic signature.[/yellow]")
    elif tier == "ACOUSTIC_ONLY_LOW":
        console.print("[dim]Some ship signature, but below confident-detection threshold.[/dim]")
    elif tier == "AMBIENT":
        console.print("[green]No vessel detected.[/green]")
    elif tier == "UNCERTAIN":
        console.print("[dim]Model abstained — uncertainty above 0.25 means insufficient evidence either way.[/dim]")

    # Tier-2 memory lookup. Optional because spinning up ChromaDB takes
    # a few hundred ms on first call and not every detection cares.
    if memory:
        _show_memory_matches(pred.get("embedding", []), console)

    # Persist this detection (local jsonl + remote gateway) on the same
    # path `os monitor` uses, so a user running ad-hoc `os detect` for
    # a real site sees it on the dashboard. Skipped when --site is
    # missing (no site context to attribute the event to) or --no-push
    # is set, and always skipped for --json output (callers piping JSON
    # tend to be scripts that don't want side effects).
    if site and not no_push and not as_json:
        det = {
            "ok": True,
            "decision_id":           f"DET-{abs(hash(str(path) + site)) % 100000:05d}",
            "site_id":               site,
            "clip":                  str(path),
            "cnn_label":             pred.get("label", "not_ship"),
            "cnn_confidence":        round(ship_prob, 3),
            "cnn_uncertainty":       round(uncertainty, 3),
            "conformal_threshold":   round(threshold, 3),
            "conformal_p":           round(ship_prob, 3),
            "conformal_pass":        bool(ship_prob >= threshold),
            "ais_vessels_in_radius": ais_radius,
            "decision_tier":         tier,
            "severity":              severity,
            "checkpoint":            _DEFAULT_CHECKPOINT,
            "summary":               f"{tier} ({severity}) · CNN p={ship_prob:.2f}",
        }
        try:
            from ..services.event_persistence import persist_detection
            cfg = _load_site_config(site)
            persist_detection(det, site, path, cfg)
        except Exception:
            # Persistence is a side effect — never fail the CLI on it.
            pass


def _load_site_config(site_id: str) -> dict:
    """Best-effort read of data/sites/{site_id}.yaml (returns {} if missing)."""
    import yaml
    p = Path("data/sites") / f"{site_id}.yaml"
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def _show_memory_matches(embedding: list, console) -> None:
    """Query ChromaDB for the closest past events. Prints up to 3 with
    timestamp, location, threat_level. Robust to ChromaDB being empty
    or unavailable — these are nice-to-have, not required."""
    if not embedding:
        return
    try:
        import asyncio
        from ..adapters.chromadb_store import ChromaDBAcousticMemory
    except Exception as e:
        console.print(f"[yellow]  memory: import failed ({e.__class__.__name__})[/yellow]")
        return

    async def _go() -> list:
        store = ChromaDBAcousticMemory()
        try:
            matches = await store.query_by_embedding(embedding, n=3)
            return matches
        finally:
            await store.close()

    try:
        matches = asyncio.run(_go())
    except Exception as e:
        console.print(f"[yellow]  memory: query failed ({e.__class__.__name__}: {e})[/yellow]")
        return

    if not matches:
        console.print()
        console.print("[dim]  memory: no comparable past events in ChromaDB (empty store).[/dim]")
        return

    console.print()
    console.print("[bold]  Closest past events in ChromaDB (Tier-2 memory)[/bold]")
    for m in matches:
        e = m.entry
        loc = f"{e.location.lat:.3f},{e.location.lon:.3f}"
        ts = e.timestamp.strftime("%Y-%m-%d %H:%M")
        console.print(
            f"    distance={m.score:.3f}  {ts}  "
            f"loc={loc}  threat={e.threat_level.value}  "
            f"vessel_type={e.vessel_type or '-'}"
        )
