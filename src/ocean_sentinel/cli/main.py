"""Ocean Sentinel CLI — `os` entrypoint.

`os onboard`         → live Gemma function-calling onboarding via Ollama
`os onboard --demo`  → scripted walkthrough with mocked tools (offline-safe)
"""
from __future__ import annotations

import time

import httpx
import typer

from ..gemma.agent import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    GemmaAgent,
)
from ..gemma.onboarding import OnboardingFlow
from .ui import (
    ask,
    confirm,
    console,
    error,
    fail,
    gemma_say,
    ok,
    show_banner,
    site_registered,
    step_header,
    table,
    tool_call,
    tool_call_static,
    warn,
)

app = typer.Typer(
    name="os",
    help="Ocean Sentinel — acoustic dark-vessel detection.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Multi-command shell. Forces Typer to keep subcommands as subcommands
    even when only one is registered (otherwise it collapses to direct call)."""


@app.command()
def onboard(
    demo: bool = typer.Option(
        False, "--demo", help="Scripted walkthrough with mocked tools (offline-safe)."
    ),
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", help="Ollama model id for live mode."
    ),
    host: str = typer.Option(
        DEFAULT_HOST, "--host", help="Ollama host URL."
    ),
) -> None:
    """Walk through site onboarding with Gemma."""
    if demo:
        _run_demo()
    else:
        _run_live(model=model, host=host)


def _check_ollama(host: str) -> str | None:
    """Return None if Ollama is reachable, else an error message string."""
    try:
        r = httpx.get(f"{host}/api/tags", timeout=2.0)
        r.raise_for_status()
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


class _CliUI:
    """Concrete UISink that delegates to the cli/ui.py module-level functions.
    Defined as a class so it satisfies the typing.Protocol dispatch and is
    trivially mockable in tests."""

    step_header   = staticmethod(step_header)
    gemma_say     = staticmethod(gemma_say)
    ask           = staticmethod(ask)
    confirm       = staticmethod(confirm)
    tool_call_static = staticmethod(tool_call_static)
    ok            = staticmethod(ok)
    fail          = staticmethod(fail)
    warn          = staticmethod(warn)
    error         = staticmethod(error)


def _run_live(*, model: str, host: str) -> None:
    show_banner()

    err = _check_ollama(host)
    if err:
        error(f"Cannot reach Ollama at {host} — {err}")
        warn("Start it with `ollama serve`, then retry. "
             "For offline preview run `os onboard --demo`.")
        raise typer.Exit(1)

    console.print(f"  [grey50]model[/] [bold]{model}[/]   "
                  f"[grey50]host[/] [bold]{host}[/]\n")

    agent = GemmaAgent(model=model, host=host)
    ui = _CliUI()
    flow = OnboardingFlow(agent=agent, ui=ui)
    flow.offer_resume()

    ctx = flow.run()

    if ctx.last_completed_step == 8 and ctx.registered_yaml_path:
        site_registered(
            ctx.site_id,
            {
                "location":    f"{ctx.lat:.4f} N, {abs(ctx.lon):.4f} W"
                                if ctx.lat is not None and ctx.lon is not None else "—",
                "depth":       f"{ctx.depth_m} m" if ctx.depth_m else "—",
                "model":       f"cnn-v7.4 + adapter (val_acc {ctx.adapter_val_acc:.3f})"
                                if ctx.adapter_val_acc else "cnn-v7.4",
                "threshold":   f"p ≥ {ctx.conformal_threshold_p:.2f}"
                                if ctx.conformal_threshold_p else "—",
                "sensitivity": ctx.sensitivity or "—",
                "alerts to":   ctx.alert_email or "(none)",
            },
        )


# ── Demo flow ────────────────────────────────────────────────────────────
TOTAL_STEPS = 8


def _run_demo() -> None:
    show_banner()

    # Step 1 ──────────────────────────────────────────────────────────────
    step_header(1, TOTAL_STEPS, "Site discovery", "where is your hydrophone?")
    gemma_say(
        "Hi. Let's set up acoustic monitoring for your site. "
        "I'll walk you through 8 steps — about 5 minutes."
    )
    ask("Where is your hydrophone?", hint="lat, lon  ·  or  stream URL")

    with tool_call("validate_site_coords", {"lat": 36.7128, "lon": -121.9023}):
        time.sleep(1.2)
    ok("ocean confirmed · 520m depth · MBNMS boundary 8.2 km W")

    with tool_call("fetch_ais_baseline",
                   {"lat": 36.7128, "lon": -121.9023, "radius_km": 10, "days": 30}):
        time.sleep(1.6)
    table(
        [
            ("vessels / day",  "47 (avg)"),
            ("vessel mix",     "tanker 38% · fishing 29% · cargo 22%"),
            ("shipping lane",  "11.0 km N · transits every ~38 min"),
            ("peak hours",     "14:00–18:00 local"),
        ],
        title="AIS baseline · last 30d",
    )
    gemma_say(
        "Heads up — there's a shipping lane 11 km north with a transit every ~38 min. "
        "We'll need to tune the false-alarm floor for that."
    )

    # Step 2 ──────────────────────────────────────────────────────────────
    step_header(2, TOTAL_STEPS, "Acoustic baseline", "fingerprint your site")
    gemma_say("I need 5 minutes of ambient audio to fingerprint your site.")
    ask("Audio source?", hint="path to .wav  ·  or  'r' to record now")

    with tool_call("record_ambient",
                   {"source": "monterey_5min.wav", "duration_min": 5}):
        time.sleep(0.9)
    ok("loaded 300.0s · 192 kHz · float32")

    with tool_call("compute_spectral_signature", {"audio": "<300s buffer>"}):
        time.sleep(2.0)
    table(
        [
            ("dominant band",   "80–200 Hz (vessel range)"),
            ("ambient class",   "deep-water · low traffic"),
            ("median PSD",      "−74.2 dB re µPa²/Hz"),
            ("snapping shrimp", "absent (depth > 100 m)"),
        ],
        title="spectral signature",
    )

    # Step 3 ──────────────────────────────────────────────────────────────
    step_header(3, TOTAL_STEPS, "Transfer learning", "find nearest known sites")
    with tool_call("compare_to_known_sites", {"signature": "<vec[64]>"}):
        time.sleep(1.4)
    table(
        [
            ("MBARI MARS",       "cosine 0.84   ✓ recommended"),
            ("monterey-bay-aq",  "cosine 0.71"),
            ("port-townsend",    "cosine 0.43"),
        ],
        title="nearest training sites",
    )
    gemma_say(
        "Closest match is MBARI MARS — same depth class, similar ambient. "
        "I recommend a light fine-tune on the last 2 layers using your sample. "
        "About 3 minutes on your machine."
    )
    if not confirm("Proceed with the recommended adapter?"):
        error("aborted by user.")
        raise typer.Exit(1)

    # Step 4 ──────────────────────────────────────────────────────────────
    step_header(4, TOTAL_STEPS, "Per-site adapter", "fine-tune last 2 layers")
    tool_call_static("finetune_adapter",
                     {"site": "monterey-test", "epochs": 10, "lr": 3e-4})
    for ep in range(1, 11):
        time.sleep(0.16)
        loss = 1.42 - 0.09 * ep
        acc  = 0.62 + 0.028 * ep
        console.print(
            f"      [grey50]epoch[/] [bold white]{ep:>2}[/]/10  "
            f"[grey50]loss[/] [white]{loss:.3f}[/]  "
            f"[grey50]val_acc[/] [white]{acc:.3f}[/]"
        )
    console.print()
    ok("adapter trained · val_acc 0.892 · saved data/sites/monterey-test/adapter.pt")

    # Step 5 ──────────────────────────────────────────────────────────────
    step_header(5, TOTAL_STEPS, "Conformal calibration", "false-alarm budget")
    with tool_call("calibrate_conformal",
                   {"site": "monterey-test", "n": 200, "alpha": 0.05}):
        time.sleep(1.5)
    table(
        [
            ("threshold p",       "0.71"),
            ("coverage",          "95.4% (target 95.0%)"),
            ("expected FA rate",  "1 false alarm per ~50 ambient hours"),
        ],
        title="conformal threshold",
    )

    # Step 6 ──────────────────────────────────────────────────────────────
    step_header(6, TOTAL_STEPS, "Alert policy", "sensitivity + channels")
    gemma_say(
        "How sensitive should we be? 'high' fires on borderline detections, "
        "'medium' is balanced, 'low' fires only when we're very sure."
    )
    sensitivity = ask("Sensitivity?", hint="high · medium · low", default="medium")
    email_in = ask("Alert email?", hint="leave blank to skip")
    email = email_in or "—"
    with tool_call("set_alert_policy", {"sensitivity": sensitivity, "email": email}):
        time.sleep(0.7)
    ok(f"sensitivity = {sensitivity} · email = {email}")

    # Step 7 ──────────────────────────────────────────────────────────────
    step_header(7, TOTAL_STEPS, "Test detection", "run pipeline on a sample")
    with tool_call("simulate_detection",
                   {"site": "monterey-test", "clip": "test_clip.wav"}):
        time.sleep(1.9)
    table(
        [
            ("CNN confidence",  "0.82  (ship)"),
            ("conformal pass",  "yes  (p = 0.78  >  0.71)"),
            ("AIS context",     "0 vessels in 10 km radius"),
            ("decision tier",   "DARK_VESSEL"),
            ("severity",        "HIGH"),
        ],
        title="pipeline trace",
    )
    with tool_call("explain_decision",
                   {"id": "DRY-RUN-0001", "modality": "spectrogram + text"}):
        time.sleep(1.6)
    gemma_say(
        "Here's what I see: blade-rate harmonics at 2.4 Hz consistent with low-speed "
        "trawling, broadband signature 18 dB above your ambient floor, zero AIS "
        "contacts in radius. CNN agrees, conformal threshold passed comfortably. "
        "If this happened in production, I'd page you immediately."
    )

    # Step 8 ──────────────────────────────────────────────────────────────
    step_header(8, TOTAL_STEPS, "Register site", "save configuration")
    site_id = ask("Site ID?", hint="short, kebab-case", default="monterey-test")
    with tool_call("register_site", {"site_id": site_id, "config": "<resolved>"}):
        time.sleep(0.6)
    ok(f"wrote data/sites/{site_id}.yaml")

    site_registered(
        site_id,
        {
            "location":    "36.7128 N, 121.9023 W",
            "depth":       "520 m",
            "model":       "cnn-v7.4 + adapter (val_acc 0.892)",
            "threshold":   "p ≥ 0.71 (95% coverage)",
            "sensitivity": sensitivity,
            "alerts to":   email if email != "—" else "(none)",
        },
    )


if __name__ == "__main__":
    app()