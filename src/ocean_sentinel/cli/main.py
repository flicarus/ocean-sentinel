"""Ocean Sentinel CLI — `os` entrypoint.

`os onboard`                  → live Gemma function-calling onboarding via Ollama
`os onboard --demo`           → scripted walkthrough with mocked tools (offline-safe)
`os refresh <site>`           → re-fit adapter + recalibrate on accumulated ambient
`os test <site>`              → run bundled known-label samples
`os monitor <site> --watch …` → operational mode: watch a folder, alert on each new clip
`os monitor <site> --replay …`→ one-shot: process every clip in a folder, exit
"""
from __future__ import annotations

import time
from pathlib import Path

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
def monitor(
    site_id: str = typer.Argument(..., help="Site identifier (kebab-case)."),
    watch_dir: str | None = typer.Option(
        None, "--watch", help="Directory to watch for new .wav files (continuous mode).",
    ),
    replay_dir: str | None = typer.Option(
        None, "--replay", help="Directory to process once and exit (demo / batch mode).",
    ),
    interval: float = typer.Option(
        2.0, "--interval", help="Seconds between watch polls.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Print every clip including AMBIENT decisions.",
    ),
) -> None:
    """Operational mode for an onboarded hydrophone site.

    Two modes:

      --watch <dir>    poll for new .wav files, process each as it lands
                       (use this for live deployment)
      --replay <dir>   process every .wav once in mtime order, then exit
                       (use this for demo / batch verification)

    Each detection is appended to data/sites/<site>/events.jsonl which
    the dashboard at /api/events/feed reads directly. Failures (corrupt
    audio, unreadable file) are logged to errors.jsonl and don't
    interrupt the loop.

    If the site has not been onboarded, Gemma offers to run the Site
    Onboarding Protocol first — without per-site calibration the alerts
    would silently use the global threshold.
    """
    from .test_command import is_site_onboarded
    from .monitor_command import replay, watch, _site_dir

    if (watch_dir is None) == (replay_dir is None):
        error("Specify exactly one of --watch <dir> or --replay <dir>.")
        raise typer.Exit(1)

    target_dir = Path(watch_dir or replay_dir)
    if not target_dir.exists() or not target_dir.is_dir():
        error(f"directory not found: {target_dir}")
        raise typer.Exit(1)

    show_banner()
    console.print(f"  [grey50]site[/]   [bold]{site_id}[/]")
    if watch_dir:
        console.print(f"  [grey50]watch[/]  [bold]{watch_dir}[/]   "
                      f"[grey50]poll[/] [bold]{interval:.1f}s[/]")
    else:
        console.print(f"  [grey50]replay[/] [bold]{replay_dir}[/]")
    console.print()

    if not is_site_onboarded(site_id):
        gemma_say(
            f"Site '{site_id}' isn't onboarded yet. Without a per-site "
            f"adapter and threshold, alerts would use the global v7.4 "
            f"calibration — accurate enough for distribution-typical "
            f"sites but unreliable for anything OOD.\n\n"
            f"The Site Onboarding Protocol is a short interactive "
            f"walkthrough (about 5 minutes). Strongly recommended before "
            f"running live monitoring."
        )
        if confirm(f"Run the Site Onboarding Protocol for '{site_id}' now?",
                   default_yes=True):
            err_check = _check_ollama(DEFAULT_HOST)
            if err_check:
                error(f"Cannot reach Ollama at {DEFAULT_HOST} — {err_check}")
                warn("Onboarding requires `ollama serve` running with "
                     f"{DEFAULT_MODEL} pulled.")
                raise typer.Exit(1)
            _run_live(model=DEFAULT_MODEL, host=DEFAULT_HOST)
            if not is_site_onboarded(site_id):
                warn(f"Onboarding did not complete; not starting monitor.")
                raise typer.Exit(1)
            console.print()
            ok(f"site '{site_id}' onboarded — starting monitor...")
            console.print()
        else:
            warn(
                f"OK — onboarding skipped. Monitor will run with the "
                f"global v7.4 threshold; alerts may be noisy. When you're "
                f"ready: `os onboard`, then re-run this command."
            )
            console.print()

    events_path = _site_dir(site_id) / "events.jsonl"

    def _print_event(file: Path, event: dict | None) -> None:
        if event is None:
            console.print(f"  [red]✗[/] {file.name}  failed (see errors.jsonl)")
            return
        tier = event.get("decision_tier") or "—"
        cnn_p = event.get("cnn_confidence") or 0.0
        thresh = event.get("conformal_threshold") or 0.0
        if tier in ("DARK_VESSEL", "CONFIRMED_VESSEL", "ACOUSTIC_ONLY_LOW"):
            colour = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "cyan"}.get(
                event.get("severity", "LOW"), "cyan",
            )
            console.print(
                f"  [{colour}]●[/] {file.name}  "
                f"[bold]{tier}[/] ({event.get('severity', '—')}) · "
                f"p={cnn_p:.2f} > {thresh:.2f}  "
                f"[grey50]{event.get('id')}[/]"
            )
        elif tier == "AMBIENT":
            if verbose:
                console.print(
                    f"  [grey50]·[/] {file.name}  AMBIENT  p={cnn_p:.2f} ≤ {thresh:.2f}"
                )
        else:  # UNCERTAIN
            console.print(
                f"  [yellow]?[/] {file.name}  {tier}  p={cnn_p:.2f}"
            )

    if replay_dir is not None:
        events = replay(site_id=site_id, folder=target_dir, on_event=_print_event)
        console.print()
        ok(f"replayed {len(events)} clip(s) → {events_path}")
        return

    # Watch mode — runs until Ctrl-C
    console.print(f"  [grey50]events →[/] [bold]{events_path}[/]")
    console.print(f"  [grey50]Ctrl-C to stop[/]\n")
    try:
        for _ in watch(site_id=site_id, folder=target_dir,
                       poll_interval_s=interval, on_event=_print_event):
            pass
    except KeyboardInterrupt:
        console.print()
        ok("monitor stopped")


@app.command()
def test(
    site_id: str = typer.Argument(..., help="Site identifier (kebab-case)."),
) -> None:
    """Run bundled known-label samples through the calibrated pipeline.

    Five samples ship with the package:
    - 3 real DeepShip vessel clips (tug, cargo, passenger) → expected ship
    - 2 deterministically-synthesised ambient clips → expected not_ship

    The site's per-site adapter and conformal threshold are loaded
    automatically. Output shows per-sample expected vs actual decision
    plus an aggregate score, so you can confirm — without trusting
    marketing copy — that your calibrated pipeline actually works on
    real audio.

    If the site has not been onboarded yet, Gemma offers to run the
    Site Onboarding Protocol first — without per-site calibration the
    results would silently fall back to the global threshold and look
    misleading.
    """
    from .test_command import run_tests, is_site_onboarded

    show_banner()
    console.print(f"  [grey50]site[/] [bold]{site_id}[/]")
    console.print(f"  [grey50]samples[/] [bold]bundled (5)[/]\n")

    if not is_site_onboarded(site_id):
        gemma_say(
            f"I can't find a per-site calibration for '{site_id}' — this "
            f"site hasn't been onboarded into SOFAR AI yet. Without "
            f"calibration the test would fall back to the global "
            f"threshold trained on a different acoustic distribution, "
            f"and the numbers would likely be unreliable.\n\n"
            f"The Site Onboarding Protocol is a short interactive walkthrough "
            f"(about 5 minutes). It records 3 minutes of your site's "
            f"ambient, fits a tiny per-site adapter on top of the base "
            f"CNN, and calibrates a false-alarm threshold with provable "
            f"guarantees. Strongly recommended before running tests."
        )
        if confirm(f"Run the Site Onboarding Protocol for '{site_id}' now?",
                   default_yes=True):
            err_check = _check_ollama(DEFAULT_HOST)
            if err_check:
                error(f"Cannot reach Ollama at {DEFAULT_HOST} — {err_check}")
                warn("Onboarding requires `ollama serve` running with "
                     f"{DEFAULT_MODEL} pulled. Start it and re-run "
                     f"`os onboard` first, then retry `os test {site_id}`.")
                raise typer.Exit(1)
            _run_live(model=DEFAULT_MODEL, host=DEFAULT_HOST)
            if not is_site_onboarded(site_id):
                warn(
                    f"Onboarding did not complete for '{site_id}' "
                    f"(no adapter.pt found). Skipping test."
                )
                raise typer.Exit(1)
            console.print()
            ok(f"site '{site_id}' onboarded — running test now...")
            console.print()
        else:
            warn(
                f"OK — onboarding skipped. Without per-site calibration "
                f"the test would produce misleading numbers; not running "
                f"it. When you're ready: `os onboard` (then `os test "
                f"{site_id}`)."
            )
            raise typer.Exit(0)

    with tool_call("os.test", {"site_id": site_id}):
        result = run_tests(site_id=site_id)

    if not result.get("ok"):
        fail(result.get("error", "test failed"))
        raise typer.Exit(1)

    rows: list[tuple[str, str]] = []
    for r in result["results"]:
        check = "✓" if r.get("correct") else "✗"
        if r.get("error"):
            value = f"[red]{check} ERROR · {r['error']}[/]"
        else:
            tier = r.get("decision_tier", "?")
            p = r.get("ship_prob", 0.0)
            colour = "green" if r.get("correct") else "red"
            value = (
                f"[{colour}]{check}[/]  expected={r['expected_label']:<8}  "
                f"got={tier}  p={p:.2f}"
            )
        rows.append((r["path"], value))

    table(rows, title=f"os test · {site_id}")

    n_ok = result["n_correct"]
    n_total = result["n_samples"]
    if n_ok == n_total:
        ok(f"all {n_total} samples classified as expected · pipeline working")
    elif n_ok >= n_total - 1:
        ok(
            f"{n_ok}/{n_total} samples classified as expected · 1 near-threshold "
            f"miss is normal on OOD sites — see `os refresh` if persistent"
        )
    else:
        warn(
            f"{n_ok}/{n_total} samples classified as expected — your site may "
            f"need refresh. Check decision_tier vs expected_label above."
        )


@app.command()
def refresh(
    site_id: str = typer.Argument(..., help="Site identifier (kebab-case)."),
    add: str | None = typer.Option(
        None, "--add", help="Path to a new ambient .wav to fold into the corpus.",
    ),
    trust_max: float = typer.Option(
        0.7,
        "--trust-max",
        help="Max ship_prob for windows to keep as 'trusted ambient' "
             "(higher = more permissive, but risks contaminating with real ships).",
    ),
    alpha: float = typer.Option(
        0.05, "--alpha", help="Target false-alarm rate for the new threshold."
    ),
) -> None:
    """Re-fit per-site adapter and recalibrate conformal threshold on
    the user's accumulated ambient.

    Run this periodically (e.g. weekly) once the hydrophone has collected
    more ambient than the original 3-minute onboarding sample. The system
    will:

    1. Combine the original onboarding ambient with everything in
       data/sites/<site_id>/ambient/*.wav (and the optional --add path).
    2. Filter out windows that look like real vessel events using the
       current model — we don't want to teach the adapter that real
       ships are ambient.
    3. Re-fit the small per-site adapter.
    4. Recalibrate the conformal threshold on the new (post-adapter)
       ambient distribution.

    Reports the before/after threshold so you can see the calibration
    tighten as more data accumulates.
    """
    from ..gemma.refresh import refresh_site

    show_banner()
    console.print(f"  [grey50]site[/] [bold]{site_id}[/]")
    if add:
        console.print(f"  [grey50]+add[/] [bold]{add}[/]")
    console.print()

    with tool_call("refresh_site",
                   {"site_id": site_id, "trust_max": trust_max, "alpha": alpha}):
        result = refresh_site(
            site_id=site_id,
            additional_ambient_path=add,
            trust_max_ship_prob=trust_max,
            alpha=alpha,
        )

    if not result.get("ok"):
        fail(result.get("error", "refresh failed"))
        raise typer.Exit(1)

    before = result["before"]
    after = result["after"]
    filt = result["filter"]

    rows = [
        ("ambient files combined",   str(result["n_ambient_files"])),
        ("total ambient seconds",    str(result["ambient_total_seconds"])),
        ("trusted windows",
         f"{filt['n_trusted_windows']}/{filt['n_total_windows']}"
         + ("  (fallback)" if filt.get("filter_fallback") else "")),
        ("threshold (before)",
         f"{before['threshold']:.3f}" if before.get("threshold") else "—"),
        ("threshold (after)",        f"{after['threshold']:.3f}"),
        ("median ambient ship_prob (before adapter retrain)",
         f"{after['adapter_median_amb_p_before']:.3f}"),
        ("median ambient ship_prob (after adapter retrain)",
         f"{after['adapter_median_amb_p_after']:.3f}"),
        ("held-out vessel recall (after)",
         f"{after['holdout_recall']:.2f}"),
        ("adapter epochs",           str(after["adapter_epochs"])),
    ]
    table(rows, title=f"refresh · {site_id}")

    threshold_change = ""
    if before.get("threshold") is not None:
        delta = after["threshold"] - before["threshold"]
        threshold_change = f"  ({'+' if delta >= 0 else ''}{delta:+.3f})"
    ok(f"site refreshed · threshold {after['threshold']:.3f}{threshold_change}")


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