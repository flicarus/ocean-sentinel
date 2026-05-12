"""`os bench` — inference latency benchmark on the user's hardware.

Reproducible single-command benchmark. Useful to:
- Verify the claimed real-time factor on a new install
- Compare hardware (MPS vs CPU vs CUDA) without writing throwaway code
- Sanity-check before deploying to a low-spec edge device

Runs N warmup + N timed predictions on a synthetic spectrogram and
reports median + p95 latencies. Synthetic spec rather than a sample
file keeps the benchmark portable and stops it being skewed by audio
decode time.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import typer


def bench_command(
    n: int = typer.Option(
        50, "--n", help="Number of timed predictions after warmup."
    ),
    warmup: int = typer.Option(
        5, "--warmup", help="Warmup predictions (excluded from timings)."
    ),
    site: str | None = typer.Option(
        None, "--site",
        help="Site_id to exercise per-site threshold path. Skip for the default."
    ),
) -> None:
    """Measure CNN inference latency on this machine.

    Output:
        median_ms  / p50  — typical case
        p95_ms            — 95-percentile tail
        realtime_factor   — 60s clip / median_ms

    The clip is synthetic (random spectrogram, shape matches the trained
    model's input) so the result reflects pure model + per-site lookup
    cost, not librosa decoding.
    """
    from ..gemma.cnn_inference import _load_classifier, _DEFAULT_CHECKPOINT
    from .ui import console

    console.rule("[bold]Ocean Sentinel · inference benchmark")

    classifier = _load_classifier(_DEFAULT_CHECKPOINT)

    # Generate a spectrogram that matches training-time stats (mean ~ -20
    # dB, std ~ 5 dB). Random content doesn't matter for latency — we
    # just need a tensor of the right shape that survives normalisation.
    spec = np.random.randn(128, 313).astype(np.float32) * 5 - 20

    console.print(f"  warmup     : {warmup} predictions")
    console.print(f"  timed      : {n} predictions")
    console.print(f"  site       : {site or '(default 0.5)'}")
    console.print()

    # Warmup — first few MPS calls include kernel compilation.
    with console.status("[dim]warming up...[/dim]", spinner="dots"):
        for _ in range(warmup):
            classifier.predict(spec, source_id=site)

    # Timed
    timings_ms: list[float] = []
    with console.status(f"[dim]running {n} timed inferences...[/dim]", spinner="dots"):
        for _ in range(n):
            t0 = time.perf_counter()
            classifier.predict(spec, source_id=site)
            timings_ms.append((time.perf_counter() - t0) * 1000)

    timings_ms.sort()
    median = timings_ms[len(timings_ms) // 2]
    p95 = timings_ms[int(len(timings_ms) * 0.95)]
    p99 = timings_ms[int(len(timings_ms) * 0.99) if len(timings_ms) > 100 else -1]
    mean = sum(timings_ms) / len(timings_ms)
    realtime_factor = 60_000.0 / median

    console.print(f"  median (p50): [bold green]{median:6.2f} ms[/bold green]")
    console.print(f"  p95         : {p95:6.2f} ms")
    console.print(f"  p99         : {p99:6.2f} ms")
    console.print(f"  mean        : {mean:6.2f} ms")
    console.print()
    console.print(f"  real-time   : [bold]{realtime_factor:.0f}×[/bold] "
                  f"(60s clip → {median:.2f}ms)")
    console.print(f"  throughput  : {1000.0 / median:.1f} clips/sec sustained")

    # A nudge for users on older hardware: the v7.6 model is ~2.3M params,
    # which CPU can serve at ~50 ms/clip — still well above real-time.
    if median > 20:
        console.print()
        console.print("[yellow]  Note: latency above 20ms suggests CPU fallback. "
                      "Set up MPS (Mac) or CUDA (NVIDIA) for ~5ms throughput.[/yellow]")
    elif median <= 10:
        console.print()
        console.print("[green]  Excellent — under 10ms means >5000× real-time, "
                      "comfortably edge-deployable.[/green]")
