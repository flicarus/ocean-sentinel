"""`os info` — show what's deployed.

A one-shot status command. Useful after install, after a model swap, or
when triaging a "why is this returning weird numbers?" question. Reports:

- Model checkpoint path and size
- Default conformal threshold (global)
- Per-site thresholds loaded (count + the actual map)
- Detected inference device (mps / cuda / cpu)
- Eval scores from data/eval/*.json when available

Read-only. Never modifies state.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer


def info_command() -> None:
    """Show the deployed model + calibration status."""
    from ..gemma.cnn_inference import (
        _DEFAULT_CHECKPOINT,
        _DEFAULT_CONFORMAL,
        _DEFAULT_SITE_THRESHOLDS,
    )
    from .ui import console

    console.rule("[bold]Ocean Sentinel · deployed configuration")

    # ── Model ─────────────────────────────────────────────────────────
    ckpt = Path(_DEFAULT_CHECKPOINT)
    if ckpt.exists():
        size_mb = ckpt.stat().st_size / 1024 / 1024
        console.print(f"  Model checkpoint   : [green]{ckpt}[/green]  ({size_mb:.1f} MB)")
    else:
        console.print(f"  Model checkpoint   : [red]MISSING[/red]  ({ckpt})")
        console.print("    → first run will fail. Train v7.6 or fix _DEFAULT_CHECKPOINT.")

    # ── Device ────────────────────────────────────────────────────────
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps (Apple Silicon GPU)"
        elif torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        else:
            device = "cpu"
        console.print(f"  Inference device   : [cyan]{device}[/cyan]")
    except Exception as e:
        console.print(f"  Inference device   : [yellow]could not probe ({e.__class__.__name__})[/yellow]")

    # ── Conformal calibration (global) ─────────────────────────────────
    conf_path = Path(_DEFAULT_CONFORMAL)
    if conf_path.exists():
        try:
            conf = json.loads(conf_path.read_text())
            thr = conf.get("threshold")
            alpha = conf.get("alpha")
            n = conf.get("n_calibration")
            console.print(f"  Conformal (global) : threshold {thr:.3f}  α={alpha}  n_cal={n}")
        except Exception as e:
            console.print(f"  Conformal (global) : [yellow]parse error ({e})[/yellow]")
    else:
        console.print(f"  Conformal (global) : [yellow]not found at {conf_path}[/yellow]")

    # ── Per-site thresholds ────────────────────────────────────────────
    thr_path = Path(_DEFAULT_SITE_THRESHOLDS)
    if thr_path.exists():
        try:
            thr_doc = json.loads(thr_path.read_text())
            per_site = thr_doc.get("per_site_thresholds", {})
            console.print(f"  Per-site thresholds: [green]{len(per_site)} sites loaded[/green]")
            for site, t in sorted(per_site.items(), key=lambda x: x[1]):
                emoji = "↓" if t < 0.5 else "↑"
                console.print(f"      {emoji} {site:<32} {t:.2f}")
        except Exception as e:
            console.print(f"  Per-site thresholds: [yellow]parse error ({e})[/yellow]")
    else:
        console.print(f"  Per-site thresholds: [dim]none (will use default 0.5 everywhere)[/dim]")

    # ── Eval scores when available ────────────────────────────────────
    console.print()
    console.print("[bold]  Latest eval results[/bold]")
    for name, path_str in [
        ("v7.5 baseline",                 "data/eval/per_site_v7_5.json"),
        ("v7.6 vanilla",                  "data/eval/per_site_v7_6.json"),
        ("v7.6 + calibration (honest)",   "data/calibration/per_site_thresholds_v7_6_honest.json"),
    ]:
        p = Path(path_str)
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
            if "overall" in d:
                acc = d["overall"]["accuracy"]
                n = d["overall"]["n"]
                console.print(f"    {name:<32} {acc:.1%}   (n={n:,})  {path_str}")
            elif "overall_test_acc_tuned" in d:
                acc = d["overall_test_acc_tuned"]
                n = d.get("test_set_n", "?")
                console.print(f"    {name:<32} {acc:.1%}   (n={n:,})  {path_str}")
        except Exception:
            continue

    # ── Inference perf ────────────────────────────────────────────────
    perf_path = Path("data/eval/inference_perf_v7_6.json")
    if perf_path.exists():
        try:
            perf = json.loads(perf_path.read_text())
            med = perf["calibrated"]["median_ms"]
            p95 = perf["calibrated"]["p95_ms"]
            rt = perf["realtime_factor"]
            console.print(f"  Latency (calibrated): {med:.2f} ms median, {p95:.2f} ms p95  (real-time factor {rt:.0f}×)")
        except Exception:
            pass

    console.print()
    console.print("[dim]Run `os detect <audio.wav>` to verify end-to-end on your hardware.[/dim]")
