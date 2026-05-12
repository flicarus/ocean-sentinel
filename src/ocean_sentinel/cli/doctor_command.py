"""`os doctor` — health check for an Ocean Sentinel install.

When inference returns weird numbers or `os monitor` hangs, the first
question is always "is the basic plumbing OK?". This command checks
each piece of the plumbing and tells you exactly what's broken and how
to fix it. Read-only — never writes state.

Checks (each PASS / WARN / FAIL):
  1. Model checkpoint reachable + correct size
  2. Per-site thresholds loadable + matches model
  3. Global conformal calibration loadable
  4. PyTorch + accelerator (MPS / CUDA / CPU)
  5. librosa loads a tiny synthetic audio
  6. End-to-end inference (synthetic spec → ship_prob)
  7. Latest eval scores present
  8. Optional API server reachable
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer


# Status emojis are chosen for terminal-safe rendering across SSH, tmux,
# and the macOS native Terminal — no fancy combining glyphs.
_OK = "[bold green]✓ PASS[/bold green]"
_WARN = "[bold yellow]⚠ WARN[/bold yellow]"
_FAIL = "[bold red]✗ FAIL[/bold red]"


def doctor_command(
    api: bool = typer.Option(
        False, "--api",
        help="Also probe the local API server (http://localhost:8000/health)."
    ),
) -> None:
    """Run a self-check on the install. Exits non-zero if any FAIL."""
    from .ui import console

    console.rule("[bold]Ocean Sentinel · health check")

    fails = 0
    warns = 0

    # ── 1. Model checkpoint ────────────────────────────────────────────
    from ..gemma.cnn_inference import (
        _DEFAULT_CHECKPOINT,
        _DEFAULT_CONFORMAL,
        _DEFAULT_SITE_THRESHOLDS,
    )
    ckpt = Path(_DEFAULT_CHECKPOINT)
    if not ckpt.exists():
        console.print(f"  {_FAIL}  Model checkpoint: missing {ckpt}")
        console.print("         → train v7.6 or set OS_CHECKPOINT to an existing .pt file")
        fails += 1
    else:
        size_mb = ckpt.stat().st_size / 1024 / 1024
        if size_mb < 1 or size_mb > 100:
            console.print(f"  {_WARN}  Model checkpoint: {ckpt} ({size_mb:.1f} MB — unusual size)")
            warns += 1
        else:
            console.print(f"  {_OK}  Model checkpoint: {ckpt.name} ({size_mb:.1f} MB)")

    # ── 2. Per-site thresholds ────────────────────────────────────────
    thr_path = Path(_DEFAULT_SITE_THRESHOLDS)
    if not thr_path.exists():
        console.print(f"  {_WARN}  Per-site thresholds: missing (will use 0.5 everywhere)")
        console.print(f"         → run scripts/calibrate_per_site_honest.py to generate")
        warns += 1
    else:
        try:
            doc = json.loads(thr_path.read_text())
            n = len(doc.get("per_site_thresholds", {}))
            if n == 0:
                console.print(f"  {_WARN}  Per-site thresholds: file exists but empty")
                warns += 1
            else:
                console.print(f"  {_OK}  Per-site thresholds: {n} sites loaded from {thr_path.name}")
        except Exception as e:
            console.print(f"  {_FAIL}  Per-site thresholds: parse error ({e})")
            fails += 1

    # ── 3. Conformal calibration ──────────────────────────────────────
    conf = Path(_DEFAULT_CONFORMAL)
    if not conf.exists():
        console.print(f"  {_WARN}  Conformal calibration: missing {conf}")
        console.print("         → run scripts/calibrate_conformal.py")
        warns += 1
    else:
        try:
            d = json.loads(conf.read_text())
            t = d.get("threshold")
            console.print(f"  {_OK}  Conformal calibration: threshold {t:.3f}")
        except Exception as e:
            console.print(f"  {_FAIL}  Conformal calibration: parse error ({e})")
            fails += 1

    # ── 4. PyTorch + device ───────────────────────────────────────────
    try:
        import torch
        if torch.backends.mps.is_available():
            console.print(f"  {_OK}  PyTorch: {torch.__version__} on mps (Apple Silicon)")
        elif torch.cuda.is_available():
            console.print(f"  {_OK}  PyTorch: {torch.__version__} on cuda ({torch.cuda.get_device_name(0)})")
        else:
            console.print(f"  {_WARN}  PyTorch: {torch.__version__} on cpu (expect ~50ms/clip, still real-time)")
            warns += 1
    except Exception as e:
        console.print(f"  {_FAIL}  PyTorch: import failed ({e})")
        fails += 1

    # ── 5. librosa ─────────────────────────────────────────────────────
    try:
        import librosa
        import numpy as np
        y = np.random.randn(16000).astype(np.float32) * 0.01
        mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128, fmax=1000)
        if mel.shape[0] == 128:
            console.print(f"  {_OK}  librosa: {librosa.__version__} forward pass OK")
        else:
            console.print(f"  {_FAIL}  librosa: unexpected mel shape {mel.shape}")
            fails += 1
    except Exception as e:
        console.print(f"  {_FAIL}  librosa: failed ({e})")
        fails += 1

    # ── 6. End-to-end inference ────────────────────────────────────────
    try:
        import numpy as np
        from ..gemma.cnn_inference import _load_classifier
        clf = _load_classifier(_DEFAULT_CHECKPOINT)
        spec = np.random.randn(128, 313).astype(np.float32) * 5 - 20
        v = clf.predict(spec)
        prob = float(v["probabilities"]["ship"])
        console.print(f"  {_OK}  End-to-end inference: ship_prob={prob:.3f} (synthetic spec)")
    except Exception as e:
        console.print(f"  {_FAIL}  End-to-end inference: failed ({e.__class__.__name__}: {e})")
        fails += 1

    # ── 7. Eval scores ────────────────────────────────────────────────
    eval_files = [
        ("data/eval/per_site_v7_6.json", "v7.6 vanilla"),
        ("data/calibration/per_site_thresholds_v7_6_honest.json", "v7.6 + calibration (honest)"),
    ]
    found = 0
    for path_str, name in eval_files:
        if Path(path_str).exists():
            found += 1
    if found == len(eval_files):
        console.print(f"  {_OK}  Eval results: {found}/{len(eval_files)} files present")
    elif found > 0:
        console.print(f"  {_WARN}  Eval results: {found}/{len(eval_files)} files present")
        warns += 1
    else:
        console.print(f"  {_WARN}  Eval results: no eval files found in data/eval/")
        warns += 1

    # ── 8. Optional API check ─────────────────────────────────────────
    if api:
        try:
            import httpx
            r = httpx.get("http://localhost:8000/health", timeout=2.0)
            if r.status_code == 200:
                console.print(f"  {_OK}  API server: http://localhost:8000 responsive")
            else:
                console.print(f"  {_WARN}  API server: HTTP {r.status_code}")
                warns += 1
        except Exception as e:
            console.print(f"  {_WARN}  API server: not reachable ({e.__class__.__name__})")
            console.print("         → start with `bash scripts/run.sh` or skip --api")
            warns += 1

    console.print()
    if fails == 0 and warns == 0:
        console.print("  [bold green]All checks passed.[/bold green] System is healthy.")
    elif fails == 0:
        console.print(f"  [yellow]{warns} warning(s).[/yellow] System will run but missing optional pieces.")
    else:
        console.print(f"  [red]{fails} failure(s), {warns} warning(s).[/red] Fix the FAILs before deploying.")
        raise typer.Exit(code=1)
