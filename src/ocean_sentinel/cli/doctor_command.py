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
from pathlib import Path

import typer
from rich.box import HEAVY, ROUNDED
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text


# Column geometry — keep labels aligned across every check line.
_LABEL_WIDTH = 24


def _header(console) -> None:
    from .ui import GHOST

    line = Text()
    line.append("Ocean Sentinel", style="bold default")
    line.append("  ·  ", style=GHOST)
    line.append("health check", style="grey70")

    console.print()
    console.print(Padding(line, (0, 2)))
    console.print(Padding(Text("─" * 72, style=GHOST), (0, 2)))
    console.print()


def _check(console, status: str, label: str, detail: str, hint: str = "") -> None:
    """Render one check row in the shared palette.

    status: "ok" | "warn" | "fail"
    """
    from .ui import AMBER, GREEN, RED, DIM, SOFT

    icon, style = {
        "ok":   ("✓", GREEN),
        "warn": ("⚠", AMBER),
        "fail": ("✗", RED),
    }[status]

    row = Text()
    row.append(f"    {icon}  ", style=style)
    row.append(label.ljust(_LABEL_WIDTH), style="default")
    row.append(detail, style=SOFT)
    console.print(row)

    if hint:
        h = Text()
        h.append("         → ", style=DIM)
        h.append(hint, style=DIM)
        console.print(h)


def _summary(console, fails: int, warns: int) -> None:
    from .ui import AMBER, GHOST, GREEN, RED, TEAL, DIM

    if fails == 0 and warns == 0:
        headline = Text()
        headline.append("✓  ", style=f"bold {GREEN}")
        headline.append("All checks passed.  ", style="default")
        headline.append("System is healthy.", style=DIM)
        border, box = TEAL, HEAVY
    elif fails == 0:
        headline = Text()
        headline.append("⚠  ", style=f"bold {AMBER}")
        headline.append(f"{warns} warning(s).  ", style="default")
        headline.append("System will run but missing optional pieces.", style=DIM)
        border, box = AMBER, ROUNDED
    else:
        headline = Text()
        headline.append("✗  ", style=f"bold {RED}")
        headline.append(f"{fails} failure(s), {warns} warning(s).  ", style="default")
        headline.append("Fix the FAILs before deploying.", style=DIM)
        border, box = RED, HEAVY

    panel = Panel(
        headline,
        box=box,
        border_style=border,
        padding=(1, 3),
    )
    console.print()
    console.print(Padding(panel, (0, 2)))


def doctor_command(
    api: bool = typer.Option(
        False, "--api",
        help="Also probe the local API server (http://localhost:8000/health)."
    ),
) -> None:
    """Run a self-check on the install. Exits non-zero if any FAIL."""
    from .ui import console

    _header(console)

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
        _check(console, "fail", "Model checkpoint", f"missing {ckpt}",
               hint="train v7.6 or set OS_CHECKPOINT to an existing .pt file")
        fails += 1
    else:
        size_mb = ckpt.stat().st_size / 1024 / 1024
        if size_mb < 1 or size_mb > 100:
            _check(console, "warn", "Model checkpoint",
                   f"{ckpt.name} · {size_mb:.1f} MB (unusual size)")
            warns += 1
        else:
            _check(console, "ok", "Model checkpoint",
                   f"{ckpt.name} · {size_mb:.1f} MB")

    # ── 2. Per-site thresholds ────────────────────────────────────────
    thr_path = Path(_DEFAULT_SITE_THRESHOLDS)
    if not thr_path.exists():
        _check(console, "warn", "Per-site thresholds",
               "missing (will use 0.5 everywhere)",
               hint="run scripts/calibrate_per_site_honest.py to generate")
        warns += 1
    else:
        try:
            doc = json.loads(thr_path.read_text())
            n = len(doc.get("per_site_thresholds", {}))
            if n == 0:
                _check(console, "warn", "Per-site thresholds",
                       "file exists but empty")
                warns += 1
            else:
                _check(console, "ok", "Per-site thresholds",
                       f"{n} sites loaded from {thr_path.name}")
        except Exception as e:
            _check(console, "fail", "Per-site thresholds", f"parse error ({e})")
            fails += 1

    # ── 3. Conformal calibration ──────────────────────────────────────
    conf = Path(_DEFAULT_CONFORMAL)
    if not conf.exists():
        _check(console, "warn", "Conformal calibration", f"missing {conf}",
               hint="run scripts/calibrate_conformal.py")
        warns += 1
    else:
        try:
            d = json.loads(conf.read_text())
            t = d.get("threshold")
            _check(console, "ok", "Conformal calibration",
                   f"threshold {t:.3f}")
        except Exception as e:
            _check(console, "fail", "Conformal calibration",
                   f"parse error ({e})")
            fails += 1

    # ── 4. PyTorch + device ───────────────────────────────────────────
    try:
        import torch
        if torch.backends.mps.is_available():
            _check(console, "ok", "PyTorch",
                   f"{torch.__version__} on mps · Apple Silicon")
        elif torch.cuda.is_available():
            _check(console, "ok", "PyTorch",
                   f"{torch.__version__} on cuda · {torch.cuda.get_device_name(0)}")
        else:
            _check(console, "warn", "PyTorch",
                   f"{torch.__version__} on cpu (~50ms/clip, still real-time)")
            warns += 1
    except Exception as e:
        _check(console, "fail", "PyTorch", f"import failed ({e})")
        fails += 1

    # ── 5. librosa ─────────────────────────────────────────────────────
    try:
        import librosa
        import numpy as np
        y = np.random.randn(16000).astype(np.float32) * 0.01
        mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128, fmax=1000)
        if mel.shape[0] == 128:
            _check(console, "ok", "librosa",
                   f"{librosa.__version__} forward pass OK")
        else:
            _check(console, "fail", "librosa", f"unexpected mel shape {mel.shape}")
            fails += 1
    except Exception as e:
        _check(console, "fail", "librosa", f"failed ({e})")
        fails += 1

    # ── 6. End-to-end inference ────────────────────────────────────────
    try:
        import numpy as np
        from ..gemma.cnn_inference import _load_classifier
        clf = _load_classifier(_DEFAULT_CHECKPOINT)
        spec = np.random.randn(128, 313).astype(np.float32) * 5 - 20
        v = clf.predict(spec)
        prob = float(v["probabilities"]["ship"])
        _check(console, "ok", "End-to-end inference",
               f"ship_prob={prob:.3f} on synthetic spec")
    except Exception as e:
        _check(console, "fail", "End-to-end inference",
               f"failed ({e.__class__.__name__}: {e})")
        fails += 1

    # ── 7. Eval scores ────────────────────────────────────────────────
    eval_files = [
        ("data/eval/per_site_v7_6.json", "v7.6 vanilla"),
        ("data/calibration/per_site_thresholds_v7_6_honest.json", "v7.6 + calibration (honest)"),
    ]
    found = sum(1 for path_str, _ in eval_files if Path(path_str).exists())
    if found == len(eval_files):
        _check(console, "ok", "Eval results",
               f"{found}/{len(eval_files)} files present")
    elif found > 0:
        _check(console, "warn", "Eval results",
               f"{found}/{len(eval_files)} files present")
        warns += 1
    else:
        _check(console, "warn", "Eval results",
               "no eval files found in data/eval/")
        warns += 1

    # ── 8. Optional API check ─────────────────────────────────────────
    if api:
        try:
            import httpx
            r = httpx.get("http://localhost:8000/health", timeout=2.0)
            if r.status_code == 200:
                _check(console, "ok", "API server",
                       "http://localhost:8000 responsive")
            else:
                _check(console, "warn", "API server", f"HTTP {r.status_code}")
                warns += 1
        except Exception as e:
            _check(console, "warn", "API server",
                   f"not reachable ({e.__class__.__name__})",
                   hint="start with `bash scripts/run.sh` or skip --api")
            warns += 1

    _summary(console, fails, warns)

    if fails > 0:
        raise typer.Exit(code=1)
