"""`refresh_site` — re-fit per-site adapter + recalibrate conformal on
accumulated ambient.

The motivation is that onboarding produces a *snapshot* (adapter +
threshold from the first 3 minutes the user ever recorded). In
production, hydrophones run continuously — each day brings hours of new
ambient that should improve the model. Without refresh, the system
freezes at the Day-1 calibration forever.

What this does
--------------
1. Combines the original onboarding ambient with any new ambient clips
   the user supplies (or auto-discovers in `data/sites/{id}/ambient/`).
2. Filters the combined audio through the *current* adapter+threshold
   to drop windows that look like vessel events. We can't recalibrate
   on a corpus that contains real ships — that would teach the model
   to call ships ambient. The trust filter keeps only windows the
   current system would have classified as not_ship.
3. Re-fits the per-site adapter on the trusted ambient.
4. Recalibrates the conformal threshold on the (post-adapter) trusted
   distribution.
5. Returns before/after numbers — threshold change, FA rate change,
   median ship_prob change — so the user sees the improvement.

What this does NOT do
---------------------
- Update CNN backbone weights (still frozen).
- Use confirmed/rejected events from the dashboard (separate feature).
- Phone home / send anything off-machine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import structlog

from .site_adapter import train_site_adapter
from .conformal import calibrate_conformal_real

log = structlog.get_logger()


_TARGET_SR_HZ = 16_000
_WINDOW_S = 60.0
_HOP_S = 30.0


def _gather_ambient_paths(site_id: str, additional: str | None) -> list[Path]:
    """Find all ambient audio sources for the site.

    Order: original from site config (if present in data/sites/{id}.yaml),
    then everything in data/sites/{id}/ambient/*.wav, then the explicit
    `additional` path. De-duplicated, preserving order.
    """
    paths: list[Path] = []

    site_yaml = Path("data/sites") / f"{site_id}.yaml"
    if site_yaml.exists():
        try:
            import yaml
            cfg = yaml.safe_load(site_yaml.read_text()) or {}
            orig = cfg.get("ambient_source")
            if orig and Path(orig).exists():
                paths.append(Path(orig))
        except Exception:
            pass

    extra_dir = Path("data/sites") / site_id / "ambient"
    if extra_dir.exists():
        for p in sorted(extra_dir.glob("*.wav")):
            paths.append(p)

    if additional:
        ap = Path(additional)
        if ap.exists():
            paths.append(ap)

    # De-duplicate while preserving order
    seen = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def _concat_ambient_to_tmp(
    paths: list[Path], site_id: str, max_total_s: float = 1800.0,
) -> tuple[Path, float]:
    """Concatenate ambient files into one WAV (capped at max_total_s).
    Returns (tmp_path, total_seconds_loaded)."""
    site_dir = Path("data/sites") / site_id
    site_dir.mkdir(parents=True, exist_ok=True)
    tmp = site_dir / "_combined_ambient.wav"

    chunks = []
    total_s = 0.0
    for p in paths:
        if total_s >= max_total_s:
            break
        remaining = max_total_s - total_s
        try:
            y, sr = librosa.load(str(p), sr=_TARGET_SR_HZ, mono=True, duration=remaining)
        except Exception:
            continue
        if y.size == 0:
            continue
        chunks.append(y)
        total_s += y.size / sr

    if not chunks:
        return tmp, 0.0
    combined = np.concatenate(chunks).astype(np.float32)
    sf.write(str(tmp), combined, _TARGET_SR_HZ)
    return tmp, total_s


def _filter_trusted_windows(
    combined_path: Path,
    site_id: str,
    trust_max_ship_prob: float,
) -> tuple[Path, dict[str, Any]]:
    """Run the current per-site model over the combined ambient and keep
    only windows with ship_prob < trust_max_ship_prob.

    Returns (path_to_trusted_wav, stats). If too few windows pass the
    filter we fall back to "use the bottom 80% by ship_prob" so the
    refresh can still run on a confused site (where everything looks
    like ship to the model).
    """
    from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier

    y, sr = librosa.load(str(combined_path), sr=_TARGET_SR_HZ, mono=True)
    win = int(sr * _WINDOW_S)
    hop = int(sr * _HOP_S)
    if y.size < win:
        return combined_path, {
            "n_total_windows": 0,
            "n_trusted_windows": 0,
            "filter_fallback": False,
        }

    starts = list(range(0, y.size - win + 1, hop))
    clf = CNNV7Classifier("data/models/cnn_v7_4.pt")
    adapter_path = Path("data/sites") / site_id / "adapter.pt"
    if adapter_path.exists():
        clf.set_site_adapter(adapter_path)
    else:
        clf.set_site_adapter(None)

    ship_probs: list[float] = []
    for s in starts:
        chunk = y[s : s + win]
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=1000)
        spec = librosa.power_to_db(mel, ref=1.0)
        pred = clf.predict(spec, source_id=site_id)
        p = (pred["confidence"] if pred["label"] == "ship"
             else 1.0 - pred["confidence"])
        ship_probs.append(p)

    keep_mask = [p < trust_max_ship_prob for p in ship_probs]
    n_total = len(starts)
    n_trusted = sum(keep_mask)
    fallback = False

    # Fallback: if hard filter rejects too much, keep bottom 80% by ship_prob.
    # This handles the case where the model is currently confused on this
    # site — we still need SOMETHING to recalibrate against.
    if n_trusted < max(5, int(0.2 * n_total)):
        fallback = True
        sorted_idx = np.argsort(ship_probs)
        cutoff = max(5, int(0.8 * n_total))
        keep_idx = set(int(i) for i in sorted_idx[:cutoff].tolist())
        keep_mask = [i in keep_idx for i in range(n_total)]
        n_trusted = sum(keep_mask)

    # Stitch only the trusted windows back into a contiguous WAV
    trusted_chunks = [y[s : s + win] for s, keep in zip(starts, keep_mask) if keep]
    if not trusted_chunks:
        return combined_path, {
            "n_total_windows": n_total,
            "n_trusted_windows": 0,
            "filter_fallback": fallback,
            "median_ship_prob_in_corpus": float(np.median(ship_probs)) if ship_probs else 0.0,
        }
    trusted_y = np.concatenate(trusted_chunks).astype(np.float32)
    trusted_path = combined_path.with_name("_trusted_ambient.wav")
    sf.write(str(trusted_path), trusted_y, sr)

    return trusted_path, {
        "n_total_windows": n_total,
        "n_trusted_windows": n_trusted,
        "filter_fallback": fallback,
        "median_ship_prob_in_corpus": round(float(np.median(ship_probs)), 4),
    }


def _read_current_state(site_id: str) -> dict[str, Any]:
    """Read the *current* adapter + conformal so we can report before/after."""
    import json
    state: dict[str, Any] = {}
    conf_path = Path("data/sites") / site_id / "conformal.json"
    if conf_path.exists():
        try:
            d = json.loads(conf_path.read_text())
            state["threshold"] = d.get("threshold")
            state["alpha"] = d.get("alpha")
            state["n_calibration"] = d.get("n_calibration")
            state["median_ambient_ship_prob"] = d.get("median_ambient_ship_prob")
        except Exception:
            pass
    adapter_path = Path("data/sites") / site_id / "adapter.pt"
    state["has_adapter"] = adapter_path.exists()
    return state


def refresh_site(
    site_id: str,
    *,
    additional_ambient_path: str | None = None,
    trust_max_ship_prob: float = 0.7,
    alpha: float = 0.05,
    max_total_s: float = 1800.0,
) -> dict[str, Any]:
    """Re-fit per-site adapter and recalibrate conformal on accumulated
    ambient.

    Parameters
    ----------
    site_id : str
        The site to refresh. Must already be onboarded
        (data/sites/{site_id}.yaml present).
    additional_ambient_path : str | None
        Path to a new ambient .wav to fold into the calibration corpus.
        If None, only the originally-onboarded ambient + anything in
        data/sites/{id}/ambient/ is used.
    trust_max_ship_prob : float
        Windows with ship_prob ≥ this value are dropped before retraining.
        Prevents real vessel events from contaminating the ambient
        distribution. Default 0.7 — half-way between the threshold and
        the high-confidence vessel zone.
    alpha : float
        Target false-alarm rate for the recalibrated conformal threshold.
    max_total_s : float
        Cap on total audio loaded (memory bound). Default 30 minutes.
    """
    # ── 1. find ambient ────────────────────────────────────────────────
    paths = _gather_ambient_paths(site_id, additional_ambient_path)
    if not paths:
        return {
            "ok": False,
            "error": (
                f"no ambient audio found for site {site_id}. "
                f"pass --add <path> or drop .wav files into "
                f"data/sites/{site_id}/ambient/"
            ),
        }

    state_before = _read_current_state(site_id)
    if not state_before.get("has_adapter"):
        log.info("refresh_no_existing_adapter", site_id=site_id)

    # ── 2. concat + filter ─────────────────────────────────────────────
    combined_path, total_s = _concat_ambient_to_tmp(
        paths, site_id, max_total_s=max_total_s,
    )
    if total_s < _WINDOW_S * 3:
        return {
            "ok": False,
            "error": (
                f"combined ambient is only {total_s:.0f}s — "
                f"need ≥{_WINDOW_S * 3:.0f}s to recalibrate."
            ),
        }

    trusted_path, filter_stats = _filter_trusted_windows(
        combined_path, site_id, trust_max_ship_prob=trust_max_ship_prob,
    )
    if filter_stats["n_trusted_windows"] < 5:
        return {
            "ok": False,
            "error": (
                f"only {filter_stats['n_trusted_windows']} trusted "
                f"windows after filtering — refresh aborted."
            ),
            "filter_stats": filter_stats,
        }

    # ── 3. re-fit adapter (warm-started from existing) ────────────────
    train_result = train_site_adapter(
        site_id=site_id, ambient_source=str(trusted_path),
        warm_start=True,
    )
    if not train_result.get("ok"):
        return {"ok": False, "error": f"adapter retrain failed: {train_result.get('error')}"}

    # ── 4. recalibrate conformal ───────────────────────────────────────
    cal = calibrate_conformal_real(
        site_id=site_id,
        ambient_source=str(trusted_path),
        alpha=alpha,
    )
    if not cal.get("ok"):
        return {"ok": False, "error": f"recalibration failed: {cal.get('error')}"}

    # ── 5. report ──────────────────────────────────────────────────────
    return {
        "ok": True,
        "site_id": site_id,
        "n_ambient_files": len(paths),
        "ambient_total_seconds": round(total_s, 1),
        "filter": filter_stats,
        "before": {
            "threshold":            state_before.get("threshold"),
            "n_calibration":        state_before.get("n_calibration"),
            "median_ambient_ship_prob": state_before.get("median_ambient_ship_prob"),
            "had_adapter":          state_before.get("has_adapter", False),
        },
        "after": {
            "threshold":            cal.get("threshold_p"),
            "n_calibration":        cal.get("n_calibration"),
            "median_ambient_ship_prob": cal.get("median_ambient_ship_prob"),
            "adapter_epochs":       train_result["epochs_trained"],
            "adapter_early_stopped": train_result["early_stopped"],
            "adapter_median_amb_p_before": train_result["median_ship_prob_before"],
            "adapter_median_amb_p_after":  train_result["median_ship_prob_after"],
            "holdout_recall":       train_result["holdout_recall_after"],
        },
        "summary": (
            f"refresh complete · {filter_stats['n_trusted_windows']}/"
            f"{filter_stats['n_total_windows']} windows trusted · "
            f"threshold {state_before.get('threshold', '—')} → "
            f"{cal['threshold_p']:.2f}"
        ),
    }
