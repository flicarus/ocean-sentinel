"""Saliency diagnostic — investigate WHAT the CNN is actually keying on,
and why Grad-CAM hot zones don't fall in the 28-37 Hz blade-rate band.

Two experiments:

  Experiment 1 — Negative control comparison:
    Render Grad-CAM for a known-true-negative (MBARI deep canyon ambient,
    no surface vessels), a SanctSound true-ambient, the OC01 disputed
    chunk, and a ShipsEar clear positive. If the model genuinely
    discriminates, Grad-CAM patterns should differ between ships and
    ambients — not just confidence scores.

  Experiment 2 — Attention vs preprocessing:
    Visualise raw spectrogram, the high-pass-filtered input the model
    actually sees, and the Grad-CAM, side by side. Compute mean attention
    per frequency band — quantifies "model puts X% of its attention in
    the blade-rate band, Y% in engine band, Z% in cavitation/surface
    band". Replaces visual impressions with numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import structlog
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")

from ocean_sentinel.models.cnn import OceanSentinelCNN
from ocean_sentinel.services.cnn_classifier import (
    _CROP_FRAMES, _LOW_FREQ_MASK, _MEL_FREQS, _MEL_N, LABELS,
    _resolve_training_source_id, _select_device,
)

log = structlog.get_logger()

CKPT = Path("data/models/cnn_v6.pt")
OUT = Path("data/diagnostic/saliency")

# Experiment 1: 4 samples, 2 known negatives, 1 disputed, 1 clear positive
EXP1 = [
    ("data/spectrograms/mbari_20231017_14592_0.npy",
     "mbari", "TRUE NEGATIVE — MBARI deep canyon, 891m depth, no surface vessels"),
    ("data/spectrograms/sanctsound_oc01_SanctSound_OC01_01_671399974_20190308T190000Z_1.npy",
     "sanctsound", "OC01 chunk index 1 (mostly ambient assumed)"),
    ("data/spectrograms/sanctsound_oc01_SanctSound_OC01_01_671399974_20190308T190000Z_0.npy",
     "sanctsound", "OC01 disputed chunk (CNN said ship 91%)"),
    ("data/spectrograms/shipsear_0_0_0_0_0_1.npy",
     "shipsear", "TRUE POSITIVE — ShipsEar known vessel"),
]

# Frequency bands of interest, by domain knowledge
BANDS = [
    ("blade_rate", 5, 50, "Cargo blade-rate (high-pass kills 0-80 Hz)"),
    ("engine_lo",  80, 200, "Engine harmonic low"),
    ("engine_mid", 200, 500, "Engine harmonic mid"),
    ("cavitation", 500, 1000, "Cavitation / surface noise"),
]


# ── shared helpers ───────────────────────────────────────────────────────

def preprocess(spec_raw: np.ndarray, profiles: dict, source_id: str | None) -> tuple[np.ndarray, str | None]:
    spec = np.asarray(spec_raw, dtype=np.float32)
    if spec.shape[1] > _CROP_FRAMES:
        start = (spec.shape[1] - _CROP_FRAMES) // 2
        spec = spec[:, start : start + _CROP_FRAMES]
    spec = spec.copy()
    high_mean = float(spec[~_LOW_FREQ_MASK].mean())
    spec[_LOW_FREQ_MASK, :] = high_mean
    resolved = _resolve_training_source_id(source_id, set(profiles))
    if resolved is not None:
        spec = spec - profiles[resolved][:, None]
    return spec, resolved


class GradCAM:
    def __init__(self, model: OceanSentinelCNN, target: torch.nn.Module) -> None:
        self.model = model
        self._a: torch.Tensor | None = None
        self._g: torch.Tensor | None = None
        target.register_forward_hook(lambda *a: setattr(self, "_a", a[2].detach()))
        target.register_full_backward_hook(lambda *a: setattr(self, "_g", a[2][0].detach()))

    def __call__(self, x: torch.Tensor, cls: int) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(x)
        out["vessel"][0, cls].backward()
        alphas = self._g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alphas * self._a).sum(dim=1))
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[2:],
                            mode="bilinear", align_corners=False).squeeze(1)
        cam = cam[0].cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) if cam.max() > cam.min() else cam


def run_one(model, cam_obj, profiles, temperature, spec_path, source_id):
    """Returns dict with raw spec, processed spec, grad-cam, prediction."""
    spec_raw = np.load(spec_path).astype(np.float32)
    if spec_raw.shape[1] > _CROP_FRAMES:
        s = (spec_raw.shape[1] - _CROP_FRAMES) // 2
        spec_show = spec_raw[:, s : s + _CROP_FRAMES]
    else:
        spec_show = spec_raw
    spec_proc, profile_applied = preprocess(spec_raw, profiles, source_id)
    tensor = torch.from_numpy(spec_proc).unsqueeze(0).unsqueeze(0).float()
    tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
    tensor = tensor.to(next(model.parameters()).device)
    tensor.requires_grad_(True)
    out = model(tensor)
    logits = out["vessel"] / temperature
    probs = torch.softmax(logits, dim=1)[0]
    pred = int(probs.argmax().item())
    cam = cam_obj(tensor, pred)
    return {
        "spec_raw": spec_show,
        "spec_proc": spec_proc,
        "cam": cam,
        "pred": LABELS[pred],
        "conf": float(probs[pred].item()),
        "profile": profile_applied,
    }


# ── EXPERIMENT 1 ─────────────────────────────────────────────────────────

def experiment_1(model, cam_obj, profiles, temperature) -> None:
    """Side-by-side Grad-CAM for 4 samples."""
    print("\n=== Experiment 1: Negative control comparison ===")
    n = len(EXP1)
    fig, axes = plt.subplots(n, 2, figsize=(15, 3.0 * n))
    extent = (0, _CROP_FRAMES, _MEL_FREQS[0], _MEL_FREQS[-1])

    for i, (path, sid, narrative) in enumerate(EXP1):
        if not Path(path).exists():
            print(f"  SKIP missing: {path}")
            continue
        r = run_one(model, cam_obj, profiles, temperature, path, sid)
        # left: raw spec
        axes[i, 0].imshow(r["spec_raw"], aspect="auto", origin="lower",
                          cmap="magma", extent=extent)
        axes[i, 0].set_ylabel("Hz")
        axes[i, 0].set_title(f"Spectrogram — {narrative}", fontsize=9)
        for hz in (28, 37, 80):  # 28-37 = blade-rate; 80 = high-pass cutoff
            color = "cyan" if hz < 50 else "red"
            axes[i, 0].axhline(hz, color=color, linewidth=0.8, alpha=0.6, linestyle="--")

        # right: grad-cam overlay
        axes[i, 1].imshow(r["spec_raw"], aspect="auto", origin="lower",
                          cmap="magma", extent=extent, alpha=0.5)
        axes[i, 1].imshow(r["cam"], aspect="auto", origin="lower",
                          cmap="jet", extent=extent, alpha=0.5)
        axes[i, 1].set_title(
            f"Grad-CAM   →   PRED: {r['pred']}  (p={r['conf']:.2f}, profile={r['profile']})",
            fontsize=9,
        )
        for hz in (28, 37, 80):
            color = "cyan" if hz < 50 else "red"
            axes[i, 1].axhline(hz, color=color, linewidth=0.8, alpha=0.6, linestyle="--")

        print(f"  {Path(path).name[:60]}: {r['pred']} (p={r['conf']:.2f})")

    fig.suptitle(
        "Experiment 1 — Grad-CAM negative control. "
        "Cyan = blade-rate band 28-37 Hz. Red = high-pass cutoff 80 Hz "
        "(everything below this is flattened before model sees it).",
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = OUT / "exp1_negative_control.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── EXPERIMENT 2 ─────────────────────────────────────────────────────────

def attention_per_band(cam: np.ndarray) -> dict[str, float]:
    """Compute mean attention in each frequency band, normalized to %."""
    # cam shape: (128 mel bins, 157 time frames)
    band_attention = {}
    total = cam.sum()
    if total <= 0:
        return {b[0]: 0.0 for b in BANDS}
    for name, lo, hi, _ in BANDS:
        mask = (_MEL_FREQS >= lo) & (_MEL_FREQS < hi)
        band_sum = cam[mask, :].sum() if mask.any() else 0.0
        band_attention[name] = float(band_sum / total * 100.0)
    return band_attention


def experiment_2(model, cam_obj, profiles, temperature) -> None:
    """For the disputed OC01 chunk: visualise raw vs filtered input + per-band attention."""
    print("\n=== Experiment 2: Attention vs preprocessing on OC01 disputed chunk ===")
    path = EXP1[2][0]      # OC01 disputed (the one CNN said ship)
    sid = EXP1[2][1]
    if not Path(path).exists():
        print(f"  missing: {path}")
        return

    r = run_one(model, cam_obj, profiles, temperature, path, sid)
    band_pct = attention_per_band(r["cam"])

    print("  Per-band attention distribution:")
    print(f"     Total attention → 100%  (uniform reference would be 25% per band of 4)")
    for name, lo, hi, label in BANDS:
        bar = "█" * int(band_pct[name])
        print(f"     {name:11s} {lo:>4}-{hi:>4} Hz   {band_pct[name]:5.1f}%   {bar}")

    # Plot: raw spec | model-input spec (low freqs flattened) | grad-cam | per-band bar chart
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2, 1])

    extent = (0, _CROP_FRAMES, _MEL_FREQS[0], _MEL_FREQS[-1])

    ax_raw = fig.add_subplot(gs[0, 0])
    ax_raw.imshow(r["spec_raw"], aspect="auto", origin="lower", cmap="magma", extent=extent)
    ax_raw.set_title("Raw spectrogram (full freq range — what PSD analysis sees)")
    ax_raw.set_ylabel("Hz")
    for hz in (28, 37, 80):
        ax_raw.axhline(hz, color=("cyan" if hz < 50 else "red"), linewidth=0.8, alpha=0.6, linestyle="--")

    ax_proc = fig.add_subplot(gs[1, 0])
    ax_proc.imshow(r["spec_proc"], aspect="auto", origin="lower", cmap="viridis", extent=extent)
    ax_proc.set_title("Model input (high-pass < 80 Hz flattened — what CNN actually sees)")
    ax_proc.set_ylabel("Hz")
    for hz in (28, 37, 80):
        ax_proc.axhline(hz, color=("cyan" if hz < 50 else "red"), linewidth=0.8, alpha=0.6, linestyle="--")

    ax_cam = fig.add_subplot(gs[2, 0])
    ax_cam.imshow(r["spec_raw"], aspect="auto", origin="lower", cmap="magma", extent=extent, alpha=0.5)
    ax_cam.imshow(r["cam"], aspect="auto", origin="lower", cmap="jet", extent=extent, alpha=0.5)
    ax_cam.set_title(f"Grad-CAM  →  PRED: {r['pred']} (p={r['conf']:.2f})")
    ax_cam.set_ylabel("Hz")
    ax_cam.set_xlabel("Time (frames)")
    for hz in (28, 37, 80):
        ax_cam.axhline(hz, color=("cyan" if hz < 50 else "red"), linewidth=0.8, alpha=0.6, linestyle="--")

    # Per-band bar
    ax_bar = fig.add_subplot(gs[:, 1])
    names = [b[0] for b in BANDS]
    pcts = [band_pct[n] for n in names]
    labels = [f"{b[0]}\n{b[1]}-{b[2]} Hz" for b in BANDS]
    colors = ["#888888" if BANDS[i][2] <= 80 else "tab:blue" for i in range(len(BANDS))]
    bars = ax_bar.barh(range(len(BANDS)), pcts, color=colors)
    ax_bar.set_yticks(range(len(BANDS)))
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("% of total attention")
    ax_bar.set_title("Where attention falls (by band)")
    for i, p in enumerate(pcts):
        ax_bar.text(p + 0.5, i, f"{p:.1f}%", va="center", fontsize=9)

    # Annotate the gray band for blade-rate to make the high-pass story explicit
    ax_bar.text(
        0.98, 0.02,
        "Gray bar = blade-rate band\n(high-pass kills it before model)",
        transform=ax_bar.transAxes,
        ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    fig.suptitle(
        "Experiment 2 — OC01 disputed chunk: how the high-pass filter shapes what CNN sees",
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = OUT / "exp2_attention_bands.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = _select_device(None)

    model = OceanSentinelCNN().to(device)
    state = torch.load(str(CKPT), map_location=device)
    model.load_state_dict(state)
    model.eval()

    profiles_path = CKPT.with_suffix(".profiles.npz")
    profiles: dict[str, np.ndarray] = {}
    if profiles_path.exists():
        with np.load(profiles_path) as data:
            profiles = {k: data[k].astype(np.float32) for k in data.files}

    import json
    temp_path = CKPT.with_suffix(".temperature.json")
    temperature = float(json.loads(temp_path.read_text())["temperature"]) if temp_path.exists() else 1.0

    cam_obj = GradCAM(model, model.backbone.block3.conv)

    experiment_1(model, cam_obj, profiles, temperature)
    experiment_2(model, cam_obj, profiles, temperature)

    print("\nDone. Open both PNGs in Preview to interpret.")


if __name__ == "__main__":
    main()
