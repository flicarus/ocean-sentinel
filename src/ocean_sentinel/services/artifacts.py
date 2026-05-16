"""Render the visual artefacts the dashboard needs for one detection.

For every event written by `os monitor`, the dashboard wants:
  - a log-mel spectrogram PNG (what the analyst sees)
  - the same spectrogram with a Grad-CAM overlay (what the CNN attended to)

Both are produced from the *same* preprocessed input the CNN actually
saw — high-pass + center-crop to 157 frames — so the overlay is faithful
to the model's reasoning, not a marketing render.

This module is intentionally side-effect free: it returns PNG bytes. The
caller decides where to put them (Supabase Storage, /public, /tmp, etc).

Heavy imports (matplotlib, torch) are deferred until first call so the
CLI startup path stays cold-load fast — `os --help` should not import
torch.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

# Match training preprocessing exactly. Drift here silently misaligns the
# heatmap with the spectrogram pixels.
_TARGET_SR_HZ = 16_000
_N_MELS = 128
_FMAX_HZ = 1_000
_CROP_FRAMES = 157
_DURATION_S = 60.0


def _load_audio(clip: Path) -> tuple[np.ndarray, int]:
    import librosa
    return librosa.load(
        str(clip), sr=_TARGET_SR_HZ, mono=True, duration=_DURATION_S,
    )


def _mel_log(y: np.ndarray, sr: int) -> np.ndarray:
    """Return log-mel power in dB, shape (128, T). Matches CNNV7 training."""
    import librosa
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=_N_MELS, fmax=_FMAX_HZ)
    return librosa.power_to_db(mel, ref=1.0)


def _center_crop(spec: np.ndarray, target_frames: int = _CROP_FRAMES) -> np.ndarray:
    """Crop or edge-pad to exactly target_frames columns — mirrors the
    classifier's preprocess() so the heatmap aligns with what the model saw."""
    if spec.shape[1] > target_frames:
        s = (spec.shape[1] - target_frames) // 2
        return spec[:, s : s + target_frames]
    if spec.shape[1] < target_frames:
        pad = target_frames - spec.shape[1]
        return np.pad(spec, ((0, 0), (0, pad)), mode="edge")
    return spec


def _preprocess_for_model(spec: np.ndarray) -> np.ndarray:
    """Replicate CNNV7Classifier.predict preprocessing for Grad-CAM input.

    The classifier high-pass-masks 0-80 Hz and z-scores. The masking is
    why Grad-CAM never lights up below 80 Hz — the model literally
    cannot see it.
    """
    from .cnn_classifier import _LOW_FREQ_MASK
    spec = _center_crop(spec).copy()
    high_mean = float(spec[~_LOW_FREQ_MASK].mean())
    spec[_LOW_FREQ_MASK, :] = high_mean
    spec = (spec - spec.mean()) / (spec.std() + 1e-8)
    return spec


def render_spectrogram_png(clip: Path, *, title: str | None = None) -> bytes:
    """Return PNG bytes of the log-mel spectrogram for `clip`.

    This is the same view the CNN sees (modulo the high-pass mask), so
    analyst inspection and model inspection share a coordinate system.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display

    y, sr = _load_audio(clip)
    log_mel = _mel_log(y, sr)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    img = librosa.display.specshow(
        log_mel, sr=sr, fmax=_FMAX_HZ, x_axis="time", y_axis="mel", ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title or f"{clip.name} — log-mel (fmax {_FMAX_HZ} Hz)",
                 fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_saliency_png(
    clip: Path,
    *,
    checkpoint: str | Path,
    target_class: int = 1,
    title: str | None = None,
) -> bytes:
    """Return PNG bytes of a Grad-CAM heatmap overlaid on the spectrogram.

    `target_class=1` attributes to the 'ship' logit, which is what we want
    for a positive detection (shows what convinced the model it was a
    ship). For an AMBIENT prediction the caller can pass target_class=0
    to attribute to the 'not_ship' logit instead.

    The CAM is taken from the last backbone block (block4) — the largest
    receptive field still spatially resolved — and upsampled bilinearly
    to the (128, 157) input grid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from ocean_sentinel.models.cnn_v7 import OceanSentinelV7

    y, sr = _load_audio(clip)
    log_mel = _mel_log(y, sr)
    spec_disp = _center_crop(log_mel)             # for display
    spec_norm = _preprocess_for_model(log_mel)    # for the model

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = OceanSentinelV7().to(device)
    state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Hook block4 — last backbone block, B x 256 x 8 x (T/16).
    acts: dict[str, Any] = {"a": None, "g": None}

    def fwd(_m, _i, o):
        acts["a"] = o.detach()

    def bwd(_m, _gi, go):
        acts["g"] = go[0].detach()

    h1 = model.backbone.block4.register_forward_hook(fwd)
    h2 = model.backbone.block4.register_full_backward_hook(bwd)

    try:
        x = (
            torch.from_numpy(spec_norm)
            .unsqueeze(0).unsqueeze(0).float()
            .to(device).requires_grad_(True)
        )
        out = model(x)
        # Evidential head — backprop the chosen class's evidence logit.
        evidence = out["evidence"]
        model.zero_grad()
        evidence[0, target_class].backward()

        alphas = acts["g"].mean(dim=(2, 3), keepdim=True)        # (1, 256, 1, 1)
        cam = F.relu((alphas * acts["a"]).sum(dim=1))            # (1, h, w)
        cam = F.interpolate(
            cam.unsqueeze(1), size=x.shape[2:],
            mode="bilinear", align_corners=False,
        ).squeeze(1)[0].cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Posterior for the title.
        with torch.no_grad():
            alpha = F.softplus(out["evidence"]) + 1.0
            S = alpha.sum()
            ship_prob = float((alpha[0, 1] / S).item())
    finally:
        h1.remove()
        h2.remove()

    # Render: spectrogram in grayscale underneath, jet CAM overlaid.
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    vmin = float(np.percentile(spec_disp, 5))
    vmax = float(np.percentile(spec_disp, 99))
    ax.imshow(spec_disp, aspect="auto", origin="lower", cmap="gray",
              alpha=0.55, vmin=vmin, vmax=vmax)
    ax.imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.55)
    ax.set_xlabel("time frames")
    ax.set_ylabel("mel bin (low → high frequency)")
    ax.set_title(
        title or
        f"Grad-CAM (class={'ship' if target_class == 1 else 'not_ship'})"
        f"  ·  ship_prob={ship_prob:.2f}",
        fontsize=10,
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
