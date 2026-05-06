"""Grad-CAM saliency visualisation for the binary ship CNN.

Hooks the last conv layer (backbone.block3.conv) and computes class-activation
maps over the input mel spectrogram, showing *which time/frequency regions*
drove the model's verdict.  For ship-positive predictions we expect the
heatmap to concentrate around vessel-acoustic structure: low-frequency
narrow-band tones (engine harmonics, propeller blade-rate ~20-50 Hz),
broadband cavitation, and rumble bands — not the broadband hiss that fills
ambient ocean noise.

The preprocessing pipeline (high-pass, profile subtraction, z-score, centre
crop) mirrors `CNNClassifier.predict` exactly so the Grad-CAM reflects the
same input the production model sees.

Usage:
    PYTHONPATH=src venv/bin/python scripts/saliency_gradcam.py \\
        --ckpt data/models/cnn_v6.pt \\
        --spec data/spectrograms/ais_bush-point_2024-06-15_0.npy \\
              data/spectrograms/shipsear_0_0_0_0_0_1.npy \\
        --source-id ais-correlated-bush-point shipsear \\
        --out data/diagnostic/saliency/

Defaults render a curated showcase set (one sample per training source).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
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

# Default showcase: one ship sample per training source. Picked manually so
# the reader sees Grad-CAM on diverse hardware/geographies, not five
# near-identical ShipsEar clips.
SHOWCASE = [
    ("data/spectrograms/shipsear_0_0_0_0_0_1.npy",          "shipsear",                     "ship"),
    ("data/spectrograms/ais_bush-point_2024-06-15_0.npy",   "ais-correlated-bush-point",    "ship"),
    ("data/spectrograms/ais_orcasound-lab_2024-01-10_0.npy", "ais-correlated-orcasound-lab", "ship"),
    ("data/spectrograms/deepship_cargo_103_0.npy",          "deepship",                     "ship (LOHO)"),
    ("data/spectrograms/sanctsound_oc01_SanctSound_OC01_01_671399974_20190308T190000Z_0.npy", "sanctsound", "ambient"),
]


def preprocess(spec: np.ndarray, profiles: dict, source_id: str | None) -> np.ndarray:
    """Mirror CNNClassifier.predict preprocessing — center crop + high-pass +
    profile subtract + z-score. Returns a (1, 1, 128, 157) tensor-shape array."""
    spec = np.asarray(spec, dtype=np.float32)
    if spec.ndim != 2 or spec.shape[0] != _MEL_N:
        raise ValueError(f"Expected (128, T) spectrogram, got {spec.shape}")
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
    """Hooks a target conv layer; on call computes the class-activation map."""

    def __init__(self, model: OceanSentinelCNN, target_layer: torch.nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output: tuple) -> None:
        self._gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, target_class: int) -> np.ndarray:
        """Returns a (H, W) Grad-CAM map normalized 0-1, upsampled to x's
        spatial size. `target_class` is the logit index to attribute to."""
        self.model.zero_grad()
        out = self.model(x)
        score = out["vessel"][0, target_class]
        score.backward()

        # alpha_k = global avg of gradient over spatial dims for each channel
        alphas = self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (alphas * self._activations).sum(dim=1)            # (B, H, W)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        cam = cam[0].cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def render(
    spec_raw: np.ndarray,
    spec_processed: np.ndarray,
    cam: np.ndarray,
    verdict: dict,
    source_id: str | None,
    profile_applied: str | None,
    title: str,
    out_path: Path,
) -> None:
    """Save a 3-row figure: raw mel spectrogram, processed (model input),
    Grad-CAM overlay. Annotated with verdict + source."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    extent = (0, spec_processed.shape[1], _MEL_FREQS[0], _MEL_FREQS[-1])
    im_kwargs = dict(aspect="auto", origin="lower", cmap="magma", extent=extent)

    # Row 1 — raw input mel spectrogram (centre-cropped to model window).
    if spec_raw.shape[1] > _CROP_FRAMES:
        start = (spec_raw.shape[1] - _CROP_FRAMES) // 2
        spec_show = spec_raw[:, start : start + _CROP_FRAMES]
    else:
        spec_show = spec_raw
    axes[0].imshow(spec_show, **im_kwargs)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Raw mel spectrogram (model input window)")
    # Mark the cargo-ship blade-rate band — submission-video annotation.
    for hz in (28, 37):
        axes[0].axhline(hz, color="cyan", linewidth=0.8, alpha=0.6, linestyle="--")

    # Row 2 — what the model actually sees after high-pass + profile sub + z-score.
    axes[1].imshow(spec_processed, **{**im_kwargs, "cmap": "viridis"})
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title(
        f"Model input (high-pass + profile subtraction + z-score)"
        f"  —  profile={profile_applied or 'none (LOHO)'}"
    )

    # Row 3 — Grad-CAM overlay over the raw spectrogram.
    axes[2].imshow(spec_show, **im_kwargs, alpha=0.6)
    axes[2].imshow(cam, aspect="auto", origin="lower", cmap="jet",
                   alpha=0.45, extent=extent)
    axes[2].set_xlabel("Time (frames, 1 frame ≈ 0.32s)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_title(
        f"Grad-CAM:  predicted = {verdict['label'].upper()}  "
        f"(p={verdict['confidence']:.2f},  source_id={source_id})"
    )

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("data/models/cnn_v6.pt"))
    ap.add_argument("--spec", type=Path, nargs="+", default=None,
                    help="Spectrogram .npy paths. Default: built-in showcase.")
    ap.add_argument("--source-id", nargs="+", default=None,
                    help="Source IDs aligned with --spec. For showcase, ignored.")
    ap.add_argument("--out", type=Path, default=Path("data/diagnostic/saliency"))
    args = ap.parse_args()

    device = _select_device(None)

    # Load model
    model = OceanSentinelCNN().to(device)
    state = torch.load(str(args.ckpt), map_location=device)
    model.load_state_dict(state)
    model.eval()
    log.info("model_loaded", ckpt=str(args.ckpt), device=str(device))

    # Load profiles + temperature
    profiles_path = args.ckpt.with_suffix(".profiles.npz")
    profiles: dict[str, np.ndarray] = {}
    if profiles_path.exists():
        with np.load(profiles_path) as data:
            profiles = {k: data[k].astype(np.float32) for k in data.files}
    temperature = 1.0
    temp_path = args.ckpt.with_suffix(".temperature.json")
    if temp_path.exists():
        temperature = float(json.loads(temp_path.read_text())["temperature"])

    # Wire Grad-CAM on the last conv layer.
    cam = GradCAM(model, model.backbone.block3.conv)

    # Build work list
    if args.spec:
        if not args.source_id or len(args.source_id) != len(args.spec):
            sys.exit("--spec and --source-id lists must be the same length")
        items = [(p, sid, "(custom)") for p, sid in zip(args.spec, args.source_id)]
    else:
        items = [(Path(p), sid, lbl) for p, sid, lbl in SHOWCASE]

    rendered = 0
    for spec_path, source_id, narrative in items:
        if not spec_path.exists():
            log.warning("spec_missing", path=str(spec_path))
            continue
        spec_raw = np.load(spec_path).astype(np.float32)
        spec_proc, profile_applied = preprocess(spec_raw, profiles, source_id)

        tensor = torch.from_numpy(spec_proc).unsqueeze(0).unsqueeze(0).float().to(device)
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        tensor.requires_grad_(True)

        # Forward without no_grad so backward works.
        out = model(tensor)
        logits = out["vessel"] / temperature
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())

        cam_map = cam(tensor, pred_idx)

        verdict = {"label": LABELS[pred_idx], "confidence": float(probs[pred_idx].item())}

        title = (
            f"{spec_path.stem}   |   source: {source_id}   |   "
            f"narrative: {narrative}"
        )
        out_path = args.out / f"{spec_path.stem}.png"
        render(spec_raw, spec_proc, cam_map, verdict, source_id, profile_applied, title, out_path)
        rendered += 1
        log.info(
            "saliency_rendered",
            source=source_id,
            verdict=verdict["label"],
            confidence=round(verdict["confidence"], 3),
            output=str(out_path),
        )

    print(f"\nRendered {rendered} saliency PNGs in {args.out}/")


if __name__ == "__main__":
    main()
