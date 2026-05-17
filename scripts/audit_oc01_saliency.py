"""Grad-CAM saliency for v7.6 on three illustrative oc01 chunks.

Three panels for the case-study Part II page — each pairs the model's
own attention map with the AIS ground truth:

  A. TRUE POSITIVE: chunk during JOSCO HUIZHOU passage window
     - label=ship, AIS=JOSCO @ ~7 km, model should attend low-freq engine band
  B. MISLABELED AMBIENT: chunk in hour 07 of "ambient control" recording
     - label=not_ship, but AIS shows WIND SONG / BALOS within 7 km
     - what does the model see?
  C. CLEAN AMBIENT: chunk in hour 05 of "ambient" recording
     - label=not_ship, AIS shows no vessel within 25 km
     - model should produce no attention spike

Output: oceansentinelfrontend/public/case-study/saliency_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocean_sentinel.models.cnn_v7 import OceanSentinelV7
from ocean_sentinel.services.cnn_v7_classifier import (
    _MEL_N, _MEL_FREQS, _LOW_FREQ_MASK, TARGET_FRAMES,
)

CKPT = ROOT / "data/models/cnn_v7_6.pt"
CORPUS = ROOT / "data/training/sanctsound_corrected.jsonl"
FRONT = Path("/Users/jakub/oceansentinelfrontend/public/case-study")
FRONT.mkdir(parents=True, exist_ok=True)

# ── chunk selection ────────────────────────────────────────────────────────
# Each panel = (label, target_event_id_substring, panel_title, ais_blurb, panel_letter)
PANELS = [
    {
        "letter": "A",
        "event_substr": "20190309T115949Z_2",
        "title": "JOSCO HUIZHOU passage",
        "label_truth": "ship",
        "ais_blurb": "JOSCO HUIZHOU @ 6.81 km · 2019-03-09 12:39:55 UTC",
    },
    {
        "letter": "B",
        # hour 07 of "ambient" recording — chunk 90+ should be in 07 UTC
        "event_substr": "20190309T055952Z_100",
        "title": "Mislabeled ambient — hour 07",
        "label_truth": "not_ship (label) · vessel ≤7 km (AIS)",
        "ais_blurb": "WIND SONG / BALOS · vessel within 7 km · MarineCadastre",
    },
    {
        "letter": "C",
        # hour 05 — first chunks of "ambient" recording
        "event_substr": "20190309T055952Z_0",
        "title": "Clean ambient — hour 05",
        "label_truth": "not_ship",
        "ais_blurb": "No vessel within 25 km · MarineCadastre",
    },
]


class GradCAM:
    """Grad-CAM on v7 backbone block4 output (B, 256, 8, T/16)."""

    def __init__(self, model: OceanSentinelV7) -> None:
        self.model = model
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        # Hook on last backbone block
        target = model.backbone.block4
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, input, output):
        self._acts = output.detach()

    def _bwd_hook(self, module, grad_input, grad_output):
        self._grads = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(x)
        # Evidential output: evidence (B, 2) — pick the target class
        evidence = out["evidence"]
        # Backward on the target evidence
        evidence[0, class_idx].backward()
        # Global-average-pool grads to get channel weights
        alphas = self._grads.mean(dim=(2, 3), keepdim=True)  # (1, 256, 1, 1)
        # Weighted combination of activations
        cam = F.relu((alphas * self._acts).sum(dim=1))  # (1, 8, T/16)
        # Upsample to input size (128, T)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[2:], mode="bilinear",
                            align_corners=False).squeeze(1)[0]
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def preprocess(spec_raw: np.ndarray) -> np.ndarray:
    """Match cnn_v7_classifier.predict preprocessing."""
    spec = np.asarray(spec_raw, dtype=np.float32)
    if spec.shape[1] > TARGET_FRAMES:
        s = (spec.shape[1] - TARGET_FRAMES) // 2
        spec = spec[:, s:s + TARGET_FRAMES]
    elif spec.shape[1] < TARGET_FRAMES:
        pad = TARGET_FRAMES - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad)), mode="edge")
    spec = spec.copy()
    high_mean = float(spec[~_LOW_FREQ_MASK].mean())
    spec[_LOW_FREQ_MASK, :] = high_mean
    spec = (spec - spec.mean()) / (spec.std() + 1e-8)
    return spec


def load_chunk_by_substring(substr: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Find chunk matching event-id substring; return (event_id, raw_spec, normalized)."""
    with open(CORPUS) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if substr not in d.get("event_id", ""):
                continue
            spec_raw = np.load(ROOT / d["spectrogram_path"])
            spec_norm = preprocess(spec_raw)
            return d["event_id"], spec_raw, spec_norm
    raise RuntimeError(f"No chunk matching: {substr}")


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = OceanSentinelV7().to(device)
    state = torch.load(str(CKPT), map_location=device)
    model.load_state_dict(state)
    model.eval()
    cam = GradCAM(model)

    print(f"loaded v7.6 on {device}")

    # Per-site threshold for oc01
    thr_doc = json.loads((ROOT / "data/calibration/per_site_thresholds_v7_6.json").read_text())
    oc01_thr = thr_doc.get("per_site_thresholds", {}).get("oc01", 0.5)
    print(f"oc01 threshold: {oc01_thr}")

    fig = plt.figure(figsize=(14, 11), dpi=160)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.18,
                           width_ratios=[1, 1])

    for i, panel in enumerate(PANELS):
        eid, spec_raw, spec_norm = load_chunk_by_substring(panel["event_substr"])
        # Display only the ORIGINAL chunk width (skip the padding region —
        # padded columns are just edge-repeated and look like a stripe artifact).
        original_w = spec_raw.shape[1]
        spec_disp = spec_raw  # display as captured, full original width

        x = torch.from_numpy(spec_norm).unsqueeze(0).unsqueeze(0).float().to(device)
        # Forward for prediction
        with torch.no_grad():
            out = model(x)
            alpha = F.softplus(out["evidence"]) + 1.0
            S = alpha.sum()
            ship_prob = float((alpha[0, 1] / S).item())
        pred = "ship" if ship_prob > oc01_thr else "not_ship"

        # Grad-CAM for the predicted class (so we see what drove the prediction)
        cam_class = 1 if pred == "ship" else 0
        # Need fresh forward for backward — done inside __call__
        heatmap = cam(x.clone().requires_grad_(True), cam_class)
        # Crop heatmap to the original chunk width (drop the padding tail)
        heatmap_disp = heatmap[:, :original_w]

        print(f"  panel {panel['letter']}: {eid}  pred={pred}  prob={ship_prob:.3f}  threshold={oc01_thr}")

        # ── plot ───────────────────────────────────────────────────────
        ax_spec = fig.add_subplot(gs[i, 0])
        ax_cam = fig.add_subplot(gs[i, 1])

        # Spectrogram (left)
        ax_spec.imshow(spec_disp, aspect="auto", origin="lower", cmap="magma",
                       vmin=np.percentile(spec_disp, 5), vmax=np.percentile(spec_disp, 99))
        ax_spec.set_ylabel("mel bin\n(low → high freq)", fontsize=9, color="#9CA3AF")
        ax_spec.set_xlabel("time (frames)", fontsize=9, color="#9CA3AF")
        ax_spec.tick_params(colors="#9CA3AF", labelsize=8)
        for s in ax_spec.spines.values():
            s.set_color("#E5E7EB")
        title_color = "#EF4444" if pred == "ship" else "#00D4C8"
        ax_spec.set_title(
            f"({panel['letter']}) {panel['title']}\n"
            f"label: {panel['label_truth']}",
            fontsize=10, color="#111", loc="left",
        )

        # Heatmap overlay (right) — both at original chunk width, no padding artifact
        ax_cam.imshow(spec_disp, aspect="auto", origin="lower", cmap="gray",
                      alpha=0.5,
                      vmin=np.percentile(spec_disp, 5), vmax=np.percentile(spec_disp, 99))
        ax_cam.imshow(heatmap_disp, aspect="auto", origin="lower", cmap="jet", alpha=0.55)
        ax_cam.set_xlabel("time (frames)", fontsize=9, color="#9CA3AF")
        ax_cam.tick_params(colors="#9CA3AF", labelsize=8, labelleft=False)
        for s in ax_cam.spines.values():
            s.set_color("#E5E7EB")
        ax_cam.set_title(
            f"v7.6 prediction: [{pred}] ship_prob={ship_prob:.3f}  ·  thr={oc01_thr}\n"
            f"{panel['ais_blurb']}",
            fontsize=9, color=title_color, loc="left",
        )

        # Annotate freq bands on the heatmap — show where the MODEL actually
        # attends (cavitation + lower engine), NOT the blade-rate band that
        # Part I PSD analysis highlighted. The blade-rate band is masked
        # out in preprocessing (_LOW_FREQ_MASK), so model has zero access
        # to it — it must be honest about that.
        engine_hi = int(np.searchsorted(_MEL_FREQS, 500))
        engine_lo = int(np.searchsorted(_MEL_FREQS, 50))
        cavitation_lo = engine_hi
        ax_cam.axhline(engine_lo, ls=":", c="white", alpha=0.5, lw=0.7)
        ax_cam.axhline(engine_hi, ls=":", c="white", alpha=0.5, lw=0.7)
        ax_cam.text(2, cavitation_lo + 2, "cavitation band ↑ (500-1000 Hz)",
                    fontsize=7, color="white", alpha=0.9)
        ax_cam.text(2, engine_lo + 2, "engine band ↑ (50-500 Hz)",
                    fontsize=7, color="white", alpha=0.9)
        ax_cam.text(2, 1, "↓ blade-rate band masked (model can't see 0-50 Hz)",
                    fontsize=7, color="#FBBF24", alpha=0.95)

    fig.suptitle(
        "What v7.6 sees · OC01 chunks · paired with MarineCadastre AIS ground truth",
        fontsize=12, fontweight=600, color="#111", y=0.995,
    )
    out_path = FRONT / "saliency_oc01.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
