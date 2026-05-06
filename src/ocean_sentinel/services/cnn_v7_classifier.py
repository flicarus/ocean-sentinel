"""Tier-1 CNN classifier v7 — temporal-aware multi-task with uncertainty.

Same role as CNNClassifier v6: take a mel spectrogram, return a verdict.
What's new compared to v6:
  - Backbone is OceanSentinelV7 (~2.3M params, ResNet+Transformer) trained on
    AIS-corrected labels including SanctSound where v6 had ~12% recall.
  - Output dict carries `uncertainty` from the evidential head — when the
    Dirichlet alphas are flat the model says "I don't know" rather than
    forcing a confident-looking softmax.
  - Auxiliary heads expose vessel_type and distance bucket alongside the
    binary verdict.

Preprocessing must match scripts/train_cnn_v7.py SpecDataset exactly. If we
drift here, calibration silently degrades.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import structlog
import torch
import torch.nn.functional as F

from ocean_sentinel.models.cnn_v7 import OceanSentinelV7

log = structlog.get_logger()


LABELS: tuple[str, ...] = ("not_ship", "ship")
DISTANCE_LABELS: tuple[str, ...] = ("none", "far", "medium", "close")
VESSEL_TYPE_LABELS: tuple[str, ...] = (
    "none", "cargo_ship", "tanker", "fishing_vessel", "passenger_vessel",
)

_HIGH_PASS_CUTOFF_HZ = 80.0
_MEL_N = 128
_MEL_FMAX = 1000.0
_MEL_FREQS = librosa.mel_frequencies(n_mels=_MEL_N, fmax=_MEL_FMAX)
_LOW_FREQ_MASK = _MEL_FREQS < _HIGH_PASS_CUTOFF_HZ
TARGET_FRAMES = 313


def _select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CNNV7Classifier:
    """Loads OceanSentinelV7 once, predicts on spectrograms with uncertainty.

    The evidential head outputs Dirichlet alphas. When `total_evidence` (sum
    of alphas - K) is small the model has weak evidence on either side; we
    surface that as `uncertainty`. Callers can threshold on uncertainty to
    abstain rather than commit to a guess.
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        device: str | None = None,
        evidence_threshold: float = 0.5,
    ) -> None:
        self._device = _select_device(device)
        self._model = OceanSentinelV7()
        state = torch.load(str(ckpt_path), map_location=self._device)
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()
        self._ckpt_path = str(ckpt_path)
        self._evidence_threshold = evidence_threshold
        log.info(
            "cnn_v7_loaded",
            ckpt=self._ckpt_path, device=str(self._device),
            params=sum(p.numel() for p in self._model.parameters()),
        )

    @torch.no_grad()
    def predict(
        self,
        spectrogram: np.ndarray,
        source_id: str | None = None,
    ) -> dict:
        """Run v7 on a single mel spectrogram (128 x T, absolute-dB).

        `source_id` is accepted for parity with v6's interface but ignored —
        v7 was trained on a more diverse corpus and doesn't rely on per-source
        profile subtraction.
        """
        del source_id  # parity with v6 interface

        spec = np.asarray(spectrogram, dtype=np.float32)
        if spec.ndim != 2 or spec.shape[0] != _MEL_N:
            raise ValueError(
                f"Expected 2D mel spectrogram with {_MEL_N} bins, got {spec.shape}"
            )

        # Crop or pad to TARGET_FRAMES.
        if spec.shape[1] > TARGET_FRAMES:
            start = (spec.shape[1] - TARGET_FRAMES) // 2
            spec = spec[:, start:start + TARGET_FRAMES]
        elif spec.shape[1] < TARGET_FRAMES:
            pad = TARGET_FRAMES - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad)), mode="edge")

        spec = spec.copy()
        high_mean = float(spec[~_LOW_FREQ_MASK].mean())
        spec[_LOW_FREQ_MASK, :] = high_mean

        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(self._device)

        out = self._model(x)
        evidence = out["evidence"][0]                       # (2,)
        alpha = F.softplus(evidence) + 1.0
        S = alpha.sum()
        probs = alpha / S
        pred_idx = int(probs.argmax().item())

        # Total evidence beyond uniform prior; high → confident, low → uncertain.
        total_evidence = float((alpha.sum() - 2).item())
        uncertainty = 2.0 / float(S.item())                 # K / S; in (0, 1]

        vessel_type_logits = out["vessel_type"][0]
        distance_logits = out["distance"][0]
        vessel_type_idx = int(vessel_type_logits.argmax().item())
        distance_idx = int(distance_logits.argmax().item())

        return {
            "label": LABELS[pred_idx],
            "confidence": float(probs[pred_idx].item()),
            "probabilities": {
                LABELS[i]: float(p.item()) for i, p in enumerate(probs)
            },
            "uncertainty": uncertainty,
            "total_evidence": total_evidence,
            "abstain": total_evidence < self._evidence_threshold,
            "vessel_type": VESSEL_TYPE_LABELS[vessel_type_idx],
            "vessel_type_probs": {
                VESSEL_TYPE_LABELS[i]: float(p.item())
                for i, p in enumerate(F.softmax(vessel_type_logits, dim=0))
            },
            "distance": DISTANCE_LABELS[distance_idx],
            "distance_probs": {
                DISTANCE_LABELS[i]: float(p.item())
                for i, p in enumerate(F.softmax(distance_logits, dim=0))
            },
            "embedding": out["embedding"][0].cpu().tolist(),
            "checkpoint": self._ckpt_path,
        }
