"""Tier-1 CNN classifier — fast binary ship/not_ship triage on spectrograms.

The preprocessing here MUST match SpecDataset.__getitem__ in scripts/train_cnn.py.
Any drift between training and inference normalization silently poisons
predictions, so this module is the single source of truth for inference-time
spectrogram preprocessing.

Per-source frequency profile subtraction is applied when a `<ckpt>.profiles.npz`
file is found alongside the checkpoint AND a known source_id is supplied.
For unknown source_ids we fall back to no subtraction — equivalent to the
LOHO held-out evaluation path during training.
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import structlog
import torch

from ocean_sentinel.models.cnn import OceanSentinelCNN

log = structlog.get_logger()


LABELS: tuple[str, ...] = ("not_ship", "ship")

# Match training defaults (scripts/train_cnn.py constants).
_HIGH_PASS_CUTOFF_HZ = 80.0
_MEL_N = 128
_MEL_FMAX = 1000.0
_MEL_FREQS = librosa.mel_frequencies(n_mels=_MEL_N, fmax=_MEL_FMAX)
_LOW_FREQ_MASK = _MEL_FREQS < _HIGH_PASS_CUTOFF_HZ
_CROP_FRAMES = 157


def _select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_training_source_id(
    live_source_id: str | None,
    available: set[str],
) -> str | None:
    """Map an adapter's live source_id to its training-data equivalent.

    Returns None when no profile exists for this source (e.g. a hydrophone
    we never trained on, or one whose bootstrap data was filtered out).
    """
    if not live_source_id:
        return None
    if live_source_id in available:
        return live_source_id
    # Orcasound live ids are "orcasound_<node>" (so the lab node ends up as
    # "orcasound_orcasound_lab" because its raw node name is "orcasound_lab").
    # Training data uses "ais-correlated-<node-with-dashes>".
    if live_source_id.startswith("orcasound_"):
        node = live_source_id.removeprefix("orcasound_")
        candidate = f"ais-correlated-{node.replace('_', '-')}"
        if candidate in available:
            return candidate
    return None


class CNNClassifier:
    """Loads a trained OceanSentinelCNN once, predicts on spectrograms."""

    def __init__(self, ckpt_path: str | Path, device: str | None = None) -> None:
        self._device = _select_device(device)
        self._model = OceanSentinelCNN()
        state = torch.load(str(ckpt_path), map_location=self._device)
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()
        self._ckpt_path = str(ckpt_path)

        profiles_path = Path(ckpt_path).with_suffix(".profiles.npz")
        if profiles_path.exists():
            with np.load(profiles_path) as data:
                self._profiles: dict[str, np.ndarray] = {
                    k: data[k].astype(np.float32, copy=False) for k in data.files
                }
            log.info(
                "cnn_profiles_loaded",
                path=str(profiles_path),
                sources=sorted(self._profiles),
            )
        else:
            self._profiles = {}
            log.warning(
                "cnn_profiles_missing",
                expected=str(profiles_path),
                consequence=(
                    "inference will run without per-source profile subtraction; "
                    "known-source accuracy may drop several pp vs val benchmark"
                ),
            )

        # Temperature scaling: post-hoc calibration scalar fitted by
        # scripts/calibrate_cnn.py. Logits are divided by T before softmax
        # so reported confidences match real accuracy.
        temperature_path = Path(ckpt_path).with_suffix(".temperature.json")
        if temperature_path.exists():
            self._temperature = float(
                json.loads(temperature_path.read_text())["temperature"]
            )
            log.info(
                "cnn_temperature_loaded",
                path=str(temperature_path),
                T=self._temperature,
            )
        else:
            self._temperature = 1.0
            log.warning(
                "cnn_temperature_missing",
                expected=str(temperature_path),
                consequence="confidences may be miscalibrated (typically overconfident)",
            )

        log.info(
            "cnn_classifier_loaded",
            ckpt=self._ckpt_path,
            device=str(self._device),
            profiles=len(self._profiles),
            temperature=self._temperature,
        )

    @torch.no_grad()
    def predict(
        self,
        spectrogram: np.ndarray,
        source_id: str | None = None,
    ) -> dict:
        """Classify a single mel spectrogram (128 x T, absolute-dB).

        `source_id` is the live adapter's `source_id` attribute. If we have
        a cached profile for it, we subtract that profile after high-pass
        and before z-score — matching SpecDataset.__getitem__ exactly.

        Returns a dict with the predicted label, confidence, full
        probabilities, the 64-dim shared backbone embedding, and which
        training source_id (if any) we resolved the live source to.
        """
        spec = np.asarray(spectrogram, dtype=np.float32)
        if spec.ndim != 2 or spec.shape[0] != _MEL_N:
            raise ValueError(
                f"Expected 2D mel spectrogram with {_MEL_N} bins, "
                f"got shape {spec.shape}"
            )

        # 60s chunks (313 frames) → center-crop to the 157-frame training window.
        if spec.shape[1] > _CROP_FRAMES:
            start = (spec.shape[1] - _CROP_FRAMES) // 2
            spec = spec[:, start:start + _CROP_FRAMES]

        # High-pass: replace sub-80Hz bins with the mean of the rest.
        spec = spec.copy()
        high_mean = float(spec[~_LOW_FREQ_MASK].mean())
        spec[_LOW_FREQ_MASK, :] = high_mean

        # Profile subtraction: fixes the train/inference distribution shift
        # for hydrophones the model was trained on. Unknown sources fall
        # through with no subtraction (the LOHO path).
        resolved = _resolve_training_source_id(source_id, set(self._profiles))
        if resolved is not None:
            spec = spec - self._profiles[resolved][:, None]

        tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        tensor = tensor.to(self._device)

        out = self._model(tensor)
        logits = out["vessel"] / self._temperature
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        embedding = out["embedding"][0].cpu().tolist()

        return {
            "label": LABELS[pred_idx],
            "confidence": float(probs[pred_idx].item()),
            "probabilities": {
                LABELS[i]: float(p.item()) for i, p in enumerate(probs)
            },
            "embedding": embedding,
            "checkpoint": self._ckpt_path,
            "profile_applied": resolved,
        }
