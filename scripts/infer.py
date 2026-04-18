"""CNN inference wrapper — load checkpoint, predict on a spectrogram."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from ocean_sentinel.models.cnn import OceanSentinelCNN


LABELS = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


def load_model(ckpt_path: str = "data/models/cnn_v2.pt", device: str = "cpu") -> OceanSentinelCNN:
    """Load a trained checkpoint into an OceanSentinelCNN and set to eval mode."""
    model = OceanSentinelCNN()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.train(False)
    return model


def normalize(spec: np.ndarray) -> torch.Tensor:
    """Per-sample z-score. Matches SpecDataset.__getitem__ exactly -- any
    drift between training and inference normalization will silently poison
    predictions.
    """
    tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()  # (1, 1, F, T)
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)


@torch.no_grad()
def predict(model: OceanSentinelCNN, spec_path: str) -> dict:
    spec = np.load(spec_path)
    tensor = normalize(spec)

    out = model(tensor)
    probs = torch.softmax(out["vessel"], dim=1)[0]
    pred_idx = probs.argmax().item()
    embedding = out["embedding"][0]

    return {
        "label": LABELS[pred_idx],
        "confidence": probs[pred_idx].item(),
        "probabilities": {LABELS[i]: p.item() for i, p in enumerate(probs)},
        "embedding": embedding.tolist(),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/infer.py <spectrogram.npy>")
        sys.exit(1)

    spec = sys.argv[1]
    model = load_model()
    result = predict(model, spec)

    print(f"Prediction: {result['label']} (confidence: {result['confidence']:.1%})")
    print(f"All classes:")
    for label, prob in result["probabilities"].items():
        print(f"  {label:10s}: {prob:.1%}")
    print(f"Embedding shape: {len(result['embedding'])}")