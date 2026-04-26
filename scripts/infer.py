"""CNN inference CLI — load checkpoint, predict on a spectrogram."""
from __future__ import annotations

import sys

import numpy as np

from ocean_sentinel.services.cnn_classifier import CNNClassifier


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/infer.py <spectrogram.npy> [ckpt]")
        sys.exit(1)

    spec_path = sys.argv[1]
    ckpt = sys.argv[2] if len(sys.argv) >= 3 else "data/models/cnn_v6.pt"

    spec = np.load(spec_path)
    classifier = CNNClassifier(ckpt)
    result = classifier.predict(spec)

    print(f"Prediction: {result['label']} (confidence: {result['confidence']:.1%})")
    print("All classes:")
    for label, prob in result["probabilities"].items():
        print(f"  {label:10s}: {prob:.1%}")
    print(f"Embedding shape: {len(result['embedding'])}")


if __name__ == "__main__":
    main()
