from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → MaxPool. One down-sampling stage."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))

class Backbone(nn.Module):
    """Shared feature extractor. Three ConvBlocks + global average pool.

    Input:  (B, 1, 128, T)   mel spectrogram
    Output: (B, 64)          embedding vector (for RAG + task heads)
    """

    def __init__(self) -> None:
        super().__init__()
        self.block1 = ConvBlock(in_channels=1,  out_channels=16)
        self.block2 = ConvBlock(in_channels=16, out_channels=32)
        self.block3 = ConvBlock(in_channels=32, out_channels=64)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)                    # (B, 16, 64,  T/2)
        x = self.block2(x)                    # (B, 32, 32,  T/4)
        x = self.block3(x)                    # (B, 64, 16,  T/8)
        x = self.global_pool(x)               # (B, 64, 1,   1)
        return torch.flatten(x, start_dim=1)  # (B, 64)


class VesselHead(nn.Module):
    """Classifies a shared embedding into 5 threat levels.

    Output: raw logits. Pass directly to nn.CrossEntropyLoss (which applies
    log-softmax internally) or apply softmax yourself to get probabilities.
    """

    NUM_CLASSES = 5  # NONE, LOW, MEDIUM, HIGH, CRITICAL

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, self.NUM_CLASSES),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)
    

class OceanSentinelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.vessel_head = VesselHead(feature_dim=64)
        # self.species_head = SpeciesHead(feature_dim=64) # Week 3


    def forward(self, spec):
        features = self.backbone(spec)
        return {
            "vessel": self.vessel_head(features),
            "embedding": features,
        }
    
if __name__ == "__main__":
    model = OceanSentinelCNN()
    n = sum(p.numel() for p in model.parameters())
    print(f"OceanSentinelCNN has {n:,} parameters")
    x = torch.randn(2, 1, 128, 1876)
    out = model(x)
    print(f"vessel: {out['vessel'].shape}")
    print(f"embedding: {out['embedding'].shape}")


