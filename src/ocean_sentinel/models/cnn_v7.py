"""v7 — temporal-aware ship/ambient CNN with multi-task heads + evidential uncertainty.

What changes vs v6:
  - Input window grows from 5s (157 frames) to 60s (1876 frames). Vessel
    signatures evolve over tens of seconds (Doppler shift, harmonic drift,
    speed changes). 5s clips force the model to act on a snapshot; 60s
    lets it reason about temporal structure.
  - Conv backbone scales from 25k params (3 blocks, 16/32/64 ch) to ~1.2M
    (4 ResNet blocks, 32/64/128/256 ch). With v6 we were data-bound, not
    capacity-bound on training set; OOD generalization needs a richer
    feature space.
  - A Transformer encoder sits on top of the conv features. Self-attention
    is bidirectional by construction — every time-step attends to every
    other time-step. This implements "what happened before / what's about
    to happen" automatically; the model learns where to look.
  - Multi-head output: binary ship/ambient (primary), vessel type,
    distance bucket. Auxiliary heads regularize the shared backbone and
    give finer-grained outputs at inference.
  - Evidential head (Sensoy et al. 2018) — outputs Dirichlet alphas, not
    plain logits. Total evidence is calibrated; when the model is
    genuinely uncertain (small alphas, high entropy), it can say so
    explicitly rather than being forced to a confident-looking softmax.
    This is the "model hesitates" capability.

Param budget: ~2.3M total.
  backbone:    ~1.2M
  transformer: ~1.0M
  heads:       ~0.07M
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Conv backbone — ResNet-style, deeper and wider than v6.
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Two 3x3 convs + skip connection + 2x2 max-pool downsampling.

    The skip connection is what makes this a "Res"Block — gradient flows
    around the conv stack, so deeper networks don't hit the vanishing-
    gradient wall. We use a 1x1 conv on the skip path when channel counts
    differ between input and output.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Project skip to match out_channels when they differ.
        if in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False,
            )
        else:
            self.skip = nn.Identity()

        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return self.pool(out)


class ConvBackbone(nn.Module):
    """Four ResBlocks. Spectrogram (B, 1, 128, T) -> features (B, 256, 8, T/16).

    Frequency dim shrinks 128 -> 8 (16x), giving each output position a
    receptive field that spans roughly an octave. Time dim shrinks T -> T/16
    so a 1876-frame input becomes ~117 time positions for the transformer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.block1 = ResBlock(in_channels=1,   out_channels=32)
        self.block2 = ResBlock(in_channels=32,  out_channels=64)
        self.block3 = ResBlock(in_channels=64,  out_channels=128)
        self.block4 = ResBlock(in_channels=128, out_channels=256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)   # (B, 32,  64, T/2)
        x = self.block2(x)   # (B, 64,  32, T/4)
        x = self.block3(x)   # (B, 128, 16, T/8)
        x = self.block4(x)   # (B, 256,  8, T/16)
        return x


# ---------------------------------------------------------------------------
# Temporal transformer — bidirectional context over conv features.
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding. Lets the transformer know the order
    of time steps — without this, attention would treat the sequence as a
    bag of features.

    We use sinusoidal (vs learned) so the model generalizes to slightly
    different sequence lengths than it saw in training.
    """

    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class TemporalEncoder(nn.Module):
    """Stack of TransformerEncoderLayers. Each layer = self-attention over
    time + feedforward. Bidirectional — every position sees every other
    position. This is what gives the model "before/after awareness".

    2 layers is intentionally modest; with our dataset size (10-30k samples
    even after expansion) deeper transformers would overfit. We can scale
    later if data justifies it.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = self.pos_enc(x)
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Output heads.
# ---------------------------------------------------------------------------

class EvidentialHead(nn.Module):
    """Outputs Dirichlet alphas instead of softmax probs.

    Standard softmax always yields a confident-looking distribution
    [p, 1-p]; it can't say "I don't know". Evidential output is a Dirichlet
    over the simplex — when alphas are large the model is confident, when
    they're near 1 the distribution is uniform and we know the model
    doesn't have evidence to commit.

    Train with Sensoy 2018 evidential loss (handled in train_cnn_v7.py).
    At inference: alpha = softplus(logits) + 1; mean_p = alpha / sum(alpha);
    uncertainty = K / sum(alpha) where K = num_classes.
    """

    def __init__(self, feature_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Returns raw evidence; the loss/inference code applies softplus + 1.
        return self.fc(features)


class ClassifierHead(nn.Module):
    """Plain MLP classification head. Used for vessel type and distance —
    auxiliary tasks where standard cross-entropy is fine."""

    def __init__(
        self, feature_dim: int, num_classes: int, hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


# ---------------------------------------------------------------------------
# Top-level model — backbone + transformer + multi-head outputs.
# ---------------------------------------------------------------------------

class OceanSentinelV7(nn.Module):
    """Temporal-aware multi-task ship classifier.

    Forward returns a dict so train and inference code can pick the heads
    they need. Embedding is exposed for downstream RAG / similarity work.
    """

    NUM_VESSEL_TYPES = 5     # cargo, tanker, fishing, passenger, none
    NUM_DISTANCE_BINS = 4    # close (<=5km), medium (5-15km), far (15-50km), none

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ConvBackbone()
        self.temporal = TemporalEncoder(d_model=256, nhead=8, num_layers=2)

        # Heads operate on the temporally-pooled embedding.
        self.vessel_head = EvidentialHead(feature_dim=256, num_classes=2)
        self.type_head = ClassifierHead(
            feature_dim=256, num_classes=self.NUM_VESSEL_TYPES,
        )
        self.distance_head = ClassifierHead(
            feature_dim=256, num_classes=self.NUM_DISTANCE_BINS,
        )

    def forward(self, spec: torch.Tensor) -> dict:
        # spec: (B, 1, 128, T)  — mel spectrogram
        feats = self.backbone(spec)                  # (B, 256, 8, T/16)

        # Pool freq dim, transpose to sequence: (B, T/16, 256)
        feats = feats.mean(dim=2)                    # (B, 256, T/16)
        feats = feats.transpose(1, 2)                # (B, T/16, 256)

        # Temporal context across time steps.
        feats = self.temporal(feats)                 # (B, T/16, 256)

        # Aggregate to one embedding per clip — mean pool over time.
        embedding = feats.mean(dim=1)                # (B, 256)

        # Heads.
        evidence = self.vessel_head(embedding)       # raw evidence; loss applies softplus+1
        vessel_type_logits = self.type_head(embedding)
        distance_logits = self.distance_head(embedding)

        return {
            "evidence": evidence,                    # (B, 2)
            "vessel_type": vessel_type_logits,       # (B, 5)
            "distance": distance_logits,             # (B, 4)
            "embedding": embedding,                  # (B, 256)
        }
