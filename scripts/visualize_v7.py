"""Visualize OceanSentinelV7 — three views.

1. torchinfo text summary (per-layer params + shapes; runs instantly).
2. ONNX export — drag the .onnx file onto https://netron.app (or open in
   the desktop Netron) for an interactive node graph.
3. Matplotlib block diagram — high-level data flow with shapes annotated.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torchinfo import summary

from ocean_sentinel.models.cnn_v7 import OceanSentinelV7


OUT_DIR = Path("data/diagnostic/v7")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def text_summary(model: torch.nn.Module) -> None:
    print("=" * 100)
    print("TORCHINFO SUMMARY")
    print("=" * 100)
    s = summary(
        model,
        input_size=(1, 1, 128, 313),
        col_names=("input_size", "output_size", "num_params", "params_percent"),
        col_width=20,
        depth=3,
        verbose=1,
    )
    # torchinfo prints to stdout; also save text version.
    (OUT_DIR / "v7_torchinfo.txt").write_text(str(s))
    print(f"\nSaved -> {OUT_DIR / 'v7_torchinfo.txt'}")


def export_onnx(model: torch.nn.Module) -> None:
    print()
    print("=" * 100)
    print("ONNX EXPORT")
    print("=" * 100)
    model.eval()
    dummy = torch.randn(1, 1, 128, 313)
    out_path = OUT_DIR / "cnn_v7.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["spectrogram"],
        output_names=["evidence", "vessel_type", "distance", "embedding"],
        dynamic_axes={
            "spectrogram": {0: "batch", 3: "time"},
            "evidence": {0: "batch"},
            "vessel_type": {0: "batch"},
            "distance": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  saved {out_path}  ({size_mb:.2f} MB)")
    print(f"  -> drag this file onto https://netron.app/")
    print(f"  -> or open in the Netron desktop app for the interactive graph")


def block_diagram() -> None:
    """Matplotlib high-level data flow diagram."""
    print()
    print("=" * 100)
    print("BLOCK DIAGRAM")
    print("=" * 100)

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Helper: draw a labeled block.
    def block(x, y, w, h, label, sub=None, color="#dde6f0", lw=1.5):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
            linewidth=lw, edgecolor="#3a4f63", facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (1.5 if sub else 0),
                label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#1a2a3a")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.0, sub,
                    ha="center", va="center",
                    fontsize=8, color="#5a6a7a", style="italic")

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#3a4f63"))
        if label:
            ax.text((x1 + x2) / 2 + 0.5, (y1 + y2) / 2,
                    label, fontsize=8, color="#3a4f63",
                    rotation=0)

    # Title.
    ax.text(50, 96,
            "OceanSentinelV7 — temporal-aware ship classifier (~2.3M params)",
            ha="center", va="center", fontsize=14, fontweight="bold")

    # Input.
    block(40, 86, 20, 5, "Input mel spectrogram",
          "(B, 1, 128, T)   T≈313 frames @ 16 kHz, hop 512", "#fff4d6")

    # Conv backbone (4 ResBlocks).
    block(35, 70, 30, 12, "ConvBackbone (ResNet-style)",
          "4 ResBlocks: 1→32→64→128→256 ch\n"
          "16x freq downsample, 16x time downsample\n"
          "out: (B, 256, 8, T/16)   ~1.2M params",
          "#d6ebff")

    # Freq pool + transpose.
    block(35, 60, 30, 6, "Frequency pool + reshape",
          "mean over freq dim → (B, T/16, 256)", "#e8e8e8")

    # Temporal transformer.
    block(20, 40, 60, 14, "TemporalEncoder (Transformer)",
          "PositionalEncoding + 2× TransformerEncoderLayer\n"
          "d_model=256, heads=8, FFN=512   ~1.0M params\n"
          "self-attention is BIDIRECTIONAL — every step\n"
          "sees every other step (past + future context)",
          "#fde2d6")

    # Mean pool over time.
    block(35, 30, 30, 5, "Mean pool over time",
          "(B, T/16, 256) → (B, 256)", "#e8e8e8")

    # Heads (4 outputs).
    head_y = 12
    head_w = 16
    head_h = 9

    block(2, head_y, head_w, head_h,
          "EvidentialHead\n(primary)",
          "(B, 2)\nsoftplus+1 → Dirichlet α\nbinary ship/ambient + uncertainty",
          "#d6f0d6")

    block(22, head_y, head_w, head_h,
          "ClassifierHead\nvessel_type",
          "(B, 5)\ncargo, tanker, fishing,\npassenger, none",
          "#fff0d6")

    block(42, head_y, head_w, head_h,
          "ClassifierHead\ndistance",
          "(B, 4)\nclose ≤10 km, medium 10-30,\nfar 30-50, none",
          "#fff0d6")

    block(62, head_y, head_w, head_h,
          "Embedding",
          "(B, 256)\nshared backbone\nfeatures (for RAG / sim)",
          "#e8d6f0")

    # Arrows top to bottom.
    arrow(50, 86, 50, 82)            # input → backbone
    arrow(50, 70, 50, 66)            # backbone → freq pool
    arrow(50, 60, 50, 54)            # freq pool → transformer
    arrow(50, 40, 50, 35)            # transformer → mean pool
    arrow(50, 30, 50, 25)            # mean pool → heads (junction)

    # Branch arrows to four heads.
    arrow(50, 25, 10, head_y + head_h)
    arrow(50, 25, 30, head_y + head_h)
    arrow(50, 25, 50, head_y + head_h)
    arrow(50, 25, 70, head_y + head_h)

    # Legend / annotation.
    ax.text(50, 4,
            "Self-attention implements 'before/after awareness' automatically.\n"
            "Evidential head outputs Dirichlet alphas — when alphas≈1 the model says 'I don't know'.",
            ha="center", va="center", fontsize=9, color="#3a4f63",
            style="italic")

    out_path = OUT_DIR / "v7_block_diagram.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def receptive_field_walk() -> None:
    """One-line trace through the conv backbone showing how (128, T)
    shrinks to (8, T/16). Helpful for understanding what each block does."""
    print()
    print("=" * 100)
    print("LAYER-BY-LAYER SHAPE TRACE")
    print("=" * 100)
    model = OceanSentinelV7().eval()
    x = torch.randn(1, 1, 128, 313)
    print(f"  input                          {tuple(x.shape)}")
    x1 = model.backbone.block1(x);  print(f"  after ResBlock1 (32 ch)        {tuple(x1.shape)}")
    x2 = model.backbone.block2(x1); print(f"  after ResBlock2 (64 ch)        {tuple(x2.shape)}")
    x3 = model.backbone.block3(x2); print(f"  after ResBlock3 (128 ch)       {tuple(x3.shape)}")
    x4 = model.backbone.block4(x3); print(f"  after ResBlock4 (256 ch)       {tuple(x4.shape)}")
    x5 = x4.mean(dim=2);            print(f"  after freq mean-pool           {tuple(x5.shape)}")
    x5 = x5.transpose(1, 2);        print(f"  after transpose to (B, T, C)   {tuple(x5.shape)}")
    x6 = model.temporal(x5);        print(f"  after TemporalEncoder          {tuple(x6.shape)}")
    x7 = x6.mean(dim=1);            print(f"  after mean pool over time      {tuple(x7.shape)}")


def main() -> None:
    model = OceanSentinelV7()
    text_summary(model)
    receptive_field_walk()
    export_onnx(model)
    block_diagram()
    print()
    print(f"All artifacts in: {OUT_DIR}/")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
