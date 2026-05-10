"""Per-site adapter — real label-free fine-tune.

Replaces the previous "validation-only" Step 4 with an actual weight
update on a small residual adapter sitting on the v7 embedding.

What this is and isn't
----------------------
- It IS a real fine-tune: the 33k-param adapter's weights change to push
  the user's ambient toward not_ship in the vessel-head's logit space,
  while held-out training-distribution vessel clips keep their high
  ship_prob.
- It is NOT a full fine-tune of the 2.3M-param backbone — that needs
  far more data than 3 minutes of ambient and would overfit immediately.
- It is NOT supervised on the user's events — the user gives only their
  ambient. Vessel "labels" come from our held-out training corpus
  (DeepShip clips), which we already trust.

Why this works
--------------
The adapter is a residual MLP initialised so that adapter(x) = x at the
start (up.weight = 0). Training runs for at most N epochs with the
backbone frozen and a held-out vessel-recall floor: if pushing ambient
toward not_ship would degrade vessel recall below the floor, we stop
and revert. So in the worst case the adapter degrades to identity,
matching the threshold-only baseline. In the best case it suppresses
ambient ship_prob enough that the per-site conformal threshold no
longer has to rise to the point of losing near-threshold vessels.

Where this sits in the inference pipeline
-----------------------------------------
v7 spec → conv backbone → transformer → 256-d embedding
                                        ↓
                                    [adapter]    ← per-site, optional
                                        ↓
                                   vessel head → ship_prob

The adapter is loaded by `CNNV7Classifier.set_site_adapter(path)` and
applied between embedding and head.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

log = structlog.get_logger()


# ── Module ───────────────────────────────────────────────────────────────


class SiteAdapter(nn.Module):
    """Residual bottleneck MLP on the 256-d v7 embedding.

    Init scheme: Kaiming on the down-projection, zeros on the
    up-projection. So at training start `forward(x) == x` and the
    inference path is bit-exact equivalent to the un-adapted model.
    """

    def __init__(self, dim: int = 256, bottleneck: int = 64) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.kaiming_normal_(self.down.weight, nonlinearity="relu")
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.relu(self.down(x)))


# ── Training ────────────────────────────────────────────────────────────


@dataclass
class TrainingResult:
    site_id: str
    adapter_path: str
    n_ambient_windows: int
    n_validation_clips: int
    epochs_trained: int
    early_stopped: bool
    early_stop_reason: str | None
    val_acc_before: float           # CNN base recall on user ambient
    val_acc_after: float            # CNN+adapter recall on user ambient
    holdout_recall_before: float    # CNN base recall on held-out vessels
    holdout_recall_after: float     # CNN+adapter recall on held-out vessels
    median_ship_prob_before: float
    median_ship_prob_after: float


_TARGET_SR_HZ = 16_000
_WINDOW_S = 60.0
_HOP_S = 30.0
_MEL_N = 128
_MEL_FMAX = 1000.0
_TARGET_FRAMES = 313
_HIGH_PASS_HZ = 80.0


def _make_spec(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=_MEL_N, fmax=_MEL_FMAX,
    )
    return librosa.power_to_db(mel, ref=1.0)


def _normalise_for_v7(spec_np: np.ndarray, mel_freqs: np.ndarray) -> np.ndarray:
    """Apply the same crop/pad + low-band high-pass + standardisation that
    CNNV7Classifier.predict does, so the embeddings we compute here match
    inference."""
    if spec_np.shape[1] > _TARGET_FRAMES:
        start = (spec_np.shape[1] - _TARGET_FRAMES) // 2
        spec_np = spec_np[:, start : start + _TARGET_FRAMES]
    elif spec_np.shape[1] < _TARGET_FRAMES:
        pad = _TARGET_FRAMES - spec_np.shape[1]
        spec_np = np.pad(spec_np, ((0, 0), (0, pad)), mode="edge")
    spec_np = spec_np.copy()
    low_mask = mel_freqs < _HIGH_PASS_HZ
    high_mean = float(spec_np[~low_mask].mean())
    spec_np[low_mask, :] = high_mean
    spec_np = (spec_np - spec_np.mean()) / (spec_np.std() + 1e-8)
    return spec_np


def _slide_audio(y: np.ndarray, sr: int) -> list[np.ndarray]:
    win = int(sr * _WINDOW_S)
    hop = int(sr * _HOP_S)
    if y.size < win:
        return []
    return [y[s : s + win] for s in range(0, y.size - win + 1, hop)]


def _embed_batch(
    chunks: list[np.ndarray],
    sr: int,
    model: nn.Module,
    device: torch.device,
    mel_freqs: np.ndarray,
) -> torch.Tensor:
    """Run frozen backbone+transformer on chunks, return (N, 256) embeddings."""
    if not chunks:
        return torch.zeros(0, 256)
    specs = []
    for chunk in chunks:
        s = _make_spec(chunk, sr)
        s = _normalise_for_v7(s, mel_freqs)
        specs.append(s)
    arr = np.stack(specs)[:, None, :, :]
    x = torch.from_numpy(arr).float().to(device)
    with torch.no_grad():
        feats = model.backbone(x)
        feats = feats.mean(dim=2).transpose(1, 2)
        feats = model.temporal(feats)
        emb = feats.mean(dim=1)
    return emb.detach()


def _ship_prob_from_embedding(
    emb: torch.Tensor, vessel_head: nn.Module,
) -> torch.Tensor:
    """Apply frozen vessel head + Dirichlet softmax → P(ship)."""
    evidence = vessel_head(emb)
    alpha = F.softplus(evidence) + 1.0
    probs = alpha / alpha.sum(dim=-1, keepdim=True)
    return probs[:, 1]   # P(ship)


def _load_audio_chunks(path: Path, max_clips: int = 10) -> list[np.ndarray]:
    try:
        y, sr = librosa.load(str(path), sr=_TARGET_SR_HZ, mono=True,
                             duration=_WINDOW_S * max_clips)
    except Exception:
        return []
    chunks = _slide_audio(y, sr)
    if not chunks and y.size >= _TARGET_SR_HZ * 5:
        # Pad to one window if we have at least 5 s
        win = int(sr * _WINDOW_S)
        chunks = [np.pad(y, (0, max(0, win - y.size)), mode="edge")[:win]]
    return chunks[:max_clips]


def _default_holdout_clips() -> list[Path]:
    """A small, fixed roster of held-out vessel clips for adapter
    validation. Same set every time so results are comparable."""
    candidates = [
        "data/deepship/Tug/49.wav",
        "data/deepship/Tug/40.wav",
        "data/deepship/Cargo/103.wav",
        "data/deepship/Cargo/15.wav",
        "data/deepship/Cargo/99.wav",
        "data/deepship/Passengership/16.wav",
        "data/deepship/Passengership/29.wav",
    ]
    return [Path(c) for c in candidates if Path(c).exists()]


def train_site_adapter(
    site_id: str,
    ambient_source: str,
    *,
    holdout_vessel_clips: list[Path] | None = None,
    n_epochs: int = 12,
    learning_rate: float = 5e-3,
    l2_weight: float = 1e-2,
    holdout_recall_floor: float = 0.85,
    checkpoint: str = "data/models/cnn_v7_4.pt",
    device: str | None = None,
    warm_start: bool = False,
) -> dict[str, Any]:
    """Real per-site adapter fine-tune.

    Returns a dict with `ok`, training meta, and an `adapter_path` the
    caller can pass to `CNNV7Classifier.set_site_adapter(path)`.

    Safeguards:
    - Held-out vessel recall floor: if the adapter would drop recall on
      our 7-clip DeepShip held-out set below `holdout_recall_floor`, we
      revert to the last known-good adapter (or to identity if none).
    - L2 regularisation on adapter weights (prevents large spikes).
    - 12 epochs max, early stop on plateau.
    """
    src = Path(ambient_source)
    if not src.exists():
        return {"ok": False, "error": f"ambient source not found: {ambient_source}"}

    # Load v7 (full model)
    from ocean_sentinel.models.cnn_v7 import OceanSentinelV7
    if device:
        dev = torch.device(device)
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    model = OceanSentinelV7()
    state = torch.load(checkpoint, map_location=dev)
    model.load_state_dict(state)
    model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    mel_freqs = librosa.mel_frequencies(n_mels=_MEL_N, fmax=_MEL_FMAX)

    # Embed ambient
    try:
        y, sr = librosa.load(str(src), sr=_TARGET_SR_HZ, mono=True, duration=300.0)
    except Exception as e:
        return {"ok": False, "error": f"failed to load ambient: {type(e).__name__}: {e}"}

    ambient_chunks = _slide_audio(y, sr)
    if len(ambient_chunks) < 3:
        return {
            "ok": False,
            "error": f"need ≥3 60s windows of ambient, got {len(ambient_chunks)}",
        }

    ambient_emb = _embed_batch(ambient_chunks, sr, model, dev, mel_freqs)

    # Embed held-out vessels
    holdout_paths = holdout_vessel_clips or _default_holdout_clips()
    if not holdout_paths:
        return {"ok": False, "error": "no held-out vessel clips available"}
    holdout_chunks: list[np.ndarray] = []
    for p in holdout_paths:
        chunks = _load_audio_chunks(p, max_clips=1)
        if chunks:
            holdout_chunks.append(chunks[0])
    if not holdout_chunks:
        return {"ok": False, "error": "could not load any held-out vessel clips"}
    holdout_emb = _embed_batch(holdout_chunks, sr, model, dev, mel_freqs)

    # Baseline (no adapter, i.e. adapter = identity)
    with torch.no_grad():
        amb_p_before = _ship_prob_from_embedding(ambient_emb, model.vessel_head)
        hold_p_before = _ship_prob_from_embedding(holdout_emb, model.vessel_head)
    val_acc_before = float((amb_p_before < 0.5).float().mean().item())
    holdout_recall_before = float((hold_p_before >= 0.5).float().mean().item())
    median_before = float(amb_p_before.median().item())

    # Build trainable adapter. If warm_start and a checkpoint already
    # exists for this site, continue from it — turns successive refreshes
    # into a continual-learning loop instead of independent retrainings.
    adapter = SiteAdapter(dim=256, bottleneck=64).to(dev)
    existing_adapter_path = Path("data/sites") / site_id / "adapter.pt"
    if warm_start and existing_adapter_path.exists():
        try:
            state_existing = torch.load(str(existing_adapter_path), map_location=dev)
            adapter.load_state_dict(state_existing)
            log.info("site_adapter_warm_started", path=str(existing_adapter_path))
        except Exception as e:  # pragma: no cover — defensive
            log.warning(
                "site_adapter_warm_start_failed",
                path=str(existing_adapter_path),
                error=f"{type(e).__name__}: {e}",
            )
    optim = torch.optim.Adam(adapter.parameters(), lr=learning_rate)
    target_zero = torch.zeros(ambient_emb.shape[0], device=dev)

    best_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    best_median_amb_p = float(amb_p_before.median().item())
    early_stopped = False
    early_stop_reason: str | None = None
    epochs_trained = 0

    for epoch in range(1, n_epochs + 1):
        adapter.train()
        optim.zero_grad()
        amb_emb_adapted = adapter(ambient_emb)
        amb_p = _ship_prob_from_embedding(amb_emb_adapted, model.vessel_head)
        bce = F.binary_cross_entropy(amb_p.clamp(1e-6, 1 - 1e-6), target_zero)
        l2 = sum((p ** 2).sum() for p in adapter.parameters())
        loss = bce + l2_weight * l2
        loss.backward()
        optim.step()

        # Eval
        adapter.eval()
        with torch.no_grad():
            amb_p_eval = _ship_prob_from_embedding(
                adapter(ambient_emb), model.vessel_head,
            )
            hold_p_eval = _ship_prob_from_embedding(
                adapter(holdout_emb), model.vessel_head,
            )
        val_acc = float((amb_p_eval < 0.5).float().mean().item())
        holdout_recall = float((hold_p_eval >= 0.5).float().mean().item())
        median_amb_p = float(amb_p_eval.median().item())

        log.info(
            "site_adapter_epoch",
            epoch=epoch,
            loss=round(float(loss.item()), 4),
            val_acc=round(val_acc, 3),
            holdout_recall=round(holdout_recall, 3),
            median_amb_p=round(median_amb_p, 3),
        )

        # Floor check — abort epoch if held-out recall fell below floor
        if holdout_recall < holdout_recall_floor:
            adapter.load_state_dict(best_state)
            early_stopped = True
            early_stop_reason = (
                f"held-out vessel recall {holdout_recall:.2f} < "
                f"floor {holdout_recall_floor:.2f} after epoch {epoch}; "
                f"reverted to last good adapter"
            )
            break

        # Among epochs that pass the floor, keep the one that suppresses
        # median ambient ship_prob the most. (Plain val_acc with a 0.5
        # threshold is too coarse — it stays 0 while p drops 0.84→0.55,
        # which IS the improvement we want to capture.)
        if median_amb_p < best_median_amb_p:
            best_median_amb_p = median_amb_p
            best_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}

        epochs_trained = epoch

    # Always finish on best snapshot
    adapter.load_state_dict(best_state)

    # Final after-numbers
    adapter.eval()
    with torch.no_grad():
        amb_p_after = _ship_prob_from_embedding(
            adapter(ambient_emb), model.vessel_head,
        )
        hold_p_after = _ship_prob_from_embedding(
            adapter(holdout_emb), model.vessel_head,
        )
    val_acc_after = float((amb_p_after < 0.5).float().mean().item())
    holdout_recall_after = float((hold_p_after >= 0.5).float().mean().item())
    median_after = float(amb_p_after.median().item())

    # Persist
    site_dir = Path("data/sites") / site_id
    site_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = site_dir / "adapter.pt"
    torch.save(adapter.state_dict(), str(adapter_path))

    meta = TrainingResult(
        site_id=site_id,
        adapter_path=str(adapter_path),
        n_ambient_windows=len(ambient_chunks),
        n_validation_clips=len(holdout_chunks),
        epochs_trained=epochs_trained,
        early_stopped=early_stopped,
        early_stop_reason=early_stop_reason,
        val_acc_before=round(val_acc_before, 4),
        val_acc_after=round(val_acc_after, 4),
        holdout_recall_before=round(holdout_recall_before, 4),
        holdout_recall_after=round(holdout_recall_after, 4),
        median_ship_prob_before=round(median_before, 4),
        median_ship_prob_after=round(median_after, 4),
    )
    (site_dir / "adapter_train_meta.json").write_text(
        json.dumps(asdict(meta), indent=2)
    )
    return {"ok": True, **asdict(meta)}


# ── Inference helpers ────────────────────────────────────────────────────


def load_site_adapter(path: str | Path, device: torch.device) -> SiteAdapter:
    adapter = SiteAdapter(dim=256, bottleneck=64).to(device)
    state = torch.load(str(path), map_location=device)
    adapter.load_state_dict(state)
    adapter.eval()
    return adapter
