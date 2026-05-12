"""Train OceanSentinelV7.2 — same architecture, hard-negative oversampling.

What's different vs train_cnn_v7_1.py
-------------------------------------
v7.1 already had focal + KL coef 0.05 + cross-site mix + session split.
v7.2 adds targeted hard-negative oversampling on top:
  - Loads errors v7.1 made on the training pool (from
    scripts/extract_hard_negatives.py).
  - Replicates them in the dataset by --hard-neg-oversample (default 5).
  - Plus: any newly-pulled bulk JSONLs (data/training/v7_bulk/*.jsonl)
    are picked up automatically.

Focal loss already up-weights hard samples *per batch*, but only on
what the dataloader serves. Replicating wrong rows lets the dataloader
serve them more often, then focal compounds on top. The gradient budget
ends up concentrated on what v7.1 actually fails on, not on the easy
in-distribution majority that v7.1 already nails.

Usage:
    PYTHONPATH=src venv/bin/python scripts/train_cnn_v7_2.py \\
        --epochs 30 --batch-size 32 --lr 3e-4 \\
        --hard-negatives data/training/v7_bulk/hard_negatives.jsonl \\
        --hard-neg-oversample 5 \\
        --out data/models/cnn_v7_2.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import structlog
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, "src")

from ocean_sentinel.models.cnn_v7 import OceanSentinelV7

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Constants — match the inference path (decision/window.py + cnn_v7_classifier).
# ---------------------------------------------------------------------------

MEL_N = 128
MEL_FMAX = 1000.0
MEL_FREQS = librosa.mel_frequencies(n_mels=MEL_N, fmax=MEL_FMAX)
HIGH_PASS_CUTOFF_HZ = 80.0
LOW_FREQ_MASK = MEL_FREQS < HIGH_PASS_CUTOFF_HZ
LABEL_TO_IDX = {"not_ship": 0, "ship": 1}
DISTANCE_TO_IDX = {"none": 0, "far": 1, "medium": 2, "close": 3}
VESSEL_TYPE_TO_IDX = {
    "none": 0, "cargo_ship": 1, "tanker": 2,
    "fishing_vessel": 3, "passenger_vessel": 4,
}
DEFAULT_TARGET_FRAMES = 313


# ---------------------------------------------------------------------------
# Augmentation: SpecAugment + cross-site ambient mixing.
# ---------------------------------------------------------------------------

def specaugment(spec: np.ndarray, time_mask_max: int = 30, freq_mask_max: int = 18,
                n_time_masks: int = 2, n_freq_masks: int = 2) -> np.ndarray:
    """SpecAugment (Park et al. 2019) — replace random time/freq strips
    with the spec mean. Forces redundant cues across time + frequency."""
    spec = spec.copy()
    fill = float(spec.mean())
    T = spec.shape[1]
    F_ = spec.shape[0]
    for _ in range(n_time_masks):
        if T > 0:
            t = np.random.randint(0, time_mask_max + 1)
            t0 = np.random.randint(0, max(1, T - t))
            spec[:, t0:t0 + t] = fill
    for _ in range(n_freq_masks):
        if F_ > 0:
            f = np.random.randint(0, freq_mask_max + 1)
            f0 = np.random.randint(0, max(1, F_ - f))
            spec[f0:f0 + f, :] = fill
    return spec


def mix_ambient(ship_spec: np.ndarray, ambient_spec: np.ndarray,
                scale_range: tuple[float, float] = (0.3, 0.7)) -> np.ndarray:
    """Combine a ship spec with ambient from a different site.

    Specs are absolute mel dB (ref=1.0). To mix in dB space we go through
    linear power: P_combined = P_ship + scale * P_ambient, then back to
    dB. The scale factor controls how loud the ambient is relative to
    the ship — lower scale = ship dominates, higher = noisier mix.

    The ambient is cropped/padded to match ship's time axis, so the mix
    is a per-frame elementwise combine.
    """
    if ambient_spec.shape != ship_spec.shape:
        T = ship_spec.shape[1]
        if ambient_spec.shape[1] >= T:
            start = np.random.randint(0, ambient_spec.shape[1] - T + 1)
            ambient_spec = ambient_spec[:, start:start + T]
        else:
            pad = T - ambient_spec.shape[1]
            ambient_spec = np.pad(
                ambient_spec, ((0, 0), (0, pad)), mode="edge",
            )
    scale = np.random.uniform(*scale_range)
    # dB → linear power → mix → dB. Add small epsilon to avoid log(0).
    p_ship = np.power(10.0, ship_spec / 10.0)
    p_amb = np.power(10.0, ambient_spec / 10.0)
    p_mix = p_ship + scale * p_amb
    return 10.0 * np.log10(p_mix + 1e-12)


# ---------------------------------------------------------------------------
# Dataset.
# ---------------------------------------------------------------------------

class SpecDataset(Dataset):
    """Loads (spectrogram, labels) tuples from JSONL manifests, with
    optional cross-site ambient mixing for ship samples."""

    def __init__(
        self,
        jsonl_paths: list[Path],
        skip_sanctsound_in_primary: bool = False,
        augment: bool = False,
        target_frames: int = DEFAULT_TARGET_FRAMES,
        exclude_source_substrings: tuple[str, ...] = (),
        cross_site_mix: bool = False,
        cross_site_mix_prob: float = 0.5,
    ) -> None:
        self.augment = augment
        self.target_frames = target_frames
        self.cross_site_mix = cross_site_mix
        self.cross_site_mix_prob = cross_site_mix_prob

        self.rows: list[dict] = []
        n_filtered = 0
        for p in jsonl_paths:
            if not p.exists():
                continue
            is_primary = "v7.jsonl" in p.name
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "spectrogram_path" not in row or "label" not in row:
                        continue
                    if (
                        is_primary
                        and skip_sanctsound_in_primary
                        and (row.get("provenance") or {}).get("source_id") == "sanctsound"
                    ):
                        continue
                    if exclude_source_substrings:
                        sid = (row.get("provenance") or {}).get("source_id", "")
                        if any(s in sid for s in exclude_source_substrings):
                            n_filtered += 1
                            continue
                    self.rows.append(row)
        if n_filtered:
            log.info("dataset_filtered", n_filtered=n_filtered,
                     exclude=list(exclude_source_substrings))
        log.info("dataset_loaded", n=len(self.rows), files=len(jsonl_paths))

        # Index ambient samples by source so cross-site mixing can pick
        # ambient from a *different* site than the current ship sample.
        self._ambient_by_source: dict[str, list[str]] = defaultdict(list)
        if cross_site_mix:
            for r in self.rows:
                if r.get("label") == "not_ship":
                    src = (r.get("provenance") or {}).get("source_id", "unknown")
                    self._ambient_by_source[src].append(r["spectrogram_path"])
            log.info(
                "cross_site_mix_pool",
                sources=list(self._ambient_by_source.keys()),
                total_ambient=sum(len(v) for v in self._ambient_by_source.values()),
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        spec = np.load(row["spectrogram_path"]).astype(np.float32)

        # Center-crop or right-pad to target_frames.
        if spec.shape[1] > self.target_frames:
            start = (spec.shape[1] - self.target_frames) // 2
            spec = spec[:, start:start + self.target_frames]
        elif spec.shape[1] < self.target_frames:
            pad = self.target_frames - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad)), mode="edge")

        # Cross-site ambient mix BEFORE high-pass / z-score, so the mix
        # happens in raw absolute dB space (matches mix_ambient's math).
        # Only ship samples get mixed — mixing ambient with ambient adds
        # nothing useful and risks erasing the not_ship signal.
        if (
            self.augment
            and self.cross_site_mix
            and row.get("label") == "ship"
            and self._ambient_by_source
            and random.random() < self.cross_site_mix_prob
        ):
            own_src = (row.get("provenance") or {}).get("source_id", "unknown")
            other_sources = [s for s in self._ambient_by_source if s != own_src]
            if other_sources:
                src = random.choice(other_sources)
                amb_path = random.choice(self._ambient_by_source[src])
                amb = np.load(amb_path).astype(np.float32)
                spec = mix_ambient(spec, amb)

        # High-pass: zero out sub-80 Hz (replace with mean of >80 Hz).
        spec = spec.copy()
        high_mean = float(spec[~LOW_FREQ_MASK].mean())
        spec[LOW_FREQ_MASK, :] = high_mean

        # Z-score normalize per sample.
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)

        if self.augment:
            spec = specaugment(spec)

        binary_idx = LABEL_TO_IDX.get(row.get("label"), -100)
        distance_idx = DISTANCE_TO_IDX.get(row.get("distance_bucket"), -100)
        vessel_type_str = (row.get("gemma_verdict") or {}).get("vessel_type")
        vessel_type_idx = VESSEL_TYPE_TO_IDX.get(vessel_type_str, -100)
        source_id = (row.get("provenance") or {}).get("source_id") or row.get("source_id") or "unknown"

        return {
            "spec": torch.from_numpy(spec).unsqueeze(0).float(),
            "binary": torch.tensor(binary_idx, dtype=torch.long),
            "distance": torch.tensor(distance_idx, dtype=torch.long),
            "vessel_type": torch.tensor(vessel_type_idx, dtype=torch.long),
            "source_id": source_id,
        }


# ---------------------------------------------------------------------------
# Loss: evidential MSE+KL with optional focal weighting.
# ---------------------------------------------------------------------------

def evidential_loss(
    evidence: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    annealing_coef: float = 0.05,
    class_weights: torch.Tensor | None = None,
    focal: bool = True,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Evidential MSE + KL (Sensoy 2018), plus an optional focal term.

    annealing_coef controls how strongly we push wrong-class evidence
    toward the uniform Dirichlet. v7's default of 1.0 collapsed the
    evidential head to constant ~0.975 confidence; v7.1 defaults to 0.05.

    Focal weighting (Lin 2017) multiplies per-sample loss by
    (1 - p_true)^gamma, so easy samples (p_true close to 1) contribute
    almost no gradient and the model has to actually learn the hard
    cases instead of riding on dataset memorization.
    """
    alpha = F.softplus(evidence) + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    y = F.one_hot(target, num_classes=num_classes).float()

    err = (y - p).pow(2).sum(dim=1)
    var = (p * (1 - p) / (S + 1)).sum(dim=1)
    mse = err + var

    alpha_tilde = y + (1 - y) * alpha
    K = num_classes
    sum_alpha_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(sum_alpha_tilde)
        - torch.lgamma(torch.tensor(float(K)))
        - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
        + ((alpha_tilde - 1) * (
            torch.digamma(alpha_tilde) - torch.digamma(sum_alpha_tilde)
        )).sum(dim=1, keepdim=True)
    ).squeeze(1)

    per_sample = mse + annealing_coef * kl

    if focal:
        # P(true class) per sample. Detached so focal weight doesn't
        # propagate gradient through itself — only through the wrapped
        # MSE+KL term.
        p_true = (p * y).sum(dim=1).detach()
        focal_weight = (1.0 - p_true).pow(focal_gamma)
        per_sample = per_sample * focal_weight

    if class_weights is not None:
        w = class_weights[target]
        per_sample = per_sample * w
    return per_sample.mean()


def compute_class_weights(rows: list[dict], num_classes: int = 2) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for r in rows:
        idx = LABEL_TO_IDX.get(r.get("label"), -1)
        if 0 <= idx < num_classes:
            counts[idx] += 1
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


# ---------------------------------------------------------------------------
# Session-level split — keeps adjacent offsets from one recording out of
# both sides of the train/val split.
# ---------------------------------------------------------------------------

def session_key(row: dict) -> str:
    """A 'session' = one recording session. We bucket by source_file when
    available (ais-correlated bootstrap), else source_id + capture date.
    Rows with the same key go to the same side of the split."""
    prov = row.get("provenance") or {}
    src_file = prov.get("source_file")
    if src_file:
        return str(src_file)
    src_id = prov.get("source_id", "unknown")
    capture = row.get("audio_capture_start") or ""
    # Bucket to day so 60s windows from one recording stay together.
    day = capture[:10] if capture else "no-date"
    return f"{src_id}@{day}"


def session_split(
    rows: list[dict], val_frac: float, seed: int,
) -> tuple[list[int], list[int]]:
    sessions: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        sessions[session_key(r)].append(i)

    sess_names = list(sessions.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(sess_names)

    target_val = int(len(rows) * val_frac)
    val_idx: list[int] = []
    train_idx: list[int] = []
    for name in sess_names:
        if len(val_idx) < target_val:
            val_idx.extend(sessions[name])
        else:
            train_idx.extend(sessions[name])
    log.info(
        "session_split",
        n_sessions=len(sess_names),
        val_sessions=sum(1 for n in sess_names
                         if any(i in set(val_idx) for i in sessions[n][:1])),
        n_train=len(train_idx), n_val=len(val_idx),
    )
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Eval.
# ---------------------------------------------------------------------------

def evaluate(model: OceanSentinelV7, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct = 0
    total = 0
    confusion = Counter()
    with torch.no_grad():
        for batch in loader:
            spec = batch["spec"].to(device)
            target = batch["binary"].to(device)
            mask = target != -100
            if mask.sum() == 0:
                continue
            out = model(spec)
            alpha = F.softplus(out["evidence"]) + 1.0
            S = alpha.sum(dim=1, keepdim=True)
            preds = (alpha / S).argmax(dim=1)
            correct += (preds[mask] == target[mask]).sum().item()
            total += mask.sum().item()
            for p, t in zip(preds[mask].cpu().tolist(), target[mask].cpu().tolist()):
                confusion[(t, p)] += 1
    acc = correct / max(total, 1)
    return {"binary_acc": acc, "binary_n": total, "confusion": dict(confusion)}


# ---------------------------------------------------------------------------
# Train.
# ---------------------------------------------------------------------------

def train(args) -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("device", device=str(device))

    paths = [Path(args.primary_jsonl)]
    bulk_dir = Path("data/training/v7_bulk")
    if bulk_dir.exists():
        # Skip the hard_negatives JSONL here — we load it separately and
        # oversample. If it ends up in `paths` too it would be counted 1×
        # AND oversampled, which double-counts.
        for p in sorted(bulk_dir.glob("*.jsonl")):
            if "hard_negatives" in p.name:
                continue
            paths.append(p)
    sanct_corrected = Path("data/training/sanctsound_corrected.jsonl")
    if sanct_corrected.exists() and not args.exclude_sanctsound:
        paths.append(sanct_corrected)
    sanct_diverse = Path("data/training/sanctsound_diverse.jsonl")
    if sanct_diverse.exists():
        paths.append(sanct_diverse)
    log.info("training_files", files=[str(p) for p in paths])

    train_dataset = SpecDataset(
        paths, augment=args.augment, target_frames=args.target_frames,
        cross_site_mix=args.cross_site_mix,
        cross_site_mix_prob=args.cross_site_mix_prob,
    )
    val_dataset = SpecDataset(
        paths, augment=False, target_frames=args.target_frames,
    )
    if len(train_dataset) == 0:
        sys.exit("no training data found")

    if args.session_split:
        train_idx, val_idx = session_split(
            train_dataset.rows, args.val_frac, args.seed,
        )
    else:
        rng = np.random.default_rng(args.seed)
        all_idx = np.arange(len(train_dataset))
        rng.shuffle(all_idx)
        val_n = int(len(train_dataset) * args.val_frac)
        val_idx = all_idx[:val_n].tolist()
        train_idx = all_idx[val_n:].tolist()

    # Hard-negative oversampling — done AFTER session_split so duplicates
    # never leak into val. We append the hard-neg rows to train_dataset's
    # row list, then add the new indices to train_idx only. val_dataset
    # is left untouched, so val metrics stay honest.
    hard_neg_added = 0
    hn_path = Path(args.hard_negatives) if args.hard_negatives else None
    if hn_path and hn_path.exists() and args.hard_neg_oversample > 0:
        hn_rows: list[dict] = []
        with hn_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    hn_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        hn_rows = [
            r for r in hn_rows
            if "spectrogram_path" in r and Path(r["spectrogram_path"]).exists()
        ]
        for _ in range(args.hard_neg_oversample):
            offset = len(train_dataset.rows)
            train_dataset.rows.extend(hn_rows)
            train_idx.extend(range(offset, offset + len(hn_rows)))
        hard_neg_added = len(hn_rows) * args.hard_neg_oversample
        log.info(
            "hard_negatives_loaded",
            unique=len(hn_rows),
            oversample=args.hard_neg_oversample,
            total_added=hard_neg_added,
            path=str(hn_path),
        )
    elif args.hard_negatives:
        log.warning("hard_negatives_missing", path=args.hard_negatives)

    train_set = torch.utils.data.Subset(train_dataset, train_idx)
    val_set = torch.utils.data.Subset(val_dataset, val_idx)
    log.info("split", train=len(train_set), val=len(val_set),
             session_split=args.session_split,
             hard_neg_added=hard_neg_added)

    # ── v7.6 — balanced site sampling ─────────────────────────────────
    # v7.5 trained on 83% MBARI pool, which dilutered gradient signal for
    # under-represented sites (point-robinson 2.06% -> 0.81% share between
    # v7.4 and v7.5). That caused 100% -> 13% regression there.
    # WeightedRandomSampler with weight = 1/site_count rebalances effective
    # gradient updates so each site contributes proportionally regardless
    # of raw row count.
    if args.balanced_sampler:
        site_counts: dict[str, int] = defaultdict(int)
        train_rows = [train_dataset.rows[i] for i in train_idx]
        for r in train_rows:
            src = (r.get("provenance") or {}).get("source_id", "unknown")
            # normalize sanctsound-* and ais-correlated-* to physical site
            if src.startswith("sanctsound-more-60s-"):
                src = src[len("sanctsound-more-60s-"):]
            elif src.startswith("sanctsound-diverse-"):
                src = src[len("sanctsound-diverse-"):]
            elif src.startswith("sanctsound-corrected-"):
                src = src[len("sanctsound-corrected-"):]
            elif src.startswith("ais-correlated-"):
                src = src[len("ais-correlated-"):]
            elif src.startswith("mbari"):
                src = "mbari"
            site_counts[src] += 1
        # weight per training-row = 1/count_of_its_site
        sample_weights = []
        for r in train_rows:
            src = (r.get("provenance") or {}).get("source_id", "unknown")
            if src.startswith("sanctsound-more-60s-"): src = src[len("sanctsound-more-60s-"):]
            elif src.startswith("sanctsound-diverse-"): src = src[len("sanctsound-diverse-"):]
            elif src.startswith("sanctsound-corrected-"): src = src[len("sanctsound-corrected-"):]
            elif src.startswith("ais-correlated-"): src = src[len("ais-correlated-"):]
            elif src.startswith("mbari"): src = "mbari"
            sample_weights.append(1.0 / max(1, site_counts[src]))
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        log.info("balanced_sampler", n_sites=len(site_counts),
                 site_counts=dict(site_counts))
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers,
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers,
        )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = OceanSentinelV7().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model", params=f"{n_params:,}")

    class_weights = compute_class_weights(train_dataset.rows).to(device)
    log.info("class_weights", values=class_weights.cpu().tolist())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    best_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            spec = batch["spec"].to(device)
            binary = batch["binary"].to(device)
            distance = batch["distance"].to(device)

            mask_b = binary != -100
            if mask_b.sum() == 0:
                continue

            out = model(spec)

            loss_b = evidential_loss(
                out["evidence"][mask_b], binary[mask_b],
                annealing_coef=args.kl_coef,
                class_weights=class_weights,
                focal=args.focal,
                focal_gamma=args.focal_gamma,
            )
            mask_d = distance != -100
            loss_d = torch.tensor(0.0, device=device)
            if mask_d.sum() > 0:
                loss_d = F.cross_entropy(out["distance"][mask_d], distance[mask_d])
            loss = loss_b + 0.3 * loss_d

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)
        metrics = evaluate(model, val_loader, device)
        log.info(
            "epoch_done",
            epoch=epoch + 1,
            train_loss=round(train_loss, 4),
            val_binary_acc=round(metrics["binary_acc"], 4),
            val_binary_n=metrics["binary_n"],
        )

        if metrics["binary_acc"] > best_acc:
            best_acc = metrics["binary_acc"]
            torch.save(model.state_dict(), out_path)
            log.info("checkpoint_saved", path=str(out_path), acc=best_acc)

    print(f"\nBest val binary accuracy: {best_acc:.4f}")
    print(f"Saved to: {out_path}")


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-jsonl", default="data/training/gemma_labels.v7.jsonl")
    ap.add_argument("--out", default="data/models/cnn_v7_6.pt")
    ap.add_argument(
        "--hard-negatives", default="data/training/v7_bulk/hard_negatives.jsonl",
        help="JSONL of hard-negative rows from extract_hard_negatives.py. "
             "Pass empty string to skip oversampling.",
    )
    ap.add_argument(
        "--hard-neg-oversample", type=int, default=5,
        help="How many extra copies of each hard negative to add to the "
             "train pool. 0 = disabled.",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-frames", type=int, default=DEFAULT_TARGET_FRAMES)
    ap.add_argument("--exclude-sanctsound", action="store_true")
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no-augment", dest="augment", action="store_false")

    # New v7.1 levers
    ap.add_argument("--focal", action="store_true", default=True,
                    help="Focal weighting on evidential loss (default: on)")
    ap.add_argument("--no-focal", dest="focal", action="store_false")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--kl-coef", type=float, default=0.05,
                    help="Evidential KL coefficient. v7 used 1.0 (collapsed); "
                         "v7.1 default 0.05.")
    ap.add_argument("--cross-site-mix", action="store_true", default=True,
                    help="Mix ambient from other sites into ship samples")
    ap.add_argument("--no-cross-site-mix", dest="cross_site_mix", action="store_false")
    ap.add_argument("--cross-site-mix-prob", type=float, default=0.5)
    ap.add_argument("--session-split", action="store_true", default=True,
                    help="Train/val split by recording session, not by row")
    ap.add_argument("--no-session-split", dest="session_split", action="store_false")
    ap.add_argument("--balanced-sampler", action="store_true", default=False,
                    help="Use WeightedRandomSampler with weight=1/site_count "
                         "to equalize per-site gradient signal. Fixes the "
                         "v7.5 issue where 83%% MBARI pool diluted other sites.")

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    cli()
