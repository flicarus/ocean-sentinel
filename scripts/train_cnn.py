from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torch.utils.data import DataLoader, Subset


LABEL_MAP = {"not_ship": 0, "ship": 1}

# All 5s chunks at 16kHz land at 157 frames with the current analyzer.
# AIS-correlated rows are 60s -> 313 frames; center-crop them to match.
CROP_FRAMES = 157

# High-pass cutoff: zero low-freq mel bins at load time to kill the
# location-specific DC rumble (< ~80 Hz carries most of the "which
# hydrophone is this" signal). Mask is computed once from librosa's
# mel filterbank — same fmax/n_mels as AudioAnalyzer.
_HIGH_PASS_CUTOFF_HZ = 80.0
_MEL_N = 128
_MEL_FMAX = 1000.0
_MEL_FREQS = librosa.mel_frequencies(n_mels=_MEL_N, fmax=_MEL_FMAX)
_LOW_FREQ_MASK = _MEL_FREQS < _HIGH_PASS_CUTOFF_HZ  # (128,) bool

# SpecAugment params (Park et al. 2019, modest for 128x157 inputs).
_SPECAUG_FREQ_PARAM = 20   # max mask width in mel bins
_SPECAUG_TIME_PARAM = 20   # max mask width in frames
_SPECAUG_N_MASKS = 2       # number of (freq + time) mask pairs


class SpecDataset(Dataset):
    """Reads one training pair per JSONL line: (.npy spectrogram, binary label).

    `sources` filters by `provenance.source_id` in the new rich schema.

    Three preprocessing steps are applied in __getitem__:
      1. High-pass mask (always): low-freq mel bins < 80 Hz are replaced with
         the per-sample mean of the remaining bins. Kills the
         location-specific DC rumble.
      2. Per-source freq-profile subtraction (when source_profiles set):
         subtract the source's mean frequency profile (128,) broadcast over
         time. Kills the location-specific spectral tilt that the CNN
         otherwise uses as a shortcut. Profile MUST be computed from training
         entries only to avoid leakage — see `compute_source_freq_profiles`.
      3. SpecAugment (augment=True only): frequency + time masks drop random
         bands/intervals. Regularizer — forces the model to spread features
         across the spectrogram.
    """

    def __init__(
        self,
        jsonl_path: str,
        sources: set[str] | None = None,
        representation_version: str | None = None,
        augment: bool = False,
    ) -> None:
        lines = Path(jsonl_path).read_text().splitlines()
        all_entries = [json.loads(l) for l in lines if l.strip()]

        def keep(e: dict) -> bool:
            if e.get("label") not in LABEL_MAP:
                return False
            if sources is not None and e.get("provenance", {}).get("source_id") not in sources:
                return False
            if (
                representation_version is not None
                and e.get("representation_version") != representation_version
            ):
                return False
            return True

        self.entries = [e for e in all_entries if keep(e)]
        self.augment = augment
        self.source_profiles: dict[str, np.ndarray] | None = None
        skipped = len(all_entries) - len(self.entries)
        if skipped:
            print(f"SpecDataset: skipped {skipped} entries (filter + malformed)")
        label_dist = {}
        for e in self.entries:
            lbl = e["label"]
            label_dist[lbl] = label_dist.get(lbl, 0) + 1
        print(
            f"SpecDataset: loaded {len(self.entries)} pairs "
            f"(augment={augment}), distribution: {label_dist}"
        )


    def __len__(self) -> int:
        return len(self.entries)

    @property
    def source_to_idx(self) -> dict[str, int]:
        """Stable int id per source_id (lazily built, sorted for determinism)."""
        if not hasattr(self, "_src_to_idx"):
            all_srcs = sorted({
                e["provenance"]["source_id"] for e in self.entries
            })
            self._src_to_idx = {s: i for i, s in enumerate(all_srcs)}
        return self._src_to_idx


    def set_source_profiles(
        self, profiles: dict[str, np.ndarray] | None,
    ) -> None:
        """Install per-source frequency profiles for tilt subtraction."""
        self.source_profiles = profiles

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        entry = self.entries[idx]
        spec = np.load(entry["spectrogram_path"]).astype(np.float32, copy=True)
        if spec.shape[1] > CROP_FRAMES:
            start = (spec.shape[1] - CROP_FRAMES) // 2
            spec = spec[:, start:start + CROP_FRAMES]

        # High-pass: overwrite sub-80Hz bins with the mean of the remaining
        # bins so they carry no information, but without introducing a
        # sharp discontinuity that the z-score step would amplify.
        high_mean = float(spec[~_LOW_FREQ_MASK].mean())
        spec[_LOW_FREQ_MASK, :] = high_mean

        # Per-source freq-profile subtraction — removes location-specific DC
        # tilt. profile has shape (128,); broadcast across time.
        if self.source_profiles is not None:
            src_id = entry["provenance"]["source_id"]
            profile = self.source_profiles.get(src_id)
            if profile is not None:
                spec = spec - profile[:, None]

        if self.augment:
            spec = _apply_specaugment(spec.copy())

        tensor = torch.from_numpy(spec).unsqueeze(0).float()      # (1, 128, T)
        # Per-sample z-score normalization: makes inputs comparable across
        # sources and stabilizes training. Epsilon guards against silent chunks.
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        label = LABEL_MAP[entry["label"]]
        source_idx = self.source_to_idx[entry["provenance"]["source_id"]]
        return tensor, label, source_idx


def compute_source_freq_profiles(
    entries: list[dict],
) -> dict[str, np.ndarray]:
    """Time-averaged (128,) spectrum per source_id over the given entries.

    Applies the same high-pass mask as __getitem__ so the profile lives in
    the same space as the samples we'll subtract it from. Call this on
    TRAINING entries only — otherwise val leaks into normalization.
    """
    sums: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(_MEL_N, dtype=np.float64)
    )
    counts: Counter[str] = Counter()
    for entry in entries:
        src = entry["provenance"]["source_id"]
        spec = np.load(entry["spectrogram_path"]).astype(np.float32, copy=False)
        if spec.shape[1] > CROP_FRAMES:
            start = (spec.shape[1] - CROP_FRAMES) // 2
            spec = spec[:, start:start + CROP_FRAMES]
        high_mean = float(spec[~_LOW_FREQ_MASK].mean())
        # Copy so we don't mutate the cached file-backed array on disk-backed paths
        spec = spec.copy()
        spec[_LOW_FREQ_MASK, :] = high_mean
        sums[src] += spec.mean(axis=1)
        counts[src] += 1
    profiles = {
        src: (sums[src] / counts[src]).astype(np.float32)
        for src in sums
    }
    print("Source freq profiles (train-only):")
    for src, p in profiles.items():
        print(f"  {src:35s} n={counts[src]:5d}  mean_db={p.mean():+.2f}  "
              f"range=[{p.min():+.2f}, {p.max():+.2f}]")
    return profiles


def _apply_specaugment(spec: np.ndarray) -> np.ndarray:
    """Park et al. 2019 SpecAugment — frequency + time masking.

    Applies N mask pairs in-place. Masked cells are set to the spectrogram's
    mean (same convention as high-pass) to avoid introducing large dB drops.
    """
    n_mels, n_frames = spec.shape
    fill = float(spec.mean())
    for _ in range(_SPECAUG_N_MASKS):
        f = np.random.randint(0, _SPECAUG_FREQ_PARAM + 1)
        if f > 0:
            f0 = np.random.randint(0, max(1, n_mels - f))
            spec[f0:f0 + f, :] = fill
        t = np.random.randint(0, _SPECAUG_TIME_PARAM + 1)
        if t > 0:
            t0 = np.random.randint(0, max(1, n_frames - t))
            spec[:, t0:t0 + t] = fill
    return spec


def _session_key(entry: dict) -> str:
    """Canonical 'session' key per source.

    Sources name their chunks differently, so session grouping is source-aware:
      - ShipsEar chunks (shipsear/0_0/0_0_N.wav) share a parent dir per
        recording; group by that parent.
      - MBARI chunks carry their offset in the URL (...wav#offset=34522s),
        so source_file is already per 60s sample.
      - Anything else: trust source_file as-is.
    """
    src_id = entry["provenance"]["source_id"]
    src_file = entry["provenance"]["source_file"]
    if src_id == "shipsear":
        return src_file.rsplit("/", 1)[0]
    return src_file


def _group_indices_by_session(entries: list[dict]) -> dict[str, list[int]]:
    """Group chunk indices by original recording ('session').

    All 5s chunks carved from the same recording share a group, so the
    session-level split can keep them on the same side of train/val.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        groups[_session_key(entry)].append(idx)
    return dict(groups)


def _split_session_indices(
    entries: list[dict], val_frac: float = 0.2,
) -> tuple[list[int], list[int]]:
    """Session-level 80/20 split, stratified by binary label.

    Within each label (ship / not_ship), shuffle sessions with seed=42 and
    fill val to ~val_frac of that label's chunks. A 1.5x target cap prevents
    a single giant session (e.g. ShipsEar class C, ~843 chunks) from
    dominating val and starving train of that recording's signal.
    """
    groups = _group_indices_by_session(entries)
    group_label = {k: entries[v[0]]["label"] for k, v in groups.items()}

    by_label: dict[str, list[str]] = defaultdict(list)
    for k, lbl in group_label.items():
        by_label[lbl].append(k)

    rng = random.Random(42)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for lbl in sorted(by_label):
        keys = sorted(by_label[lbl])
        rng.shuffle(keys)
        total = sum(len(groups[k]) for k in keys)
        target = int(total * val_frac)
        cap = int(target * 1.5)
        val_so_far = 0
        unassigned: list[str] = []
        for k in keys:
            size = len(groups[k])
            if val_so_far < target and val_so_far + size <= cap:
                val_indices.extend(groups[k])
                val_so_far += size
            else:
                unassigned.append(k)

        # Minority-class safety net: if the cap starved val of this label
        # entirely (e.g. ShipsEar class E concentrated in 1-2 big sessions),
        # force-move the SMALLEST remaining session to val so we at least
        # report an honest confusion matrix for this class.
        if val_so_far == 0 and unassigned:
            smallest = min(unassigned, key=lambda k: len(groups[k]))
            val_indices.extend(groups[smallest])
            val_so_far += len(groups[smallest])
            unassigned.remove(smallest)
            print(
                f"  split fallback: label={lbl!r} cap starved val, "
                f"forced smallest session ({len(groups[smallest])} chunks) into val"
            )

        for k in unassigned:
            train_indices.extend(groups[k])

    train_labels = Counter(entries[i]["label"] for i in train_indices)
    val_labels = Counter(entries[i]["label"] for i in val_indices)
    print(
        f"Session-level stratified split: {len(groups)} sessions -> "
        f"train={len(train_indices)} {dict(train_labels)}, "
        f"val={len(val_indices)} {dict(val_labels)}"
    )
    return train_indices, val_indices


def _split_held_out_sources(
    entries: list[dict], held_out: set[str],
) -> tuple[list[int], list[int]]:
    """Leave-One-Source-Out split: val = entries whose provenance.source_id
    is in `held_out`; train = everything else. Honest generalization test.

    This bypasses session-level stratification because the question it
    answers is different: can the model transfer to an entirely unseen
    location? We want val to be as diverse within the held-out sources
    as possible.
    """
    train_indices: list[int] = []
    val_indices: list[int] = []
    for idx, e in enumerate(entries):
        if e["provenance"]["source_id"] in held_out:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    train_labels = Counter(entries[i]["label"] for i in train_indices)
    val_labels = Counter(entries[i]["label"] for i in val_indices)
    print(
        f"Leave-one-source-out split: held_out={sorted(held_out)} -> "
        f"train={len(train_indices)} {dict(train_labels)}, "
        f"val={len(val_indices)} {dict(val_labels)}"
    )
    if not val_indices:
        raise SystemExit(
            f"held_out sources {sorted(held_out)} matched no entries. "
            f"Check --held-out-sources against provenance.source_id."
        )
    return train_indices, val_indices


def make_loaders(
    dataset_train: SpecDataset,
    dataset_val: SpecDataset,
    batch_size: int = 16,
    val_frac: float = 0.2,
    held_out_sources: set[str] | None = None,
):
    """Build train/val loaders sharing one split but different augmentation.

    `dataset_train` should have augment=True, `dataset_val` augment=False.
    Both must contain the identical entries in the same order — the split
    indices reference positions, not rows.

    If `held_out_sources` is provided, val is constructed from those sources
    only (LOHO = leave-one-source-out honest generalization benchmark).
    Otherwise, falls back to session-level stratified split.
    """
    assert len(dataset_train.entries) == len(dataset_val.entries), (
        "train and val dataset instances must carry identical entries"
    )
    if held_out_sources:
        train_indices, val_indices = _split_held_out_sources(
            dataset_train.entries, held_out=set(held_out_sources),
        )
    else:
        train_indices, val_indices = _split_session_indices(
            dataset_train.entries, val_frac=val_frac,
        )
    train_ds = Subset(dataset_train, train_indices)
    val_ds = Subset(dataset_val, val_indices)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def compute_class_weights(entries: list[dict], indices: list[int]) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss.

    Prevents the model from defaulting to the majority class under severe
    imbalance (e.g. ShipsEar-only: 9:1 ship:not_ship). Indexed by LABEL_MAP.
    """
    counts = Counter(entries[i]["label"] for i in indices)
    total = sum(counts.values())
    weights = torch.zeros(len(LABEL_MAP), dtype=torch.float32)
    for lbl, idx in LABEL_MAP.items():
        n = counts.get(lbl, 0)
        # Inverse frequency, normalized so weights average to 1
        weights[idx] = total / (len(LABEL_MAP) * n) if n > 0 else 0.0
    print(f"Class weights for loss: {dict(zip(LABEL_MAP, weights.tolist()))}")
    return weights



def train_one_epoch(
    model, loader, loss_fn, optimizer, device,
    mixup_alpha: float = 0.0,
    cross_source_only: bool = False,
):
    """Train one epoch.

    Mixup (Zhang et al. 2018): blend pairs of samples within each batch so
    the model is forced to learn features that survive blending. When
    `cross_source_only=True`, only blend pairs whose source_id differs —
    same-source pairs pass through pristine. This concentrates the
    regularization signal where it matters (cross-location shortcut).

    Per-sample λ is drawn per mixed pair, so same-source pairs get λ=1
    (no mix) and cross-source pairs get λ ~ Beta(α, α). Loss is the
    sample-wise weighted sum.

    `loss_fn` is expected to have reduction='none' OR we handle standard
    reduction separately. To keep both paths clean, we call the loss
    function with reduction='none' equivalent by constructing it ourselves.
    """
    model.train()
    total_loss = 0.0
    correct = total = 0
    use_mixup = mixup_alpha > 0.0
    # Replicate loss_fn behaviour but unreduced: we need per-sample losses
    # to apply per-sample lambdas. Build a parallel no-reduction variant.
    weight = getattr(loss_fn, "weight", None)
    ce_none = nn.CrossEntropyLoss(weight=weight, reduction="none")
    for batch in loader:
        x, y, src = batch
        x, y, src = x.to(device), y.to(device), src.to(device)
        B = x.size(0)
        optimizer.zero_grad()
        if use_mixup:
            perm = torch.randperm(B, device=device)
            lam_scalar = float(np.random.beta(mixup_alpha, mixup_alpha))
            lam = torch.full((B,), lam_scalar, device=device, dtype=x.dtype)
            if cross_source_only:
                same_src = src == src[perm]
                # Same-source pairs: λ=1.0 → no mix, loss_a dominates, no y_b effect
                lam = torch.where(same_src, torch.ones_like(lam), lam)
            lam4 = lam.view(B, 1, 1, 1)
            x_mix = lam4 * x + (1.0 - lam4) * x[perm]
            y_a, y_b = y, y[perm]
            logits = model(x_mix)["vessel"]
            loss_a = ce_none(logits, y_a)                # (B,)
            loss_b = ce_none(logits, y_b)                # (B,)
            per_sample = lam * loss_a + (1.0 - lam) * loss_b
            loss = per_sample.mean()
            # Accuracy: pick the dominant target per-sample (λ >= 0.5 => y_a)
            dom = torch.where(lam >= 0.5, y_a, y_b)
            correct += (logits.argmax(dim=1) == dom).sum().item()
        else:
            logits = model(x)["vessel"]
            loss = loss_fn(logits, y)
            correct += (logits.argmax(dim=1) == y).sum().item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * B
        total += B
    return total_loss / total, correct / total

@torch.no_grad()                               # no gradients in eval (saves memory)
def evaluate(model, loader, loss_fn, device, n_classes: int = 2):
    """Evaluate on a loader. Returns:
        (avg_loss, accuracy, macro_recall, per_class_recall dict)

    macro_recall is robust to class imbalance — averages per-class recall
    instead of per-sample accuracy. This is the right metric to select
    checkpoints on when val is imbalanced (like LOHO held-out ShipsEar
    where val is 9:1 ship:not_ship).
    """
    model.eval()
    total_loss = 0.0
    correct = total = 0
    per_class_correct = np.zeros(n_classes, dtype=np.int64)
    per_class_total = np.zeros(n_classes, dtype=np.int64)
    for batch in loader:
        x, y = batch[0], batch[1]  # source_idx is batch[2] if present — ignore here
        x, y = x.to(device), y.to(device)
        logits = model(x)["vessel"]
        loss = loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * x.size(0)
        correct += (preds == y).sum().item()
        total += y.size(0)
        for c in range(n_classes):
            mask = y == c
            per_class_total[c] += int(mask.sum().item())
            per_class_correct[c] += int(((preds == y) & mask).sum().item())
    inv = {v: k for k, v in LABEL_MAP.items()}
    per_recall = {
        inv[c]: (per_class_correct[c] / per_class_total[c]
                 if per_class_total[c] > 0 else float("nan"))
        for c in range(n_classes)
    }
    valid = [r for r in per_recall.values() if r == r]  # drop NaN
    macro_recall = float(sum(valid) / len(valid)) if valid else 0.0
    return total_loss / total, correct / total, macro_recall, per_recall


@torch.no_grad()
def _confusion_matrix(model, loader, device, n_classes: int = 2) -> np.ndarray:
    """Build a raw NxN confusion matrix over a dataloader."""
    model.train(False)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for batch in loader:
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)
        preds = model(x)["vessel"].argmax(dim=1)
        for t, p in zip(y.tolist(), preds.tolist()):
            cm[t, p] += 1
    return cm


def _print_confusion(cm: np.ndarray) -> None:
    inv = {v: k for k, v in LABEL_MAP.items()}
    labels = [inv[i] for i in range(cm.shape[0])]
    header = "truth\\pred".ljust(12) + "".join(f"{l:>9s}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        total = int(row.sum())
        cells = "".join(f"{c:>9d}" for c in row)
        acc = (row[i] / total) if total else 0.0
        print(f"{labels[i]:<12s}{cells}   ({int(row[i])}/{total} = {acc:.1%})")


def main() -> None:
    import argparse

    from ocean_sentinel.models.cnn import OceanSentinelCNN

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/training/gemma_labels.jsonl")
    ap.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Filter by provenance.source_id. Default: all sources.",
    )
    ap.add_argument("--representation", default="abs_db_v1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ckpt", default="data/models/cnn_v2.pt")
    ap.add_argument("--no-augment", action="store_true",
                    help="Disable SpecAugment on the training split.")
    ap.add_argument("--no-source-norm", action="store_true",
                    help="Disable per-source freq-profile subtraction.")
    ap.add_argument(
        "--held-out-sources", nargs="+", default=None,
        help="Leave-one-source-out honest benchmark: val is ONLY these "
             "provenance.source_ids; train is everything else.",
    )
    ap.add_argument(
        "--class-weights", action="store_true",
        help="Use inverse-frequency class weights in CrossEntropyLoss. "
             "Prevents majority-class default under severe imbalance.",
    )
    ap.add_argument(
        "--mixup-alpha", type=float, default=0.0,
        help="Mixup Beta(alpha, alpha) — blends batched samples to break "
             "source-identity shortcuts. 0.0 disables. Typical: 0.2-0.4.",
    )
    ap.add_argument(
        "--cross-source-mixup", action="store_true",
        help="Only mix cross-source pairs. Same-source pairs pass pristine "
             "(λ=1). Concentrates regularization on cross-location transfer.",
    )
    args = ap.parse_args()

    sources = set(args.sources) if args.sources else None
    ds_train = SpecDataset(
        args.jsonl,
        sources=sources,
        representation_version=args.representation,
        augment=not args.no_augment,
    )
    ds_val = SpecDataset(
        args.jsonl,
        sources=sources,
        representation_version=args.representation,
        augment=False,
    )
    if len(ds_train) < 20:
        raise SystemExit(
            f"Dataset too small ({len(ds_train)}). Run bootstrap_shipsear.py first."
        )

    # Resolve training indices consistently with whichever split mode is
    # active. Profiles and class-weights are computed from these indices.
    held_out = set(args.held_out_sources) if args.held_out_sources else None
    if held_out:
        train_idx, _ = _split_held_out_sources(ds_train.entries, held_out)
    else:
        train_idx, _ = _split_session_indices(ds_train.entries)

    # Compute source freq-profiles from TRAINING indices only, then install
    # on both datasets so val samples are normalized with the same profiles
    # (no val leakage — profiles never see val entries).
    if not args.no_source_norm:
        train_entries = [ds_train.entries[i] for i in train_idx]
        profiles = compute_source_freq_profiles(train_entries)
        ds_train.set_source_profiles(profiles)
        ds_val.set_source_profiles(profiles)

    train_loader, val_loader = make_loaders(
        ds_train, ds_val,
        batch_size=args.batch_size,
        held_out_sources=held_out,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = OceanSentinelCNN().to(device)

    if args.class_weights:
        weights = compute_class_weights(ds_train.entries, train_idx).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Selector: by default pick checkpoint on macro-recall (balanced under
    # class imbalance) rather than raw val_loss. val_loss on imbalanced val
    # happily saves "predict majority always" shortcuts.
    best_macro = -1.0
    best_snap = {"epoch": 0, "loss": float("inf"), "acc": 0.0,
                 "macro": 0.0, "per_class": {}}
    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            mixup_alpha=args.mixup_alpha,
            cross_source_only=args.cross_source_mixup,
        )
        val_loss, val_acc, val_macro, per_recall = evaluate(
            model, val_loader, loss_fn, device,
        )
        per_str = " ".join(f"{k}={v:.1%}" for k, v in per_recall.items())
        print(
            f"epoch {epoch:2d}/{args.epochs}  "
            f"train_loss={train_loss:.3f} train_acc={train_acc:.1%}  "
            f"val_loss={val_loss:.3f} val_acc={val_acc:.1%} "
            f"macro={val_macro:.1%}  [{per_str}]"
        )
        if val_macro > best_macro:
            best_macro = val_macro
            best_snap = {
                "epoch": epoch, "loss": val_loss, "acc": val_acc,
                "macro": val_macro, "per_class": dict(per_recall),
            }
            torch.save(model.state_dict(), ckpt_path)
            print(
                f"  saved checkpoint "
                f"(macro={val_macro:.1%}, val_acc={val_acc:.1%})"
            )

    print(
        f"\nBest (by macro-recall): epoch {best_snap['epoch']}  "
        f"macro={best_snap['macro']:.1%}  val_acc={best_snap['acc']:.1%}  "
        f"per_class={best_snap['per_class']}"
    )
    print(f"Checkpoint: {ckpt_path}")

    # Reload best weights, build confusion matrix on validation set
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    cm = _confusion_matrix(model, val_loader, device)
    print("\nValidation confusion matrix:")
    _print_confusion(cm)


if __name__ == "__main__":
    main()
