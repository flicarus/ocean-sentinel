"""Proper held-out evaluation on clips the CNN has never seen.

What this answers
-----------------
"Does v7.4 actually generalise, or is it memorising?"

Three groups:

  IN-DISTRIBUTION + UNSEEN
    Vessel clips from DeepShip (Cargo, Passengership, Tanker) whose
    *specific* (class, number) tuple is NOT in any training JSONL. Site
    distribution = same as training; the model has just never seen
    these particular recordings.

  OOD SYNTHETIC
    The tropical reef ambient we built earlier. Site distribution NOT
    in training; ambient is guaranteed-no-ships by construction.

  AMBIENT (sanity floor)
    Two deterministic synthetic ambient clips (seeds 101, 202) whose
    label is mathematically guaranteed not_ship.

For each clip we run:
  1. Raw CNN  — base v7.4, no per-site adapter, global conformal threshold
  2. Full pipeline — same as `simulate_detection` (decision tier mapping
     on top of CNN output)

We report per-group accuracy on both, plus a per-clip breakdown so any
specific failure can be inspected.
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from ocean_sentinel.gemma.cnn_inference import simulate_detection
from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier


CKPT = "data/models/cnn_v7_4.pt"
SR = 16_000
CLIP_SECONDS = 60


# ── unseen clip discovery ──────────────────────────────────────────────


def _trained_deepship_ids() -> set[tuple[str, str]]:
    """Read every training JSONL and collect (class, number) tuples for
    DeepShip clips actually used."""
    out: set[tuple[str, str]] = set()
    for jsonl in Path("data/training").rglob("*.jsonl"):
        try:
            for line in jsonl.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                eid = (row.get("event_id") or "").lower()
                if "deepship" in eid:
                    parts = eid.split("_")
                    if len(parts) >= 3:
                        out.add((parts[1], parts[2]))
        except Exception:
            pass
    return out


def _find_unseen_deepship(per_class: int = 5) -> list[tuple[str, Path, float]]:
    """Pick a panel of unseen DeepShip vessels. Returns list of
    (label, path, offset_s) — same path can appear at multiple offsets
    when the underlying recording is long enough.

    Default: up to 5 clips per class (Cargo / Passengership / Tanker)
    × up to 2 offsets each (30 s and 90 s) = up to 30 vessel evaluations.
    """
    trained = _trained_deepship_ids()
    out: list[tuple[str, Path, float]] = []
    for cls in ("Cargo", "Passengership", "Tanker"):
        folder = Path(f"data/deepship/{cls}")
        if not folder.exists():
            continue
        kept = 0
        for p in sorted(folder.glob("*.wav"),
                         key=lambda x: int(x.stem) if x.stem.isdigit() else 9999):
            if (cls.lower(), p.stem) in trained:
                continue
            try:
                duration = librosa.get_duration(path=str(p))
            except Exception:
                continue
            if duration < CLIP_SECONDS:
                continue
            # primary window
            out.append(("ship", p, 30.0))
            # second window if recording is long enough — captures
            # within-clip variability the model has to handle in
            # production (engine load shifts, vessels transiting).
            if duration >= 60 + 90:
                out.append(("ship", p, 90.0))
            kept += 1
            if kept >= per_class:
                break
    return out


# ── synthetic OOD + ambient ────────────────────────────────────────────


def _synth_reef(out: Path, duration_s: int = CLIP_SECONDS) -> Path:
    rng = np.random.RandomState(2026)
    n = SR * duration_s
    y = np.zeros(n, dtype=np.float32)
    n_clicks = int(30 * duration_s)
    click_template = np.exp(-np.linspace(0, 6, SR // 200)).astype(np.float32)
    t_click = np.arange(len(click_template)) / SR
    click_template *= np.sin(2 * np.pi * 4_000 * t_click)
    positions = rng.randint(0, n - len(click_template), size=n_clicks)
    for pos in positions:
        end = pos + len(click_template)
        if end < n:
            y[pos:end] += 0.4 * click_template * rng.uniform(0.5, 1.0)
    chorus = rng.randn(n).astype(np.float32) * 0.15
    Y = np.fft.rfft(chorus)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    chorus = np.fft.irfft(
        Y * ((freqs > 200) & (freqs < 800)).astype("float32"), n=n,
    ).astype(np.float32)
    y += chorus
    y += 0.05 * rng.randn(n).astype(np.float32)
    y = y / max(1.0, float(np.max(np.abs(y))) * 1.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y, SR)
    return out


def _synth_ambient(out: Path, seed: int) -> Path:
    rng = np.random.RandomState(seed)
    n = SR * CLIP_SECONDS
    y = (0.1 * rng.randn(n)).astype(np.float32)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    y = np.fft.irfft(
        Y * (1.0 / (1.0 + freqs / 100.0) ** 0.5).astype("float32"), n=n,
    ).astype(np.float32)
    y = y / max(1.0, float(np.max(np.abs(y))) * 1.05) * 0.4
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y, SR)
    return out


# ── inference helpers ──────────────────────────────────────────────────


def _raw_cnn_predict(clf: CNNV7Classifier, clip: Path,
                     offset: float = 0.0) -> dict:
    y, sr = librosa.load(
        str(clip), sr=SR, mono=True,
        offset=offset, duration=CLIP_SECONDS,
    )
    if y.size < sr * 5:
        return {"ok": False, "error": "too short"}
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=1000)
    spec = librosa.power_to_db(mel, ref=1.0)
    pred = clf.predict(spec, source_id="eval-unseen")
    p = pred["confidence"] if pred["label"] == "ship" else 1.0 - pred["confidence"]
    return {
        "ok": True,
        "ship_prob": float(p),
        "label": pred["label"],
        "uncertainty": float(pred["uncertainty"]),
    }


# ── runner ─────────────────────────────────────────────────────────────


def _ship_label_from_tier(tier: str) -> str:
    if tier in {"DARK_VESSEL", "CONFIRMED_VESSEL", "ACOUSTIC_ONLY_LOW"}:
        return "ship"
    if tier == "AMBIENT":
        return "not_ship"
    return "uncertain"


def main() -> None:
    print("=" * 76)
    print("HELD-OUT EVALUATION — SAMPLES NEVER IN TRAINING")
    print("=" * 76)

    # Build the eval set
    eval_set: list[dict] = []

    # IN-DISTRIBUTION + UNSEEN (DeepShip clips not in any training jsonl)
    for label, p, offset in _find_unseen_deepship(per_class=5):
        eval_set.append({
            "group": "in_dist_unseen", "label": label, "path": p,
            "offset": offset,
        })

    # OOD synthetic reef (a site we have NEVER trained on)
    reef_path = Path("data/synthetic/eval_reef.wav")
    _synth_reef(reef_path)
    eval_set.append({
        "group": "ood_reef", "label": "not_ship", "path": reef_path,
    })

    # AMBIENT sanity floor
    amb1 = Path("data/synthetic/eval_ambient_a.wav")
    amb2 = Path("data/synthetic/eval_ambient_b.wav")
    _synth_ambient(amb1, seed=101)
    _synth_ambient(amb2, seed=202)
    eval_set.append({"group": "ambient_synth", "label": "not_ship", "path": amb1})
    eval_set.append({"group": "ambient_synth", "label": "not_ship", "path": amb2})

    print(f"\nEval clips: {len(eval_set)}")

    # Run inference (raw CNN with NO per-site adapter — most pessimistic)
    clf = CNNV7Classifier(CKPT)
    clf.set_site_adapter(None)

    rows = []
    for item in eval_set:
        offset = item.get("offset", 0.0)
        raw = _raw_cnn_predict(clf, item["path"], offset=offset)
        # full pipeline doesn't take offset; uses the default 60s window
        full = simulate_detection(site_id="eval-unseen", clip=str(item["path"]))
        if not raw.get("ok") or not full.get("ok"):
            continue
        raw_pred = "ship" if raw["ship_prob"] >= 0.5 else "not_ship"
        full_label = _ship_label_from_tier(full.get("decision_tier", ""))
        rows.append({
            **item,
            "offset":     offset,
            "ship_prob":  raw["ship_prob"],
            "uncertainty": raw["uncertainty"],
            "raw_pred":   raw_pred,
            "raw_correct": raw_pred == item["label"],
            "tier":       full["decision_tier"],
            "full_pred":  full_label,
            "full_correct": full_label == item["label"],
        })

    # ── per-clip table ────────────────────────────────────────────────
    print(f"\n{'─' * 76}\nPer-clip breakdown\n{'─' * 76}")
    print(f"{'group':<16} {'label':<8} {'off':>4} {'p':>6} {'unc':>5} "
          f"{'raw':<8} {'tier':<20} {'r✓':>2} {'f✓':>2}  clip")
    for r in rows:
        rok = "✓" if r["raw_correct"] else "✗"
        fok = "✓" if r["full_correct"] else "✗"
        off = f"{r.get('offset', 0):.0f}s"
        path_label = Path(r["path"]).parent.name + "/" + Path(r["path"]).name
        print(f"{r['group']:<16} {r['label']:<8} {off:>4} "
              f"{r['ship_prob']:>6.2f} {r['uncertainty']:>5.2f} "
              f"{r['raw_pred']:<8} {r['tier']:<20} {rok:>2} {fok:>2}  "
              f"{path_label}")

    # ── per-group accuracy ────────────────────────────────────────────
    groups = sorted({r["group"] for r in rows})
    print(f"\n{'─' * 76}\nAccuracy by group\n{'─' * 76}")
    print(f"{'group':<22} {'n':>4} {'raw_acc':>10} {'full_acc':>10}")
    for g in groups:
        gr = [r for r in rows if r["group"] == g]
        if not gr:
            continue
        n = len(gr)
        raw_n = sum(r["raw_correct"] for r in gr)
        full_n = sum(r["full_correct"] for r in gr)
        print(f"{g:<22} {n:>4} "
              f"{raw_n}/{n} {raw_n / n * 100:>5.1f}%  "
              f"{full_n}/{n} {full_n / n * 100:>5.1f}%")

    # ── overall ────────────────────────────────────────────────────────
    n = len(rows)
    raw_n = sum(r["raw_correct"] for r in rows)
    full_n = sum(r["full_correct"] for r in rows)
    print(f"\n{'═' * 76}\n  OVERALL")
    print(f"  raw CNN:        {raw_n}/{n} = {raw_n / n * 100:.1f}%")
    print(f"  full pipeline:  {full_n}/{n} = {full_n / n * 100:.1f}%")
    print(f"  ↑ full pipeline counts UNCERTAIN as fail (no decision made)")
    print(f"{'═' * 76}")


if __name__ == "__main__":
    main()
