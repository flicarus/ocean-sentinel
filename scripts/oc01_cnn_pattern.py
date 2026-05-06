"""OC01 — does the CNN actually discriminate AIS-positive from AIS-negative?

PSD analysis showed that low-freq peaks at OC01 are not unique to vessels —
the site has near-constant 5-50 Hz spectral activity. So PSD alone cannot
separate AIS-positive from AIS-negative chunks.

But the CNN looks at >80 Hz only (engine harmonics + cavitation). If it
genuinely captures vessel-specific high-freq features, it should call
"ship" on AIS-positive chunks and "ambient" on AIS-negative chunks even
though their low-freq spectra look identical.

If it doesn't separate them, then CNN is probably just memorizing OC01
site signature and our "two independent channels" claim is empty.

Method:
  1. For each of 6 candidate hours (4 close-range AIS-positive,
     2 AIS-empty), extract 60s audio from FLAC.
  2. Resample to 16 kHz (training rate).
  3. Split into twelve 5-second chunks.
  4. Compute mel spectrogram per chunk (matches AudioAnalyzer params).
  5. Run cnn_v6.pt with source_id='sanctsound' on each.
  6. Report per-hour: mean ship-prob, fraction classified as ship,
     max ship-prob.
  7. Render bar chart comparing groups.

Output: data/diagnostic/sanctsound/audio/cnn_pattern_results.png
        + stdout summary table.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, "src")
from ocean_sentinel.services.cnn_classifier import CNNClassifier


FLAC_DIR = Path("data/sanctsound/oc01")
FLAC_FILES = {
    datetime(2019, 3, 8, 19, 0, 0, tzinfo=timezone.utc):
        FLAC_DIR / "SanctSound_OC01_01_671399974_20190308T190000Z.flac",
    datetime(2019, 3, 8, 23, 59, 55, tzinfo=timezone.utc):
        FLAC_DIR / "SanctSound_OC01_01_671399974_20190308T235955Z.flac",
    datetime(2019, 3, 9, 5, 59, 52, tzinfo=timezone.utc):
        FLAC_DIR / "SanctSound_OC01_01_671399974_20190309T055952Z.flac",
    datetime(2019, 3, 9, 11, 59, 49, tzinfo=timezone.utc):
        FLAC_DIR / "SanctSound_OC01_01_671399974_20190309T115949Z.flac",
}

# From oc01_psd_pattern.py output. The disputed window is 12:14-12:44 UTC
# (10 misclassified chunks per the original analysis). Other close hours
# only know "≤10km vessel during the hour" — vessel could be present any
# minute within that hour, so we widen the sweep to catch it.
GROUPS = {
    "close": [
        # DISPUTED — sample the known misclassified window directly.
        datetime(2019, 3, 9, 12, 14, 0, tzinfo=timezone.utc),
        datetime(2019, 3, 8, 19, 5, 0, tzinfo=timezone.utc),
        datetime(2019, 3, 8, 23, 5, 0, tzinfo=timezone.utc),
        datetime(2019, 3, 9, 0, 5, 0, tzinfo=timezone.utc),
    ],
    "empty": [
        datetime(2019, 3, 9, 4, 5, 0, tzinfo=timezone.utc),
        datetime(2019, 3, 9, 9, 5, 0, tzinfo=timezone.utc),
    ],
}

CKPT = Path("data/models/cnn_v6.pt")
CHUNK_SECONDS = 5
SWEEP_SECONDS = 300  # 5 min = 60 chunks per hour
TARGET_SR = 16000
N_MELS = 128
F_MAX = 1000.0
OUT_PNG = Path("data/diagnostic/sanctsound/audio/cnn_pattern_results.png")


def find_audio_offset(target):
    candidates = []
    for start, path in FLAC_FILES.items():
        delta_s = (target - start).total_seconds()
        if delta_s < 0:
            continue
        if delta_s + 600 > 6 * 3600:
            continue
        candidates.append((path, delta_s))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1])
    return candidates[0]


def load_and_resample(path, offset_seconds, duration_seconds):
    info = sf.info(str(path))
    sr = info.samplerate
    start_frame = int(offset_seconds * sr)
    n_frames = int(duration_seconds * sr)
    audio, _ = sf.read(str(path), start=start_frame,
                       frames=n_frames, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio


def make_spec(samples, sr=TARGET_SR):
    """Match AudioAnalyzer._make_spectrogram exactly."""
    mel = librosa.feature.melspectrogram(
        y=samples, sr=sr, n_mels=N_MELS, fmax=F_MAX,
    )
    return librosa.power_to_db(mel, ref=1.0)


def main():
    if not CKPT.exists():
        sys.exit(f"missing checkpoint {CKPT}")

    print(f"Loading CNN: {CKPT}")
    clf = CNNClassifier(CKPT)
    print()

    rows: list[dict] = []
    for group, hours in GROUPS.items():
        for h in hours:
            match = find_audio_offset(h)
            if match is None:
                print(f"  [skip] {h.isoformat()} — no audio coverage")
                continue
            path, base_offset = match
            sweep_offset = base_offset
            print(f"== {group:5s}  {h.strftime('%m-%d %H:00 UTC')}  "
                  f"file={path.name}, offset={sweep_offset:.0f}s ==")

            audio = load_and_resample(path, sweep_offset, SWEEP_SECONDS)
            samples_per_chunk = CHUNK_SECONDS * TARGET_SR
            n_chunks = SWEEP_SECONDS // CHUNK_SECONDS

            chunk_probs = []
            for i in range(n_chunks):
                chunk = audio[i * samples_per_chunk:(i + 1) * samples_per_chunk]
                if len(chunk) < samples_per_chunk:
                    break
                spec = make_spec(chunk)
                # spec should be (128, ~157). predict() handles crop.
                result = clf.predict(spec, source_id="sanctsound")
                p_ship = result["probabilities"]["ship"]
                chunk_probs.append(p_ship)
            # Print only chunks with notable probability to keep stdout sane.
            print(f"  ran {n_chunks} chunks; chunks where ship_p>0.5:")
            ship_count = sum(1 for p in chunk_probs if p > 0.5)
            for i, p in enumerate(chunk_probs):
                if p > 0.5:
                    print(f"    chunk {i:3d} (t+{i*CHUNK_SECONDS:3d}s): ship_p={p:.3f}")
            if ship_count == 0:
                print("    (none)")

            arr = np.array(chunk_probs)
            stats = {
                "group": group,
                "hour": h,
                "mean_ship_p": float(arr.mean()),
                "max_ship_p": float(arr.max()),
                "min_ship_p": float(arr.min()),
                "frac_ship": float((arr > 0.5).mean()),
                "n_chunks": len(arr),
                "all_probs": arr,
            }
            print(
                f"  -> mean={stats['mean_ship_p']:.3f}  "
                f"max={stats['max_ship_p']:.3f}  "
                f"frac>0.5={stats['frac_ship']:.2f}  "
                f"({stats['n_chunks']} chunks)\n"
            )
            rows.append(stats)

    print("\n== Summary table ==")
    print(f"{'group':<7s}  {'hour':<14s}  {'mean p':>7s}  {'max p':>7s}  "
          f"{'frac>0.5':>9s}  {'n':>3s}")
    for r in rows:
        print(
            f"{r['group']:<7s}  {r['hour'].strftime('%m-%d %H:00 UTC'):<14s}  "
            f"{r['mean_ship_p']:>7.3f}  {r['max_ship_p']:>7.3f}  "
            f"{r['frac_ship']:>9.2f}  {r['n_chunks']:>3d}"
        )

    print("\n== Group means ==")
    for g in ("close", "empty"):
        gs = [r for r in rows if r["group"] == g]
        if not gs:
            continue
        all_probs = np.concatenate([r["all_probs"] for r in gs])
        print(
            f"  {g:5s}:  mean ship_p across all chunks = "
            f"{all_probs.mean():.3f}  ({len(all_probs)} chunks total)  "
            f"frac>0.5 = {(all_probs > 0.5).mean():.2f}"
        )

    print("\n== Render plot ==")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: per-hour bars (mean + max)
    ax = axes[0]
    labels = [f"{r['group']}\n{r['hour'].strftime('%m-%d %H:00')}" for r in rows]
    x = np.arange(len(rows))
    means = [r["mean_ship_p"] for r in rows]
    maxes = [r["max_ship_p"] for r in rows]
    colors = ["crimson" if r["group"] == "close" else "steelblue" for r in rows]

    width = 0.4
    ax.bar(x - width / 2, means, width, color=colors, alpha=0.6, label="mean ship_p")
    ax.bar(x + width / 2, maxes, width, color=colors, alpha=1.0, label="max ship_p")
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", label="ship threshold (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("CNN ship probability")
    ax.set_title(
        "CNN ship probability per hour (12 × 5s chunks each).\n"
        "Red = close-range (AIS≤10km).  Blue = empty (no AIS in 50km).",
        fontsize=10,
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Right: chunk-level distributions
    ax = axes[1]
    for r in rows:
        c = "crimson" if r["group"] == "close" else "steelblue"
        ax.scatter(
            np.full_like(r["all_probs"], rows.index(r)),
            r["all_probs"], color=c, alpha=0.7, s=40,
        )
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("CNN ship probability per 5s chunk")
    ax.set_title(
        "Per-chunk ship probability — does the spread differ between groups?",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "OC01 — does the CNN discriminate AIS-positive from AIS-negative?",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT_PNG}")


if __name__ == "__main__":
    main()
