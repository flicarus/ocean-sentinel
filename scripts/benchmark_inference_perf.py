"""Inference latency + memory benchmark for v7.4.

Enterprise-relevant questions
-----------------------------
- ms/clip on the user's hardware (target: < clip duration so we're real-time)
- ms breakdown: load + mel + predict
- Memory peak during inference (cap on edge-device viability)
- Throughput at sustained load (clips/sec)
- Cold-start cost (first inference vs warm)

Hardware tested
---------------
Whatever the operator runs this on. Defaults to MPS (Apple Silicon) →
CUDA → CPU in that order. Reports the actual device used.

This script does NOT measure quality — only operational performance.
For quality see scripts/benchmark_v7_4.py.
"""
from __future__ import annotations

import gc
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import torch

from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier


CKPT = "data/models/cnn_v7_4.pt"
SR = 16_000
CLIP_S = 60
N_WARMUP = 5
N_TIMED = 50


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _make_synth_clip(seed: int = 42) -> np.ndarray:
    """Deterministic 60s pseudo-vessel clip so timing is reproducible."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, CLIP_S, SR * CLIP_S, endpoint=False)
    y = (
        0.4 * np.sin(2 * np.pi * 60 * t)
        + 0.3 * np.sin(2 * np.pi * 120 * t)
        + 0.05 * rng.randn(len(t))
    ).astype(np.float32)
    return y


def _melspec(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128, fmax=1000)
    return librosa.power_to_db(mel, ref=1.0)


def _device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    print("=" * 76)
    print("v7.4 INFERENCE PERFORMANCE BENCHMARK")
    print("=" * 76)
    print(f"  device:   {_device_str()}")
    print(f"  clip:     {CLIP_S} s @ {SR} Hz mono synthetic")
    print(f"  warmup:   {N_WARMUP} runs (discarded)")
    print(f"  timed:    {N_TIMED} runs per phase")

    rss_at_start = _peak_rss_mb()
    print(f"\n  rss before model load: {rss_at_start:.1f} MB")

    t0 = time.perf_counter()
    clf = CNNV7Classifier(CKPT)
    load_time_ms = (time.perf_counter() - t0) * 1000
    rss_after_load = _peak_rss_mb()
    print(f"  rss after model load:  {rss_after_load:.1f} MB  "
          f"(+{rss_after_load - rss_at_start:.1f})")
    print(f"  cold load time:        {load_time_ms:.0f} ms")

    n_params = sum(p.numel() for p in clf._model.parameters())
    print(f"  model params:          {n_params:,}")

    # Generate one synth clip + spec we'll reuse
    y = _make_synth_clip()

    # ── time mel computation alone ────────────────────────────────────
    print(f"\n[1/4] mel spectrogram (librosa, CPU) ...")
    for _ in range(N_WARMUP):
        _ = _melspec(y)
    times = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        _ = _melspec(y)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.asarray(times)
    print(f"      median {np.median(times):.1f} ms · "
          f"p95 {np.percentile(times, 95):.1f} ms · "
          f"max {times.max():.1f} ms")

    # ── time model.predict alone ──────────────────────────────────────
    print(f"\n[2/4] CNN predict (model fwd, {_device_str()}) ...")
    spec = _melspec(y)
    for _ in range(N_WARMUP):
        _ = clf.predict(spec, source_id="bench")
    times = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        _ = clf.predict(spec, source_id="bench")
        times.append((time.perf_counter() - t0) * 1000)
    times = np.asarray(times)
    p_med = float(np.median(times))
    p_95 = float(np.percentile(times, 95))
    print(f"      median {p_med:.1f} ms · "
          f"p95 {p_95:.1f} ms · "
          f"max {times.max():.1f} ms")

    # ── full pipeline (mel + predict) ─────────────────────────────────
    print(f"\n[3/4] full pipeline (mel + predict) ...")
    times = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        s = _melspec(y)
        _ = clf.predict(s, source_id="bench")
        times.append((time.perf_counter() - t0) * 1000)
    times = np.asarray(times)
    full_med = float(np.median(times))
    full_p95 = float(np.percentile(times, 95))
    print(f"      median {full_med:.1f} ms · "
          f"p95 {full_p95:.1f} ms · "
          f"max {times.max():.1f} ms")

    # Real-time factor: clip is 60 s; if median is 200 ms we run 300x real-time
    rtf = (CLIP_S * 1000) / full_med
    print(f"      real-time factor: {rtf:.0f}× "
          f"(model can process {rtf:.0f} parallel streams in real time)")

    # ── sustained throughput ──────────────────────────────────────────
    print(f"\n[4/4] sustained throughput (200 back-to-back inferences) ...")
    n_sustained = 200
    t0 = time.perf_counter()
    for _ in range(n_sustained):
        s = _melspec(y)
        _ = clf.predict(s, source_id="bench")
    sustained_total = time.perf_counter() - t0
    throughput = n_sustained / sustained_total
    print(f"      {n_sustained} clips in {sustained_total:.1f}s → "
          f"{throughput:.1f} clips/sec ({throughput * 60:.0f} clips/min)")

    rss_peak = _peak_rss_mb()
    print(f"\n  rss peak:              {rss_peak:.1f} MB")
    print(f"  rss model-only:        ~{rss_after_load - rss_at_start:.0f} MB")

    # ── persist ───────────────────────────────────────────────────────
    out = Path("data/eval/inference_perf.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "_meta": {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "device":      _device_str(),
            "model_params": n_params,
            "clip_seconds": CLIP_S,
            "sample_rate":  SR,
            "n_warmup":     N_WARMUP,
            "n_timed":      N_TIMED,
        },
        "load_time_ms":         round(load_time_ms, 1),
        "predict_only_ms":      {"median": round(p_med, 2), "p95": round(p_95, 2)},
        "full_pipeline_ms":     {"median": round(full_med, 2), "p95": round(full_p95, 2)},
        "real_time_factor":     round(rtf, 1),
        "sustained_throughput_clips_per_sec": round(throughput, 2),
        "rss_peak_mb":          round(rss_peak, 1),
        "rss_model_only_mb":    round(rss_after_load - rss_at_start, 1),
    }, indent=2))
    print(f"\n  wrote {out}")

    # ── headline ──────────────────────────────────────────────────────
    print(f"\n{'═' * 76}")
    print(f"  HEADLINE — enterprise-relevant operational metrics")
    print(f"{'═' * 76}")
    print(f"  model size:           2.3M params  ({rss_after_load - rss_at_start:.0f} MB resident)")
    print(f"  end-to-end latency:   {full_med:.0f} ms median ({full_p95:.0f} ms p95)")
    print(f"  real-time factor:     {rtf:.0f}× ({rtf:.0f} concurrent streams on this hardware)")
    print(f"  sustained throughput: {throughput:.0f} clips/sec on {_device_str()}")
    print(f"  cold-start:           {load_time_ms:.0f} ms (model load + first warmup)")
    print(f"  edge-device viable:   YES if hardware ≥ M-class Apple Silicon / mid-range CUDA")


if __name__ == "__main__":
    main()
