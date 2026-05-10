"""Live smoke test: simulate_detection → explain_decision (multimodal).

Runs the real CNN v7.4 pipeline on two audio clips (a known-vessel and a
quieter clip) and asks Gemma 4 multimodal to narrate each result.

Usage:
    PYTHONPATH=src venv/bin/python scripts/smoke_explain_decision.py
"""
from __future__ import annotations

import json
from pathlib import Path

from ocean_sentinel.gemma.cnn_inference import simulate_detection
from ocean_sentinel.gemma.explanations import explain_decision_real

CLIPS = [
    ("vessel-tug", "data/deepship/Tug/49.wav"),
    ("ambient-mbari", "data/mbari/sample_60s.wav"),
]


def run_one(label: str, clip: str) -> None:
    print(f"\n{'=' * 70}\n{label}: {clip}\n{'=' * 70}")
    if not Path(clip).exists():
        print(f"  SKIP: clip not found")
        return

    det = simulate_detection(site_id="smoke-test", clip=clip)
    print(f"\n[simulate_detection]")
    print(json.dumps(
        {k: v for k, v in det.items() if k != "summary"},
        indent=2,
    ))
    if not det.get("ok"):
        return

    decision_id = det["decision_id"]
    print(f"\n[explain_decision]  (calling Gemma multimodal — may take ~20s)")
    out = explain_decision_real(decision_id, modality="spectrogram+text")

    print(f"\n  ok:               {out.get('ok')}")
    print(f"  modality:         {out.get('modality')}")
    print(f"  narration_source: {out.get('narration_source')}")
    print(f"  spectrogram_path: {out.get('spectrogram_path')}")
    print(f"\n  trace:")
    for k, v in out.get("trace", {}).items():
        print(f"    {k}: {v}")
    print(f"\n  explanation:")
    print(f"    {out.get('explanation')}")
    print(f"\n  summary: {out.get('summary')}")


if __name__ == "__main__":
    for label, clip in CLIPS:
        run_one(label, clip)
