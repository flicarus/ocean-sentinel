"""Honest test of Gemma 4 multimodal spectrogram reading.

Validation question: does Gemma 4 actually understand what it's looking
at, or does it just regurgitate whatever the CNN tells it?

Test protocol — Gemma is given the SPECTROGRAM ONLY, with no hint of
what the CNN decided. We compare its verdict against the CNN's verdict
and the ground-truth label. Four cases:

  1. CNN confident SHIP (truth=ship)        → does Gemma agree?
  2. CNN confident AMBIENT (truth=not_ship) → does Gemma agree?
  3. CNN uncertain (truth=not_ship)         → does Gemma break tie correctly?
  4. CNN uncertain (truth=ship)             → does Gemma break tie correctly?

If Gemma's accuracy on these four is at chance, the Ocean Intelligence
System framing is dishonest — Gemma is decorative. If Gemma can
actually distinguish, especially on the uncertain cases, multimodal
verification adds real value.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")


def render_spectrogram(spec_path: Path, out_png: Path) -> None:
    """Render a stored mel spectrogram (.npy) to a PNG that mirrors how
    a marine acoustician would view it."""
    spec = np.load(spec_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("time bin (60 seconds total)")
    ax.set_ylabel("mel frequency bin (0 Hz bottom → ~1000 Hz top)")
    ax.set_title(f"Hydrophone log-mel spectrogram ({spec.shape[0]} mels × {spec.shape[1]} frames)")
    plt.colorbar(im, ax=ax, label="dB")
    plt.tight_layout()
    plt.savefig(out_png, dpi=100)
    plt.close()


# Honest prompt — does NOT mention CNN decision. Tells Gemma how to read
# the image, then asks for a verdict + reasoning.
PROMPT = """\
You are looking at one log-mel spectrogram of a 60-second underwater
hydrophone recording. Your job is to decide whether a vessel is present.

How to read this image:
  - X axis: time, left to right, 60 seconds total
  - Y axis: frequency on a mel scale, 0 Hz at the bottom, ~1000 Hz top
  - Brighter pixels = louder at that (time, frequency)
  - Vessel signatures live in the bottom strip (below ~200 Hz)

What a vessel looks like:
  - Horizontal bright bands at low frequency that persist across most
    of the time axis (steady engine + propeller harmonics)
  - The bottom of the image is noticeably brighter than the top
  - May see stacked tones (engine fundamental + harmonics)

What ambient (no vessel) looks like:
  - Texture without persistent horizontal structure
  - Brightness scattered or roughly uniform across frequency
  - Bottom strip not dramatically brighter than the rest
  - Possible vertical streaks (rain, biological clicks, waves)

YOUR TASK: Look at the image and answer in this exact format:

VERDICT: SHIP or AMBIENT
CONFIDENCE: 0.0 to 1.0
REASONING: 1-2 sentences citing specific image features (frequency
           range, time persistence, intensity contrast).

Be honest if uncertain — say AMBIENT with confidence 0.55 rather than
forcing a verdict you don't believe.
"""


def ask_gemma(image_path: Path, host: str = "http://localhost:11434", model: str = "gemma4:e4b", timeout_s: float = 90.0) -> dict:
    """Send the spectrogram to Gemma 4 multimodal via Ollama. Returns
    a dict with verdict / confidence / reasoning / raw_text / elapsed_s."""
    try:
        import ollama
    except Exception as e:
        return {"error": f"ollama import failed: {e}"}

    t0 = time.perf_counter()
    try:
        client = ollama.Client(host=host, timeout=timeout_s)
        response = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": PROMPT,
                "images": [str(image_path)],
            }],
        )
    except Exception as e:
        return {"error": f"ollama call failed: {type(e).__name__}: {e}"}

    elapsed = time.perf_counter() - t0
    raw = (response.get("message", {}) or {}).get("content", "")

    # Parse the structured response — Gemma may add prose around it.
    verdict = None
    confidence = None
    reasoning = ""
    for line in raw.splitlines():
        ls = line.strip()
        if ls.upper().startswith("VERDICT:"):
            v = ls.split(":", 1)[1].strip().upper()
            if "SHIP" in v:
                verdict = "SHIP"
            elif "AMBIENT" in v:
                verdict = "AMBIENT"
        elif ls.upper().startswith("CONFIDENCE:"):
            try:
                confidence = float(ls.split(":", 1)[1].strip().split()[0])
            except Exception:
                pass
        elif ls.upper().startswith("REASONING:"):
            reasoning = ls.split(":", 1)[1].strip()

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "raw_text": raw,
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    cases = [
        ("CNN CONFIDENT SHIP (truth=ship)",
         "data/spectrograms/shipsear_2_2_0_2_0_43.npy",
         "ship"),
        ("CNN CONFIDENT AMBIENT (truth=not_ship)",
         "data/spectrograms/sanctsound_sb01_SanctSound_SB01_01_1678032935_20181112T190000Z_Post-Deployment_11.npy",
         "not_ship"),
        ("CNN UNCERTAIN #1 (ship_prob=0.35, truth=not_ship)",
         "data/spectrograms/sanctsound_sb01_SanctSound_SB01_01_1678032935_20181113T002444Z_116.npy",
         "not_ship"),
        ("CNN UNCERTAIN #2 (ship_prob=0.35, truth=not_ship)",
         "data/spectrograms/shipsear_4_4_0_4_0_160.npy",
         "not_ship"),
    ]

    out_dir = Path("data/eval/gemma_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (label, spec_path, truth) in enumerate(cases):
        print(f"\n{'═' * 70}")
        print(f"  CASE {i+1}: {label}")
        print(f"  spec: {spec_path}")
        print(f"  ground truth: {truth}")
        print(f"{'═' * 70}")

        spec_path_obj = Path(spec_path)
        if not spec_path_obj.exists():
            print(f"  [skip] file missing")
            continue

        png_path = out_dir / f"case_{i+1}.png"
        render_spectrogram(spec_path_obj, png_path)
        print(f"  → rendered: {png_path}")

        print(f"  → asking Gemma 4 multimodal (no CNN hint)...")
        res = ask_gemma(png_path)
        if "error" in res:
            print(f"  [FAIL] {res['error']}")
            continue

        print(f"  Gemma verdict   : {res['verdict']}")
        print(f"  Gemma confidence: {res['confidence']}")
        print(f"  Gemma reasoning : {res['reasoning']}")
        print(f"  Gemma latency   : {res['elapsed_s']}s")
        print()
        print(f"  Raw output (truncated to 300 chars):")
        print(f"    {res['raw_text'][:300]}")

        truth_label = "SHIP" if truth == "ship" else "AMBIENT"
        match = (res["verdict"] == truth_label)
        print()
        print(f"  → Gemma matches ground truth? {'✅ YES' if match else '❌ NO'}")

        results.append({
            "case": label,
            "spec": spec_path,
            "truth": truth_label,
            "gemma_verdict": res["verdict"],
            "gemma_confidence": res["confidence"],
            "gemma_reasoning": res["reasoning"],
            "matches": match,
            "elapsed_s": res["elapsed_s"],
        })

    print(f"\n{'═' * 70}")
    print("  SUMMARY")
    print(f"{'═' * 70}")
    correct = sum(1 for r in results if r["matches"])
    total = len(results)
    print(f"  Gemma 4 accuracy on 4 held-out spectrograms: {correct}/{total}")
    for r in results:
        emoji = "✅" if r["matches"] else "❌"
        print(f"    {emoji} {r['case']:<50s} → Gemma said {r['gemma_verdict']} (truth {r['truth']})")

    # Save
    json_out = out_dir / "results.json"
    json_out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {json_out}")


if __name__ == "__main__":
    main()
