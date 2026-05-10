"""Real `explain_decision` — Gemma 4 multimodal narration of a CNN decision.

What this replaces
------------------
Until now `explain_decision` returned a canned trace (made-up blade-rate,
harmonics, SNR). That worked as a placeholder but was the only tool in the
14-tool flow that fabricated numbers — uncomfortable for a system whose
core promise is "every numerical claim comes from a tool result".

What this does instead
----------------------
1. Looks up the decision record persisted by `simulate_detection` at
   `data/decisions/{decision_id}.json`.
2. Renders the clip's mel-spectrogram (same params as CNN training:
   128 mels, fmax=1 kHz, log-dB, ref=1.0) as a PNG.
3. Computes a small set of *real* spectral features that we can stand
   behind: peak frequency, spectral centroid, spectral flatness, and
   low-band (0–200 Hz, where vessel signatures live) energy fraction.
4. Calls Gemma 4 multimodal with the PNG + the numeric trace and asks
   for a 1–2 sentence operator-friendly explanation. The narration is
   *grounded in the image the model can see*, not in invented features.

Why this matters for the submission
-----------------------------------
This is the most direct showcase of Gemma 4's native multimodal
capability inside the system: the same model that orchestrates the
14-tool flow also reads our spectrograms. If Ollama is unreachable
or multimodal fails, we degrade gracefully — return the real trace
and a templated explanation, never a fabricated one.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import librosa
import matplotlib
matplotlib.use("Agg")  # headless: never open a GUI window from a tool call
import matplotlib.pyplot as plt
import numpy as np

DECISIONS_DIR = Path("data/decisions")

# Match CNN v7 training preprocessing.
_TARGET_SR_HZ = 16_000
_N_MELS = 128
_FMAX_HZ = 1_000
_MAX_DURATION_S = 60.0

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "gemma4:e4b"


# ── persistence: write the decision record so explain_decision can find it ─

def write_decision_record(decision_id: str, record: dict[str, Any]) -> Path:
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = DECISIONS_DIR / f"{decision_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def _load_decision_record(decision_id: str) -> dict[str, Any] | None:
    path = DECISIONS_DIR / f"{decision_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ── spectrogram rendering ────────────────────────────────────────────────

def _render_spectrogram_png(clip_path: Path, out_path: Path) -> None:
    """Render the clip's log-mel spectrogram to PNG.

    Matches CNN v7 training preprocessing: 16 kHz, 128 mels, fmax=1 kHz,
    log-power dB with ref=1.0. So the image Gemma reads is the *same view*
    the CNN sees.
    """
    y, sr = librosa.load(
        str(clip_path), sr=_TARGET_SR_HZ, mono=True, duration=_MAX_DURATION_S,
    )
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=_N_MELS, fmax=_FMAX_HZ)
    log_mel = librosa.power_to_db(mel, ref=1.0)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    img = librosa.display.specshow(
        log_mel, sr=sr, fmax=_FMAX_HZ, x_axis="time", y_axis="mel", ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"{clip_path.name} — log-mel (fmax={_FMAX_HZ} Hz)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ── real spectral features (no fabrication) ──────────────────────────────

def _compute_spectral_features(clip_path: Path) -> dict[str, Any]:
    """Compute a small set of features we can stand behind.

    Deliberately omits things we cannot reliably measure on noisy ocean
    audio (blade-rate, harmonic series). Vessel signatures concentrate
    in the 20–200 Hz band, so `low_band_energy_fraction` is the most
    interpretable single number.

    We ignore everything below 20 Hz: hydrophones routinely have DC
    offset and sub-acoustic platform motion that dominates the raw
    spectrum and makes naive peak/centroid reports misleading.
    """
    y, sr = librosa.load(
        str(clip_path), sr=_TARGET_SR_HZ, mono=True, duration=_MAX_DURATION_S,
    )

    # Power spectrum, averaged across time.
    n_fft = 4096
    hop = 1024
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2  # (freq, frames)
    power_avg = S.mean(axis=1) + 1e-12
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Restrict analysis to audio band: drop DC + sub-20 Hz noise.
    audio_mask = freqs >= 20.0
    a_freqs = freqs[audio_mask]
    a_power = power_avg[audio_mask]

    peak_idx = int(np.argmax(a_power))
    peak_hz = float(a_freqs[peak_idx])

    centroid = float((a_freqs * a_power).sum() / a_power.sum())

    flatness = float(np.exp(np.log(a_power).mean()) / a_power.mean())

    low_band_mask = (a_freqs >= 20.0) & (a_freqs < 200.0)
    low_band_fraction = float(a_power[low_band_mask].sum() / a_power.sum())

    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    rms_db = float(20 * np.log10(rms))

    return {
        "peak_frequency_hz": round(peak_hz, 1),
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_flatness": round(flatness, 3),
        "low_band_energy_fraction_below_200hz": round(low_band_fraction, 3),
        "rms_db": round(rms_db, 1),
        "duration_s": round(float(librosa.get_duration(y=y, sr=sr)), 1),
    }


# ── Gemma multimodal narration ───────────────────────────────────────────

_NARRATION_PROMPT_TEMPLATE = """\
You are reading ONE log-mel spectrogram of an underwater hydrophone
recording and writing a one- or two-sentence explanation for a
non-technical reserve operator.

# How to read this image

- X axis: time, left → right (clip is {duration_s:.0f} seconds long).
- Y axis: frequency on a mel scale, 0 Hz at the bottom, ~1000 Hz at the top.
- Brighter pixels = louder at that (time, frequency).
- The bottom strip of the image (below ~200 Hz on the mel axis, i.e.
  the lowest ~30% of the y-axis) is where vessel signatures live.

# What vessel signatures look like

A vessel typically shows up as **horizontal bright bands** at low
frequency that **persist across most of the time axis** — a steady,
narrow-band tone or a small set of stacked tones (engine + propeller
harmonics). The lower part of the image looks brighter than the upper
part, often noticeably so.

# What ambient (no vessel) looks like

Ambient looks **textured but unstructured**: brightness scattered or
roughly uniform across frequency, no persistent horizontal bands, often
varying or patchy across time (waves, rain, biological clicks). The
bottom strip is *not* dramatically brighter than the rest.

# Numeric context (from our pipeline — do NOT contradict)

- decision tier:               {tier} ({severity})
- CNN ship probability:        {ship_prob:.2f}  (threshold {threshold:.2f})
- conformal pass:              {conformal_pass}
- AIS vessels in radius:       {ais}
- peak frequency:              {peak_hz:.0f} Hz
- spectral centroid:           {centroid_hz:.0f} Hz
- low-band energy (<200 Hz):   {low_band_pct:.0f}% of total power
- spectral flatness:           {flatness:.2f}  (1.0 = white noise, 0 = pure tone)

# Your task

Look at the image. In **at most two sentences**:

1. Say what you actually see (e.g. "a steady bright band near the bottom
   across most of the clip" / "diffuse energy with no persistent
   horizontal band").
2. Say why that is or isn't consistent with the {tier} label.

Hard rules:
- Do NOT invent specific blade rates, exact harmonic frequencies, or a
  vessel type (tanker, fishing, etc.) — you cannot tell those from one
  spectrogram.
- Reference AT MOST one of the numeric values above.
- If what you see and the label disagree, say so plainly ("the
  spectrogram looks ambiguous despite the {tier} label").
- No greeting, no preamble, no markdown headers. Two sentences max.
"""


def _call_gemma_multimodal(
    *,
    image_path: Path,
    context: dict[str, Any],
    host: str,
    model: str,
    timeout_s: float = 60.0,
) -> str | None:
    """Call Gemma with the spectrogram + numeric context. Returns the text
    body, or None on any failure (ollama down, model timeout, etc.)."""
    try:
        import ollama
    except Exception:
        return None

    prompt = _NARRATION_PROMPT_TEMPLATE.format(
        tier=context.get("decision_tier", "UNKNOWN"),
        severity=context.get("severity", "n/a"),
        ship_prob=float(context.get("cnn_confidence", 0.0)),
        threshold=float(context.get("conformal_threshold", 0.0)),
        conformal_pass=context.get("conformal_pass", False),
        ais=int(context.get("ais_vessels_in_radius", 0)),
        peak_hz=float(context.get("peak_frequency_hz", 0.0)),
        centroid_hz=float(context.get("spectral_centroid_hz", 0.0)),
        low_band_pct=100.0 * float(context.get(
            "low_band_energy_fraction_below_200hz", 0.0,
        )),
        flatness=float(context.get("spectral_flatness", 0.0)),
        duration_s=float(context.get("duration_s", 0.0)),
    )

    try:
        client = ollama.Client(host=host, timeout=timeout_s)
        response = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }],
        )
    except Exception:
        return None

    msg = response.get("message", {}) or {}
    text = (msg.get("content") or "").strip()
    return text or None


def _templated_narration(features: dict[str, Any], decision: dict[str, Any]) -> str:
    """Honest fallback when multimodal is unavailable. Uses only real
    measured features — no invented blade-rate or harmonics."""
    tier = decision.get("decision_tier", "UNKNOWN")
    pct_low = 100.0 * float(features.get("low_band_energy_fraction_below_200hz", 0.0))
    centroid = float(features.get("spectral_centroid_hz", 0.0))
    if tier in {"DARK_VESSEL", "ACOUSTIC_ONLY_LOW"}:
        return (
            f"Clip energy concentrates below 200 Hz "
            f"({pct_low:.0f}% of total power; spectral centroid "
            f"{centroid:.0f} Hz), which is consistent with "
            f"a vessel-like low-frequency source."
        )
    if tier == "AMBIENT":
        return (
            f"Energy is spread across the band (centroid {centroid:.0f} Hz, "
            f"low-band fraction {pct_low:.0f}%) without a dominant low-frequency "
            f"signature."
        )
    return (
        f"Spectral centroid {centroid:.0f} Hz, low-band fraction "
        f"{pct_low:.0f}%. Evidence is mixed; flagging for review is appropriate."
    )


# ── main entry ───────────────────────────────────────────────────────────

def explain_decision_real(
    decision_id: str,
    modality: str = "spectrogram+text",
    *,
    host: str = _DEFAULT_HOST,
    model: str = _DEFAULT_MODEL,
    use_multimodal: bool | None = None,
) -> dict[str, Any]:
    """Real implementation of the explain_decision tool.

    Parameters
    ----------
    decision_id : str
        Returned previously by `simulate_detection`. Looked up at
        `data/decisions/{decision_id}.json`.
    modality : str
        "text" — features + templated narration only.
        "spectrogram" — render PNG, return path; no narration.
        "spectrogram+text" (default) — render PNG, call Gemma multimodal
        if reachable, fall back to templated narration if not.
    use_multimodal : bool | None
        Override for tests / debugging. If None, defaults to True iff
        modality requests text.

    Returns
    -------
    dict with `ok`, `decision_id`, `modality`, `trace` (real numeric features
    + decision context), `spectrogram_path` (when rendered), `explanation`,
    `narration_source` ('gemma_multimodal' or 'templated' or 'none'),
    and `summary`.
    """
    record = _load_decision_record(decision_id)
    if record is None:
        return {
            "ok": False,
            "error": (
                f"decision {decision_id} not found in {DECISIONS_DIR}. "
                f"Call simulate_detection first."
            ),
        }

    clip_path = Path(record.get("clip", ""))
    if not clip_path.exists():
        return {
            "ok": False,
            "error": f"clip referenced by decision is missing: {clip_path}",
        }

    if modality not in {"text", "spectrogram", "spectrogram+text"}:
        return {
            "ok": False,
            "error": (
                f"modality must be one of text|spectrogram|spectrogram+text, "
                f"got {modality}"
            ),
        }

    features = _compute_spectral_features(clip_path)

    spec_path: Path | None = None
    if modality in {"spectrogram", "spectrogram+text"}:
        spec_path = DECISIONS_DIR / f"{decision_id}.png"
        try:
            _render_spectrogram_png(clip_path, spec_path)
        except Exception as e:
            return {
                "ok": False,
                "error": f"spectrogram render failed: {type(e).__name__}: {e}",
            }

    want_text = modality in {"text", "spectrogram+text"}
    want_multimodal = (
        want_text
        and modality == "spectrogram+text"
        and (use_multimodal if use_multimodal is not None else True)
    )

    explanation: str | None = None
    narration_source = "none"

    if want_multimodal and spec_path is not None:
        ctx = {**record, **features}
        explanation = _call_gemma_multimodal(
            image_path=spec_path, context=ctx, host=host, model=model,
        )
        if explanation:
            narration_source = "gemma_multimodal"

    if want_text and explanation is None:
        explanation = _templated_narration(features, record)
        if narration_source == "none":
            narration_source = "templated"

    trace = {
        **features,
        "decision_tier": record.get("decision_tier"),
        "severity": record.get("severity"),
        "cnn_confidence": record.get("cnn_confidence"),
        "conformal_threshold": record.get("conformal_threshold"),
        "conformal_pass": record.get("conformal_pass"),
        "ais_vessels_in_radius": record.get("ais_vessels_in_radius"),
        "site_id": record.get("site_id"),
        "clip": str(clip_path),
    }

    summary_bits = [str(record.get("decision_tier", "UNKNOWN"))]
    if features.get("low_band_energy_fraction_below_200hz") is not None:
        summary_bits.append(
            f"{100*features['low_band_energy_fraction_below_200hz']:.0f}% energy <200Hz"
        )
    if narration_source == "gemma_multimodal":
        summary_bits.append("Gemma multimodal narration")

    return {
        "ok": True,
        "decision_id": decision_id,
        "modality": modality,
        "trace": trace,
        "spectrogram_path": str(spec_path) if spec_path else None,
        "explanation": explanation,
        "narration_source": narration_source,
        "summary": " · ".join(summary_bits),
    }


# Allow tests / scripts to skip the live Gemma call without monkeypatching.
def is_multimodal_disabled() -> bool:
    return os.environ.get("OS_DISABLE_MULTIMODAL", "").lower() in {"1", "true", "yes"}
