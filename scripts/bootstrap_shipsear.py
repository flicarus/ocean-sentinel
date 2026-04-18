"""Bootstrap CNN training data from the ShipsEar dataset.

Uses ShipsEar's human-annotated tags as the ground-truth label (from the
Santos-Dominguez et al. 2016 paper) -- NOT Gemma. Today's Gemma-via-Ollama
stack is too unreliable for labeling and its proper role is Tier-3 reasoner
at inference, not Tier-1 detector at training time.

For each 5-second wav:
  1. Load with soundfile, build AudioSegment with Vigo-harbor synthetic metadata
  2. AudioAnalyzer -> absolute-dB mel spectrogram + acoustic features
  3. Save .npy to data/spectrograms/
  4. Append JSONL row with threat_level from the ShipsEar class mapping

Idempotent: re-running skips event_ids already present in gemma_labels.jsonl.

Usage:
    PYTHONPATH=src venv/bin/python scripts/bootstrap_shipsear.py \\
        --src shipsear_5s_16k --limit 20   # sanity run
    PYTHONPATH=src venv/bin/python scripts/bootstrap_shipsear.py \\
        --src shipsear_5s_16k              # full ~2,223 files
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import (
    AudioSegment,
    GeoPoint,
    TimeWindow,
)
from ocean_sentinel.services.audio_analyzer import AudioAnalyzer

log = structlog.get_logger()

VIGO = GeoPoint(lat=42.24, lon=-8.72)
BASE_TIME = datetime(2013, 6, 1, tzinfo=timezone.utc)

# ShipsEar class idx -> (letter, ThreatLevel) per Santos-Dominguez 2016.
# Taxonomy:
#   A = fishing boats, trawlers, dredgers     -> MEDIUM (commercial activity)
#   B = motorboats, sailboats, pilot boats    -> LOW    (recreational)
#   C = passenger ferries, RORO               -> MEDIUM (routine commercial)
#   D = ocean liners, cargo, cruise ships     -> HIGH   (large commercial)
#   E = background / natural ambient          -> NONE
CLASS_MAP: dict[int, tuple[str, str]] = {
    0: ("A", "MEDIUM"),
    1: ("B", "LOW"),
    2: ("C", "MEDIUM"),
    3: ("D", "HIGH"),
    4: ("E", "NONE"),
}

JSONL_PATH = Path("data/training/gemma_labels.jsonl")
SPEC_DIR = Path("data/spectrograms")


def walk_dataset(root: Path) -> list[tuple[Path, int]]:
    out: list[tuple[Path, int]] = []
    for cls in sorted(CLASS_MAP):
        cls_dir = root / str(cls)
        if not cls_dir.exists():
            log.warning("bootstrap_class_dir_missing", path=str(cls_dir))
            continue
        for wav in sorted(cls_dir.rglob("*.wav")):
            out.append((wav, cls))
    return out


def build_segment(wav_path: Path, idx: int) -> AudioSegment:
    samples, sr = sf.read(wav_path, dtype="float32")
    duration_s = len(samples) / sr
    start = BASE_TIME + timedelta(seconds=idx * 10)
    return AudioSegment(
        source_file=f"shipsear/{wav_path.parent.name}/{wav_path.name}",
        location=VIGO,
        time_window=TimeWindow(start=start, end=start + timedelta(seconds=duration_s)),
        sample_rate=sr,
        samples=samples,
    )


def event_id_for(wav_path: Path, cls: int) -> str:
    return f"shipsear_{cls}_{wav_path.parent.name}_{wav_path.stem}"


def load_already_done(jsonl: Path) -> set[str]:
    if not jsonl.exists():
        return set()
    done: set[str] = set()
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["event_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def main(src: Path, limit: int) -> None:
    settings = Settings()
    analyzer = AudioAnalyzer(settings)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    done = load_already_done(JSONL_PATH)
    pairs = walk_dataset(src)
    if limit > 0:
        pairs = pairs[:limit]

    log.info(
        "bootstrap_started",
        total=len(pairs),
        already_done=len(done),
        src=str(src),
    )

    written = 0
    skipped = 0
    failed = 0
    class_count: dict[str, int] = {}

    with JSONL_PATH.open("a") as jsonl_file:
        for i, (wav_path, cls) in enumerate(pairs):
            eid = event_id_for(wav_path, cls)
            if eid in done:
                skipped += 1
                continue

            letter, threat_level = CLASS_MAP[cls]
            try:
                segment = build_segment(wav_path, i)
                analyzed, features = analyzer.analyze(segment)

                spec_path = SPEC_DIR / f"{eid}.npy"
                np.save(spec_path, analyzed.spectrogram)

                entry = {
                    "event_id": eid,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "spectrogram_path": str(spec_path),
                    "features": {
                        "engine_band_ratio": features["engine_band_ratio"],
                        "peak_frequency_hz": features["peak_frequency_hz"],
                        "spectral_flatness": features["spectral_flatness"],
                        "rms_energy": features["rms_energy"],
                        "engine_band_energy_db": features["engine_band_energy_db"],
                    },
                    "context_text": (
                        f"shipsear-bootstrap | class={letter} | "
                        f"source_file={segment.source_file}"
                    ),
                    "gemma_verdict": {
                        "threat_level": threat_level,
                        "confidence": 1.0,
                        "reasoning": f"ShipsEar ground truth: class {letter}",
                        "vessel_type": _vessel_type_for(letter),
                        "recommended_action": "none",
                        "shipsear_class": letter,
                    },
                    "source_id": "shipsear-groundtruth",
                    "representation_version": "abs_db_v1",
                }

                jsonl_file.write(json.dumps(entry) + "\n")
                jsonl_file.flush()

                class_count[threat_level] = class_count.get(threat_level, 0) + 1
                written += 1

                if (i + 1) % 100 == 0:
                    log.info(
                        "bootstrap_progress",
                        done=i + 1,
                        total=len(pairs),
                        written=written,
                        failed=failed,
                    )
            except Exception as e:
                failed += 1
                log.error("bootstrap_failed", wav=str(wav_path), error=str(e))

    print(f"\nWritten: {written}   skipped: {skipped}   failed: {failed}")
    print("Class distribution (this run):")
    for level in ("NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"):
        n = class_count.get(level, 0)
        print(f"  {level:8s}  {n}")


def _vessel_type_for(letter: str) -> str:
    return {
        "A": "fishing",
        "B": "recreational",
        "C": "ferry",
        "D": "cargo",
        "E": "none",
    }[letter]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("shipsear_5s_16k"))
    ap.add_argument("--limit", type=int, default=0, help="0 = process all files")
    args = ap.parse_args()
    main(args.src, args.limit)
