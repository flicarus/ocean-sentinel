"""Inventory the full v7 training dataset across all manifest files.

Reports counts by source, label, distance bucket, and final binary class.
Used as a sanity check before training v7 — catches issues like missing
spectrogram files, all-one-class buckets, or unexpected source skew.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

JSONL_PATHS = [
    Path("data/training/gemma_labels.v7.jsonl"),
    Path("data/training/sanctsound_corrected.jsonl"),
    Path("data/training/sanctsound_diverse.jsonl"),
] + sorted(Path("data/training/v7_bulk").glob("*.jsonl") if Path("data/training/v7_bulk").exists() else [])


def main() -> None:
    by_source = Counter()
    by_label = Counter()
    by_distance = Counter()
    by_source_x_label = defaultdict(Counter)
    missing_spec = 0
    skipped_sanctsound_in_primary = 0
    n_total = 0

    for p in JSONL_PATHS:
        if not p.exists():
            print(f"  SKIP (missing): {p}")
            continue
        is_primary = p.name == "gemma_labels.v7.jsonl"
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                source = (r.get("provenance") or {}).get("source_id") or "unknown"
                label = r.get("label", "unknown")
                distance = r.get("distance_bucket", "unset")

                # Mirror trainer's behavior — skip raw sanctsound rows in primary
                if is_primary and source == "sanctsound":
                    skipped_sanctsound_in_primary += 1
                    continue

                if not os.path.exists(r.get("spectrogram_path", "")):
                    missing_spec += 1
                    continue

                n_total += 1
                by_source[source] += 1
                by_label[label] += 1
                by_distance[distance] += 1
                by_source_x_label[source][label] += 1

    print("=" * 70)
    print(f"Total trainable rows: {n_total:,}")
    print(f"Skipped sanctsound rows in v7.jsonl: {skipped_sanctsound_in_primary} (use sanctsound_corrected instead)")
    print(f"Missing spectrogram files: {missing_spec}")
    print()
    print(f"By binary label:")
    for k, v in by_label.most_common():
        print(f"  {k:<20s} {v:>6d}  ({100*v/n_total:.1f}%)")
    print()
    print(f"By distance bucket (where set):")
    for k, v in by_distance.most_common():
        print(f"  {k:<20s} {v:>6d}")
    print()
    print(f"By source:")
    for src, n in sorted(by_source.items(), key=lambda x: -x[1]):
        breakdown = ", ".join(f"{lab}={c}" for lab, c in by_source_x_label[src].most_common())
        print(f"  {src:<35s} {n:>5d}  ({breakdown})")
    print()


if __name__ == "__main__":
    main()
