"""Score v6 against AIS-corrected SanctSound labels.

Until now we evaluated v6 against the SanctSound dataset's ambient labels
and treated disagreement as model error. The relabel script just produced
640 corrected labels (520 ship, 120 ship_distant) — let's see if v6 actually
agrees with AIS once we trust AIS as ground truth instead of SanctSound's
default labels.

This is the cleanest in-vivo answer to: "is v6 working on OOD SanctSound
data?" If accuracy on AIS-corrected labels is high, v6 was right all along
and the data was the problem. If it's still low, v6 has a real OOD gap and
v7 needs the temporal-context / multi-task fixes we're building.

Output: stdout summary table + confusion matrix.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from ocean_sentinel.services.cnn_classifier import CNNClassifier


CKPT = Path("data/models/cnn_v6.pt")
CORRECTED = Path("data/training/sanctsound_corrected.jsonl")


def main() -> None:
    print(f"Loading CNN v6: {CKPT}")
    clf = CNNClassifier(CKPT)
    print()

    rows = []
    with CORRECTED.open() as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Evaluating on {len(rows)} corrected SanctSound chunks")

    confusion = Counter()  # (true, pred) -> count
    by_label = Counter()
    by_distance = Counter()
    correct_by_label = Counter()
    confidences = {"ship_correct": [], "ship_wrong": [],
                   "amb_correct": [], "amb_wrong": []}

    for r in rows:
        spec = np.load(r["spectrogram_path"]).astype(np.float32)
        ais_label = r.get("sanctsound_ais_label", r["label"])
        true_binary = "ship" if ais_label == "ship" else (
            "ship" if ais_label == "ship_distant" else "ambient"
        )

        # Use legacy "sanctsound" source_id so v6 applies its profile.
        result = clf.predict(spec, source_id="sanctsound")
        pred_binary = "ship" if result["label"] == "ship" else "ambient"
        p_ship = result["probabilities"]["ship"]

        confusion[(true_binary, pred_binary)] += 1
        by_label[true_binary] += 1
        by_distance[r.get("distance_bucket", "unknown")] += 1
        if true_binary == pred_binary:
            correct_by_label[true_binary] += 1

        if true_binary == "ship":
            (confidences["ship_correct"] if pred_binary == "ship"
             else confidences["ship_wrong"]).append(p_ship)
        else:
            (confidences["amb_correct"] if pred_binary == "ambient"
             else confidences["amb_wrong"]).append(p_ship)

    print()
    print("=" * 60)
    print("Confusion (rows=AIS truth, cols=v6 prediction):")
    print(f"               | pred ambient | pred ship  ")
    print(f"  true ambient | {confusion[('ambient','ambient')]:>12d} | {confusion[('ambient','ship')]:>10d}")
    print(f"  true ship    | {confusion[('ship','ambient')]:>12d} | {confusion[('ship','ship')]:>10d}")
    print()
    print(f"Per-class accuracy:")
    for lab in ("ship", "ambient"):
        if by_label[lab]:
            acc = correct_by_label[lab] / by_label[lab]
            print(f"  {lab}: {correct_by_label[lab]}/{by_label[lab]} = {acc:.1%}")
    print()
    print(f"Distance-bucket breakdown of AIS truth:")
    for bucket, n in by_distance.most_common():
        print(f"  {bucket}: {n}")
    print()
    print(f"Confidence stats:")
    for k, vs in confidences.items():
        if vs:
            arr = np.array(vs)
            print(f"  {k:20s}  n={len(arr):4d}  mean={arr.mean():.3f}  "
                  f"median={np.median(arr):.3f}  std={arr.std():.3f}")


if __name__ == "__main__":
    main()
