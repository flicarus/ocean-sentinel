"""Revert sb01 labels in sanctsound_corrected.jsonl to original `not_ship`.

Background: scripts/sanctsound_ais_relabel.py uses 10km radius + 1h
window for AIS lookup. On Stellwagen Bank (sb01) every hour has a vessel
within 10km because it's a shipping corridor — so all 480 sb01 chunks
got flipped to label='ship'. But the audio doesn't actually contain
audible vessels (10km exceeds the typical hydrophone acoustic range
~3-5km). v7.1 trained on the original `not_ship` labels gets 97.9% on
sb01 audio; v7.2 trained on the flipped `ship` labels collapses to 8.3%.

This script reverts label → original_label for every sb01 row in
sanctsound_corrected.jsonl, leaving oc01 untouched (its 70%-flip ratio
suggests narrower mix of audible vs distant ships, model handles it).

Idempotent: rows where label already == original_label are unchanged.

Usage:
    venv/bin/python scripts/fix_sb01_labels.py
"""
from __future__ import annotations

import json
from pathlib import Path

JSONL = Path("data/training/sanctsound_corrected.jsonl")


def main() -> None:
    if not JSONL.exists():
        print(f"missing {JSONL}")
        return

    rows = []
    n_total = 0
    n_sb01 = 0
    n_reverted = 0

    with JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            sid = (r.get("provenance") or {}).get("source_id", "")
            if "sb01" in sid:
                n_sb01 += 1
                original = r.get("original_label")
                current = r.get("label")
                if original and current != original:
                    r["label"] = original
                    r["sb01_label_reverted"] = True
                    r["sanctsound_ais_label"] = current  # preserve old AIS label
                    n_reverted += 1
            rows.append(r)

    # Write back atomically.
    tmp = JSONL.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(JSONL)

    print(f"Total rows: {n_total}")
    print(f"sb01 rows : {n_sb01}")
    print(f"Reverted  : {n_reverted}")


if __name__ == "__main__":
    main()
