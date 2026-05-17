#!/usr/bin/env bash
# Download NOAA SanctSound audio from stations Ocean Sentinel was NOT trained on.
#
# Why this exists
# ---------------
# Every SanctSound clip on disk under data/training/ is in the v7.4
# training set (verified — 2223/2223 ShipsEar specs are trained, every
# sanctsound/diverse-sb02 etc. is trained). To honestly claim "v7.4
# generalises to held-out distributions" we need audio from sites the
# model has never seen.
#
# These four stations (ci03, fk04, gr02, hi05) are different physical
# hydrophones than anything in our training corpus. Acoustic
# environments span Channel Islands (CA), Florida Keys, Gray's Reef
# (GA), and Hawaii — different climates, different vessel traffic
# patterns, different biological soundscapes.
#
# Public access via Google Cloud Storage's noaa-passive-bioacoustic
# bucket. No auth required.
#
# Usage
# -----
#   bash scripts/download_external_eval.sh
#   PYTHONPATH=src venv/bin/python scripts/evaluate_external.py
#
# Disk usage: ~1 GB total. Download time: 15-30 min on broadband.

set -euo pipefail

OUT_DIR="data/external_eval"
mkdir -p "$OUT_DIR"

GCS="https://storage.googleapis.com/noaa-passive-bioacoustic/sanctsound/audio"

# Station / deployment / file path
declare -a TARGETS=(
    "hi05|sanctsound_hi05_01/audio/SanctSound_HI05_01_671399971_20181115T000002Z.flac"
    "ci03|sanctsound_ci03_02/audio/SanctSound_CI03_02_671391784_191015220000.flac"
    "gr02|sanctsound_gr02_02/audio/SanctSound_GR02_02_470331456_190513170000.flac"
    "mb03|01/audio/SanctSound_MB03_01_D143_181114000000.x.flac"
)

for entry in "${TARGETS[@]}"; do
    station="${entry%%|*}"
    rel_path="${entry#*|}"
    url="${GCS}/${station}/${rel_path}"
    out_file="${OUT_DIR}/${station}_$(basename "$rel_path")"

    if [[ -f "$out_file" ]]; then
        size_mb=$(du -m "$out_file" | cut -f1)
        echo "  [skip] ${out_file} already present (${size_mb} MB)"
        continue
    fi

    echo "  [pull] ${station} → ${out_file}"
    if command -v aria2c &>/dev/null; then
        aria2c -x 4 -s 4 -d "$OUT_DIR" -o "$(basename "$out_file")" "$url" || {
            echo "    aria2 failed; falling back to curl"
            curl -L --fail --progress-bar -o "$out_file" "$url"
        }
    else
        curl -L --fail --progress-bar -o "$out_file" "$url"
    fi
done

echo
echo "  All downloads complete. Files in ${OUT_DIR}/:"
ls -lh "$OUT_DIR"/*.flac 2>/dev/null || true
echo
echo "  Next step: PYTHONPATH=src venv/bin/python scripts/evaluate_external.py"
