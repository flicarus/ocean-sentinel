"""Migrate existing local training pairs into Supabase.

Reads every entry from the local gemma_labels.jsonl file and uploads it
to Supabase — metadata goes to the `training_pairs` table, the .npy
spectrogram file goes to the `spectrograms` storage bucket.

Safe to re-run: upsert=true means existing records get overwritten,
not duplicated.

Usage:
    PYTHONPATH=src venv/bin/python3 scripts/migrate_jsonl_to_supabase.py \
        --jsonl data/training/gemma_labels.jsonl
"""

import argparse
import asyncio
import json
from pathlib import Path

from ocean_sentinel.adapters.supabase_training_logger import SupabaseTrainingLogger
from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import AcousticFeatures


async def main(jsonl_path: Path) -> None:
    settings = Settings()

    if not settings.supabase_url or not settings.supabase_service_role_key:
        print("ERROR: OS_SUPABASE_URL and OS_SUPABASE_SERVICE_ROLE_KEY must be set in .env")
        return

    logger = SupabaseTrainingLogger(
        url=settings.supabase_url,
        service_role_key=settings.supabase_service_role_key,
    )

    lines = [l for l in jsonl_path.read_text().splitlines() if l.strip()]
    print(f"Found {len(lines)} entries in {jsonl_path}")

    n_ok, n_fail = 0, 0
    for line in lines:
        entry = json.loads(line)
        try:
            features = AcousticFeatures.from_analyzer_dict(entry["features"])
            await logger.log(
                event_id=entry["event_id"],
                spectrogram_path=entry["spectrogram_path"],
                features=features,
                context_text=entry.get("context_text", ""),
                gemma_verdict=entry["gemma_verdict"],
                source_id=entry.get("source_id", "legacy-jsonl"),
            )
            n_ok += 1
            print(f"  OK  {entry['event_id']}")
        except Exception as e:
            n_fail += 1
            print(f"  FAIL {entry.get('event_id')}: {e}")

    print(f"\nDone. OK={n_ok}  FAIL={n_fail}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    args = ap.parse_args()
    asyncio.run(main(args.jsonl))
