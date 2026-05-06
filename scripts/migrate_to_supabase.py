"""Backfill training_pairs from local v7.jsonl + data/spectrograms/ to Supabase.

Idempotent — re-running skips event_ids already present in the table.
Parallel uploads (ThreadPoolExecutor) push ~5500 files in 5-10 minutes.

Usage:
    PYTHONPATH=src venv/bin/python scripts/migrate_to_supabase.py --jsonl data/training/gemma_labels.v7.jsonl
    PYTHONPATH=src venv/bin/python scripts/migrate_to_supabase.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "src")

from supabase import create_client

from ocean_sentinel.config import Settings


BUCKET = "spectrograms"
TABLE = "training_pairs"
DEFAULT_JSONL = Path("data/training/gemma_labels.v7.jsonl")
WORKERS = 8


def fetch_existing_event_ids(client) -> set[str]:
    """Pull all event_ids already in training_pairs.  Paginates through results."""
    seen: set[str] = set()
    page_size = 1000
    offset = 0
    while True:
        rows = (
            client.table(TABLE)
            .select("event_id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        if not rows.data:
            break
        for r in rows.data:
            seen.add(r["event_id"])
        if len(rows.data) < page_size:
            break
        offset += page_size
    return seen


def push_one(client, row: dict) -> tuple[str, str | None]:
    """Upload spectrogram + insert metadata row.  Returns (event_id, error_or_None)."""
    event_id = row["event_id"]
    try:
        spec_path = Path(row["spectrogram_path"])
        source_id = row.get("provenance", {}).get("source_id", "unknown")
        storage_key = f"{source_id}/{spec_path.name}"

        # 1. Upload spectrogram (upsert=true so re-uploads are safe).
        if spec_path.exists():
            with spec_path.open("rb") as f:
                client.storage.from_(BUCKET).upload(
                    path=storage_key,
                    file=f.read(),
                    file_options={
                        "contentType": "application/octet-stream",
                        "upsert": "true",
                    },
                )
        else:
            return event_id, f"spectrogram missing: {spec_path}"

        # 2. Insert metadata row.
        feats = row.get("features", {})
        payload = {
            "event_id": event_id,
            "source_id": source_id,
            "orig_path": str(spec_path),
            "spectrogram_bucket": BUCKET,
            "spectrogram_key": storage_key,
            "features": {
                "engine_band_ratio": feats.get("engine_band_ratio"),
                "peak_frequency_hz": feats.get("peak_frequency_hz"),
                "spectral_flatness": feats.get("spectral_flatness"),
                "rms_energy": feats.get("rms_energy"),
                "engine_band_energy_db": feats.get("engine_band_energy_db"),
            },
            "context_text": f"bootstrap | {source_id} | {row.get('timestamp', '')}",
            # Bootstrap rows have a ground-truth label, not a Gemma verdict.
            # Keep gemma_verdict null-ish so the generated `threat_level` column
            # stays NULL — this distinguishes training data from production
            # classifications without adding a new column.
            "gemma_verdict": {"is_bootstrap": True},
            "ground_truth_label": row.get("label"),
            "ground_truth_source": source_id,
        }
        client.table(TABLE).insert(payload).execute()
        return event_id, None
    except Exception as e:
        return event_id, str(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be migrated, but don't push.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap number of rows to migrate (smoke test).")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    settings = Settings()
    if not (settings.supabase_url and settings.supabase_service_role_key):
        sys.exit("OS_SUPABASE_URL / OS_SUPABASE_SERVICE_ROLE_KEY missing in .env")

    client = create_client(settings.supabase_url, settings.supabase_service_role_key)

    print(f"Reading {args.jsonl} …")
    rows = [json.loads(l) for l in args.jsonl.open()]
    print(f"  {len(rows)} rows in JSONL")

    print("Fetching existing event_ids from Supabase …")
    existing = fetch_existing_event_ids(client)
    print(f"  {len(existing)} already in table")

    todo = [r for r in rows if r["event_id"] not in existing]
    if args.limit:
        todo = todo[: args.limit]

    print(f"\nTo migrate: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return
    if args.dry_run:
        print("[dry-run] sample event_ids that would be migrated:")
        for r in todo[:5]:
            print(f"  - {r['event_id']}")
        return

    print(f"Pushing with {args.workers} workers …\n")
    t0 = time.perf_counter()
    ok = 0
    failed: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(push_one, client, r): r["event_id"] for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            event_id, err = fut.result()
            if err:
                failed.append((event_id, err))
            else:
                ok += 1
            if i % 100 == 0 or i == len(todo):
                elapsed = time.perf_counter() - t0
                rate = i / elapsed
                eta = (len(todo) - i) / rate if rate > 0 else 0
                print(
                    f"  {i:5d}/{len(todo)}  "
                    f"ok={ok:5d}  failed={len(failed):3d}  "
                    f"({rate:.1f} rows/s, ETA {eta:.0f}s)"
                )

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s.  Migrated {ok}, failed {len(failed)}.")
    if failed:
        print("\nFirst 5 failures:")
        for eid, err in failed[:5]:
            print(f"  - {eid}: {err[:120]}")


if __name__ == "__main__":
    main()
