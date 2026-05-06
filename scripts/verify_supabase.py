"""Verify the Supabase wiring before relying on it in production paths.

Checks, in order:
  1. OS_SUPABASE_URL + OS_SUPABASE_SERVICE_ROLE_KEY are set
  2. Client connects to the project
  3. `spectrograms` bucket exists
  4. `training_pairs` table exists with the columns SupabaseTrainingLogger writes
  5. Round-trip: upload a 1-byte test object, insert + read a row, clean up

Run after pasting Maciej's credentials into .env:

    PYTHONPATH=src venv/bin/python scripts/verify_supabase.py
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, "src")

from ocean_sentinel.config import Settings


REQUIRED_COLUMNS = [
    "event_id",
    "source_id",
    "orig_path",
    "spectrogram_bucket",
    "spectrogram_key",
    "features",
    "context_text",
    "gemma_verdict",
    "ground_truth_label",
    "ground_truth_source",
]

BUCKET = "spectrograms"
TABLE = "training_pairs"


def fail(msg: str, exit_code: int = 1) -> None:
    print(f"  ❌ {msg}")
    sys.exit(exit_code)


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def step(msg: str) -> None:
    print(f"\n→ {msg}")


def main() -> None:
    print("Supabase wiring verification\n" + "=" * 40)

    # 1. Env vars --------------------------------------------------------
    step("Step 1: env vars")
    settings = Settings()
    if not settings.supabase_url:
        fail("OS_SUPABASE_URL is not set in .env")
    if not settings.supabase_service_role_key:
        fail("OS_SUPABASE_SERVICE_ROLE_KEY is not set in .env")
    ok(f"URL: {settings.supabase_url}")
    ok(f"Service role key: {settings.supabase_service_role_key[:24]}…")

    # 2. Client connects -------------------------------------------------
    step("Step 2: client connects")
    try:
        from supabase import create_client
    except ImportError:
        fail("supabase-py not installed. `venv/bin/pip install supabase`")
    try:
        client = create_client(settings.supabase_url, settings.supabase_service_role_key)
        ok("client created")
    except Exception as e:
        fail(f"client creation failed: {e}")

    # 3. Bucket exists ---------------------------------------------------
    step(f"Step 3: bucket '{BUCKET}' exists")
    try:
        buckets = client.storage.list_buckets()
        names = [getattr(b, "name", None) or b.get("name") for b in buckets]
        if BUCKET not in names:
            fail(f"bucket '{BUCKET}' not found. Existing: {names}")
        ok(f"bucket '{BUCKET}' present (all: {names})")
    except Exception as e:
        fail(f"could not list buckets: {e}")

    # 4. Table + columns -------------------------------------------------
    step(f"Step 4: table '{TABLE}' exists with required columns")
    try:
        # Cheapest probe: SELECT 0 rows, but include all columns we need.
        select_cols = ",".join(REQUIRED_COLUMNS)
        client.table(TABLE).select(select_cols).limit(0).execute()
        ok(f"table '{TABLE}' has all {len(REQUIRED_COLUMNS)} required columns")
    except Exception as e:
        fail(f"table check failed: {e}\n     Expected columns: {REQUIRED_COLUMNS}")

    # 5. Round-trip ------------------------------------------------------
    step("Step 5: round-trip (upload + insert + select + cleanup)")
    test_id = f"_verify_{uuid.uuid4().hex[:8]}"
    storage_key = f"_verify/{test_id}.bin"
    payload = {
        "event_id": test_id,
        "source_id": "verify_supabase",
        "orig_path": "test://bytes",
        "spectrogram_bucket": BUCKET,
        "spectrogram_key": storage_key,
        "features": {"test": 1.0},
        "context_text": "verify_supabase smoke test",
        "gemma_verdict": {"threat_level": "NONE", "confidence": 0.0, "reasoning": "test"},
        "ground_truth_label": None,
        "ground_truth_source": None,
    }
    try:
        client.storage.from_(BUCKET).upload(
            path=storage_key,
            file=b"\x00",
            file_options={"contentType": "application/octet-stream", "upsert": "true"},
        )
        ok("storage upload OK")

        client.table(TABLE).insert(payload).execute()
        ok("table insert OK")

        rows = client.table(TABLE).select("event_id").eq("event_id", test_id).execute()
        if not rows.data:
            fail("inserted row not visible on read-back")
        ok(f"read-back OK ({len(rows.data)} row)")

        # Cleanup — never leave _verify_* rows in production tables.
        client.table(TABLE).delete().eq("event_id", test_id).execute()
        client.storage.from_(BUCKET).remove([storage_key])
        ok("cleanup OK")
    except Exception as e:
        fail(f"round-trip failed: {e}")

    print("\n" + "=" * 40)
    print("All checks passed. Supabase is wired up and ready.")


if __name__ == "__main__":
    main()
