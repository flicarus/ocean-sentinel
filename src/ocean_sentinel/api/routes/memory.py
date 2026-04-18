from pathlib import Path
from fastapi import APIRouter, Request

from ocean_sentinel.adapters.supabase_training_logger import SupabaseTrainingLogger

router = APIRouter()


@router.get("/stats")
async def memory_stats(request: Request):
    memory = request.app.state.memory
    chromadb_n = await memory.count()

    training_logger = request.app.state.training_logger

    if isinstance(training_logger, SupabaseTrainingLogger):
        # Pull real count from Supabase — shared across both machines
        result = training_logger._client.table("training_pairs") \
            .select("id", count="exact") \
            .execute()
        training_n = result.count

        # Count spectrograms in the storage bucket
        files = training_logger._client.storage \
            .from_(training_logger._bucket) \
            .list()
        spec_n = len(files) if files else 0
    else:
        # Fallback: count local files
        jsonl_path = Path("data/training/gemma_labels.jsonl")
        training_n = (
            sum(1 for line in jsonl_path.open() if line.strip())
            if jsonl_path.exists() else 0
        )
        spec_dir = Path("data/spectrograms")
        spec_n = len(list(spec_dir.glob("*.npy"))) if spec_dir.exists() else 0

    return {
        "chromadb_entries": chromadb_n,
        "training_pairs": training_n,
        "spectrograms": spec_n,
    }
