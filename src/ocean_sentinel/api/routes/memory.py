from pathlib import Path
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/stats")
async def memory_stats(request: Request):
    memory = request.app.state.memory
    chromadb_n = await memory.count()

    jsonl_path = Path("data/training/gemma_labels.jsonl")
    jsonl_n = (
        sum(1 for line in jsonl_path.open() if line.strip())
        if jsonl_path.exists() else 0
    )

    spec_dir = Path("data/spectrograms")
    spec_n = len(list(spec_dir.glob("*.npy"))) if spec_dir.exists() else 0

    return {
        "chromadb_entries": chromadb_n,
        "training_pairs": jsonl_n,
        "spectrograms": spec_n,
    }
