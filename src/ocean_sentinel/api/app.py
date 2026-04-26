from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ocean_sentinel.config import Settings
from ocean_sentinel.adapters.gfw import GFWAdapter
from ocean_sentinel.adapters.copernicus import CopernicusAdapter
from ocean_sentinel.adapters.chromadb_store import ChromaDBAcousticMemory
from ocean_sentinel.adapters.persistence import SQLiteEventStore
from ocean_sentinel.adapters.training_logger import JSONLTrainingLogger
from ocean_sentinel.adapters.supabase_training_logger import SupabaseTrainingLogger
from ocean_sentinel.api.routes import health, events, alerts, dashboard, pipeline, logs, memory
from ocean_sentinel.api.middleware import RequestIDMiddleware, ErrorHandlerMiddleware
from ocean_sentinel.logging import configure_logging
from ocean_sentinel.services.cnn_classifier import CNNClassifier

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = Settings()

    store = SQLiteEventStore(settings.database_url.replace("sqlite+aiosqlite:///", ""))
    await store.init()

    memory = ChromaDBAcousticMemory(persist_dir="data/chromadb")

    if settings.supabase_url and settings.supabase_service_role_key:
        training_logger = SupabaseTrainingLogger(
            url=settings.supabase_url,
            service_role_key=settings.supabase_service_role_key,
        )
    else:
        training_logger = JSONLTrainingLogger(output_dir="data/training")

    cnn_ckpt = Path(settings.cnn_checkpoint_path)
    if cnn_ckpt.exists():
        cnn = CNNClassifier(cnn_ckpt)
    else:
        log.warning("cnn_checkpoint_missing", path=str(cnn_ckpt))
        cnn = None

    app.state.settings = settings
    app.state.gfw = GFWAdapter(settings)
    app.state.copernicus = CopernicusAdapter(settings)
    app.state.store = store
    app.state.memory = memory
    app.state.training_logger = training_logger
    app.state.cnn = cnn

    yield

    await app.state.gfw.close()
    await memory.close()
    await store.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Ocean Sentinel",
        description="Multimodal intelligence layer for illegal fishing detection",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(RequestIDMiddleware)

    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(events.router, prefix="/events", tags=["events"])
    app.include_router(alerts.router, prefix="/alerts", tags=["alerts"])
    app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
    app.include_router(pipeline.router, prefix="/pipeline", tags=["pipeline"])
    app.include_router(logs.router, prefix="/logs", tags=["logs"])
    app.include_router(memory.router, prefix="/memory", tags=["memory"])

    return app


app = create_app()