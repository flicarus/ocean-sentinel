"""SSE endpoint that streams structlog events to the dashboard in real-time."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter
from starlette.responses import StreamingResponse

router = APIRouter()

# In-memory broadcast — each connected client gets a queue.
# When a log event fires, it's pushed to every queue.
_subscribers: list[asyncio.Queue] = []


def broadcast_processor(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict,
) -> dict:
    """structlog processor that pushes every log entry to SSE subscribers."""
    if _subscribers:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": method_name,
            "event": event_dict.get("event", ""),
            "details": {
                k: _safe_serialize(v)
                for k, v in event_dict.items()
                if k != "event"
            },
        }
        for queue in _subscribers:
            try:
                queue.put_nowait(entry)
            except asyncio.QueueFull:
                pass  # drop if client is too slow

    return event_dict


def _safe_serialize(value: object) -> object:
    """Convert non-JSON-serializable values to strings."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


async def _event_stream(queue: asyncio.Queue):
    """Yield SSE-formatted events from the queue."""
    try:
        while True:
            entry = await queue.get()
            data = json.dumps(entry)
            yield f"data: {data}\n\n"
    except asyncio.CancelledError:
        pass


@router.get("/stream")
async def stream_logs():
    """SSE endpoint — dashboard connects here to receive real-time logs."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    _subscribers.append(queue)

    async def cleanup_stream():
        try:
            async for chunk in _event_stream(queue):
                yield chunk
        finally:
            _subscribers.remove(queue)

    return StreamingResponse(
        cleanup_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
