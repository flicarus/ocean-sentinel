import uuid
import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from ocean_sentinel.exceptions import OceanSentinelError

log = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            return await call_next(request)
        except OceanSentinelError as e:
            log.error(
                "ocean_sentinel_error",
                code=e.code,
                message=e.message,
                path=request.url.path,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": e.code,
                    "message": e.message,
                    "details": e.details,
                }
            )