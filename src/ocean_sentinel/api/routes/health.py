from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/")
async def health(request: Request):
    return {
        "status": "ok",
        "environment": request.app.state.settings.environment,
    }