"""FastAPI application entrypoint for the Multimodal AI Chatbot backend."""

from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.config import get_settings
from backend.database import connection
from backend.database.connection import check_db_health, init_db
from backend.database.schemas import HealthResponse
from backend.routes import audio, chat, documents, image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.rate_limit_per_minute}/minute"])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize storage directories and database tables on startup."""
    settings.ensure_storage_dirs()
    db_ready = init_db()
    if not db_ready:
        logger.warning("Started API without live MySQL connection. Check MySQL configuration.")
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Multimodal AI Chatbot API with Hugging Face models, RAG, MySQL persistence, and safety guardrails.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    """Return a clean JSON error without exposing internal stack traces or secrets."""
    logger.error("Unhandled server error: %s", exc.__class__.__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal error occurred. Please try again."},
    )


app.include_router(chat.router)
app.include_router(image.router)
app.include_router(audio.router)
app.include_router(documents.router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Report backend, database, and model configuration status."""
    db_ok = check_db_health()
    engine_url = str(connection._engine.url) if connection._engine else ""
    db_label = "mysql" if "mysql" in engine_url else ("sqlite" if "sqlite" in engine_url else "connected")
    provider = settings.ai_provider.lower().strip()
    key_configured = bool(
        settings.gemini_api_key.strip() if provider == "gemini" else settings.hf_token.strip()
    )
    models = (
        {
            "provider": "gemini",
            "chat": settings.gemini_chat_model,
            "vision": settings.gemini_vision_model,
            "embedding": settings.gemini_embedding_model,
        }
        if provider == "gemini"
        else {
            "provider": "huggingface",
            "chat": settings.hf_chat_model,
            "vision": settings.hf_vision_model,
            "asr": settings.hf_asr_model,
            "tts": settings.hf_tts_model,
            "embedding": settings.hf_embedding_model,
        }
    )
    return HealthResponse(
        status="ok" if (db_ok and key_configured) else "degraded",
        database=db_label if db_ok else "unavailable",
        ai_provider=provider,
        api_key_configured=key_configured,
        models=models,
    )


@app.get("/api/media/{media_type}/{filename}", tags=["Media"])
def serve_media(media_type: str, filename: str):
    """Safely serve stored image or audio files with path-traversal protection."""
    if media_type == "images":
        base_dir = settings.images_dir.resolve()
    elif media_type == "audio":
        base_dir = settings.audio_dir.resolve()
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media type not found.")

    safe_name = Path(filename).name
    target = (base_dir / safe_name).resolve()
    if target.parent != base_dir or not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media file not found.")
    return FileResponse(path=str(target))
