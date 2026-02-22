import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings, configure_logging
from app.api.ingest import router as ingest_router
from app.core.embedding_model import load_embedding_model
import os

# Initialize settings
settings = get_settings()

# Configure logging
configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info(
        f"Starting {settings.SERVICE_NAME} in {settings.ENV} environment",
        extra={"service": settings.SERVICE_NAME, "env": settings.ENV},
    )
    # Initialize embedding model once (unless disabled via env for tests)
    if os.environ.get("DISABLE_EMBEDDINGS") != "1":
        try:
            app.state.embedding_model = load_embedding_model()
            logger.info("Embedding model loaded", extra={"model": app.state.embedding_model.model_name})
        except Exception:
            logger.exception("Failed to initialize embedding model")
            raise
    else:
        app.state.embedding_model = None
    yield
    # Shutdown
    logger.info(f"Shutting down {settings.SERVICE_NAME}")


# Create FastAPI application
app = FastAPI(
    title=settings.SERVICE_NAME,
    description="Ingestion microservice for RAG system",
    version="0.1.0",
    lifespan=lifespan,
)

# Register API routers
app.include_router(ingest_router)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
