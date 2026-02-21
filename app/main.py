import logging

from fastapi import FastAPI

from app.core.config import get_settings, configure_logging

# Initialize settings
settings = get_settings()

# Configure logging
configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.SERVICE_NAME,
    description="Ingestion microservice for RAG system",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize application on startup."""
    logger.info(
        f"Starting {settings.SERVICE_NAME} in {settings.ENV} environment",
        extra={"service": settings.SERVICE_NAME, "env": settings.ENV},
    )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup on shutdown."""
    logger.info(f"Shutting down {settings.SERVICE_NAME}")


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
