import logging
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    SERVICE_NAME: str = "documind-ingestion-service"
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    DATA_DIR: str = "./data"
    FAISS_INDEX_PATH: str = "./data/faiss.index"
    SQLITE_DB_PATH: str = "./data/metadata.db"
    HF_HOME: str = "/tmp/models"


def get_settings() -> Settings:
    """Load and return application settings."""
    return Settings()


def configure_logging(log_level: str) -> None:
    """Configure structured logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
