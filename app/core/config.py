import logging
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    SERVICE_NAME: str
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


def get_settings() -> Settings:
    """Load and return application settings."""
    return Settings()


def configure_logging(log_level: str) -> None:
    """Configure structured logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
