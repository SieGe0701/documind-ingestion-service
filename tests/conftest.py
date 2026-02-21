import os
from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    # Set required environment variable for config
    os.environ["SERVICE_NAME"] = "test-service"

    from app.main import app
    return TestClient(app)
