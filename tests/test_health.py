import pytest
from fastapi.testclient import TestClient


def test_health_endpoint_returns_ok(client: TestClient):
    """Test that /health endpoint returns OK status."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_endpoint_content_type(client: TestClient):
    """Test that /health endpoint returns JSON content type."""
    response = client.get("/health")

    assert response.headers["content-type"] == "application/json"
