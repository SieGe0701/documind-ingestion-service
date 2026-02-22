import os
from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    # Set required environment variable for config
    os.environ["SERVICE_NAME"] = "test-service"
    # Disable heavy embedding model load during tests and provide a dummy model
    os.environ["DISABLE_EMBEDDINGS"] = "1"

    from app.main import app
    # If startup didn't attach an embedding model (disabled), attach a dummy
    if getattr(app.state, "embedding_model", None) is None:
        from types import SimpleNamespace

        def dummy_embed(texts):
            # return fixed-dim zero vectors for testing
            return [[0.0] * 8 for _ in texts]

        app.state.embedding_model = SimpleNamespace(embed_texts=dummy_embed, model_name="dummy-test-model")
    return TestClient(app)
