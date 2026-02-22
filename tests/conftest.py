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
    os.environ["DISABLE_STORAGE"] = "1"

    from app.main import app
    from types import SimpleNamespace

    def dummy_embed(texts):
        return [[0.0] * 8 for _ in texts]

    class DummyVectorStore:
        def add_embeddings(self, embeddings, metadata_items):
            return list(range(len(embeddings)))

        def close(self):
            return None

    class DummyMetadataStore:
        def save_document(self, **kwargs):
            return None

        def save_chunks(self, **kwargs):
            return None

        def close(self):
            return None

    # Always set fresh dummy dependencies to avoid reusing closed state across tests
    app.state.embedding_model = SimpleNamespace(
        embed_texts=dummy_embed,
        model_name="dummy-test-model",
    )
    app.state.vector_store = DummyVectorStore()
    app.state.metadata_store = DummyMetadataStore()

    with TestClient(app) as test_client:
        # Re-attach dummy stores after lifespan startup to guard against state overwrite
        app.state.embedding_model = SimpleNamespace(
            embed_texts=dummy_embed,
            model_name="dummy-test-model",
        )
        app.state.vector_store = DummyVectorStore()
        app.state.metadata_store = DummyMetadataStore()
        yield test_client
