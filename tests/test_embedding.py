import pytest
from types import SimpleNamespace
from fastapi.testclient import TestClient


def test_load_embedding_model_once(monkeypatch):
    calls = {"count": 0}

    def fake_loader(model_name="m"):
        calls["count"] += 1
        return SimpleNamespace(model_name=model_name, embed_texts=lambda texts: [[0.0]] * len(texts))

    import os
    # ensure required env var for app import
    os.environ["SERVICE_NAME"] = "test-service"
    os.environ.pop("DISABLE_EMBEDDINGS", None)
    import app.main as main

    # Patch the loader on the main module before creating TestClient
    monkeypatch.setattr(main, "load_embedding_model", fake_loader)

    with TestClient(main.app) as client:
        # ensure startup completed
        client.get("/health")
        # startup should have invoked loader once
        assert calls["count"] == 1
        client.get("/health")


def test_embedding_dim_and_count(monkeypatch):
    # fake model that returns vectors of dim 4
    def fake_model_constructor(name):
        class FakeModel:
            def encode(self, texts, convert_to_numpy=True):
                import numpy as np

                arr = np.arange(len(texts) * 4).reshape(len(texts), 4).astype(float)
                return arr

        return FakeModel()

    # Patch SentenceTransformer in embedding_model module
    import app.core.embedding_model as embmod

    monkeypatch.setattr(embmod, "SentenceTransformer", fake_model_constructor)

    m = embmod.load_embedding_model("fake-model")
    texts = ["a", "b", "c"]
    embs = m.embed_texts(texts)
    assert len(embs) == len(texts)
    assert len(embs[0]) == 4

    # Check normalization (L2 norm ~=1)
    import math

    for v in embs:
        norm = math.sqrt(sum(x * x for x in v))
        assert pytest.approx(norm, rel=1e-6) == 1.0


def test_empty_input_returns_empty():
    import app.core.embedding_model as embmod

    # If SentenceTransformer not installed, constructing loader raises; emulate minimal model
    class Dummy:
        def encode(self, texts, convert_to_numpy=True):
            return []

    m = SimpleNamespace(
        model_name="dummy",
        _model=Dummy(),
        embed_texts=embmod.EmbeddingModel.embed_texts.__get__(Dummy(), Dummy),
    )
    # calling embed_texts with empty list should return []
    assert m.embed_texts([]) == []
