from typing import List

import numpy as np

from sentence_transformers import SentenceTransformer



class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {exc}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Return empty list for empty input
        if not texts:
            return []

        # Support models that might be attached as _model in tests
        model = getattr(self, "model", None) or getattr(self, "_model", None)
        if model is None:
            raise RuntimeError("No underlying model available for embeddings")

        # Call encode with a minimal set of kwargs to support test doubles
        try:
            arr = model.encode(texts, convert_to_numpy=True)
        except TypeError:
            # Fallback for models that return lists or arrays without kwargs
            arr = model.encode(texts)

        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr.tolist()


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    return EmbeddingModel(model_name)
