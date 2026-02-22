import os
from threading import Lock
from typing import List

import numpy as np

SentenceTransformer = None

_model_instance = None
_model_lock = Lock()



class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._cache_dir = os.environ.get("HF_HOME", "/tmp/models")

    def get_model(self):
        # Lazy loading keeps deployment footprint low: model files download only when embeddings are first requested.
        if self._model is not None:
            return self._model

        transformer_cls = _get_sentence_transformer_cls()
        os.makedirs(self._cache_dir, exist_ok=True)
        try:
            self._model = transformer_cls(self.model_name, cache_folder=self._cache_dir)
        except TypeError:
            self._model = transformer_cls(self.model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {exc}")

        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Return empty list for empty input
        if not texts:
            return []

        model = self.get_model()

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
    global _model_instance
    with _model_lock:
        if _model_instance is None:
            _model_instance = EmbeddingModel(model_name)
        return _model_instance


def _get_sentence_transformer_cls():
    global SentenceTransformer
    if SentenceTransformer is not None:
        return SentenceTransformer

    try:
        from sentence_transformers import SentenceTransformer as _SentenceTransformer
    except Exception:
        raise RuntimeError("sentence-transformers is not installed")

    SentenceTransformer = _SentenceTransformer
    return SentenceTransformer
