import json
import os
from typing import Dict, List

import numpy as np

try:
    import faiss
except Exception:
    faiss = None


class FaissVectorStore:
    """FAISS IndexFlatL2 store with persisted id->metadata mapping."""

    def __init__(self, index_path: str) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install faiss-cpu to enable vector storage.")

        self.index_path = index_path
        self.mapping_path = f"{index_path}.mapping.json"
        self.index = None
        self.id_mapping: Dict[str, Dict[str, int | str]] = {}

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)

        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r", encoding="utf-8") as handle:
                self.id_mapping = json.load(handle)

    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
            return

        if self.index.d != dim:
            # Recreate index when embedding dimension changes between runs.
            # This prevents runtime failures with stale persisted indexes.
            self.index = faiss.IndexFlatL2(dim)
            self.id_mapping = {}

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadata_items: List[Dict[str, int | str]],
    ) -> List[int]:
        if not embeddings:
            return []

        if len(embeddings) != len(metadata_items):
            raise ValueError("embeddings and metadata_items must have the same length")

        array = np.asarray(embeddings, dtype="float32")
        if array.ndim != 2:
            raise ValueError("embeddings must be 2-dimensional")

        self._ensure_index(array.shape[1])

        start_id = self.index.ntotal
        self.index.add(array)

        stored_ids: List[int] = []
        for offset, item in enumerate(metadata_items):
            faiss_id = int(start_id + offset)
            self.id_mapping[str(faiss_id)] = {
                "document_id": str(item["document_id"]),
                "chunk_id": int(item["chunk_id"]),
            }
            stored_ids.append(faiss_id)

        self.persist()
        return stored_ids

    def persist(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

        with open(self.mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.id_mapping, handle)

    def close(self) -> None:
        self.persist()
