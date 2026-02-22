import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

faiss = pytest.importorskip("faiss")


def _dummy_embed(texts):
    return [[0.0] * 8 for _ in texts]


@contextmanager
def _running_client():
    import app.main as main

    with TestClient(main.app) as client:
        main.app.state.embedding_model = SimpleNamespace(
            embed_texts=_dummy_embed,
            model_name="dummy-test-model",
        )
        yield client


def _ingest_text(client: TestClient, text: str, filename: str = "sample.txt") -> dict:
    response = client.post(
        "/ingest",
        files={"file": (filename, text.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    return response.json()


@pytest.fixture
def storage_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    faiss_index_path = data_dir / "faiss.index"
    sqlite_db_path = data_dir / "metadata.db"

    monkeypatch.setenv("SERVICE_NAME", "test-service")
    monkeypatch.setenv("DISABLE_EMBEDDINGS", "1")
    monkeypatch.setenv("DISABLE_STORAGE", "0")
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("FAISS_INDEX_PATH", str(faiss_index_path))
    monkeypatch.setenv("SQLITE_DB_PATH", str(sqlite_db_path))

    return {
        "faiss_index_path": faiss_index_path,
        "sqlite_db_path": sqlite_db_path,
    }


def test_faiss_index_grows_after_ingestion(storage_paths):
    text = "A" * 1300

    with _running_client() as client:
        result = _ingest_text(client, text, "grow.txt")

    index = faiss.read_index(str(storage_paths["faiss_index_path"]))
    assert index.ntotal == result["num_chunks"]
    assert index.ntotal > 0


def test_sqlite_rows_created_for_document_and_chunks(storage_paths):
    text = "B" * 1200

    with _running_client() as client:
        result = _ingest_text(client, text, "rows.txt")

    document_id = result["document_id"]
    conn = sqlite3.connect(str(storage_paths["sqlite_db_path"]))
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM documents WHERE document_id = ?", (document_id,))
    doc_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (document_id,))
    chunk_count = cur.fetchone()[0]

    conn.close()

    assert doc_count == 1
    assert chunk_count == result["num_chunks"]


def test_restart_service_data_still_present(storage_paths):
    text = "C" * 1000

    with _running_client() as client:
        result = _ingest_text(client, text, "restart.txt")

    first_index = faiss.read_index(str(storage_paths["faiss_index_path"]))
    first_total = first_index.ntotal

    conn = sqlite3.connect(str(storage_paths["sqlite_db_path"]))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    first_docs = cur.fetchone()[0]
    conn.close()

    with _running_client():
        pass

    second_index = faiss.read_index(str(storage_paths["faiss_index_path"]))
    second_total = second_index.ntotal

    conn = sqlite3.connect(str(storage_paths["sqlite_db_path"]))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    second_docs = cur.fetchone()[0]
    conn.close()

    assert first_total == result["num_chunks"]
    assert second_total == first_total
    assert first_docs == 1
    assert second_docs == first_docs


def test_multiple_documents_ingestion_works(storage_paths):
    with _running_client() as client:
        first = _ingest_text(client, "D" * 700, "doc1.txt")
        second = _ingest_text(client, "E" * 900, "doc2.txt")

    assert first["document_id"] != second["document_id"]

    conn = sqlite3.connect(str(storage_paths["sqlite_db_path"]))
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM documents")
    doc_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cur.fetchone()[0]

    conn.close()

    index = faiss.read_index(str(storage_paths["faiss_index_path"]))

    assert doc_count == 2
    assert chunk_count == first["num_chunks"] + second["num_chunks"]
    assert index.ntotal == chunk_count
