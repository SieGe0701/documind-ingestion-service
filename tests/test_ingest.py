import io
import uuid
import pytest
from fastapi.testclient import TestClient
from app.core.chunker import chunk_text


def test_ingest_pdf_file_success(client: TestClient, monkeypatch):
    """Test successful PDF ingestion (parsers mocked)."""
    pdf_content = b"%PDF-1.4\n%Mock PDF content"
    file = ("test.pdf", io.BytesIO(pdf_content), "application/pdf")

    # Mock loader to avoid real PDF parsing
    def fake_load_pdf(b):
        return "A" * len(b)

    monkeypatch.setattr("app.api.ingest.load_pdf", fake_load_pdf)

    response = client.post("/ingest", files={"file": file})

    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert "num_chunks" in data
    assert "embedding_model" in data
    uuid.UUID(data["document_id"])
    assert data["num_chunks"] == len(chunk_text(fake_load_pdf(pdf_content)))
    assert data["embedding_model"] == "dummy-test-model"


def test_ingest_txt_file_success(client: TestClient, monkeypatch):
    """Test successful TXT ingestion (parsers mocked)."""
    txt_content = b"This is a sample text file content."
    file = ("test.txt", io.BytesIO(txt_content), "text/plain")

    def fake_load_txt(b):
        # simulate decoded and normalized text length
        return "B" * len(b)

    monkeypatch.setattr("app.api.ingest.load_txt", fake_load_txt)

    response = client.post("/ingest", files={"file": file})

    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert "num_chunks" in data
    assert "embedding_model" in data
    uuid.UUID(data["document_id"])
    assert data["num_chunks"] == len(chunk_text(fake_load_txt(txt_content)))
    assert data["embedding_model"] == "dummy-test-model"


def test_ingest_unsupported_content_type_returns_400(client: TestClient):
    """Test that unsupported file type returns 400."""
    file_content = b"This is an executable file"
    file = ("test.exe", io.BytesIO(file_content), "application/x-msdownload")

    response = client.post("/ingest", files={"file": file})

    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Unsupported file type"


def test_ingest_docx_unsupported_returns_400(client: TestClient):
    """Test that DOCX files are rejected."""
    file_content = b"Mock DOCX content"
    docx_content_type = (
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document"
    )
    file = ("test.docx", io.BytesIO(file_content), docx_content_type)

    response = client.post("/ingest", files={"file": file})

    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Unsupported file type"


def test_ingest_response_structure(client: TestClient, monkeypatch):
    """Test that response contains expected fields (parsers mocked)."""
    file_content = b"Sample content"
    file = ("sample.pdf", io.BytesIO(file_content), "application/pdf")

    monkeypatch.setattr("app.api.ingest.load_pdf", lambda b: "x" * len(b))

    response = client.post("/ingest", files={"file": file})

    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert "num_chunks" in data
    assert "embedding_model" in data
    uuid.UUID(data["document_id"])
    assert isinstance(data["num_chunks"], int)


def test_ingest_without_file_returns_error(client: TestClient):
    """Test that missing file parameter returns error."""
    response = client.post("/ingest")

    assert response.status_code == 422  # Unprocessable Entity


def test_ingest_large_file(client: TestClient, monkeypatch):
    """Test ingestion of a larger file (parsers mocked)."""
    # Create a 1MB file
    file_content = b"x" * (1024 * 1024)
    file = ("large.pdf", io.BytesIO(file_content), "application/pdf")

    monkeypatch.setattr("app.api.ingest.load_pdf", lambda b: "z" * len(b))

    response = client.post("/ingest", files={"file": file})

    assert response.status_code == 200
    data = response.json()
    fake_text = "z" * len(file_content)
    expected = len(chunk_text(fake_text))
    assert data["num_chunks"] == expected
