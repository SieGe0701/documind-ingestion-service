import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from app.core.document_loader import load_pdf, load_txt
from app.core.chunker import chunk_text

logger = logging.getLogger(__name__)

router = APIRouter()

# Supported MIME types
SUPPORTED_CONTENT_TYPES = {"application/pdf", "text/plain"}


@router.post("/ingest")
async def ingest_file(request: Request, file: UploadFile = File(...)) -> dict:
    """
    Upload and process a document file.

    Supported formats: PDF, TXT

    Args:
        file: The file to ingest

    Returns:
        File metadata including filename, content type, and size

    Raises:
        HTTPException 400: If content type is not supported
    """
    # Validate content type
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type",
        )

    # Read file bytes to parse
    file_bytes = await file.read()

    # Parse based on content type
    try:
        if file.content_type == "application/pdf":
            text = load_pdf(file_bytes)
        else:
            text = load_txt(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Chunk the text (fixed-size, overlapping)
    chunks = chunk_text(text)
    num_chunks = len(chunks)

    # Embed chunks if embedding model is available
    embedding_model = getattr(request.app.state, "embedding_model", None)
    if embedding_model is not None and num_chunks > 0:
        texts = [c["text"] for c in chunks]
        embeddings = embedding_model.embed_texts(texts)
        embedding_model_name = getattr(embedding_model, "model_name", "unknown")
    else:
        embeddings = []
        embedding_model_name = getattr(embedding_model, "model_name", "unknown") if embedding_model is not None else ""

    if len(embeddings) != num_chunks:
        raise HTTPException(status_code=500, detail="Embedding count mismatch")

    document_id = str(uuid4())

    vector_store = getattr(request.app.state, "vector_store", None)
    metadata_store = getattr(request.app.state, "metadata_store", None)
    if vector_store is None or metadata_store is None:
        raise HTTPException(status_code=500, detail="Storage is not initialized")

    vector_metadata = [
        {"document_id": document_id, "chunk_id": chunk["chunk_id"]}
        for chunk in chunks
    ]

    if embeddings:
        vector_store.add_embeddings(embeddings, vector_metadata)

    metadata_store.save_document(
        document_id=document_id,
        filename=file.filename or "unknown",
        upload_timestamp=datetime.now(timezone.utc).isoformat(),
        num_chunks=num_chunks,
        embedding_model=embedding_model_name,
    )
    metadata_store.save_chunks(document_id=document_id, chunks=chunks)

    logger.info(
        f"File uploaded: {file.filename}",
        extra={
            "uploaded_filename": file.filename,
            "uploaded_content_type": file.content_type,
            "document_id": document_id,
            "num_chunks": num_chunks,
        },
    )

    return {
        "document_id": document_id,
        "num_chunks": num_chunks,
        "embedding_model": embedding_model_name,
    }
