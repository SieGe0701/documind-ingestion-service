import logging

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

    text_length = len(text)

    # Chunk the text (fixed-size, overlapping)
    chunks = chunk_text(text)
    num_chunks = len(chunks)

    # Embed chunks if embedding model is available
    embedding_model = getattr(request.app.state, "embedding_model", None)
    if embedding_model is not None and num_chunks > 0:
        texts = [c["text"] for c in chunks]
        embeddings = embedding_model.embed_texts(texts)
        embedding_dim = len(embeddings[0]) if embeddings else 0
        embedding_model_name = getattr(embedding_model, "model_name", "")
    else:
        embedding_dim = 0
        embedding_model_name = ""

    logger.info(
        f"File uploaded: {file.filename}",
        extra={
            "filename": file.filename,
            "content_type": file.content_type,
            "text_length": text_length,
            "num_chunks": num_chunks,
        },
    )

    preview = [c["text"] for c in chunks[:2]]

    return {
        "num_chunks": num_chunks,
        "chunk_preview": preview,
        "embedding_dim": embedding_dim,
        "embedding_model": embedding_model_name,
    }
