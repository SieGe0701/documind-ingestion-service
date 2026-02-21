import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

# Supported MIME types
SUPPORTED_CONTENT_TYPES = {"application/pdf", "text/plain"}


@router.post("/ingest")
async def ingest_file(file: UploadFile = File(...)) -> dict:
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

    # Read file bytes to get size
    file_bytes = await file.read()
    size_bytes = len(file_bytes)

    logger.info(
        f"File uploaded: {file.filename}",
        extra={
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": size_bytes,
        },
    )

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": size_bytes,
    }
