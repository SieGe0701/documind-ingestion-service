import io
import re
from typing import Optional

from pypdf import PdfReader


def load_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF file.

    Extracts text page-by-page, skips empty pages, and preserves
    paragraph separation.

    Args:
        file_bytes: Raw PDF file content as bytes

    Returns:
        Extracted and normalized text content

    Raises:
        ValueError: If PDF is invalid or cannot be read
    """
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)

        if not reader.pages:
            return ""

        text_parts = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text)

        full_text = "\n\n".join(text_parts)
        return _normalize_text(full_text)

    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")


def load_txt(file_bytes: bytes) -> str:
    """
    Extract text from TXT file.

    Handles encoding safely with UTF-8 and fallback encoding,
    normalizes line endings.

    Args:
        file_bytes: Raw TXT file content as bytes

    Returns:
        Extracted and normalized text content

    Raises:
        ValueError: If text cannot be decoded
    """
    # Try UTF-8 first, then fallback to latin-1
    encodings = ["utf-8", "latin-1", "iso-8859-1"]

    text: Optional[str] = None

    for encoding in encodings:
        try:
            text = file_bytes.decode(encoding)
            break
        except (UnicodeDecodeError, AttributeError):
            continue

    if text is None:
        raise ValueError("Unable to decode text file with supported encodings")

    return _normalize_text(text)


def _normalize_text(text: str) -> str:
    """
    Normalize text by collapsing whitespace and trimming.

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text
    """
    # Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple spaces
    text = re.sub(r" +", " ", text)

    # Collapse multiple newlines
    text = re.sub(r"\n\n+", "\n\n", text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text
