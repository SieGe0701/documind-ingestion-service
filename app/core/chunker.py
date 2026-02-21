from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """
    Split `text` into fixed-size character chunks with overlap.

    - `chunk_size`: maximum number of characters per chunk
    - `overlap`: number of characters that overlap between consecutive chunks

    Returns a list of dicts with keys `chunk_id` (1-based int) and `text`.
    Empty chunks are omitted. Order is preserved and deterministic.
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    # Ensure we always make progress even if overlap >= chunk_size
    step = chunk_size - overlap
    if step <= 0:
        step = 1

    chunks: List[Dict] = []
    text_len = len(text)
    start = 0
    chunk_id = 1

    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = text[start:end]
        if piece and piece.strip():
            chunks.append({"chunk_id": chunk_id, "text": piece})
            chunk_id += 1

            # If we've reached the end of the text with this chunk, stop
            if end >= text_len:
                break

        start += step

    return chunks
