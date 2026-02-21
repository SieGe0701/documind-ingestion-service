import pytest

from app.core.chunker import chunk_text


def test_short_text_less_than_chunk_size():
    text = "short text"
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["chunk_id"] == 1


def test_text_exactly_equal_to_chunk_size():
    chunk_size = 20
    text = "x" * chunk_size
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=5)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["chunk_id"] == 1


def test_long_text_overlap_respected_and_no_empty_chunks():
    # Create deterministic text of length 26
    text = "abcdefghijklmnopqrstuvwxyz"
    chunk_size = 10
    overlap = 3

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) > 1

    # Check no empty chunks and sequential chunk_ids
    for i, c in enumerate(chunks, start=1):
        assert c["text"].strip() != ""
        assert c["chunk_id"] == i

    # Check overlap between consecutive chunks
    for a, b in zip(chunks, chunks[1:]):
        assert a["text"][-overlap:] == b["text"][:overlap]


def test_deterministic_chunk_ids_for_various_sizes():
    text = "".join(str(i) for i in range(300))
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    # ids must be 1..N sequential
    ids = [c["chunk_id"] for c in chunks]
    assert ids == list(range(1, len(chunks) + 1))


def test_step_always_progresses_when_overlap_ge_chunk_size():
    text = "abcdefghij" * 5
    # overlap >= chunk_size forces step fallback to 1
    chunks = chunk_text(text, chunk_size=3, overlap=3)
    assert len(chunks) > 1
    # ensure no infinite loops and deterministic ids
    assert [c["chunk_id"] for c in chunks] == list(range(1, len(chunks) + 1))
