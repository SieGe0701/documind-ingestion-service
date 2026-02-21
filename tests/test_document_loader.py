import io
from types import SimpleNamespace
import pytest

from app.core import document_loader as dl


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


def test_load_pdf_skips_empty_pages_and_preserves_paragraphs(monkeypatch):
    # Create fake PdfReader with pages including empty and non-empty
    fake_pages = [FakePage("First page text."), FakePage("\n\n"), FakePage("Second page text.")]

    fake_reader = SimpleNamespace(pages=fake_pages)

    def fake_pdf_reader(_):
        return fake_reader

    monkeypatch.setattr(dl, "PdfReader", fake_pdf_reader)

    pdf_bytes = b"%PDF-FAKE-BYTES"
    text = dl.load_pdf(pdf_bytes)

    # Paragraph separation preserved with double newline between pages
    assert "First page text." in text
    assert "Second page text." in text
    assert "\n\n" in text


def test_load_txt_utf8_and_normalization():
    # Create utf-8 bytes with mixed line endings and extra spaces
    raw = "Line one.\r\n\r\n   Line   two.  \nLine three."
    b = raw.encode("utf-8")

    text = dl.load_txt(b)

    # Normalized line endings and collapsed spaces
    assert "Line one." in text
    assert "Line two." in text
    assert "Line three." in text
    assert "\r" not in text


def test_load_txt_fallback_encoding():
    # Create bytes that are not valid UTF-8 but valid latin-1
    # For example, 0xA3 (pound sign) is valid latin-1 but in isolation may be invalid utf-8
    b = bytes([0xA3, 0x20, 0x54, 0x65, 0x73, 0x74])  # b"\xA3 Test"

    text = dl.load_txt(b)

    assert "Test" in text
    # Ensure the non-ascii character survived via fallback decode
    assert "\u00A3" in text or "\xa3" in text
