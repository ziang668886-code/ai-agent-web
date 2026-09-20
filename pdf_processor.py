"""PDF text extraction and chunking utilities for the first RAG version."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader

from text_chunker import (
    build_chunks,
    clean_text,
    find_natural_end,
    safe_source_filename,
    split_text,
)


SCANNED_PDF_ERROR = "当前仅支持可以复制文字的 PDF，暂不支持纯扫描件。"


class PDFProcessingError(ValueError):
    """Raised when a PDF cannot be converted into usable text chunks."""


def _read_pdf_bytes(pdf_source: str | Path | bytes | BinaryIO) -> tuple[bytes, str]:
    """Read PDF bytes and infer a display-safe source filename."""
    if isinstance(pdf_source, (str, Path)):
        path = Path(pdf_source)
        return path.read_bytes(), path.name
    if isinstance(pdf_source, bytes):
        return pdf_source, "uploaded.pdf"

    filename = Path(getattr(pdf_source, "name", "uploaded.pdf")).name
    if hasattr(pdf_source, "seek"):
        pdf_source.seek(0)
    data = pdf_source.read()
    if hasattr(pdf_source, "seek"):
        pdf_source.seek(0)
    if not isinstance(data, bytes):
        raise TypeError("PDF 文件必须以二进制方式读取。")
    return data, filename


def _clean_page_text(text: str) -> str:
    """Normalize whitespace while retaining paragraph boundaries."""
    return clean_text(text)


def _find_natural_end(text: str, start: int, target_size: int) -> int:
    """Find a paragraph, line, or sentence boundary near the target size."""
    return find_natural_end(text, start, target_size)


def _split_page_text(
    text: str,
    target_size: int = 700,
    overlap: int = 100,
    min_size: int = 50,
) -> list[str]:
    """Split one page into overlapping chunks at natural text boundaries."""
    return split_text(
        text,
        target_size=target_size,
        overlap=overlap,
        min_size=min_size,
    )


def process_pdf(
    pdf_source: str | Path | bytes | BinaryIO,
    *,
    source_file: str | None = None,
    target_size: int = 700,
    overlap: int = 100,
    min_size: int = 50,
) -> dict:
    """Extract a PDF page by page and return metadata and chunks."""
    pdf_bytes, inferred_name = _read_pdf_bytes(pdf_source)
    document_id = hashlib.sha256(pdf_bytes).hexdigest()
    safe_source_file = safe_source_filename(source_file or inferred_name)

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise PDFProcessingError(f"PDF 文件读取失败：{exc}") from exc

    pages: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = _clean_page_text(page.extract_text() or "")
        except Exception as exc:
            raise PDFProcessingError(f"PDF 第 {page_number} 页文本提取失败：{exc}") from exc
        pages.append({"page_number": page_number, "text": page_text})

    if not any(page["text"] for page in pages):
        raise PDFProcessingError(SCANNED_PDF_ERROR)

    chunks = build_chunks(
        document_id=document_id,
        source_file=safe_source_file,
        source_type="pdf",
        text_units=[
            {
                "text": page["text"],
                "page_number": page["page_number"],
                "locator_type": "page",
                "locator_value": str(page["page_number"]),
            }
            for page in pages
        ],
        target_size=target_size,
        overlap=overlap,
        min_size=min_size,
    )

    if not chunks:
        raise PDFProcessingError(SCANNED_PDF_ERROR)

    return {
        "source_file": safe_source_file,
        "source_type": "pdf",
        "document_id": document_id,
        "total_pages": len(reader.pages),
        "chunks": chunks,
    }
