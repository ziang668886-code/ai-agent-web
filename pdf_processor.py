"""PDF text extraction and chunking utilities for the first RAG version."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader


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
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n\s*\n(?:\s*\n)+", "\n\n", cleaned)


def _find_natural_end(text: str, start: int, target_size: int) -> int:
    """Find a paragraph, line, or sentence boundary near the target size."""
    hard_end = min(start + target_size, len(text))
    if hard_end == len(text):
        return hard_end
    search_start = start + max(target_size // 2, 1)
    window = text[search_start:hard_end]
    boundaries = list(re.finditer(r"\n\n|\n|[。！？!?；;](?:[\"'”’）)]*)", window))
    return search_start + boundaries[-1].end() if boundaries else hard_end


def _split_page_text(
    text: str,
    target_size: int = 700,
    overlap: int = 100,
    min_size: int = 50,
) -> list[str]:
    """Split one page into overlapping chunks at natural text boundaries."""
    if target_size <= 0:
        raise ValueError("target_size 必须大于 0。")
    if overlap < 0 or overlap >= target_size:
        raise ValueError("overlap 必须大于等于 0 且小于 target_size。")
    if min_size <= 0:
        raise ValueError("min_size 必须大于 0。")
    if len(text) < min_size:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = _find_natural_end(text, start, target_size)
        chunk = text[start:end].strip()
        if len(chunk) >= min_size:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(0, end - overlap)
        while next_start < end and text[next_start].isspace():
            next_start += 1
        if next_start <= start:
            next_start = end
        if len(text) - next_start < min_size:
            break
        start = next_start
    return chunks


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
    safe_source_file = Path(source_file or inferred_name).name

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

    chunks: list[dict] = []
    chunk_index = 0
    for page in pages:
        page_chunks = _split_page_text(
            page["text"],
            target_size=target_size,
            overlap=overlap,
            min_size=min_size,
        )
        for chunk_text in page_chunks:
            chunk_seed = f"{document_id}:{page['page_number']}:{chunk_index}:{chunk_text}"
            chunk_id = hashlib.sha256(chunk_seed.encode("utf-8")).hexdigest()
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_text": chunk_text,
                    "source_file": safe_source_file,
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

    if not chunks:
        raise PDFProcessingError(SCANNED_PDF_ERROR)

    return {
        "source_file": safe_source_file,
        "document_id": document_id,
        "total_pages": len(reader.pages),
        "chunks": chunks,
    }

