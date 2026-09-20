"""Unified in-memory parser for PDF, DOCX, TXT, and Markdown documents."""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path
from typing import Any, Iterator

from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from pdf_processor import PDFProcessingError, SCANNED_PDF_ERROR, process_pdf
from text_chunker import build_chunks, clean_text, safe_source_filename


MAX_DOCUMENT_SIZE_BYTES = 10 * 1024 * 1024
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_DOCX_ZIP_ENTRIES = 5000
SUPPORTED_SOURCE_TYPES = {"pdf", "docx", "txt", "md"}
SOURCE_TYPE_BY_SUFFIX = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "md",
    ".markdown": "md",
}
SUPPORTED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/x-markdown": "md",
}


class DocumentProcessingError(ValueError):
    """Raised when an uploaded document cannot produce usable text chunks."""


def _document_id(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _source_type(source_file: str, content_type: str | None) -> str:
    suffix = Path(safe_source_filename(source_file)).suffix.lower()
    source_type = SOURCE_TYPE_BY_SUFFIX.get(suffix)
    if source_type is None and content_type:
        source_type = SUPPORTED_CONTENT_TYPES.get(content_type.lower().strip())
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise DocumentProcessingError("仅支持 PDF、Word、TXT 和 Markdown 文件。")
    return source_type


def _decode_utf8(file_bytes: bytes, source_type: str) -> str:
    try:
        return file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        label = "TXT" if source_type == "txt" else "Markdown"
        raise DocumentProcessingError(
            f"{label} 文件必须使用 UTF-8 或 UTF-8 BOM 编码。"
        ) from exc


def _validate_docx_archive(file_bytes: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            members = archive.infolist()
            if len(members) > MAX_DOCX_ZIP_ENTRIES:
                raise DocumentProcessingError("Word 文件内容过于复杂。")
            total_size = sum(member.file_size for member in members)
            if total_size > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise DocumentProcessingError("Word 文件解压后过大。")
    except DocumentProcessingError:
        raise
    except (zipfile.BadZipFile, OSError) as exc:
        raise DocumentProcessingError("Word 文件已损坏或格式无效。") from exc


def _iter_docx_blocks(document: DocumentObject) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in their original document-body order."""

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _table_text(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells = [clean_text(cell.text) for cell in row.cells]
        text_cells = [cell for cell in cells if cell]
        if text_cells:
            rows.append(" | ".join(text_cells))
    return "\n".join(rows)


def _docx_text_units(file_bytes: bytes) -> list[dict[str, Any]]:
    _validate_docx_archive(file_bytes)
    try:
        document = Document(io.BytesIO(file_bytes))
    except Exception as exc:
        raise DocumentProcessingError("Word 文件已损坏或无法读取。") from exc

    units: list[dict[str, Any]] = []
    parts: list[str] = []
    section_title: str | None = None
    section_start = 1
    block_number = 0

    def flush() -> None:
        nonlocal parts
        text = clean_text("\n\n".join(parts))
        if text:
            units.append(
                {
                    "text": text,
                    "page_number": None,
                    "locator_type": "section" if section_title else "paragraph",
                    "locator_value": section_title or str(section_start),
                }
            )
        parts = []

    for block in _iter_docx_blocks(document):
        block_number += 1
        if isinstance(block, Paragraph):
            text = clean_text(block.text)
            if not text:
                continue
            style_name = str(getattr(block.style, "name", "") or "")
            if style_name.lower().startswith("heading"):
                flush()
                section_title = text
                section_start = block_number
                parts.append(text)
            else:
                if not parts:
                    section_start = block_number
                parts.append(text)
        else:
            table_text = _table_text(block)
            if table_text:
                if not parts:
                    section_start = block_number
                parts.append(table_text)
    flush()
    return units


def _strip_markdown_noise(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s*```.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"[`*_~]", "", text)
    return clean_text(text)


def _markdown_text_units(text: str) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    section_title = "正文"
    parts: list[str] = []

    def flush() -> None:
        nonlocal parts
        body = _strip_markdown_noise("\n\n".join(parts))
        if body:
            units.append(
                {
                    "text": body,
                    "page_number": None,
                    "locator_type": "section",
                    "locator_value": section_title,
                }
            )
        parts = []

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        heading = re.match(r"^\s{0,3}#{1,3}\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush()
            section_title = _strip_markdown_noise(heading.group(1)) or "未命名章节"
            parts.append(section_title)
        else:
            parts.append(line)
    flush()
    return units


def process_document(
    file_bytes: bytes,
    source_file: str,
    content_type: str | None = None,
    *,
    target_size: int = 700,
    overlap: int = 100,
    min_size: int = 50,
) -> dict[str, Any]:
    """Parse one supported upload into uniform located chunks in memory."""

    if not isinstance(file_bytes, bytes) or not file_bytes:
        raise DocumentProcessingError("文件内容为空。")
    if len(file_bytes) > MAX_DOCUMENT_SIZE_BYTES:
        raise DocumentProcessingError("文件不能超过 10MB。")

    safe_source_file = safe_source_filename(source_file)
    source_type = _source_type(safe_source_file, content_type)
    document_id = _document_id(file_bytes)

    if source_type == "pdf":
        try:
            return process_pdf(
                file_bytes,
                source_file=safe_source_file,
                target_size=target_size,
                overlap=overlap,
                min_size=min_size,
            )
        except PDFProcessingError as exc:
            message = (
                SCANNED_PDF_ERROR
                if str(exc) == SCANNED_PDF_ERROR
                else "PDF 解析失败，请确认文件完整且包含可复制文字。"
            )
            raise DocumentProcessingError(message) from exc

    if source_type == "docx":
        text_units = _docx_text_units(file_bytes)
    elif source_type == "txt":
        text = clean_text(_decode_utf8(file_bytes, source_type))
        text_units = [
            {
                "text": text,
                "page_number": None,
                "locator_type": "chunk",
                "locator_value": None,
            }
        ]
    else:
        text_units = _markdown_text_units(_decode_utf8(file_bytes, source_type))

    chunks = build_chunks(
        document_id=document_id,
        source_file=safe_source_file,
        source_type=source_type,
        text_units=text_units,
        target_size=target_size,
        overlap=overlap,
        min_size=min_size,
    )
    if not chunks:
        label = {"docx": "Word", "txt": "TXT", "md": "Markdown"}[source_type]
        raise DocumentProcessingError(f"{label} 文件中没有可用的文本内容。")

    return {
        "source_file": safe_source_file,
        "source_type": source_type,
        "document_id": document_id,
        "chunks": chunks,
    }
