"""Shared text cleanup and chunk construction for knowledge-base documents."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any


def safe_source_filename(value: str) -> str:
    """Return a display-only basename on both Windows and Linux."""

    normalized = str(value or "uploaded").replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1].strip()
    return filename or "uploaded"


def clean_text(text: str) -> str:
    """Normalize whitespace while retaining useful paragraph boundaries."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        re.sub(r"[ \t\f\v]+", " ", line).strip()
        for line in text.split("\n")
    ]
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n\s*\n(?:\s*\n)+", "\n\n", cleaned)


def find_natural_end(text: str, start: int, target_size: int) -> int:
    """Find a paragraph, line, or sentence boundary near the target size."""

    hard_end = min(start + target_size, len(text))
    if hard_end == len(text):
        return hard_end
    search_start = start + max(target_size // 2, 1)
    window = text[search_start:hard_end]
    boundaries = list(
        re.finditer(r"\n\n|\n|[。！？!?;；](?:[\"'”’）)]*)", window)
    )
    return search_start + boundaries[-1].end() if boundaries else hard_end


def split_text(
    text: str,
    target_size: int = 700,
    overlap: int = 100,
    min_size: int = 50,
) -> list[str]:
    """Split cleaned text into overlapping chunks at natural boundaries."""

    if target_size <= 0:
        raise ValueError("target_size 必须大于 0。")
    if overlap < 0 or overlap >= target_size:
        raise ValueError("overlap 必须大于等于 0 且小于 target_size。")
    if min_size <= 0:
        raise ValueError("min_size 必须大于 0。")

    normalized = clean_text(text)
    if len(normalized) < min_size:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = find_natural_end(normalized, start, target_size)
        chunk = normalized[start:end].strip()
        if len(chunk) >= min_size:
            chunks.append(chunk)
        if end >= len(normalized):
            break

        next_start = max(0, end - overlap)
        while next_start < end and normalized[next_start].isspace():
            next_start += 1
        if next_start <= start:
            next_start = end
        if len(normalized) - next_start < min_size:
            break
        start = next_start
    return chunks


def build_chunks(
    *,
    document_id: str,
    source_file: str,
    source_type: str,
    text_units: Sequence[Mapping[str, Any]],
    target_size: int = 700,
    overlap: int = 100,
    min_size: int = 50,
) -> list[dict[str, Any]]:
    """Build uniform chunk dictionaries from located text units."""

    chunks: list[dict[str, Any]] = []
    safe_source_file = safe_source_filename(source_file)
    chunk_index = 0

    for unit in text_units:
        unit_text = unit.get("text")
        if not isinstance(unit_text, str):
            continue
        locator_type = str(unit.get("locator_type") or "chunk")
        base_locator_value = unit.get("locator_value")
        page_number = unit.get("page_number")

        for chunk_text in split_text(
            unit_text,
            target_size=target_size,
            overlap=overlap,
            min_size=min_size,
        ):
            locator_value = (
                str(chunk_index + 1)
                if locator_type == "chunk" and base_locator_value is None
                else str(base_locator_value or "")
            )
            seed = (
                f"{document_id}:{source_type}:{locator_type}:"
                f"{locator_value}:{chunk_index}:{chunk_text}"
            )
            chunk_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_text": chunk_text,
                    "source_file": safe_source_file,
                    "source_type": source_type,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "locator_type": locator_type,
                    "locator_value": locator_value,
                }
            )
            chunk_index += 1

    return chunks
