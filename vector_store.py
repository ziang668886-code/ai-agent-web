"""Visitor-isolated NumPy vector storage for the first RAG version."""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from filelock import FileLock


KNOWLEDGE_BASE_ROOT = Path(__file__).resolve().parent / "data" / "knowledge_bases"
INDEX_FILENAME = "index.npz"
REQUIRED_CHUNK_FIELDS = {
    "chunk_text",
    "document_id",
    "chunk_id",
    "source_file",
    "page_number",
    "chunk_index",
}


class VectorStoreError(RuntimeError):
    """Raised when a vector index cannot be read or written safely."""


def _validated_visitor_id(visitor_id: str) -> str:
    if not isinstance(visitor_id, str):
        raise ValueError("visitor_id 必须是 UUID 字符串。")
    try:
        parsed = uuid.UUID(visitor_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("visitor_id 格式无效。") from exc
    canonical = str(parsed)
    if visitor_id.lower() != canonical:
        raise ValueError("visitor_id 必须使用标准 UUID 格式。")
    return canonical


def _validated_document_id(document_id: str) -> str:
    if not isinstance(document_id, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}",
        document_id,
    ):
        raise ValueError("document_id 必须是有效的 SHA-256。")
    return document_id.lower()


def _index_path(visitor_id: str, *, create_directory: bool = False) -> Path:
    safe_visitor_id = _validated_visitor_id(visitor_id)
    visitor_directory = KNOWLEDGE_BASE_ROOT / safe_visitor_id
    if create_directory:
        visitor_directory.mkdir(parents=True, exist_ok=True)
    return visitor_directory / INDEX_FILENAME


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("embeddings 必须是非空的二维数组。")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("embeddings 包含无效向量。")
    return np.asarray(array / norms, dtype=np.float32)


def _load_index(index_path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(index_path, allow_pickle=False) as data:
            return {name: data[name].copy() for name in data.files}
    except Exception as exc:
        raise VectorStoreError("向量索引读取失败，文件可能已损坏。") from exc


def _chunk_arrays(chunks: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
    for index, chunk in enumerate(chunks):
        missing = REQUIRED_CHUNK_FIELDS.difference(chunk)
        if missing:
            raise ValueError(f"第 {index} 个 Chunk 缺少字段：{', '.join(sorted(missing))}")

    return {
        "chunk_texts": np.asarray([str(c["chunk_text"]) for c in chunks], dtype=np.str_),
        "document_ids": np.asarray([str(c["document_id"]) for c in chunks], dtype=np.str_),
        "chunk_ids": np.asarray([str(c["chunk_id"]) for c in chunks], dtype=np.str_),
        "source_files": np.asarray([Path(str(c["source_file"])).name for c in chunks], dtype=np.str_),
        "page_numbers": np.asarray([int(c["page_number"]) for c in chunks], dtype=np.int32),
        "chunk_indexes": np.asarray([int(c["chunk_index"]) for c in chunks], dtype=np.int32),
    }


def _atomic_save(index_path: Path, index_data: dict[str, np.ndarray]) -> None:
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=index_path.parent,
            prefix=".index-",
            suffix=".npz",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
        np.savez_compressed(temporary_path, **index_data)
        os.replace(temporary_path, index_path)
    except Exception as exc:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)
        raise VectorStoreError("向量索引保存失败。") from exc


def add_chunks(
    visitor_id: str,
    chunks: Sequence[Mapping[str, object]],
    embeddings: np.ndarray,
) -> Path:
    """Add or replace chunks in one visitor's isolated vector index."""
    if not chunks:
        raise ValueError("待写入的 chunks 不能为空。")

    normalized_embeddings = _normalize_rows(embeddings)
    if len(chunks) != normalized_embeddings.shape[0]:
        raise ValueError("Chunk 数量与 Embedding 数量不一致。")

    new_data = _chunk_arrays(chunks)
    new_data["embeddings"] = normalized_embeddings
    index_path = _index_path(visitor_id, create_directory=True)
    lock = FileLock(f"{index_path}.lock")

    with lock:
        if index_path.exists():
            existing = _load_index(index_path)
            if existing["embeddings"].shape[1] != normalized_embeddings.shape[1]:
                raise ValueError("新旧 Embedding 向量维度不一致。")
            new_ids = set(new_data["chunk_ids"].tolist())
            keep_mask = np.asarray(
                [chunk_id not in new_ids for chunk_id in existing["chunk_ids"]],
                dtype=bool,
            )
            merged = {
                key: np.concatenate((existing[key][keep_mask], new_data[key]), axis=0)
                for key in new_data
            }
        else:
            merged = new_data
        _atomic_save(index_path, merged)

    return index_path


def has_knowledge_base(visitor_id: str) -> bool:
    """Return whether a visitor has a saved NumPy index."""
    return _index_path(visitor_id).is_file()


def has_document(visitor_id: str, document_id: str) -> bool:
    """Return whether a document already exists in one visitor's index."""
    safe_document_id = _validated_document_id(document_id)
    index_path = _index_path(visitor_id)
    if not index_path.is_file():
        return False

    lock = FileLock(f"{index_path}.lock")
    with lock:
        data = _load_index(index_path)

    try:
        document_ids = data["document_ids"]
    except KeyError as exc:
        raise VectorStoreError("向量索引缺少文档标识数据。") from exc
    return bool(np.any(document_ids == safe_document_id))


def search(
    visitor_id: str,
    query_embedding: np.ndarray,
    top_k: int = 4,
) -> list[dict]:
    """Return the most similar chunks using cosine similarity."""
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0。")

    index_path = _index_path(visitor_id)
    if not index_path.is_file():
        return []

    query = np.asarray(query_embedding, dtype=np.float32)
    if query.ndim == 2 and query.shape[0] == 1:
        query = query[0]
    if query.ndim != 1 or query.size == 0:
        raise ValueError("query_embedding 必须是一维非空向量。")
    query_norm = float(np.linalg.norm(query))
    if not np.isfinite(query_norm) or query_norm == 0.0:
        raise ValueError("query_embedding 是无效向量。")
    query = np.asarray(query / query_norm, dtype=np.float32)

    lock = FileLock(f"{index_path}.lock")
    with lock:
        data = _load_index(index_path)

    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != query.shape[0]:
        raise ValueError("查询向量与索引向量维度不一致。")

    scores = embeddings @ query
    result_indexes = np.argsort(scores)[::-1][: min(top_k, len(scores))]
    return [
        {
            "score": float(scores[index]),
            "chunk_text": str(data["chunk_texts"][index]),
            "source_file": str(data["source_files"][index]),
            "page_number": int(data["page_numbers"][index]),
            "document_id": str(data["document_ids"][index]),
            "chunk_id": str(data["chunk_ids"][index]),
        }
        for index in result_indexes
    ]
