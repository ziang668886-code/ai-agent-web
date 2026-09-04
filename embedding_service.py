"""Text embedding helpers backed by Volcengine Ark."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark


EMBEDDING_MODEL = "doubao-embedding-vision-251215"


class EmbeddingServiceError(RuntimeError):
    """Raised when an embedding cannot be generated."""


def _sanitize_error(error: Exception, api_key: str) -> str:
    message = f"{type(error).__name__}: {error}"
    return message.replace(api_key, "[REDACTED]") if api_key else message


@lru_cache(maxsize=1)
def _get_client() -> tuple[Ark, str]:
    load_dotenv()
    api_key = os.getenv("ARK_API_KEY", "").strip()
    if not api_key:
        raise EmbeddingServiceError("ARK_API_KEY 未设置，无法调用 Embedding API。")
    return Ark(api_key=api_key), api_key


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise EmbeddingServiceError("Embedding API 返回了无效的零向量。")
    return np.asarray(vector / norm, dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    """Generate one normalized float32 embedding for plain text."""
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("用于生成 Embedding 的文本不能为空。")

    client, api_key = _get_client()
    try:
        response = client.multimodal_embeddings.create(
            model=EMBEDDING_MODEL,
            input=[{"type": "text", "text": cleaned_text}],
            encoding_format="float",
        )
        vector = np.asarray(response.data.embedding, dtype=np.float32)
    except Exception as exc:
        safe_message = _sanitize_error(exc, api_key)
        raise EmbeddingServiceError(f"Embedding API 调用失败：{safe_message}") from exc

    if vector.ndim != 1 or vector.size == 0:
        raise EmbeddingServiceError("Embedding API 返回的向量格式无效。")
    return _l2_normalize(vector)


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Generate normalized embeddings for multiple independent texts."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vectors = [embed_text(text) for text in texts]
    return np.stack(vectors).astype(np.float32, copy=False)


def embed_chunks(chunks: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Generate embeddings from the chunk_text field of chunk dictionaries."""
    texts: list[str] = []
    for index, chunk in enumerate(chunks):
        chunk_text = chunk.get("chunk_text")
        if not isinstance(chunk_text, str) or not chunk_text.strip():
            raise ValueError(f"第 {index} 个 Chunk 缺少有效的 chunk_text。")
        texts.append(chunk_text)
    return embed_texts(texts)

