"""Knowledge-base search tool for the current server-side visitor.

This module only performs retrieval. It does not read browser state, call the
chat model, persist chat messages, or mutate the knowledge-base index.
"""

import math
from typing import Any

from embedding_service import embed_text
from vector_store import has_knowledge_base, search


KNOWLEDGE_BASE_THRESHOLD = 0.35
"""Initial relevance threshold measured against the current test knowledge base.

Re-evaluate this value when the document collection, chunking strategy, or
embedding model changes.
"""

KNOWLEDGE_BASE_TOP_K = 4
MAX_QUERY_LENGTH = 1000


KNOWLEDGE_BASE_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "knowledge_base_search",
        "description": (
            "Search the current user's uploaded document knowledge base when the "
            "answer may depend on their documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise search query based on the user's question.",
                    "maxLength": MAX_QUERY_LENGTH,
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def _invalid_query(message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "status": "invalid_request",
        "error_code": "INVALID_QUERY",
        "message": message,
        "results": [],
    }


def _temporarily_unavailable(error_code: str) -> dict[str, Any]:
    return {
        "ok": False,
        "status": "temporarily_unavailable",
        "error_code": error_code,
        "message": "知识库检索暂时不可用。",
        "results": [],
    }


def _safe_source_file(value: Any) -> str:
    """Return a display-only filename without exposing a local directory."""

    return str(value or "未知文件").replace("\\", "/").rsplit("/", 1)[-1]


def _safe_score(value: Any) -> float | None:
    """Return one finite numeric similarity score, otherwise ``None``."""

    if value is None or isinstance(value, (bool, str, bytes)):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if math.isfinite(score) else None


def _safe_locator(item: dict[str, Any]) -> dict[str, Any]:
    """Return display-safe source metadata with legacy PDF fallbacks."""

    page_number = item.get("page_number")
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        page_number = None

    source_type = item.get("source_type")
    if source_type not in {"pdf", "docx", "txt", "md"}:
        source_type = "pdf" if page_number is not None else "txt"

    locator_type = item.get("locator_type")
    if not isinstance(locator_type, str) or not locator_type.strip():
        locator_type = "page" if source_type == "pdf" else "chunk"
    else:
        locator_type = locator_type.strip()

    locator_value = item.get("locator_value")
    if not isinstance(locator_value, str) or not locator_value.strip():
        locator_value = str(page_number) if page_number is not None else ""
    else:
        locator_value = locator_value.strip()

    return {
        "source_type": source_type,
        "page_number": page_number,
        "locator_type": locator_type,
        "locator_value": locator_value,
    }


def knowledge_base_search(
    visitor_id: str,
    query: str,
    top_k: int = KNOWLEDGE_BASE_TOP_K,
) -> dict[str, Any]:
    """Search the server-selected visitor's knowledge base.

    ``visitor_id`` is an internal, server-injected value and is intentionally
    absent from ``KNOWLEDGE_BASE_SEARCH_TOOL``. The first version fixes
    ``top_k`` at four; it is also absent from the model-visible schema.
    """

    if not isinstance(query, str):
        return _invalid_query("查询内容必须是字符串。")

    normalized_query = query.strip()
    if not normalized_query:
        return _invalid_query("查询内容不能为空。")
    if len(normalized_query) > MAX_QUERY_LENGTH:
        return _invalid_query(f"查询内容不能超过 {MAX_QUERY_LENGTH} 个字符。")
    if top_k != KNOWLEDGE_BASE_TOP_K:
        return {
            "ok": False,
            "status": "invalid_request",
            "error_code": "INVALID_TOP_K",
            "message": f"当前知识库检索固定返回最多 {KNOWLEDGE_BASE_TOP_K} 条结果。",
            "results": [],
        }

    # has_knowledge_base() performs the existing canonical UUID validation and
    # derives the visitor-isolated index path. No path is accepted from callers.
    try:
        knowledge_base_exists = has_knowledge_base(visitor_id)
    except Exception:
        return _temporarily_unavailable("INDEX_UNAVAILABLE")

    if not knowledge_base_exists:
        return {
            "ok": True,
            "status": "no_knowledge_base",
            "message": "当前用户尚未建立知识库。",
            "results": [],
        }

    try:
        query_embedding = embed_text(normalized_query)
    except Exception:
        return _temporarily_unavailable("EMBEDDING_UNAVAILABLE")

    try:
        raw_results = search(
            visitor_id,
            query_embedding,
            top_k=KNOWLEDGE_BASE_TOP_K,
        )
    except Exception:
        return _temporarily_unavailable("INDEX_UNAVAILABLE")

    valid_scores: list[float] = []
    relevant_results: list[tuple[dict[str, Any], float]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        score = _safe_score(item.get("score"))
        if score is None:
            continue
        valid_scores.append(score)
        if score >= KNOWLEDGE_BASE_THRESHOLD:
            relevant_results.append((item, score))

    top_score = max(valid_scores, default=0.0)
    if not relevant_results:
        return {
            "ok": True,
            "status": "no_relevant_results",
            "message": "当前知识库中没有找到足够相关的资料。",
            "top_score": top_score,
            "results": [],
        }

    results = []
    for rank, (item, score) in enumerate(relevant_results, start=1):
        locator = _safe_locator(item)
        results.append(
            {
                "citation": f"来源{rank}",
                "rank": rank,
                "content": str(item.get("chunk_text", "")),
                "source_file": _safe_source_file(item.get("source_file")),
                **locator,
                "score": score,
            }
        )

    return {
        "ok": True,
        "status": "results",
        "query": normalized_query,
        "result_count": len(results),
        "top_score": top_score,
        "results": results,
    }
