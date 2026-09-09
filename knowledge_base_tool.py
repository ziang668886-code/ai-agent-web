"""Knowledge-base search tool for the current server-side visitor.

This module only performs retrieval. It does not read browser state, call the
chat model, persist chat messages, or mutate the knowledge-base index.
"""

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
            "Search the current user's uploaded PDF knowledge base when the "
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

    top_score = float(raw_results[0].get("score", 0.0)) if raw_results else 0.0
    if top_score < KNOWLEDGE_BASE_THRESHOLD:
        return {
            "ok": True,
            "status": "no_relevant_results",
            "message": "当前知识库中没有找到足够相关的资料。",
            "top_score": top_score,
            "results": [],
        }

    results = []
    for rank, item in enumerate(raw_results, start=1):
        results.append(
            {
                "citation": f"来源{rank}",
                "rank": rank,
                "content": str(item.get("chunk_text", "")),
                "source_file": _safe_source_file(item.get("source_file")),
                "page_number": item.get("page_number"),
                "score": float(item.get("score", 0.0)),
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
