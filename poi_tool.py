"""Model-facing, provider-neutral search_poi Tool."""

from __future__ import annotations

from typing import Any

from poi_service import (
    MAX_ANCHOR_LENGTH,
    MAX_CITY_LENGTH,
    MAX_QUERY_LENGTH,
    POIProvider,
    search_poi_service,
)


SEARCH_POI_TOOL = {
    "type": "function",
    "function": {
        "name": "search_poi",
        "description": (
            "搜索中国城市中的真实地点、餐厅、咖啡店、商店、商场或景点。"
            "需要查询城市内地点或某个地标附近地点时使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，例如火锅店、咖啡店、餐厅、亲子景点。",
                    "maxLength": MAX_QUERY_LENGTH,
                },
                "city": {
                    "type": "string",
                    "description": "城市或行政区域，例如郑州、北京、天津武清区。",
                    "maxLength": MAX_CITY_LENGTH,
                },
                "anchor": {
                    "type": "string",
                    "description": "可选的附近地标、商圈或地址，例如故宫、武清站、二七广场。",
                    "maxLength": MAX_ANCHOR_LENGTH,
                },
            },
            "required": ["query", "city"],
            "additionalProperties": False,
        },
    },
}


def search_poi(
    query: Any,
    city: Any,
    anchor: Any = None,
    *,
    provider: POIProvider | None = None,
    **unknown_parameters: Any,
) -> dict[str, Any]:
    """Search real POIs without exposing provider details to the model."""

    if unknown_parameters:
        return {
            "ok": False,
            "status": "invalid_request",
            "error_code": "INVALID_PARAMETERS",
            "message": "地点搜索参数无效。",
            "results": [],
        }
    try:
        return search_poi_service(
            query,
            city,
            anchor,
            provider=provider,
        )
    except Exception:
        return {
            "ok": False,
            "status": "temporarily_unavailable",
            "error_code": "POI_API_UNAVAILABLE",
            "message": "地点搜索服务暂时不可用。",
            "results": [],
        }
