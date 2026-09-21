"""Model-facing, provider-neutral search_poi Tool."""

from __future__ import annotations

from typing import Any

from poi_service import (
    MAX_ANCHOR_LENGTH,
    MAX_CITY_LENGTH,
    MAX_QUERY_LENGTH,
    POIProvider,
    search_poi_by_current_location_service,
    search_poi_service,
)


SEARCH_POI_TOOL = {
    "type": "function",
    "function": {
        "name": "search_poi",
        "description": (
            "搜索中国城市中的真实地点、餐厅、咖啡店、商店、商场或景点。"
            "指定城市使用 city，地标附近可加 anchor；用户明确说‘我附近’"
            "或‘离我最近’时使用 use_current_location=true。"
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
                "use_current_location": {
                    "type": "boolean",
                    "description": "仅当用户明确要求搜索当前位置附近时设为 true。",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def search_poi(
    query: Any,
    city: Any = None,
    anchor: Any = None,
    *,
    use_current_location: bool = False,
    current_location: Any = None,
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
    if not isinstance(use_current_location, bool):
        return {
            "ok": False,
            "status": "invalid_request",
            "error_code": "INVALID_PARAMETERS",
            "message": "地点搜索参数无效。",
            "results": [],
        }
    if use_current_location:
        if city is not None or anchor is not None:
            return {
                "ok": False,
                "status": "invalid_request",
                "error_code": "INVALID_PARAMETERS",
                "message": "地点搜索参数无效。",
                "results": [],
            }
        try:
            return search_poi_by_current_location_service(
                query,
                current_location,
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
