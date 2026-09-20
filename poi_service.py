"""Provider-neutral POI search business logic and response normalization."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol


MAX_QUERY_LENGTH = 80
MAX_CITY_LENGTH = 50
MAX_ANCHOR_LENGTH = 80
MAX_POI_RESULTS = 5
DEFAULT_RADIUS_METERS = 3000
ANCHOR_CANDIDATE_LIMIT = 10


class POIServiceError(RuntimeError):
    """Raised when a POI provider is unavailable or returns invalid data."""


class POIProviderNotConfiguredError(POIServiceError):
    """Raised when the server-side POI provider has no credential."""


class POIValidationError(ValueError):
    """Stable validation error used by the service and Tool boundary."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class AnchorNotFoundError(POIServiceError):
    """Raised when the requested nearby anchor cannot be resolved."""


class AmbiguousAnchorError(POIServiceError):
    """Raised when several materially different anchors remain."""

    def __init__(self, candidates: Sequence[Mapping[str, Any]]) -> None:
        super().__init__("Anchor is ambiguous")
        self.candidates = [
            {
                "name": _optional_text(item.get("name")) or "未知地点",
                "district": _optional_text(item.get("district")),
            }
            for item in candidates[:5]
        ]


class POIProvider(Protocol):
    """Interface implemented by a real POI API adapter."""

    def search_text(
        self,
        query: str,
        city: str,
        *,
        limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        """Search POIs inside a city or administrative region."""

    def search_nearby(
        self,
        query: str,
        city: str,
        *,
        longitude: float,
        latitude: float,
        radius: int,
        limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        """Search POIs around a trusted server-selected coordinate."""


def validate_poi_inputs(
    query: Any,
    city: Any,
    anchor: Any = None,
) -> tuple[str, str, str | None]:
    """Validate and normalize model-visible POI inputs."""

    normalized_query = _validated_text(
        query,
        maximum=MAX_QUERY_LENGTH,
        error_code="INVALID_QUERY",
        message="请输入有效的地点搜索关键词。",
    )
    normalized_city = _validated_text(
        city,
        maximum=MAX_CITY_LENGTH,
        error_code="INVALID_CITY",
        message="请输入有效的城市或行政区域。",
    )
    if anchor is None:
        normalized_anchor = None
    else:
        normalized_anchor = _validated_text(
            anchor,
            maximum=MAX_ANCHOR_LENGTH,
            error_code="INVALID_ANCHOR",
            message="请输入有效的附近地标、商圈或地址。",
        )
    return normalized_query, normalized_city, normalized_anchor


def select_anchor_candidate(
    anchor: str,
    city: str,
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Select one trustworthy anchor or raise a controlled resolution error."""

    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise POIServiceError("POI provider returned invalid anchor candidates")

    valid: list[Mapping[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        if not _optional_text(item.get("name")):
            continue
        if _normalized_location(item.get("location")) is None:
            continue
        valid.append(item)

    if not valid:
        raise AnchorNotFoundError("Anchor was not found")

    city_key = _match_key(city)
    city_matched = [item for item in valid if _candidate_matches_city(item, city_key)]
    scoped = city_matched or valid
    anchor_key = _match_key(anchor)

    exact = [item for item in scoped if _match_key(item.get("name")) == anchor_key]
    exact = _deduplicate_candidates(exact)
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        # Amap can return several coordinate records for one famous landmark
        # (for example, a main POI and entrances) with the exact same name.
        # When every exact match belongs to one non-empty district, preserve
        # Amap's relevance ordering and use the first result. Cross-district
        # same-name places remain ambiguous and require user clarification.
        exact_districts = {
            district_key
            for item in exact
            if (district_key := _match_key(item.get("district")))
        }
        if len(exact_districts) == 1:
            return exact[0]
        raise AmbiguousAnchorError(exact)

    strong = [
        item
        for item in scoped
        if anchor_key in _match_key(item.get("name"))
        or _match_key(item.get("name")) in anchor_key
    ]
    strong = _deduplicate_candidates(strong)
    if len(strong) == 1:
        return strong[0]
    if len(strong) > 1:
        # These are related-name matches rather than several exact same-name
        # places. Amap already returns them in relevance order, so use its
        # first city-scoped result (for example, 故宫博物院 for “故宫”). Exact
        # same-name places were handled above and still remain ambiguous when
        # they span materially different districts or coordinates.
        return strong[0]

    scoped = _deduplicate_candidates(scoped)
    if len(scoped) == 1:
        return scoped[0]
    raise AmbiguousAnchorError(scoped)


def search_poi_service(
    query: Any,
    city: Any,
    anchor: Any = None,
    *,
    provider: POIProvider | None = None,
) -> dict[str, Any]:
    """Search POIs through an injected provider and return a stable structure."""

    try:
        normalized_query, normalized_city, normalized_anchor = validate_poi_inputs(
            query,
            city,
            anchor,
        )
    except POIValidationError as exc:
        return _invalid_request(exc.error_code, exc.message)

    created_provider = provider is None
    active_provider: POIProvider | None = provider
    try:
        if active_provider is None:
            from amap_poi_provider import AmapPOIProvider

            active_provider = AmapPOIProvider.from_environment()

        if normalized_anchor is None:
            raw_results = active_provider.search_text(
                normalized_query,
                normalized_city,
                limit=MAX_POI_RESULTS,
            )
            return _results_response(
                query=normalized_query,
                city=normalized_city,
                search_mode="city",
                anchor=None,
                raw_results=raw_results,
            )

        anchor_candidates = active_provider.search_text(
            normalized_anchor,
            normalized_city,
            limit=ANCHOR_CANDIDATE_LIMIT,
        )
        selected_anchor = select_anchor_candidate(
            normalized_anchor,
            normalized_city,
            anchor_candidates,
        )
        anchor_location = _normalized_location(selected_anchor.get("location"))
        if anchor_location is None:
            raise AnchorNotFoundError("Anchor has no usable coordinate")

        raw_results = active_provider.search_nearby(
            normalized_query,
            normalized_city,
            longitude=anchor_location["longitude"],
            latitude=anchor_location["latitude"],
            radius=DEFAULT_RADIUS_METERS,
            limit=MAX_POI_RESULTS,
        )
        return _results_response(
            query=normalized_query,
            city=normalized_city,
            search_mode="nearby",
            anchor=normalized_anchor,
            raw_results=raw_results,
        )
    except AnchorNotFoundError:
        return {
            "ok": False,
            "status": "location_required",
            "message": "没有找到指定地标，请提供更具体的地点。",
            "results": [],
        }
    except AmbiguousAnchorError as exc:
        return {
            "ok": False,
            "status": "ambiguous_location",
            "message": "找到了多个同名地点，请提供更具体的区域。",
            "candidates": exc.candidates,
            "results": [],
        }
    except Exception:
        return _temporarily_unavailable()
    finally:
        if created_provider and active_provider is not None:
            close = getattr(active_provider, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def _validated_text(
    value: Any,
    *,
    maximum: int,
    error_code: str,
    message: str,
) -> str:
    if not isinstance(value, str):
        raise POIValidationError(error_code, message)
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        raise POIValidationError(error_code, message)
    return normalized


def _invalid_request(error_code: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "status": "invalid_request",
        "error_code": error_code,
        "message": message,
        "results": [],
    }


def _temporarily_unavailable() -> dict[str, Any]:
    return {
        "ok": False,
        "status": "temporarily_unavailable",
        "error_code": "POI_API_UNAVAILABLE",
        "message": "地点搜索服务暂时不可用。",
        "results": [],
    }


def _results_response(
    *,
    query: str,
    city: str,
    search_mode: str,
    anchor: str | None,
    raw_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
        raise POIServiceError("POI provider returned invalid results")

    results: list[dict[str, Any]] = []
    for item in raw_results:
        normalized = _normalize_result(item, search_mode)
        if normalized is None:
            continue
        normalized["rank"] = len(results) + 1
        results.append(normalized)
        if len(results) >= MAX_POI_RESULTS:
            break

    if not results:
        return {
            "ok": True,
            "status": "no_results",
            "message": "没有找到符合条件的地点，可以尝试更换关键词或提供更具体的位置。",
            "query": query,
            "city": city,
            "search_mode": search_mode,
            "anchor": anchor,
            "result_count": 0,
            "results": [],
        }

    return {
        "ok": True,
        "status": "results",
        "query": query,
        "city": city,
        "search_mode": search_mode,
        "anchor": anchor,
        "result_count": len(results),
        "results": results,
    }


def _normalize_result(item: Any, search_mode: str) -> dict[str, Any] | None:
    if not isinstance(item, Mapping):
        return None
    name = _optional_text(item.get("name"))
    if not name:
        return None

    tags_value = item.get("tags")
    tags = []
    if isinstance(tags_value, Sequence) and not isinstance(tags_value, (str, bytes)):
        tags = [text for value in tags_value if (text := _optional_text(value))]

    return {
        "name": name,
        "address": _optional_text(item.get("address")),
        "district": _optional_text(item.get("district")),
        "category": _optional_text(item.get("category")),
        "distance_m": (
            _optional_number(item.get("distance_m"), minimum=0)
            if search_mode == "nearby"
            else None
        ),
        "rating": _optional_number(item.get("rating"), minimum=0),
        "cost_per_person": _optional_number(item.get("cost_per_person"), minimum=0),
        "tags": tags,
        "opening_hours": _optional_text(item.get("opening_hours")),
        "location": _normalized_location(item.get("location")),
    }


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _optional_number(value: Any, *, minimum: float | None = None) -> int | float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        return None
    return int(number) if number.is_integer() else number


def _normalized_location(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    longitude = _optional_number(value.get("longitude"))
    latitude = _optional_number(value.get("latitude"))
    if longitude is None or latitude is None:
        return None
    if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        return None
    return {"longitude": float(longitude), "latitude": float(latitude)}


def _match_key(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
    return re.sub(r"(特别行政区|自治区|自治州|省|市|区|县)$", "", text)


def _candidate_matches_city(item: Mapping[str, Any], city_key: str) -> bool:
    if not city_key:
        return True
    context = "".join(str(item.get(field) or "") for field in ("province", "city", "district"))
    context_key = _match_key(context)
    if not context_key:
        return True
    return city_key in context_key or context_key in city_key


def _deduplicate_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    unique: list[Mapping[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in candidates:
        location = _normalized_location(item.get("location"))
        if location is None:
            continue
        key = (
            _match_key(item.get("name")),
            _match_key(item.get("district")),
            round(location["longitude"], 6),
            round(location["latitude"], 6),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
