"""Amap Web Service POI 2.0 provider adapter."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from typing import Any

import httpx
from dotenv import load_dotenv

from poi_service import POIProviderNotConfiguredError, POIServiceError


AMAP_BASE_URL = "https://restapi.amap.com"
AMAP_TEXT_SEARCH_PATH = "/v5/place/text"
AMAP_AROUND_SEARCH_PATH = "/v5/place/around"
AMAP_TIMEOUT_SECONDS = 6.0
AMAP_MAX_PAGE_SIZE = 25


class AmapPOIProvider:
    """Call Amap POI 2.0 and return provider-neutral POI dictionaries."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = AMAP_TIMEOUT_SECONDS,
    ) -> None:
        if api_key is None:
            load_dotenv()
            resolved_key = os.getenv("AMAP_API_KEY", "").strip()
        else:
            resolved_key = api_key.strip() if isinstance(api_key, str) else ""
        if not resolved_key:
            raise POIProviderNotConfiguredError("Amap POI provider is not configured")

        self._api_key = resolved_key
        self._client = client or httpx.Client()
        self._owns_client = client is None
        self._timeout = timeout

    @classmethod
    def from_environment(cls) -> "AmapPOIProvider":
        """Create the production provider from server-side configuration."""

        return cls()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "AmapPOIProvider":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def search_text(
        self,
        query: str,
        city: str,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search Amap POI 2.0 by keyword within a region."""

        return self._request_pois(
            AMAP_TEXT_SEARCH_PATH,
            params={
                "keywords": query,
                "region": city,
                "city_limit": "true",
                "show_fields": "business",
                "page_size": _page_size(limit),
                "page_num": 1,
                "output": "json",
            },
        )

    def search_nearby(
        self,
        query: str,
        city: str,
        *,
        longitude: float,
        latitude: float,
        radius: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search Amap POI 2.0 around a trusted longitude/latitude pair."""

        if not _valid_coordinate(longitude, latitude):
            raise POIServiceError("Invalid nearby coordinate")
        if isinstance(radius, bool) or not isinstance(radius, int) or not 0 <= radius <= 50000:
            raise POIServiceError("Invalid nearby radius")

        return self._request_pois(
            AMAP_AROUND_SEARCH_PATH,
            params={
                "keywords": query,
                "location": f"{float(longitude):.6f},{float(latitude):.6f}",
                "radius": radius,
                "region": city,
                "city_limit": "true",
                "show_fields": "business",
                "page_size": _page_size(limit),
                "page_num": 1,
                "output": "json",
            },
        )

    def _request_pois(
        self,
        path: str,
        *,
        params: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        request_params = dict(params)
        request_params["key"] = self._api_key
        try:
            response = self._client.get(
                f"{AMAP_BASE_URL}{path}",
                params=request_params,
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise POIServiceError("Amap POI request failed") from exc

        if not isinstance(payload, Mapping):
            raise POIServiceError("Amap POI returned invalid JSON")
        if str(payload.get("status")) != "1" or str(payload.get("infocode")) != "10000":
            raise POIServiceError("Amap POI returned an unsuccessful status")

        raw_pois = payload.get("pois")
        if not isinstance(raw_pois, list):
            raise POIServiceError("Amap POI response has invalid POI data")

        results: list[dict[str, Any]] = []
        for raw_poi in raw_pois:
            if not isinstance(raw_poi, Mapping):
                continue
            converted = _convert_poi(raw_poi)
            if converted is not None:
                results.append(converted)
        return results


def _convert_poi(raw_poi: Mapping[str, Any]) -> dict[str, Any] | None:
    name = _optional_text(raw_poi.get("name"))
    if not name:
        return None
    business_value = raw_poi.get("business")
    business = business_value if isinstance(business_value, Mapping) else {}

    opening_hours = _optional_text(business.get("opentime_today"))
    if opening_hours is None:
        opening_hours = _optional_text(business.get("opentime_week"))

    return {
        "name": name,
        "address": _optional_text(raw_poi.get("address")),
        "district": _optional_text(raw_poi.get("adname")),
        "category": _optional_text(raw_poi.get("type")),
        "distance_m": _optional_number(raw_poi.get("distance")),
        "rating": _optional_number(business.get("rating")),
        "cost_per_person": _optional_number(business.get("cost")),
        "tags": _parse_tags(business.get("tag")),
        "opening_hours": opening_hours,
        "location": _parse_location(raw_poi.get("location")),
        "province": _optional_text(raw_poi.get("pname")),
        "city": _optional_text(raw_poi.get("cityname")),
    }


def _page_size(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise POIServiceError("Invalid POI result limit")
    return min(limit, AMAP_MAX_PAGE_SIZE)


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _optional_number(value: Any) -> int | float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return int(number) if number.is_integer() else number


def _parse_tags(value: Any) -> list[str]:
    text = _optional_text(value)
    if text is None:
        return []
    return [part.strip() for part in re.split(r"[|,，;；、]", text) if part.strip()]


def _parse_location(value: Any) -> dict[str, float] | None:
    if not isinstance(value, str):
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        return None
    try:
        longitude, latitude = (float(part) for part in parts)
    except ValueError:
        return None
    if not _valid_coordinate(longitude, latitude):
        return None
    return {"longitude": longitude, "latitude": latitude}


def _valid_coordinate(longitude: Any, latitude: Any) -> bool:
    if isinstance(longitude, bool) or isinstance(latitude, bool):
        return False
    try:
        longitude_value = float(longitude)
        latitude_value = float(latitude)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(longitude_value)
        and math.isfinite(latitude_value)
        and -180 <= longitude_value <= 180
        and -90 <= latitude_value <= 90
    )
