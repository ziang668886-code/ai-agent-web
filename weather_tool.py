"""Provider-neutral get_weather tool with safe, stable responses."""

from __future__ import annotations

from typing import Any

from weather_service import (
    UnknownLocationError,
    WeatherProvider,
    WeatherServiceError,
    fetch_weather,
    fetch_weather_by_coordinates,
)
from location_context import LocationContextError, validate_current_location


MAX_LOCATION_LENGTH = 100

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "查询指定城市或用户当前位置的实时天气。明确地点使用 location；"
            "用户明确说‘我这里’、‘当前位置’时使用 use_current_location=true。"
            "两个参数必须且只能提供一个。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市或地区名称，例如郑州、北京、上海。",
                    "maxLength": MAX_LOCATION_LENGTH,
                },
                "use_current_location": {
                    "type": "boolean",
                    "description": "仅当用户明确要求查询当前所在位置时设为 true。",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


def _invalid_location() -> dict[str, Any]:
    return {
        "ok": False,
        "status": "invalid_request",
        "error_code": "INVALID_LOCATION",
        "message": "请输入有效的城市或地区名称。",
    }


def _temporarily_unavailable() -> dict[str, Any]:
    return {
        "ok": False,
        "status": "temporarily_unavailable",
        "error_code": "WEATHER_API_UNAVAILABLE",
        "message": "天气服务暂时不可用。",
    }


def _location_required() -> dict[str, Any]:
    return {
        "ok": False,
        "status": "location_required",
        "message": "需要先获取当前位置，或者提供城市或具体地点。",
    }


def get_weather(
    location: str | None = None,
    *,
    use_current_location: bool = False,
    current_location: Any = None,
    provider: WeatherProvider | None = None,
) -> dict[str, Any]:
    """Return normalized weather for a server-selected provider.

    Only ``location`` and ``use_current_location`` are model-visible. Exact
    coordinates and ``provider`` are server-only dependencies.
    """

    if not isinstance(use_current_location, bool):
        return _invalid_location()
    if use_current_location and location is not None:
        return _invalid_location()
    if not use_current_location and not isinstance(location, str):
        return _invalid_location()

    normalized_location = location.strip() if isinstance(location, str) else None
    if (
        not use_current_location
        and (
            not normalized_location
            or len(normalized_location) > MAX_LOCATION_LENGTH
        )
    ):
        return _invalid_location()

    try:
        if use_current_location:
            try:
                trusted_location = validate_current_location(current_location)
            except LocationContextError:
                return _location_required()
            weather = fetch_weather_by_coordinates(
                trusted_location["latitude"],
                trusted_location["longitude"],
                label="当前位置",
                provider=provider,
            )
        else:
            weather = fetch_weather(
                normalized_location,
                provider=provider,
            )
    except UnknownLocationError:
        return _invalid_location()
    except WeatherServiceError:
        return _temporarily_unavailable()
    except Exception:
        return _temporarily_unavailable()

    return {
        "ok": True,
        "status": "success",
        "location": weather["location"],
        "weather": weather["weather"],
        "temperature_c": weather["temperature_c"],
        "feels_like_c": weather["feels_like_c"],
        "humidity": weather["humidity"],
        "wind": weather["wind"],
        "observation_time": weather["observation_time"],
    }
