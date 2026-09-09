"""Provider-neutral get_weather tool with safe, stable responses."""

from __future__ import annotations

from typing import Any

from weather_service import (
    UnknownLocationError,
    WeatherProvider,
    WeatherServiceError,
    fetch_weather,
)


MAX_LOCATION_LENGTH = 100

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "查询指定城市的实时天气或近期天气信息。当用户询问当前天气、"
            "今天或明天天气、温度、降雨、风力或天气状况等实时信息时使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市或地区名称，例如郑州、北京、上海。",
                    "maxLength": MAX_LOCATION_LENGTH,
                }
            },
            "required": ["location"],
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


def get_weather(
    location: str,
    *,
    provider: WeatherProvider | None = None,
) -> dict[str, Any]:
    """Return normalized weather for a server-selected provider.

    Only ``location`` is model-visible. ``provider`` is an internal dependency
    that will later be configured by the server, never by model arguments.
    """

    if not isinstance(location, str):
        return _invalid_location()
    normalized_location = location.strip()
    if not normalized_location or len(normalized_location) > MAX_LOCATION_LENGTH:
        return _invalid_location()

    try:
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
