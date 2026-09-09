"""QWeather provider adapter for the provider-neutral weather service."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv

from weather_service import (
    UnknownLocationError,
    WeatherProviderNotConfiguredError,
    WeatherServiceError,
)


QWEATHER_GEO_PATH = "/geo/v2/city/lookup"
QWEATHER_CURRENT_WEATHER_PATH = "/weather/v1/current/{latitude}/{longitude}"
QWEATHER_TIMEOUT_SECONDS = 6.0

_COMPASS_NAMES = {
    "n": "北风",
    "nne": "东北偏北风",
    "ne": "东北风",
    "ene": "东北偏东风",
    "e": "东风",
    "ese": "东南偏东风",
    "se": "东南风",
    "sse": "东南偏南风",
    "s": "南风",
    "ssw": "西南偏南风",
    "sw": "西南风",
    "wsw": "西南偏西风",
    "w": "西风",
    "wnw": "西北偏西风",
    "nw": "西北风",
    "nnw": "西北偏北风",
    "none": "无持续风向",
    "vrb": "风向不定",
}


class QWeatherProvider:
    """Resolve a city with GeoAPI and query QWeather Current Weather v1."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_host: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = QWEATHER_TIMEOUT_SECONDS,
    ) -> None:
        load_dotenv()
        resolved_key = (api_key or os.getenv("QWEATHER_API_KEY", "")).strip()
        resolved_host = (api_host or os.getenv("QWEATHER_API_HOST", "")).strip()
        if not resolved_key or not resolved_host:
            raise WeatherProviderNotConfiguredError(
                "QWeather provider is not configured"
            )

        self._api_key = resolved_key
        self._base_url = _normalize_api_host(resolved_host)
        self._timeout = timeout
        self._client = client or httpx.Client()
        self._owns_client = client is None

    @classmethod
    def from_environment(cls) -> "QWeatherProvider":
        """Build the provider from server-side environment variables."""

        return cls()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "QWeatherProvider":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def fetch_current_weather(self, location: str) -> Mapping[str, Any]:
        """Return current weather using the project's provider-neutral fields."""

        geo_data = self._get_json(
            QWEATHER_GEO_PATH,
            params={
                "location": location,
                "range": "cn",
                "number": 10,
                "lang": "zh",
            },
            unknown_location_on_empty=True,
        )
        locations = geo_data.get("location")
        if not isinstance(locations, list) or not locations:
            raise UnknownLocationError("Location was not found")

        selected = locations[0]
        if not isinstance(selected, Mapping):
            raise WeatherServiceError("QWeather GeoAPI returned invalid data")

        city_name = _required_text(selected, "name")
        latitude = _coordinate(selected, "lat", -90, 90)
        longitude = _coordinate(selected, "lon", -180, 180)
        weather_data = self._get_json(
            QWEATHER_CURRENT_WEATHER_PATH.format(
                latitude=latitude,
                longitude=longitude,
            ),
            params={"lang": "zh", "localTime": "true"},
        )

        return _convert_current_weather(city_name, weather_data)

    def _get_json(
        self,
        path: str,
        *,
        params: Mapping[str, Any],
        unknown_location_on_empty: bool = False,
    ) -> Mapping[str, Any]:
        try:
            response = self._client.get(
                f"{self._base_url}{path}",
                params=params,
                headers={
                    "X-QW-Api-Key": self._api_key,
                    "Accept": "application/json",
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise WeatherServiceError("QWeather request failed") from exc

        if not isinstance(payload, Mapping):
            raise WeatherServiceError("QWeather returned invalid JSON")

        # GeoAPI v2 includes a string status code. Current Weather v1 uses HTTP
        # status and does not include this legacy-style field.
        code = payload.get("code")
        if code is not None and str(code) != "200":
            if unknown_location_on_empty and str(code) in {"204", "404"}:
                raise UnknownLocationError("Location was not found")
            raise WeatherServiceError("QWeather returned an unsuccessful status")

        return payload


def _normalize_api_host(value: str) -> str:
    candidate = value if "://" in value else f"https://{value}"
    parsed = urlsplit(candidate)
    hostname = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not hostname.endswith(".qweatherapi.com")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise WeatherProviderNotConfiguredError("Invalid QWeather API Host")
    return f"https://{hostname}"


def _required_text(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise WeatherServiceError("QWeather response is missing required data")
    return value.strip()


def _required_number(data: Mapping[str, Any], field: str) -> float:
    value = data.get(field)
    if isinstance(value, bool):
        raise WeatherServiceError("QWeather response contains invalid numeric data")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WeatherServiceError(
            "QWeather response contains invalid numeric data"
        ) from exc
    if not math.isfinite(number):
        raise WeatherServiceError("QWeather response contains invalid numeric data")
    return number


def _coordinate(
    data: Mapping[str, Any],
    field: str,
    minimum: float,
    maximum: float,
) -> str:
    number = _required_number(data, field)
    if not minimum <= number <= maximum:
        raise WeatherServiceError("QWeather GeoAPI returned invalid coordinates")
    return str(data[field]).strip()


def _nested_mapping(data: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = data.get(field)
    if not isinstance(value, Mapping):
        raise WeatherServiceError("QWeather response is missing required data")
    return value


def _clean_number(number: float) -> int | float:
    return int(number) if number.is_integer() else number


def _format_wind(wind_data: Mapping[str, Any]) -> str:
    direction = _nested_mapping(wind_data, "direction")
    speed = _nested_mapping(wind_data, "speed")
    compass = _required_text(direction, "compass").lower()
    direction_text = _COMPASS_NAMES.get(compass, compass.upper())
    scale = _clean_number(_required_number(wind_data, "scale"))
    speed_value = _clean_number(_required_number(speed, "value"))
    speed_unit = _required_text(speed, "unit")
    return f"{direction_text} {scale}级 {speed_value}{speed_unit}"


def _convert_current_weather(
    city_name: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    condition = _nested_mapping(payload, "condition")
    temperature = _nested_mapping(payload, "temperature")
    feels_like = _nested_mapping(payload, "feelsLike")
    wind = _nested_mapping(payload, "wind")

    humidity_fraction = _required_number(payload, "humidity")
    if not 0 <= humidity_fraction <= 1:
        raise WeatherServiceError("QWeather response contains invalid humidity")

    # Current Weather v1 no longer returns the old v7 `obsTime` field. Keep the
    # stable field nullable instead of fabricating an observation timestamp.
    observation_time = payload.get("observationTime") or payload.get("obsTime")
    if observation_time is not None and not isinstance(observation_time, str):
        raise WeatherServiceError("QWeather response contains invalid time data")

    return {
        "location": city_name,
        "weather": _required_text(condition, "text"),
        "temperature_c": _clean_number(_required_number(temperature, "value")),
        "feels_like_c": _clean_number(_required_number(feels_like, "value")),
        "humidity": _clean_number(humidity_fraction * 100),
        "wind": _format_wind(wind),
        "observation_time": observation_time,
    }
