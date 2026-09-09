"""Provider-neutral weather data access and response normalization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol


class WeatherServiceError(RuntimeError):
    """Raised when a weather provider is unavailable or returns invalid data."""


class UnknownLocationError(WeatherServiceError):
    """Raised when the provider cannot resolve the requested location."""


class WeatherProviderNotConfiguredError(WeatherServiceError):
    """Raised when no server-side weather provider has been configured."""


class WeatherProvider(Protocol):
    """Interface implemented by a real weather API adapter."""

    def fetch_current_weather(self, location: str) -> Mapping[str, Any] | None:
        """Fetch raw current weather data for one normalized location."""


def _optional_number(value: Any, field_name: str) -> int | float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WeatherServiceError(f"Invalid numeric field: {field_name}") from exc
    if not math.isfinite(number):
        raise WeatherServiceError(f"Invalid numeric field: {field_name}")
    return int(number) if number.is_integer() else number


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_weather_response(
    raw_data: Mapping[str, Any] | None,
    requested_location: str,
) -> dict[str, Any]:
    """Convert a provider adapter result into the project's stable fields."""

    if raw_data is None:
        raise UnknownLocationError("Location was not found")
    if not isinstance(raw_data, Mapping):
        raise WeatherServiceError("Weather provider returned invalid data")
    if raw_data.get("found") is False:
        raise UnknownLocationError("Location was not found")

    location = _optional_text(raw_data.get("location")) or requested_location
    weather = _optional_text(raw_data.get("weather"))
    if not weather:
        raise WeatherServiceError("Weather provider response has no weather status")

    return {
        "location": location,
        "weather": weather,
        "temperature_c": _optional_number(
            raw_data.get("temperature_c"),
            "temperature_c",
        ),
        "feels_like_c": _optional_number(
            raw_data.get("feels_like_c"),
            "feels_like_c",
        ),
        "humidity": _optional_number(raw_data.get("humidity"), "humidity"),
        "wind": _optional_text(raw_data.get("wind")),
        "observation_time": _optional_text(raw_data.get("observation_time")),
    }


def fetch_weather(
    location: str,
    *,
    provider: WeatherProvider | None = None,
) -> dict[str, Any]:
    """Fetch weather through a server-selected provider and normalize it.

    When no provider is injected (the production path), QWeather is created
    from server-side environment variables. Tests may inject a provider without
    making network calls. Credentials and provider-specific fields never enter
    this API's normalized response.
    """

    created_provider = provider is None
    if created_provider:
        # Lazy import avoids a module cycle: the adapter reuses the service's
        # stable exception types and protocol.
        from qweather_provider import QWeatherProvider

        provider = QWeatherProvider.from_environment()

    try:
        raw_data = provider.fetch_current_weather(location)
    except UnknownLocationError:
        raise
    except Exception as exc:
        raise WeatherServiceError("Weather provider request failed") from exc
    finally:
        if created_provider:
            # The production QWeather adapter owns an httpx.Client. Tests inject
            # their provider and therefore retain control of its lifecycle.
            close = getattr(provider, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    return normalize_weather_response(raw_data, location)
