"""Server-side coarse location enrichment for the current browser session."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from location_context import (
    COARSE_LOCATION_FIELDS,
    LocationContextError,
    validate_current_location,
)


class LocationServiceError(RuntimeError):
    """Raised when coarse location context cannot be resolved safely."""


class LocationProvider(Protocol):
    def convert_wgs84_to_gcj02(
        self,
        latitude: float,
        longitude: float,
    ) -> Mapping[str, float]: ...

    def reverse_geocode(
        self,
        *,
        latitude: float,
        longitude: float,
    ) -> Mapping[str, str]: ...


def resolve_current_location(
    current_location: Any,
    *,
    provider: LocationProvider | None = None,
) -> dict[str, Any]:
    """Add only coarse administrative context to a validated session location."""

    try:
        trusted = validate_current_location(current_location)
    except LocationContextError as exc:
        raise LocationServiceError("Current location is unavailable") from exc

    if trusted.get("label"):
        return trusted

    created_provider = provider is None
    active_provider = provider
    try:
        if active_provider is None:
            from amap_poi_provider import AmapPOIProvider

            active_provider = AmapPOIProvider.from_environment()

        converted = active_provider.convert_wgs84_to_gcj02(
            trusted["latitude"],
            trusted["longitude"],
        )
        coarse = active_provider.reverse_geocode(
            latitude=converted["latitude"],
            longitude=converted["longitude"],
        )
        if not isinstance(coarse, Mapping):
            raise LocationServiceError("Reverse geocode returned invalid data")

        enriched = dict(trusted)
        for field in sorted(COARSE_LOCATION_FIELDS):
            value = coarse.get(field)
            if isinstance(value, str) and value.strip():
                enriched[field] = value.strip()
        if "label" not in enriched:
            raise LocationServiceError("Reverse geocode returned no coarse label")
        return validate_current_location(enriched)
    except LocationServiceError:
        raise
    except Exception as exc:
        raise LocationServiceError("Location service is temporarily unavailable") from exc
    finally:
        if created_provider and active_provider is not None:
            close = getattr(active_provider, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
