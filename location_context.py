"""Pure validation helpers for browser-provided location data."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


LOCATION_PERMISSION_STATUSES = frozenset(
    {"idle", "granted", "denied", "timeout", "unavailable", "unsupported"}
)
_COMPONENT_RESULT_STATUSES = LOCATION_PERMISSION_STATUSES - {"idle"}
_LOCATION_FIELDS = frozenset({"latitude", "longitude", "accuracy"})
_CURRENT_LOCATION_FIELDS = frozenset(
    {
        "latitude",
        "longitude",
        "accuracy_m",
        "coordinate_system",
        "source",
    }
)
# 允许保留的粗粒度行政字段：到区/县、街道/乡镇一级为止。门牌号、楼栋、
# 住宅小区、完整 formatted_address 以及经纬度都不属于粗粒度字段，永远不保存。
# township 是高德 addressComponent.township；该字段为空时退回同级的街道名 street。
COARSE_LOCATION_FIELDS = frozenset(
    {"province", "city", "district", "township", "street", "label"}
)


class LocationContextError(ValueError):
    """A safe, user-input validation error without browser internals."""


def _validate_number(
    value: Any,
    *,
    field_name: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LocationContextError(f"{field_name} must be a finite number")

    number = float(value)
    if not math.isfinite(number):
        raise LocationContextError(f"{field_name} must be a finite number")
    if number < minimum or (maximum is not None and number > maximum):
        raise LocationContextError(f"{field_name} is outside the allowed range")
    return number


def validate_location_payload(payload: Any) -> dict[str, Any]:
    """Validate an exact browser location payload and return trusted WGS84 data.

    The browser may only provide ``latitude``, ``longitude`` and ``accuracy``.
    Coordinate system and source are assigned by the server rather than trusted
    from browser-controlled input.
    """

    if not isinstance(payload, Mapping):
        raise LocationContextError("location payload must be an object")
    if set(payload) != _LOCATION_FIELDS:
        raise LocationContextError("location payload fields are invalid")

    latitude = _validate_number(
        payload["latitude"], field_name="latitude", minimum=-90.0, maximum=90.0
    )
    longitude = _validate_number(
        payload["longitude"],
        field_name="longitude",
        minimum=-180.0,
        maximum=180.0,
    )
    accuracy_m = _validate_number(
        payload["accuracy"], field_name="accuracy", minimum=0.0
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "accuracy_m": accuracy_m,
        "coordinate_system": "wgs84",
        "source": "browser_geolocation",
    }


def validate_geolocation_result(
    payload: Any,
) -> tuple[str, dict[str, Any] | None]:
    """Validate the exact result emitted by the browser component."""

    if not isinstance(payload, Mapping):
        raise LocationContextError("geolocation result must be an object")

    status = payload.get("status")
    if status not in _COMPONENT_RESULT_STATUSES:
        raise LocationContextError("geolocation status is invalid")

    expected_fields = {"status", "location"} if status == "granted" else {"status"}
    if set(payload) != expected_fields:
        raise LocationContextError("geolocation result fields are invalid")

    if status == "granted":
        return status, validate_location_payload(payload["location"])
    return status, None


def validate_current_location(payload: Any) -> dict[str, Any]:
    """Revalidate the trusted shape stored in the current Streamlit session."""

    if not isinstance(payload, Mapping):
        raise LocationContextError("current location must be an object")
    fields = set(payload)
    if (
        not _CURRENT_LOCATION_FIELDS.issubset(fields)
        or fields - (_CURRENT_LOCATION_FIELDS | COARSE_LOCATION_FIELDS)
    ):
        raise LocationContextError("current location fields are invalid")
    coarse_fields = fields & COARSE_LOCATION_FIELDS
    if coarse_fields and "label" not in coarse_fields:
        raise LocationContextError("current location label is missing")
    if payload.get("coordinate_system") != "wgs84":
        raise LocationContextError("current location coordinate system is invalid")
    if payload.get("source") != "browser_geolocation":
        raise LocationContextError("current location source is invalid")

    validated = validate_location_payload(
        {
            "latitude": payload["latitude"],
            "longitude": payload["longitude"],
            "accuracy": payload["accuracy_m"],
        }
    )
    for field in COARSE_LOCATION_FIELDS:
        if field not in payload:
            continue
        value = payload[field]
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > 100:
            raise LocationContextError("current location context is invalid")
        validated[field] = value.strip()
    return validated
