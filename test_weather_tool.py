"""Unit tests for the provider-neutral weather tool."""

import json
import unittest
from unittest.mock import Mock

from weather_service import UnknownLocationError
from weather_tool import GET_WEATHER_TOOL, MAX_LOCATION_LENGTH, get_weather


class WeatherToolTests(unittest.TestCase):
    def current_location(self):
        return {
            "latitude": 34.7466,
            "longitude": 113.6254,
            "accuracy_m": 15.0,
            "coordinate_system": "wgs84",
            "source": "browser_geolocation",
        }

    def test_normal_city_is_trimmed_and_provider_is_called_once(self):
        provider = Mock()
        provider.fetch_current_weather.return_value = {
            "location": "郑州",
            "weather": "晴",
        }

        result = get_weather("  郑州  ", provider=provider)

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["location"], "郑州")
        provider.fetch_current_weather.assert_called_once_with("郑州")

    def test_empty_location_is_rejected_without_provider_call(self):
        provider = Mock()

        result = get_weather("  \n ", provider=provider)

        self.assertEqual(result["error_code"], "INVALID_LOCATION")
        provider.fetch_current_weather.assert_not_called()

    def test_non_string_location_is_rejected(self):
        result = get_weather(123)  # type: ignore[arg-type]

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "INVALID_LOCATION")

    def test_overlong_location_is_rejected(self):
        provider = Mock()

        result = get_weather("城" * (MAX_LOCATION_LENGTH + 1), provider=provider)

        self.assertEqual(result["error_code"], "INVALID_LOCATION")
        provider.fetch_current_weather.assert_not_called()

    def test_provider_result_is_converted_to_stable_structure(self):
        provider = Mock()
        provider.fetch_current_weather.return_value = {
            "location": "郑州市",
            "weather": "多云",
            "temperature_c": "28",
            "feels_like_c": "29.5",
            "humidity": "45",
            "wind": "东北风 2 级",
            "observation_time": "2026-09-09T10:00:00+08:00",
            "api_key": "must-not-leak",
            "provider_internal_id": "internal",
        }

        result = get_weather("郑州", provider=provider)

        self.assertEqual(
            result,
            {
                "ok": True,
                "status": "success",
                "location": "郑州市",
                "weather": "多云",
                "temperature_c": 28,
                "feels_like_c": 29.5,
                "humidity": 45,
                "wind": "东北风 2 级",
                "observation_time": "2026-09-09T10:00:00+08:00",
            },
        )
        self.assertNotIn("must-not-leak", str(result))
        self.assertNotIn("provider_internal_id", result)

    def test_provider_timeout_is_sanitized(self):
        provider = Mock()
        provider.fetch_current_weather.side_effect = TimeoutError(
            "https://provider.example/secret"
        )

        result = get_weather("北京", provider=provider)

        self.assertEqual(result["error_code"], "WEATHER_API_UNAVAILABLE")
        self.assertNotIn("provider.example", str(result))

    def test_provider_exception_is_sanitized(self):
        provider = Mock()
        provider.fetch_current_weather.side_effect = RuntimeError(
            "api_key=secret-value"
        )

        result = get_weather("上海", provider=provider)

        self.assertEqual(result["status"], "temporarily_unavailable")
        self.assertNotIn("secret-value", str(result))

    def test_malformed_provider_response_is_temporarily_unavailable(self):
        provider = Mock()
        provider.fetch_current_weather.return_value = {
            "location": "上海",
            "provider_debug": "internal response without weather",
        }

        result = get_weather("上海", provider=provider)

        self.assertEqual(result["error_code"], "WEATHER_API_UNAVAILABLE")
        self.assertNotIn("provider_debug", str(result))

    def test_unknown_location_from_provider_is_invalid_request(self):
        provider = Mock()
        provider.fetch_current_weather.side_effect = UnknownLocationError(
            "unknown location"
        )

        result = get_weather("不存在的城市", provider=provider)

        self.assertEqual(result["error_code"], "INVALID_LOCATION")

    def test_unknown_location_response_is_invalid_request(self):
        provider = Mock()
        provider.fetch_current_weather.return_value = {"found": False}

        result = get_weather("不存在的城市", provider=provider)

        self.assertEqual(result["error_code"], "INVALID_LOCATION")

    def test_unconfigured_provider_is_temporarily_unavailable(self):
        result = get_weather("北京")

        self.assertEqual(result["error_code"], "WEATHER_API_UNAVAILABLE")

    def test_current_location_calls_coordinate_provider_path(self):
        provider = Mock()
        provider.fetch_current_weather_by_coordinates.return_value = {
            "location": "当前位置",
            "weather": "晴",
        }

        result = get_weather(
            use_current_location=True,
            current_location=self.current_location(),
            provider=provider,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["location"], "当前位置")
        provider.fetch_current_weather.assert_not_called()
        provider.fetch_current_weather_by_coordinates.assert_called_once_with(
            34.7466,
            113.6254,
            label="当前位置",
        )

    def test_current_location_missing_returns_location_required(self):
        provider = Mock()

        result = get_weather(
            use_current_location=True,
            current_location=None,
            provider=provider,
        )

        self.assertEqual(result["status"], "location_required")
        provider.fetch_current_weather_by_coordinates.assert_not_called()

    def test_location_and_current_location_are_mutually_exclusive(self):
        result = get_weather(
            "北京",
            use_current_location=True,
            current_location=self.current_location(),
        )

        self.assertEqual(result["status"], "invalid_request")

    def test_missing_location_choice_is_invalid(self):
        self.assertEqual(get_weather()["status"], "invalid_request")

    def test_current_location_result_never_returns_coordinates(self):
        provider = Mock()
        provider.fetch_current_weather_by_coordinates.return_value = {
            "location": "当前位置",
            "weather": "晴",
            "latitude": 34.7466,
            "longitude": 113.6254,
            "accuracy": 15,
        }

        result = get_weather(
            use_current_location=True,
            current_location=self.current_location(),
            provider=provider,
        )

        self.assertNotIn("latitude", result)
        self.assertNotIn("longitude", result)
        self.assertNotIn("accuracy", result)

    def test_schema_exposes_location_choice_without_coordinates(self):
        function_schema = GET_WEATHER_TOOL["function"]
        properties = function_schema["parameters"]["properties"]
        schema_text = json.dumps(GET_WEATHER_TOOL, ensure_ascii=False)

        self.assertEqual(function_schema["name"], "get_weather")
        self.assertEqual(set(properties), {"location", "use_current_location"})
        self.assertNotIn("latitude", schema_text)
        self.assertNotIn("longitude", schema_text)
        self.assertNotIn("accuracy", schema_text)
        self.assertNotIn("provider", schema_text)
        self.assertNotIn("api_key", schema_text.casefold())


if __name__ == "__main__":
    unittest.main()
