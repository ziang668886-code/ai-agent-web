"""Mock-only tests for the QWeather provider adapter."""

import json
import secrets
import unittest
from unittest.mock import Mock

import httpx

from qweather_provider import QWeatherProvider
from weather_service import (
    UnknownLocationError,
    WeatherServiceError,
    fetch_weather,
    fetch_weather_by_coordinates,
)


TEST_HOST = "test-host.qweatherapi.com"
TEST_KEY = secrets.token_hex(16)


def response(payload, status_code=200):
    request = httpx.Request("GET", f"https://{TEST_HOST}/test")
    return httpx.Response(status_code, json=payload, request=request)


def geo_payload(*locations):
    return {"code": "200", "location": list(locations)}


def current_payload(**overrides):
    payload = {
        "metadata": {"tag": "not-returned"},
        "condition": {"text": "晴", "code": "100"},
        "temperature": {"value": 28.0, "unit": "°C"},
        "feelsLike": {"value": 29.5, "unit": "°C"},
        "humidity": 0.45,
        "wind": {
            "direction": {"degree": 225, "compass": "sw"},
            "speed": {"value": 4.5, "unit": "m/s"},
            "scale": 3,
        },
        "observationTime": "2026-09-09T10:00:00+08:00",
    }
    payload.update(overrides)
    return payload


class QWeatherProviderTests(unittest.TestCase):
    def make_provider(self, *responses):
        client = Mock()
        client.get.side_effect = responses
        provider = QWeatherProvider(
            api_key=TEST_KEY,
            api_host=TEST_HOST,
            client=client,
        )
        return provider, client

    def test_zhengzhou_geo_and_current_weather_success(self):
        provider, client = self.make_provider(
            response(
                geo_payload(
                    {
                        "name": "郑州",
                        "id": "101180101",
                        "lat": "34.7466",
                        "lon": "113.6254",
                    }
                )
            ),
            response(current_payload()),
        )

        result = provider.fetch_current_weather("郑州")

        self.assertEqual(result["location"], "郑州")
        self.assertEqual(result["weather"], "晴")
        self.assertEqual(client.get.call_count, 2)
        geo_call, weather_call = client.get.call_args_list
        self.assertTrue(geo_call.args[0].endswith("/geo/v2/city/lookup"))
        self.assertEqual(geo_call.kwargs["params"]["location"], "郑州")
        self.assertTrue(
            weather_call.args[0].endswith(
                "/weather/v1/current/34.7466/113.6254"
            )
        )

    def test_multiple_geo_results_use_first_result(self):
        provider, client = self.make_provider(
            response(
                geo_payload(
                    {"name": "第一结果", "lat": "34.7", "lon": "113.6"},
                    {"name": "第二结果", "lat": "31.2", "lon": "121.5"},
                )
            ),
            response(current_payload()),
        )

        result = provider.fetch_current_weather("测试城市")

        self.assertEqual(result["location"], "第一结果")
        self.assertIn("/34.7/113.6", client.get.call_args_list[1].args[0])

    def test_coordinates_call_current_weather_without_geo_lookup(self):
        provider, client = self.make_provider(response(current_payload()))

        result = provider.fetch_current_weather_by_coordinates(
            34.7466,
            113.6254,
            label="当前位置",
        )

        self.assertEqual(result["location"], "当前位置")
        self.assertEqual(result["weather"], "晴")
        self.assertEqual(client.get.call_count, 1)
        request_url = client.get.call_args.args[0]
        self.assertTrue(
            request_url.endswith("/weather/v1/current/34.7466/113.6254")
        )
        self.assertNotIn("/geo/", request_url)

    def test_coordinate_service_returns_only_normalized_weather(self):
        provider, _ = self.make_provider(
            response(current_payload(provider_secret="do-not-return"))
        )

        result = fetch_weather_by_coordinates(
            34.7466,
            113.6254,
            label="当前位置",
            provider=provider,
        )

        self.assertEqual(result["location"], "当前位置")
        self.assertNotIn("latitude", result)
        self.assertNotIn("longitude", result)
        self.assertNotIn("provider_secret", result)

    def test_coordinate_provider_rejects_invalid_values_before_request(self):
        provider, client = self.make_provider(response(current_payload()))

        with self.assertRaises(WeatherServiceError):
            provider.fetch_current_weather_by_coordinates(
                91,
                113.6254,
                label="当前位置",
            )

        client.get.assert_not_called()

    def test_unknown_city_raises_stable_location_error(self):
        provider, _ = self.make_provider(response(geo_payload()))

        with self.assertRaises(UnknownLocationError):
            provider.fetch_current_weather("不存在的城市")

    def test_geo_timeout_is_sanitized(self):
        provider, _ = self.make_provider(httpx.ReadTimeout("secret raw error"))

        with self.assertRaisesRegex(
            WeatherServiceError, "QWeather request failed"
        ) as ctx:
            provider.fetch_current_weather("郑州")
        self.assertNotIn("secret raw error", str(ctx.exception))

    def test_geo_malformed_json_is_sanitized(self):
        malformed = Mock()
        malformed.raise_for_status.return_value = None
        malformed.json.side_effect = ValueError("raw json with secret")
        provider, _ = self.make_provider(malformed)

        with self.assertRaisesRegex(WeatherServiceError, "QWeather request failed"):
            provider.fetch_current_weather("郑州")

    def test_weather_response_converts_all_stable_fields(self):
        provider, _ = self.make_provider(
            response(
                geo_payload(
                    {"name": "郑州", "lat": "34.7", "lon": "113.6"}
                )
            ),
            response(current_payload()),
        )

        result = fetch_weather("郑州", provider=provider)

        self.assertEqual(
            result,
            {
                "location": "郑州",
                "weather": "晴",
                "temperature_c": 28,
                "feels_like_c": 29.5,
                "humidity": 45,
                "wind": "西南风 3级 4.5m/s",
                "observation_time": "2026-09-09T10:00:00+08:00",
            },
        )

    def test_weather_timeout_is_sanitized(self):
        provider, _ = self.make_provider(
            response(
                geo_payload(
                    {"name": "郑州", "lat": "34.7", "lon": "113.6"}
                )
            ),
            httpx.ReadTimeout("request contained secret"),
        )

        with self.assertRaisesRegex(
            WeatherServiceError, "QWeather request failed"
        ) as ctx:
            provider.fetch_current_weather("郑州")
        self.assertNotIn("secret", str(ctx.exception))

    def test_weather_non_success_status_is_sanitized(self):
        provider, _ = self.make_provider(
            response(
                geo_payload(
                    {"name": "郑州", "lat": "34.7", "lon": "113.6"}
                )
            ),
            response({"error": "key=secret"}, status_code=401),
        )

        with self.assertRaisesRegex(
            WeatherServiceError, "QWeather request failed"
        ) as ctx:
            provider.fetch_current_weather("郑州")
        self.assertNotIn("secret", str(ctx.exception))

    def test_weather_missing_field_is_rejected(self):
        provider, _ = self.make_provider(
            response(
                geo_payload(
                    {"name": "郑州", "lat": "34.7", "lon": "113.6"}
                )
            ),
            response(current_payload(condition={})),
        )

        with self.assertRaises(WeatherServiceError):
            provider.fetch_current_weather("郑州")

    def test_numeric_types_wind_and_time_are_converted(self):
        provider, _ = self.make_provider(
            response(
                geo_payload(
                    {"name": "郑州", "lat": "34.7", "lon": "113.6"}
                )
            ),
            response(
                current_payload(
                    temperature={"value": "28", "unit": "°C"},
                    feelsLike={"value": "30.25", "unit": "°C"},
                    humidity="0.61",
                )
            ),
        )

        result = provider.fetch_current_weather("郑州")

        self.assertEqual(result["temperature_c"], 28)
        self.assertEqual(result["feels_like_c"], 30.25)
        self.assertEqual(result["humidity"], 61)
        self.assertEqual(result["wind"], "西南风 3级 4.5m/s")
        self.assertEqual(
            result["observation_time"], "2026-09-09T10:00:00+08:00"
        )

    def test_current_v1_without_observation_time_returns_none(self):
        weather = current_payload()
        weather.pop("observationTime")
        provider, _ = self.make_provider(
            response(
                geo_payload(
                    {"name": "郑州", "lat": "34.7", "lon": "113.6"}
                )
            ),
            response(weather),
        )

        result = provider.fetch_current_weather("郑州")

        self.assertIsNone(result["observation_time"])

    def test_api_key_and_provider_fields_never_enter_result(self):
        provider, client = self.make_provider(
            response(
                geo_payload(
                    {"name": "郑州", "lat": "34.7", "lon": "113.6"}
                )
            ),
            response(current_payload(provider_secret="do-not-return")),
        )

        result = provider.fetch_current_weather("郑州")
        serialized = json.dumps(result, ensure_ascii=False)

        self.assertNotIn(TEST_KEY, serialized)
        self.assertNotIn("do-not-return", serialized)
        for call in client.get.call_args_list:
            self.assertNotIn(TEST_KEY, call.args[0])
            self.assertEqual(call.kwargs["headers"]["X-QW-Api-Key"], TEST_KEY)


if __name__ == "__main__":
    unittest.main()
