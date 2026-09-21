import sys
import types
import unittest
from unittest.mock import Mock

try:
    import httpx
except ModuleNotFoundError:
    httpx = types.ModuleType("httpx")

    class HTTPError(Exception):
        pass

    class TimeoutException(HTTPError):
        pass

    class HTTPStatusError(HTTPError):
        def __init__(self, message, **_kwargs):
            super().__init__(message)

    httpx.HTTPError = HTTPError
    httpx.TimeoutException = TimeoutException
    httpx.HTTPStatusError = HTTPStatusError
    httpx.Client = Mock
    sys.modules["httpx"] = httpx

try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv

from amap_poi_provider import (
    AMAP_AROUND_SEARCH_PATH,
    AMAP_COORDINATE_CONVERT_PATH,
    AMAP_REVERSE_GEOCODE_PATH,
    AMAP_TEXT_SEARCH_PATH,
    AmapPOIProvider,
)
from poi_service import POIProviderNotConfiguredError, POIServiceError


def success_payload(pois=None):
    return {
        "status": "1",
        "info": "OK",
        "infocode": "10000",
        "count": str(len(pois or [])),
        "pois": pois or [],
    }


def raw_poi(**overrides):
    value = {
        "name": "示例火锅",
        "address": "示例路1号",
        "pname": "天津市",
        "cityname": "天津市",
        "adname": "武清区",
        "type": "餐饮服务;中餐厅;火锅店",
        "location": "117.000001,39.000002",
        "distance": "860",
        "business": {
            "rating": "4.6",
            "cost": "92",
            "tag": "火锅|朋友聚餐",
            "opentime_today": "10:00-22:00",
        },
    }
    value.update(overrides)
    return value


def mock_client_with_payload(payload):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    client = Mock()
    client.get.return_value = response
    return client, response


class AmapPOIProviderTests(unittest.TestCase):
    def make_provider(self, payload, key="unit-test-key"):
        client, response = mock_client_with_payload(payload)
        return AmapPOIProvider(api_key=key, client=client), client, response

    def test_text_search_uses_current_v5_endpoint_and_official_parameters(self):
        provider, client, _ = self.make_provider(success_payload([raw_poi()]))
        provider.search_text("咖啡店", "郑州", limit=5)
        args, kwargs = client.get.call_args
        self.assertTrue(args[0].endswith(AMAP_TEXT_SEARCH_PATH))
        self.assertNotIn("/v3/", args[0])
        params = kwargs["params"]
        self.assertEqual(params["keywords"], "咖啡店")
        self.assertEqual(params["region"], "郑州")
        self.assertEqual(params["city_limit"], "true")
        self.assertEqual(params["show_fields"], "business")
        self.assertEqual(params["page_size"], 5)

    def test_nearby_search_uses_longitude_first_and_fixed_inputs(self):
        provider, client, _ = self.make_provider(success_payload([raw_poi()]))
        provider.search_nearby(
            "餐厅",
            "北京",
            longitude=116.397128,
            latitude=39.916527,
            radius=3000,
            limit=5,
        )
        args, kwargs = client.get.call_args
        self.assertTrue(args[0].endswith(AMAP_AROUND_SEARCH_PATH))
        params = kwargs["params"]
        self.assertEqual(params["location"], "116.397128,39.916527")
        self.assertEqual(params["radius"], 3000)
        self.assertEqual(params["region"], "北京")

    def test_gps_coordinate_conversion_uses_official_endpoint_and_order(self):
        payload = {
            "status": "1",
            "info": "OK",
            "infocode": "10000",
            "locations": "113.631000,34.751000",
        }
        provider, client, _ = self.make_provider(payload)

        result = provider.convert_wgs84_to_gcj02(34.7466, 113.6254)

        args, kwargs = client.get.call_args
        self.assertTrue(args[0].endswith(AMAP_COORDINATE_CONVERT_PATH))
        self.assertEqual(kwargs["params"]["locations"], "113.625400,34.746600")
        self.assertEqual(kwargs["params"]["coordsys"], "gps")
        self.assertEqual(result, {"longitude": 113.631, "latitude": 34.751})

    def test_reverse_geocode_returns_township_but_no_precise_address(self):
        payload = {
            "status": "1",
            "info": "OK",
            "infocode": "10000",
            "regeocode": {
                "formatted_address": "河南省郑州市金水区敏感路1号敏感小区",
                "addressComponent": {
                    "province": "河南省",
                    "city": "郑州市",
                    "district": "金水区",
                    "township": "莲湖街道",
                    "streetNumber": {"street": "敏感路", "number": "1号"},
                    "neighborhood": {"name": "敏感小区"},
                    "building": {"name": "敏感楼栋"},
                },
            },
        }
        provider, client, _ = self.make_provider(payload)

        result = provider.reverse_geocode(
            latitude=34.751,
            longitude=113.631,
        )

        args, kwargs = client.get.call_args
        self.assertTrue(args[0].endswith(AMAP_REVERSE_GEOCODE_PATH))
        self.assertEqual(kwargs["params"]["extensions"], "base")
        self.assertEqual(
            result,
            {
                "province": "河南省",
                "city": "郑州市",
                "district": "金水区",
                "township": "莲湖街道",
                "label": "河南省郑州市金水区莲湖街道",
            },
        )
        # 门牌号、街道名、小区、楼栋与完整地址都不保留。
        self.assertNotIn("敏感", str(result))
        self.assertNotIn("formatted_address", result)
        self.assertNotIn("street", result)
        self.assertNotIn("number", result)

    def test_reverse_geocode_falls_back_to_street_when_township_is_empty(self):
        payload = {
            "status": "1",
            "info": "OK",
            "infocode": "10000",
            "regeocode": {
                "formatted_address": "河南省郑州市金水区花园路1号",
                "addressComponent": {
                    "province": "河南省",
                    "city": "郑州市",
                    "district": "金水区",
                    "township": [],
                    "streetNumber": {"street": "花园路", "number": "1号"},
                },
            },
        }
        provider, _, _ = self.make_provider(payload)

        result = provider.reverse_geocode(latitude=34.751, longitude=113.631)

        self.assertEqual(result["street"], "花园路")
        self.assertNotIn("township", result)
        self.assertEqual(result["label"], "河南省郑州市金水区花园路")
        self.assertNotIn("1号", str(result))

    def test_gps_nearby_search_does_not_add_city_filter(self):
        provider, client, _ = self.make_provider(success_payload([raw_poi()]))

        provider.search_nearby(
            "餐厅",
            None,
            longitude=113.631,
            latitude=34.751,
            radius=3000,
            limit=5,
        )

        params = client.get.call_args.kwargs["params"]
        self.assertNotIn("region", params)
        self.assertNotIn("city_limit", params)

    def test_coordinate_conversion_failure_is_sanitized(self):
        provider, _, _ = self.make_provider(
            {
                "status": "1",
                "info": "OK",
                "infocode": "10000",
                "locations": "secret-invalid-location",
            }
        )

        with self.assertRaises(POIServiceError) as caught:
            provider.convert_wgs84_to_gcj02(34.7466, 113.6254)

        self.assertNotIn("secret", str(caught.exception))

    def test_all_supported_fields_are_converted(self):
        provider, _, _ = self.make_provider(success_payload([raw_poi()]))
        result = provider.search_text("火锅店", "天津", limit=5)[0]
        self.assertEqual(result["name"], "示例火锅")
        self.assertEqual(result["district"], "武清区")
        self.assertEqual(result["distance_m"], 860)
        self.assertEqual(result["rating"], 4.6)
        self.assertEqual(result["cost_per_person"], 92)
        self.assertEqual(result["tags"], ["火锅", "朋友聚餐"])
        self.assertEqual(result["opening_hours"], "10:00-22:00")
        self.assertEqual(result["location"], {"longitude": 117.000001, "latitude": 39.000002})

    def test_missing_business_fields_are_not_fabricated(self):
        provider, _, _ = self.make_provider(success_payload([raw_poi(business={})]))
        result = provider.search_text("公园", "北京", limit=5)[0]
        self.assertIsNone(result["rating"])
        self.assertIsNone(result["cost_per_person"])
        self.assertEqual(result["tags"], [])
        self.assertIsNone(result["opening_hours"])

    def test_weekly_opening_hours_is_used_only_when_actually_returned(self):
        provider, _, _ = self.make_provider(
            success_payload([raw_poi(business={"opentime_week": "周一至周五 09:00-18:00"})])
        )
        result = provider.search_text("商场", "北京", limit=5)[0]
        self.assertEqual(result["opening_hours"], "周一至周五 09:00-18:00")

    def test_invalid_location_is_safely_converted_to_none(self):
        provider, _, _ = self.make_provider(success_payload([raw_poi(location="bad")]))
        result = provider.search_text("火锅店", "天津", limit=5)[0]
        self.assertIsNone(result["location"])

    def test_invalid_optional_numbers_are_safely_converted_to_none(self):
        provider, _, _ = self.make_provider(
            success_payload([raw_poi(distance="unknown", business={"rating": "-", "cost": []})])
        )
        result = provider.search_text("火锅店", "天津", limit=5)[0]
        self.assertIsNone(result["distance_m"])
        self.assertIsNone(result["rating"])
        self.assertIsNone(result["cost_per_person"])

    def test_timeout_is_converted_to_safe_provider_error(self):
        client = Mock()
        client.get.side_effect = httpx.TimeoutException("timeout with secret")
        provider = AmapPOIProvider(api_key="unit-test-key", client=client)
        with self.assertRaisesRegex(POIServiceError, "Amap POI request failed") as caught:
            provider.search_text("咖啡店", "郑州", limit=5)
        self.assertNotIn("secret", str(caught.exception))

    def test_http_error_is_converted_to_safe_provider_error(self):
        client, response = mock_client_with_payload(success_payload())
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "raw response secret",
            request=Mock(),
            response=Mock(),
        )
        provider = AmapPOIProvider(api_key="unit-test-key", client=client)
        with self.assertRaises(POIServiceError) as caught:
            provider.search_text("咖啡店", "郑州", limit=5)
        self.assertNotIn("raw response secret", str(caught.exception))

    def test_amap_unsuccessful_business_status_is_rejected(self):
        payload = {"status": "0", "info": "INVALID_USER_KEY", "infocode": "10001", "pois": []}
        provider, _, _ = self.make_provider(payload, key="super-secret-key")
        with self.assertRaises(POIServiceError) as caught:
            provider.search_text("咖啡店", "郑州", limit=5)
        message = str(caught.exception)
        self.assertNotIn("INVALID_USER_KEY", message)
        self.assertNotIn("10001", message)
        self.assertNotIn("super-secret-key", message)

    def test_malformed_json_is_safely_rejected(self):
        provider, _, response = self.make_provider(success_payload())
        response.json.side_effect = ValueError("raw JSON secret")
        with self.assertRaises(POIServiceError) as caught:
            provider.search_text("咖啡店", "郑州", limit=5)
        self.assertNotIn("raw JSON secret", str(caught.exception))

    def test_missing_api_key_is_a_controlled_configuration_error(self):
        with self.assertRaises(POIProviderNotConfiguredError):
            AmapPOIProvider(api_key="", client=Mock())

    def test_api_key_is_sent_but_never_returned_in_results(self):
        provider, client, _ = self.make_provider(success_payload([raw_poi()]), key="super-secret-key")
        result = provider.search_text("火锅店", "天津", limit=5)
        self.assertEqual(client.get.call_args.kwargs["params"]["key"], "super-secret-key")
        self.assertNotIn("super-secret-key", str(result))

    def test_raw_provider_fields_are_not_returned(self):
        value = raw_poi(id="provider-id", raw_secret="do-not-return")
        provider, _, _ = self.make_provider(success_payload([value]))
        result = provider.search_text("火锅店", "天津", limit=5)[0]
        self.assertNotIn("id", result)
        self.assertNotIn("raw_secret", result)
        self.assertNotIn("do-not-return", str(result))

    def test_invalid_pois_container_is_rejected(self):
        provider, _, _ = self.make_provider(
            {"status": "1", "info": "OK", "infocode": "10000", "pois": {}}
        )
        with self.assertRaises(POIServiceError):
            provider.search_text("咖啡店", "郑州", limit=5)


if __name__ == "__main__":
    unittest.main()
