import unittest

from location_service import LocationServiceError, resolve_current_location


CURRENT_LOCATION = {
    "latitude": 34.7466,
    "longitude": 113.6254,
    "accuracy_m": 20.0,
    "coordinate_system": "wgs84",
    "source": "browser_geolocation",
}


class FakeLocationProvider:
    def __init__(self, *, error=None):
        self.error = error
        self.convert_calls = []
        self.reverse_calls = []

    def convert_wgs84_to_gcj02(self, latitude, longitude):
        self.convert_calls.append((latitude, longitude))
        if self.error:
            raise self.error
        return {"longitude": 113.631, "latitude": 34.751}

    def reverse_geocode(self, *, latitude, longitude):
        self.reverse_calls.append((latitude, longitude))
        if self.error:
            raise self.error
        return {
            "province": "河南省",
            "city": "郑州市",
            "district": "金水区",
            "label": "河南省郑州市金水区",
            "formatted_address": "不应保存的详细地址",
            "number": "1号",
            "neighborhood": "不应保存的小区",
            "building": "不应保存的楼栋",
        }


class LocationServiceTests(unittest.TestCase):
    def test_valid_gps_is_converted_then_reverse_geocoded(self):
        provider = FakeLocationProvider()

        result = resolve_current_location(CURRENT_LOCATION, provider=provider)

        self.assertEqual(provider.convert_calls, [(34.7466, 113.6254)])
        self.assertEqual(provider.reverse_calls, [(34.751, 113.631)])
        self.assertEqual(result["province"], "河南省")
        self.assertEqual(result["city"], "郑州市")
        self.assertEqual(result["district"], "金水区")
        self.assertEqual(result["label"], "河南省郑州市金水区")

    def test_precise_reverse_geocode_fields_are_not_retained(self):
        result = resolve_current_location(
            CURRENT_LOCATION,
            provider=FakeLocationProvider(),
        )

        self.assertNotIn("formatted_address", result)
        self.assertNotIn("street", result)
        self.assertNotIn("number", result)
        self.assertNotIn("neighborhood", result)
        self.assertNotIn("building", result)
        self.assertNotIn("不应保存", str(result))

    def test_township_level_is_kept_in_context_and_label(self):
        provider = FakeLocationProvider()
        provider.reverse_geocode = lambda **_kwargs: {
            "province": "河南省",
            "city": "郑州市",
            "district": "中原区",
            "township": "莲湖街道",
            "label": "河南省郑州市中原区莲湖街道",
        }

        result = resolve_current_location(CURRENT_LOCATION, provider=provider)

        self.assertEqual(result["province"], "河南省")
        self.assertEqual(result["city"], "郑州市")
        self.assertEqual(result["district"], "中原区")
        self.assertEqual(result["township"], "莲湖街道")
        self.assertEqual(result["label"], "河南省郑州市中原区莲湖街道")
        self.assertNotIn("street", result)
        self.assertNotIn("formatted_address", result)
        self.assertNotIn("number", result)

    def test_street_fallback_is_kept_without_house_number(self):
        provider = FakeLocationProvider()
        provider.reverse_geocode = lambda **_kwargs: {
            "province": "河南省",
            "city": "郑州市",
            "district": "金水区",
            "street": "花园路",
            "label": "河南省郑州市金水区花园路",
            "number": "1号",
        }

        result = resolve_current_location(CURRENT_LOCATION, provider=provider)

        self.assertEqual(result["street"], "花园路")
        self.assertEqual(result["label"], "河南省郑州市金水区花园路")
        self.assertNotIn("number", result)

    def test_existing_coarse_context_skips_provider_calls(self):
        provider = FakeLocationProvider()
        enriched = {
            **CURRENT_LOCATION,
            "province": "河南省",
            "city": "郑州市",
            "district": "金水区",
            "label": "河南省郑州市金水区",
        }

        result = resolve_current_location(enriched, provider=provider)

        self.assertEqual(result, enriched)
        self.assertEqual(provider.convert_calls, [])
        self.assertEqual(provider.reverse_calls, [])

    def test_missing_gps_does_not_call_provider(self):
        provider = FakeLocationProvider()

        with self.assertRaises(LocationServiceError):
            resolve_current_location(None, provider=provider)

        self.assertEqual(provider.convert_calls, [])
        self.assertEqual(provider.reverse_calls, [])

    def test_conversion_failure_is_sanitized(self):
        provider = FakeLocationProvider(error=RuntimeError("secret coordinate"))

        with self.assertRaises(LocationServiceError) as caught:
            resolve_current_location(CURRENT_LOCATION, provider=provider)

        self.assertNotIn("secret", str(caught.exception))

    def test_reverse_geocode_failure_is_sanitized(self):
        provider = FakeLocationProvider()

        def fail_reverse(**_kwargs):
            raise RuntimeError("secret reverse response")

        provider.reverse_geocode = fail_reverse

        with self.assertRaises(LocationServiceError) as caught:
            resolve_current_location(CURRENT_LOCATION, provider=provider)

        self.assertNotIn("secret", str(caught.exception))

    def test_reverse_geocode_without_label_is_rejected(self):
        provider = FakeLocationProvider()
        provider.reverse_geocode = lambda **_kwargs: {}

        with self.assertRaises(LocationServiceError):
            resolve_current_location(CURRENT_LOCATION, provider=provider)


if __name__ == "__main__":
    unittest.main()
