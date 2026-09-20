import inspect
import unittest

import poi_tool
from poi_tool import SEARCH_POI_TOOL, search_poi


class FakeProvider:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def search_text(self, query, city, *, limit):
        self.calls.append(("text", query, city, limit))
        if self.error:
            raise self.error
        return self.results

    def search_nearby(self, *args, **kwargs):
        self.calls.append(("nearby", args, kwargs))
        if self.error:
            raise self.error
        return self.results


class POIToolTests(unittest.TestCase):
    def assert_invalid(self, result, code):
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "invalid_request")
        self.assertEqual(result["error_code"], code)
        self.assertEqual(result["results"], [])

    def test_empty_query_is_rejected(self):
        self.assert_invalid(search_poi("  ", "郑州", provider=FakeProvider()), "INVALID_QUERY")

    def test_non_string_query_is_rejected(self):
        for value in (True, 1, [], {}):
            with self.subTest(value=value):
                self.assert_invalid(search_poi(value, "郑州", provider=FakeProvider()), "INVALID_QUERY")

    def test_overlong_query_is_rejected(self):
        self.assert_invalid(search_poi("餐" * 81, "郑州", provider=FakeProvider()), "INVALID_QUERY")

    def test_empty_city_is_rejected(self):
        self.assert_invalid(search_poi("咖啡店", " ", provider=FakeProvider()), "INVALID_CITY")

    def test_non_string_city_is_rejected(self):
        for value in (False, 1, [], {}):
            with self.subTest(value=value):
                self.assert_invalid(search_poi("咖啡店", value, provider=FakeProvider()), "INVALID_CITY")

    def test_overlong_city_is_rejected(self):
        self.assert_invalid(search_poi("咖啡店", "郑" * 51, provider=FakeProvider()), "INVALID_CITY")

    def test_non_string_anchor_is_rejected(self):
        for value in (True, 1, [], {}):
            with self.subTest(value=value):
                self.assert_invalid(search_poi("餐厅", "北京", value, provider=FakeProvider()), "INVALID_ANCHOR")

    def test_empty_anchor_is_rejected(self):
        self.assert_invalid(search_poi("餐厅", "北京", "  ", provider=FakeProvider()), "INVALID_ANCHOR")

    def test_overlong_anchor_is_rejected(self):
        self.assert_invalid(search_poi("餐厅", "北京", "地" * 81, provider=FakeProvider()), "INVALID_ANCHOR")

    def test_unknown_parameter_is_rejected(self):
        result = search_poi("咖啡店", "郑州", provider=FakeProvider(), radius=1000)
        self.assert_invalid(result, "INVALID_PARAMETERS")

    def test_valid_city_search_is_trimmed_and_delegated(self):
        provider = FakeProvider(
            [{"name": "示例咖啡", "location": {"longitude": 113.1, "latitude": 34.1}}]
        )
        result = search_poi(" 咖啡店 ", " 郑州 ", provider=provider)
        self.assertTrue(result["ok"])
        self.assertEqual(provider.calls, [("text", "咖啡店", "郑州", 5)])

    def test_provider_failure_returns_safe_status(self):
        result = search_poi("咖啡店", "郑州", provider=FakeProvider(error=RuntimeError("secret")))
        self.assertEqual(result["status"], "temporarily_unavailable")
        self.assertNotIn("secret", str(result))

    def test_schema_exposes_only_query_city_and_anchor(self):
        function = SEARCH_POI_TOOL["function"]
        parameters = function["parameters"]
        self.assertEqual(function["name"], "search_poi")
        self.assertEqual(set(parameters["properties"]), {"query", "city", "anchor"})
        self.assertEqual(parameters["required"], ["query", "city"])
        self.assertFalse(parameters["additionalProperties"])

    def test_tool_has_no_streamlit_database_or_visitor_dependency(self):
        source = inspect.getsource(poi_tool).casefold()
        self.assertNotIn("streamlit", source)
        self.assertNotIn("database", source)
        self.assertNotIn("visitor_id", source)


if __name__ == "__main__":
    unittest.main()
