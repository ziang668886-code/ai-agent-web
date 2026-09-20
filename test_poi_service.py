import unittest

from poi_service import DEFAULT_RADIUS_METERS, search_poi_service


def poi(name="示例地点", **overrides):
    result = {
        "name": name,
        "address": "示例路1号",
        "district": "示例区",
        "category": "餐饮服务;中餐厅",
        "distance_m": "860",
        "rating": "4.6",
        "cost_per_person": "92",
        "tags": ["火锅", "朋友聚餐"],
        "opening_hours": "10:00-22:00",
        "location": {"longitude": 117.0, "latitude": 39.0},
        "province": "天津市",
        "city": "天津市",
    }
    result.update(overrides)
    return result


class FakeProvider:
    def __init__(self, text_results=None, nearby_results=None, error=None):
        self.text_results = [] if text_results is None else text_results
        self.nearby_results = [] if nearby_results is None else nearby_results
        self.error = error
        self.text_calls = []
        self.nearby_calls = []

    def search_text(self, query, city, *, limit):
        self.text_calls.append({"query": query, "city": city, "limit": limit})
        if self.error:
            raise self.error
        return self.text_results

    def search_nearby(self, query, city, **kwargs):
        self.nearby_calls.append({"query": query, "city": city, **kwargs})
        if self.error:
            raise self.error
        return self.nearby_results


class POIServiceTests(unittest.TestCase):
    def test_city_search_returns_normalized_result(self):
        provider = FakeProvider(text_results=[poi("郑州咖啡")])
        result = search_poi_service("咖啡店", "郑州", provider=provider)
        self.assertEqual(result["status"], "results")
        self.assertEqual(result["search_mode"], "city")
        self.assertIsNone(result["anchor"])
        self.assertEqual(result["results"][0]["rank"], 1)

    def test_nearby_search_uses_trusted_anchor_and_fixed_radius(self):
        anchor = poi(
            "故宫",
            city="北京市",
            district="东城区",
            location={"longitude": 116.397128, "latitude": 39.916527},
        )
        provider = FakeProvider(text_results=[anchor], nearby_results=[poi("故宫餐厅")])
        result = search_poi_service("餐厅", "北京", "故宫", provider=provider)
        self.assertEqual(result["status"], "results")
        self.assertEqual(result["search_mode"], "nearby")
        call = provider.nearby_calls[0]
        self.assertEqual(call["radius"], DEFAULT_RADIUS_METERS)
        self.assertEqual(call["limit"], 5)
        self.assertAlmostEqual(call["longitude"], 116.397128)
        self.assertAlmostEqual(call["latitude"], 39.916527)

    def test_anchor_no_result_returns_location_required(self):
        result = search_poi_service("餐厅", "北京", "不存在", provider=FakeProvider())
        self.assertEqual(result["status"], "location_required")
        self.assertEqual(result["results"], [])

    def test_anchor_with_invalid_coordinate_returns_location_required(self):
        provider = FakeProvider(text_results=[poi("故宫", location=None)])
        result = search_poi_service("餐厅", "北京", "故宫", provider=provider)
        self.assertEqual(result["status"], "location_required")

    def test_distinct_same_name_anchors_are_ambiguous(self):
        provider = FakeProvider(
            text_results=[
                poi("万达广场", district="金水区", location={"longitude": 113.1, "latitude": 34.1}),
                poi("万达广场", district="中原区", location={"longitude": 113.2, "latitude": 34.2}),
            ]
        )
        result = search_poi_service("餐厅", "郑州", "万达广场", provider=provider)
        self.assertEqual(result["status"], "ambiguous_location")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(set(result["candidates"][0]), {"name", "district"})

    def test_unique_exact_anchor_wins_over_fuzzy_candidate(self):
        provider = FakeProvider(
            text_results=[
                poi("故宫", city="北京市", district="东城区"),
                poi("故宫文化商店", city="北京市", district="东城区", location={"longitude": 117, "latitude": 40}),
            ],
            nearby_results=[poi("餐厅")],
        )
        result = search_poi_service("餐厅", "北京", "故宫", provider=provider)
        self.assertEqual(result["status"], "results")
        self.assertEqual(len(provider.nearby_calls), 1)

    def test_same_exact_anchor_in_one_district_uses_provider_first_result(self):
        provider = FakeProvider(
            text_results=[
                poi("故宫", city="北京市", district="东城区"),
                poi(
                    "故宫",
                    city="北京市",
                    district="东城区",
                    location={"longitude": 116.3972, "latitude": 39.9166},
                ),
            ],
            nearby_results=[poi("餐厅")],
        )
        result = search_poi_service("餐厅", "北京", "故宫", provider=provider)
        self.assertEqual(result["status"], "results")
        self.assertEqual(provider.nearby_calls[0]["longitude"], 117.0)

    def test_unique_fuzzy_anchor_is_trusted(self):
        provider = FakeProvider(
            text_results=[poi("故宫博物院", city="北京市", district="东城区")],
            nearby_results=[poi("餐厅")],
        )
        result = search_poi_service("餐厅", "北京", "故宫", provider=provider)
        self.assertEqual(result["status"], "results")

    def test_related_anchor_names_in_one_district_use_provider_first_result(self):
        provider = FakeProvider(
            text_results=[
                poi("故宫博物院", city="北京市", district="东城区"),
                poi(
                    "故宫博物院午门",
                    city="北京市",
                    district="东城区",
                    location={"longitude": 116.3972, "latitude": 39.9166},
                ),
            ],
            nearby_results=[poi("餐厅")],
        )
        result = search_poi_service("餐厅", "北京", "故宫", provider=provider)
        self.assertEqual(result["status"], "results")
        self.assertEqual(provider.nearby_calls[0]["longitude"], 117.0)

    def test_related_anchor_names_use_provider_relevance_order(self):
        provider = FakeProvider(
            text_results=[
                poi("故宫博物院", city="北京市", district="东城区"),
                poi(
                    "故宫文化馆",
                    city="北京市",
                    district="朝阳区",
                    location={"longitude": 116.5, "latitude": 39.9},
                ),
            ],
            nearby_results=[poi("餐厅")],
        )
        result = search_poi_service("餐厅", "北京", "故宫", provider=provider)
        self.assertEqual(result["status"], "results")
        self.assertEqual(provider.nearby_calls[0]["longitude"], 117.0)

    def test_no_poi_results_is_not_an_exception(self):
        result = search_poi_service("咖啡店", "郑州", provider=FakeProvider())
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "no_results")
        self.assertEqual(result["result_count"], 0)

    def test_provider_results_are_truncated_to_five(self):
        provider = FakeProvider(text_results=[poi(f"地点{i}") for i in range(8)])
        result = search_poi_service("景点", "北京", provider=provider)
        self.assertEqual(result["result_count"], 5)
        self.assertEqual([item["rank"] for item in result["results"]], [1, 2, 3, 4, 5])

    def test_missing_optional_business_fields_remain_empty(self):
        provider = FakeProvider(
            text_results=[poi(rating=None, cost_per_person=None, opening_hours=None, tags=None)]
        )
        item = search_poi_service("公园", "北京", provider=provider)["results"][0]
        self.assertIsNone(item["rating"])
        self.assertIsNone(item["cost_per_person"])
        self.assertIsNone(item["opening_hours"])
        self.assertEqual(item["tags"], [])

    def test_city_search_never_exposes_distance(self):
        provider = FakeProvider(text_results=[poi(distance_m=12)])
        item = search_poi_service("餐厅", "北京", provider=provider)["results"][0]
        self.assertIsNone(item["distance_m"])

    def test_nearby_real_distance_is_preserved(self):
        provider = FakeProvider(text_results=[poi("故宫")], nearby_results=[poi(distance_m="860")])
        item = search_poi_service("餐厅", "北京", "故宫", provider=provider)["results"][0]
        self.assertEqual(item["distance_m"], 860)

    def test_bad_result_coordinate_becomes_none(self):
        provider = FakeProvider(text_results=[poi(location={"longitude": 999, "latitude": 39})])
        item = search_poi_service("咖啡店", "郑州", provider=provider)["results"][0]
        self.assertIsNone(item["location"])

    def test_provider_exception_is_sanitized(self):
        provider = FakeProvider(error=RuntimeError("provider raw secret"))
        result = search_poi_service("咖啡店", "郑州", provider=provider)
        self.assertEqual(result["status"], "temporarily_unavailable")
        self.assertNotIn("provider raw secret", str(result))

    def test_provider_extra_fields_are_not_returned(self):
        provider = FakeProvider(text_results=[poi(provider_id="secret-id", raw_payload="secret")])
        item = search_poi_service("咖啡店", "郑州", provider=provider)["results"][0]
        self.assertEqual(
            set(item),
            {
                "rank", "name", "address", "district", "category", "distance_m",
                "rating", "cost_per_person", "tags", "opening_hours", "location",
            },
        )


if __name__ == "__main__":
    unittest.main()
