import unittest

from poi_service import (
    DEFAULT_RADIUS_METERS,
    search_poi_by_current_location_service,
    search_poi_service,
)


CURRENT_LOCATION = {
    "latitude": 34.7466,
    "longitude": 113.6254,
    "accuracy_m": 20.0,
    "coordinate_system": "wgs84",
    "source": "browser_geolocation",
}


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
    def __init__(
        self,
        text_results=None,
        nearby_results=None,
        error=None,
        converted=None,
    ):
        self.text_results = [] if text_results is None else text_results
        self.nearby_results = [] if nearby_results is None else nearby_results
        self.error = error
        self.converted = converted or {"longitude": 113.631, "latitude": 34.751}
        self.text_calls = []
        self.nearby_calls = []
        self.convert_calls = []

    def convert_wgs84_to_gcj02(self, latitude, longitude):
        self.convert_calls.append({"latitude": latitude, "longitude": longitude})
        if self.error:
            raise self.error
        return self.converted

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

    def test_current_location_converts_then_uses_real_nearby_search(self):
        provider = FakeProvider(nearby_results=[poi("附近餐厅", distance_m="320")])

        result = search_poi_by_current_location_service(
            "餐厅",
            CURRENT_LOCATION,
            provider=provider,
        )

        self.assertEqual(result["status"], "results")
        self.assertEqual(result["city"], "当前位置")
        self.assertEqual(result["results"][0]["distance_m"], 320)
        self.assertEqual(provider.text_calls, [])
        self.assertEqual(len(provider.convert_calls), 1)
        call = provider.nearby_calls[0]
        self.assertIsNone(call["city"])
        self.assertEqual(call["radius"], DEFAULT_RADIUS_METERS)
        self.assertEqual(call["longitude"], 113.631)
        self.assertEqual(call["latitude"], 34.751)

    def test_current_location_missing_does_not_call_provider(self):
        provider = FakeProvider()

        result = search_poi_by_current_location_service(
            "咖啡店",
            None,
            provider=provider,
        )

        self.assertEqual(result["status"], "location_required")
        self.assertEqual(provider.convert_calls, [])
        self.assertEqual(provider.nearby_calls, [])

    def test_coordinate_conversion_failure_is_temporarily_unavailable(self):
        provider = FakeProvider(error=RuntimeError("raw coordinate secret"))

        result = search_poi_by_current_location_service(
            "公园",
            CURRENT_LOCATION,
            provider=provider,
        )

        self.assertEqual(result["status"], "temporarily_unavailable")
        self.assertNotIn("secret", str(result))

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


class VagueLeisureIntentTests(unittest.TestCase):
    def nearby_query(self, query):
        provider = FakeProvider(nearby_results=[poi("示例地点")])
        search_poi_by_current_location_service(
            query,
            CURRENT_LOCATION,
            provider=provider,
        )
        return provider.nearby_calls[0]["query"]

    def test_vague_leisure_intent_is_mapped_to_a_real_category(self):
        cases = (
            ("好玩的", "景点"),
            ("有什么好玩的", "景点"),
            ("附近有什么好玩的", "景点"),
            ("好玩的地方", "景点"),
            ("附近去哪玩", "景点"),
            ("附近有什么可以逛的", "购物中心"),
            ("附近有什么休闲娱乐", "休闲娱乐"),
        )
        for query, expected in cases:
            with self.subTest(query=query):
                self.assertEqual(self.nearby_query(query), expected)

    def test_vague_leisure_intent_never_reaches_the_provider_verbatim(self):
        self.assertNotEqual(self.nearby_query("我附近有什么好玩的？"), "我附近有什么好玩的？")

    def test_leisure_phrase_with_extra_intent_is_not_rewritten(self):
        self.assertEqual(
            self.nearby_query("好玩又便宜的地方"),
            "好玩又便宜的地方",
        )

    def test_concrete_categories_are_never_rewritten(self):
        for query in ("咖啡店", "餐厅", "商场", "博物馆", "公园", "景点", "火锅店", "小吃"):
            with self.subTest(query=query):
                self.assertEqual(self.nearby_query(query), query)

    def test_vague_leisure_city_search_keeps_the_explicit_city(self):
        provider = FakeProvider(text_results=[poi("北京景点")])

        result = search_poi_service("好玩的", "北京", provider=provider)

        self.assertEqual(provider.text_calls[0]["city"], "北京")
        self.assertEqual(provider.text_calls[0]["query"], "景点")
        self.assertEqual(result["city"], "北京")
        self.assertEqual(result["status"], "results")


class PoiCategoryRelevanceTests(unittest.TestCase):
    def names(self, result):
        return [item["name"] for item in result["results"]]

    def test_park_query_demotes_results_whose_category_is_not_a_park(self):
        provider = FakeProvider(
            nearby_results=[
                poi("示例公园里小区", category="商务住宅;住宅区;住宅小区"),
                poi("人民公园", category="风景名胜;公园广场;公园"),
                poi("公园路便利店", category="购物服务;便利店"),
            ]
        )

        result = search_poi_by_current_location_service(
            "公园",
            CURRENT_LOCATION,
            provider=provider,
        )

        self.assertEqual(
            self.names(result),
            ["人民公园", "示例公园里小区", "公园路便利店"],
        )
        self.assertEqual([item["rank"] for item in result["results"]], [1, 2, 3])
        # 类别不符只降权，不会把结果丢掉。
        self.assertEqual(result["result_count"], 3)

    def test_results_without_category_are_not_demoted(self):
        provider = FakeProvider(
            nearby_results=[
                poi("无分类公园", category=None),
                poi("住宅区", category="商务住宅;住宅区"),
            ]
        )

        result = search_poi_by_current_location_service(
            "公园",
            CURRENT_LOCATION,
            provider=provider,
        )

        self.assertEqual(self.names(result), ["无分类公园", "住宅区"])

    def test_matching_category_results_keep_amap_order(self):
        provider = FakeProvider(
            nearby_results=[
                poi("甲餐厅", category="餐饮服务;中餐厅"),
                poi("乙咖啡", category="餐饮服务;咖啡厅"),
            ]
        )

        result = search_poi_by_current_location_service(
            "咖啡店",
            CURRENT_LOCATION,
            provider=provider,
        )

        self.assertEqual(self.names(result), ["甲餐厅", "乙咖啡"])

    def test_mall_query_demotes_unrelated_categories(self):
        provider = FakeProvider(
            nearby_results=[
                poi("商场停车场", category="交通设施服务;停车场"),
                poi("示例购物中心", category="购物服务;商场"),
            ]
        )

        result = search_poi_by_current_location_service(
            "商场",
            CURRENT_LOCATION,
            provider=provider,
        )

        self.assertEqual(self.names(result), ["示例购物中心", "商场停车场"])

    def test_non_category_query_is_left_alone(self):
        provider = FakeProvider(
            nearby_results=[
                poi("甲火锅", category="餐饮服务;中餐厅"),
                poi("乙烧烤", category="购物服务;便利店"),
            ]
        )

        result = search_poi_by_current_location_service(
            "火锅店",
            CURRENT_LOCATION,
            provider=provider,
        )

        self.assertEqual(self.names(result), ["甲火锅", "乙烧烤"])


class PoiEmptyResultMessageTests(unittest.TestCase):
    def test_current_location_empty_result_does_not_ask_for_a_city(self):
        result = search_poi_by_current_location_service(
            "好玩的",
            CURRENT_LOCATION,
            provider=FakeProvider(),
        )

        self.assertEqual(result["status"], "no_results")
        self.assertEqual(result["city"], "当前位置")
        self.assertIn("3km", result["message"])
        for phrase in ("城市", "地标", "具体地点", "提供", "地址"):
            self.assertNotIn(phrase, result["message"])

    def test_city_empty_result_keeps_the_generic_message(self):
        result = search_poi_service("咖啡店", "郑州", provider=FakeProvider())

        self.assertEqual(result["status"], "no_results")
        self.assertNotIn("3km", result["message"])
        self.assertNotIn("当前位置", result["message"])


if __name__ == "__main__":
    unittest.main()
