import math
import unittest

from location_context import (
    LocationContextError,
    validate_current_location,
    validate_geolocation_result,
    validate_location_payload,
)


class ValidateLocationPayloadTests(unittest.TestCase):
    def test_valid_location(self):
        result = validate_location_payload(
            {"latitude": 34.7466, "longitude": 113.6254, "accuracy": 12.5}
        )
        self.assertEqual(result["latitude"], 34.7466)
        self.assertEqual(result["longitude"], 113.6254)
        self.assertEqual(result["accuracy_m"], 12.5)

    def test_assigns_wgs84_server_side(self):
        result = validate_location_payload(
            {"latitude": 0, "longitude": 0, "accuracy": 0}
        )
        self.assertEqual(result["coordinate_system"], "wgs84")

    def test_assigns_browser_source_server_side(self):
        result = validate_location_payload(
            {"latitude": 0, "longitude": 0, "accuracy": 0}
        )
        self.assertEqual(result["source"], "browser_geolocation")

    def test_accepts_minimum_latitude(self):
        self.assertEqual(
            validate_location_payload(
                {"latitude": -90, "longitude": 0, "accuracy": 1}
            )["latitude"],
            -90.0,
        )

    def test_accepts_maximum_latitude(self):
        self.assertEqual(
            validate_location_payload(
                {"latitude": 90, "longitude": 0, "accuracy": 1}
            )["latitude"],
            90.0,
        )

    def test_rejects_latitude_below_minimum(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": -90.001, "longitude": 0, "accuracy": 1}
            )

    def test_rejects_latitude_above_maximum(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": 90.001, "longitude": 0, "accuracy": 1}
            )

    def test_accepts_minimum_longitude(self):
        self.assertEqual(
            validate_location_payload(
                {"latitude": 0, "longitude": -180, "accuracy": 1}
            )["longitude"],
            -180.0,
        )

    def test_accepts_maximum_longitude(self):
        self.assertEqual(
            validate_location_payload(
                {"latitude": 0, "longitude": 180, "accuracy": 1}
            )["longitude"],
            180.0,
        )

    def test_rejects_longitude_below_minimum(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": 0, "longitude": -180.001, "accuracy": 1}
            )

    def test_rejects_longitude_above_maximum(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": 0, "longitude": 180.001, "accuracy": 1}
            )

    def test_accepts_zero_accuracy(self):
        self.assertEqual(
            validate_location_payload(
                {"latitude": 0, "longitude": 0, "accuracy": 0}
            )["accuracy_m"],
            0.0,
        )

    def test_rejects_negative_accuracy(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": 0, "longitude": 0, "accuracy": -0.1}
            )

    def test_rejects_string_number(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": "34.7", "longitude": 113.6, "accuracy": 1}
            )

    def test_rejects_boolean_number(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": True, "longitude": 113.6, "accuracy": 1}
            )

    def test_rejects_nan(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": math.nan, "longitude": 0, "accuracy": 1}
            )

    def test_rejects_longitude_nan(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": 0, "longitude": math.nan, "accuracy": 1}
            )

    def test_rejects_positive_infinity(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": 0, "longitude": math.inf, "accuracy": 1}
            )

    def test_rejects_negative_infinity(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": 0, "longitude": 0, "accuracy": -math.inf}
            )

    def test_rejects_accuracy_nan(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": 0, "longitude": 0, "accuracy": math.nan}
            )

    def test_rejects_accuracy_positive_infinity(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": 0, "longitude": 0, "accuracy": math.inf}
            )

    def test_rejects_accuracy_boolean(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": 0, "longitude": 0, "accuracy": False}
            )

    def test_rejects_none_coordinate(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {"latitude": None, "longitude": 0, "accuracy": 1}
            )

    def test_rejects_non_mapping(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload([34.7, 113.6, 10])

    def test_rejects_missing_field(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload({"latitude": 34.7, "longitude": 113.6})

    def test_rejects_unknown_extra_field(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {
                    "latitude": 34.7,
                    "longitude": 113.6,
                    "accuracy": 10,
                    "visitor_id": "not-trusted",
                }
            )

    def test_rejects_browser_supplied_coordinate_system(self):
        with self.assertRaises(LocationContextError):
            validate_location_payload(
                {
                    "latitude": 34.7,
                    "longitude": 113.6,
                    "accuracy": 10,
                    "coordinate_system": "gcj02",
                }
            )


class ValidateGeolocationResultTests(unittest.TestCase):
    def test_accepts_granted_result(self):
        status, location = validate_geolocation_result(
            {
                "status": "granted",
                "location": {"latitude": 34.7, "longitude": 113.6, "accuracy": 8},
            }
        )
        self.assertEqual(status, "granted")
        self.assertEqual(location["accuracy_m"], 8.0)

    def test_accepts_denied_without_location(self):
        self.assertEqual(validate_geolocation_result({"status": "denied"}), ("denied", None))

    def test_accepts_timeout_without_location(self):
        self.assertEqual(
            validate_geolocation_result({"status": "timeout"}), ("timeout", None)
        )

    def test_accepts_unavailable_without_location(self):
        self.assertEqual(
            validate_geolocation_result({"status": "unavailable"}),
            ("unavailable", None),
        )

    def test_accepts_unsupported_without_location(self):
        self.assertEqual(
            validate_geolocation_result({"status": "unsupported"}),
            ("unsupported", None),
        )

    def test_rejects_unknown_status(self):
        with self.assertRaises(LocationContextError):
            validate_geolocation_result({"status": "precise"})

    def test_rejects_failure_with_location(self):
        with self.assertRaises(LocationContextError):
            validate_geolocation_result(
                {
                    "status": "denied",
                    "location": {"latitude": 0, "longitude": 0, "accuracy": 1},
                }
            )

    def test_rejects_granted_without_location(self):
        with self.assertRaises(LocationContextError):
            validate_geolocation_result({"status": "granted"})

    def test_rejects_extra_result_field(self):
        with self.assertRaises(LocationContextError):
            validate_geolocation_result(
                {"status": "timeout", "raw_error": "sensitive browser detail"}
            )

    def test_rejects_non_mapping_without_crashing_internals(self):
        with self.assertRaises(LocationContextError):
            validate_geolocation_result(None)


class ValidateCurrentLocationTests(unittest.TestCase):
    def setUp(self):
        self.location = {
            "latitude": 34.7466,
            "longitude": 113.6254,
            "accuracy_m": 18.5,
            "coordinate_system": "wgs84",
            "source": "browser_geolocation",
        }

    def test_valid_session_location_is_revalidated(self):
        self.assertEqual(validate_current_location(self.location), self.location)

    def test_rejects_wrong_coordinate_system(self):
        self.location["coordinate_system"] = "gcj02"
        with self.assertRaises(LocationContextError):
            validate_current_location(self.location)

    def test_rejects_wrong_source(self):
        self.location["source"] = "model"
        with self.assertRaises(LocationContextError):
            validate_current_location(self.location)

    def test_rejects_extra_session_field(self):
        self.location["visitor_id"] = "not-trusted"
        with self.assertRaises(LocationContextError):
            validate_current_location(self.location)

    def test_rejects_missing_accuracy(self):
        self.location.pop("accuracy_m")
        with self.assertRaises(LocationContextError):
            validate_current_location(self.location)

    def test_rejects_invalid_coordinate(self):
        self.location["longitude"] = 181
        with self.assertRaises(LocationContextError):
            validate_current_location(self.location)

    def test_accepts_coarse_reverse_geocode_context(self):
        self.location.update(
            {
                "province": "河南省",
                "city": "郑州市",
                "district": "金水区",
                "label": "河南省郑州市金水区",
            }
        )

        result = validate_current_location(self.location)

        self.assertEqual(result["label"], "河南省郑州市金水区")

    def test_rejects_precise_address_field(self):
        self.location["formatted_address"] = "敏感详细地址"
        with self.assertRaises(LocationContextError):
            validate_current_location(self.location)


if __name__ == "__main__":
    unittest.main()
