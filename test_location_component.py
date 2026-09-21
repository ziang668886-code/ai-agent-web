"""Tests for the permission-aware browser geolocation component."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import location_component as component
from location_context import validate_geolocation_result


JS = component._COMPONENT_JS


class ComponentBrowserContractTests(unittest.TestCase):
    """Structural checks on the shipped browser JavaScript."""

    def test_manual_and_auto_share_exactly_one_geolocation_call_site(self):
        self.assertEqual(
            JS.count("navigator.geolocation.getCurrentPosition("),
            1,
        )
        # 自动路径只允许有一个调用点，手动路径通过 onclick 复用同一个函数。
        self.assertEqual(JS.count("requestPosition();"), 1)
        self.assertIn("button.onclick = requestPosition;", JS)

    def test_auto_location_requires_granted_permission_and_server_approval(self):
        auto_call = JS.index("requestPosition();")
        guard = JS[max(0, auto_call - 400) : auto_call]

        self.assertIn("state === 'granted'", guard)
        self.assertIn("data?.auto_attempt", guard)

    def test_prompt_state_never_triggers_a_position_request(self):
        # 唯一自动调用点在 granted 守卫之后，prompt 分支没有可执行的调用。
        auto_call = JS.index("requestPosition();")
        self.assertNotIn("'prompt'", JS[:auto_call])
        guard = JS[max(0, auto_call - 400) : auto_call]
        self.assertIn("state === 'granted'", guard)
        self.assertIn("data?.auto_attempt", guard)
        self.assertIn("!autoAttempted", guard)

    def test_missing_permissions_api_keeps_manual_location_available(self):
        fallback = JS.index("typeof permissions.query !== 'function'")

        self.assertNotIn("requestPosition();", JS[:fallback])
        self.assertIn("button.onclick = requestPosition;", JS)

    def test_permission_query_failure_is_contained(self):
        self.assertIn("} catch (error) {", JS)
        self.assertIn(".catch(() => {});", JS)

    def test_unsupported_geolocation_reports_a_safe_status(self):
        self.assertIn("window.isSecureContext", JS)
        self.assertIn("reportPermissionState('unsupported');", JS)
        self.assertIn("finish(id, 'unsupported');", JS)

    def test_permission_state_is_never_persisted_in_the_browser(self):
        for forbidden in (
            "localStorage",
            "sessionStorage",
            "document.cookie",
            "indexedDB",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, JS)

    def test_reported_payload_keeps_the_validated_location_shape(self):
        self.assertIn("finish(id, 'granted', {", JS)
        self.assertIn("latitude: position.coords.latitude", JS)
        self.assertIn("longitude: position.coords.longitude", JS)
        self.assertIn("accuracy: position.coords.accuracy", JS)

        status, location = validate_geolocation_result(
            {
                "status": "granted",
                "location": {
                    "latitude": 34.7466,
                    "longitude": 113.6254,
                    "accuracy": 20.0,
                },
            }
        )

        self.assertEqual(status, "granted")
        self.assertEqual(
            set(location),
            {
                "latitude",
                "longitude",
                "accuracy_m",
                "coordinate_system",
                "source",
            },
        )


def finish_body():
    """Return the shared terminal path of the component JavaScript."""

    return JS.split("const finish = ", 1)[1].split("const requestPosition", 1)[0]


class RequestLifecycleTests(unittest.TestCase):
    """The auto attempt must reach a terminal state from every outcome."""

    def test_a_watchdog_guarantees_the_request_finishes(self):
        # getCurrentPosition 的 timeout 在等待权限决定期间不计时，浏览器可能
        # 两个回调都不触发。看门狗走同一个 finish()，因此加载态一定会清除。
        self.assertIn(
            "setTimeout(() => finish(id, 'timeout'), REQUEST_WATCHDOG_MS)",
            JS,
        )
        self.assertIn("clearTimeout(watchdog);", finish_body())

    def test_every_terminal_path_clears_the_loading_state(self):
        body = finish_body()
        self.assertIn("button.disabled = false;", body)
        self.assertIn("progress.textContent = '';", body)
        # success / denied / unavailable / timeout / unsupported 都只经过 finish()。
        self.assertEqual(JS.count("button.disabled = false;"), 1)
        self.assertEqual(JS.count("progress.textContent = '';"), 1)

    def test_late_callbacks_cannot_resurrect_the_loading_state(self):
        self.assertIn("if (id !== requestId)", finish_body())
        self.assertIn("requestId += 1;", JS)
        # 编号与看门狗必须在模块作用域：组件被重新调用时也要保留，
        # 否则进行中的请求会被误判为过期，加载态反而清不掉。
        module_scope = JS[: JS.index("export default function")]
        self.assertIn("let requestId = 0;", module_scope)
        self.assertIn("let watchdog = null;", module_scope)

    def test_loading_is_shown_only_after_the_browser_starts(self):
        start = JS.index("const requestPosition = ")
        body = JS[start : JS.index("button.onclick = requestPosition;")]
        self.assertIn("progress.textContent = '正在获取位置…';", body)
        self.assertIn("setTriggerValue('location_request_state', 'started')", body)
        # 组件挂载本身绝不显示“正在获取位置”。
        self.assertNotIn("正在获取位置", JS[:start])

    def test_auto_location_fires_at_most_once_per_component_instance(self):
        self.assertIn("let autoAttempted = false;", JS)
        auto_call = JS.index("requestPosition();")
        guard = JS[max(0, auto_call - 400) : auto_call]
        self.assertIn("autoAttempted = true;", guard)

    def test_manual_button_is_wired_before_any_early_return(self):
        # 手动入口在 unsupported / Permissions API 缺失等提前返回之前就已绑定，
        # 因此任何自动失败或被看门狗终止之后都还能手动重试。
        onclick = JS.index("button.onclick = requestPosition;")
        for early_return in (
            "if (!supported) {\n                reportPermissionState('unsupported');",
            "if (!permissions || typeof permissions.query !== 'function') {",
        ):
            with self.subTest(early_return=early_return):
                self.assertLess(onclick, JS.index(early_return))


class AutoAttemptRaceTests(unittest.TestCase):
    """The mid-request rerun must not disarm the in-flight request."""

    def test_auto_attempt_stays_armed_while_the_request_is_in_flight(self):
        consumed_mid_flight = component.should_consume_auto_attempt(
            already_attempted=False,
            has_location_result=False,
            request_finished=False,
        )

        self.assertFalse(consumed_mid_flight)
        # 因此 data.auto_attempt 保持 True，进行中的请求不会被重新配置。
        self.assertTrue(
            component.should_attempt_auto_location(
                has_current_location=False,
                already_attempted=consumed_mid_flight,
            )
        )

    def test_a_reported_result_concludes_the_attempt(self):
        self.assertTrue(
            component.should_consume_auto_attempt(
                already_attempted=False,
                has_location_result=True,
                request_finished=True,
            )
        )

    def test_a_finished_request_without_a_result_also_concludes_it(self):
        # 浏览器报告请求结束但没给结果时也不能留下永久 pending。
        self.assertTrue(
            component.should_consume_auto_attempt(
                already_attempted=False,
                has_location_result=False,
                request_finished=True,
            )
        )

    def test_a_concluded_attempt_is_never_repeated(self):
        self.assertFalse(
            component.should_consume_auto_attempt(
                already_attempted=True,
                has_location_result=True,
                request_finished=True,
            )
        )
        self.assertFalse(
            component.should_attempt_auto_location(
                has_current_location=False,
                already_attempted=True,
            )
        )

    def test_a_pending_request_never_survives_a_finished_report(self):
        for outcome in ("granted", "denied", "timeout", "unavailable"):
            with self.subTest(outcome=outcome):
                pending = component.resolve_request_state_update(
                    stored_pending=True,
                    reported_state="finished",
                )
                self.assertIs(pending, False)


class RequestStateUpdateTests(unittest.TestCase):
    def test_started_marks_the_request_pending(self):
        self.assertIs(
            component.resolve_request_state_update(
                stored_pending=False,
                reported_state="started",
            ),
            True,
        )

    def test_finished_clears_the_pending_request(self):
        self.assertIs(
            component.resolve_request_state_update(
                stored_pending=True,
                reported_state="finished",
            ),
            False,
        )

    def test_an_unchanged_request_state_does_not_rerun(self):
        for pending, state in ((True, "started"), (False, "finished")):
            with self.subTest(state=state):
                self.assertIsNone(
                    component.resolve_request_state_update(
                        stored_pending=pending,
                        reported_state=state,
                    )
                )

    def test_untrusted_request_states_are_ignored(self):
        for value in (None, "", "STARTED", "pending", 1, ["started"], {}):
            with self.subTest(value=value):
                self.assertIsNone(
                    component.resolve_request_state_update(
                        stored_pending=False,
                        reported_state=value,
                    )
                )

    def test_request_state_never_carries_coordinates(self):
        self.assertEqual(component.REQUEST_STATES, {"started", "finished"})


class AutoLocationGuardTests(unittest.TestCase):
    def test_new_session_without_location_may_attempt_once(self):
        self.assertTrue(
            component.should_attempt_auto_location(
                has_current_location=False,
                already_attempted=False,
            )
        )

    def test_existing_location_suppresses_the_automatic_attempt(self):
        self.assertFalse(
            component.should_attempt_auto_location(
                has_current_location=True,
                already_attempted=False,
            )
        )

    def test_an_attempt_is_never_repeated_in_the_same_session(self):
        self.assertFalse(
            component.should_attempt_auto_location(
                has_current_location=False,
                already_attempted=True,
            )
        )

    def test_manual_relocate_is_not_gated_by_the_auto_guard(self):
        # 自动守卫只影响自动定位，按钮标签仍然允许用户重新定位。
        self.assertEqual(
            component.location_button_label(
                attempt_status="granted",
                has_current_location=True,
            ),
            component.RELOCATE_LABEL,
        )


class PermissionStateUpdateTests(unittest.TestCase):
    def test_a_changed_permission_state_is_stored(self):
        self.assertEqual(
            component.resolve_permission_state_update("unknown", "granted"),
            "granted",
        )

    def test_an_unchanged_permission_state_does_not_rerun(self):
        for state in ("granted", "prompt", "denied", "unsupported", "unknown"):
            with self.subTest(state=state):
                self.assertIsNone(
                    component.resolve_permission_state_update(state, state)
                )

    def test_untrusted_browser_values_are_ignored(self):
        for value in (None, "", "GRANTED", "some-other", 42, ["granted"], {}):
            with self.subTest(value=value):
                self.assertIsNone(
                    component.resolve_permission_state_update("unknown", value)
                )


class LocationStatusMessageTests(unittest.TestCase):
    def messages(self, **overrides):
        arguments = {
            "attempt_status": "idle",
            "permission_state": "unknown",
            "has_current_location": False,
        }
        arguments.update(overrides)
        return component.location_status_message(**arguments)

    def test_denied_permission_explains_the_browser_setting(self):
        message = self.messages(permission_state="denied")
        self.assertIn("浏览器已禁止位置权限", message)
        self.assertIn("网站权限设置", message)
        # 不声称可以绕过浏览器权限，只保留手动输入城市的出口。
        self.assertIn("手动输入城市", message)

    def test_denied_attempt_status_is_reported_without_a_permission_state(self):
        self.assertEqual(
            self.messages(attempt_status="denied"),
            component.DENIED_MESSAGE,
        )

    def test_unsupported_environment_keeps_its_message(self):
        self.assertEqual(
            self.messages(attempt_status="unsupported"),
            component.UNSUPPORTED_MESSAGE,
        )
        self.assertEqual(
            self.messages(permission_state="unsupported"),
            component.UNSUPPORTED_MESSAGE,
        )

    def test_timeout_and_unavailable_keep_their_messages(self):
        self.assertEqual(
            self.messages(attempt_status="timeout"),
            component.TIMEOUT_MESSAGE,
        )
        self.assertEqual(
            self.messages(attempt_status="unavailable"),
            component.UNAVAILABLE_MESSAGE,
        )

    def test_an_existing_location_shows_no_warning(self):
        self.assertIsNone(
            self.messages(
                attempt_status="denied",
                permission_state="denied",
                has_current_location=True,
            )
        )

    def test_prompt_and_unknown_states_show_no_extra_copy(self):
        self.assertIsNone(self.messages(permission_state="prompt"))
        self.assertIsNone(self.messages(permission_state="unknown"))


class ComponentStorageBoundaryTests(unittest.TestCase):
    def test_the_component_never_persists_coordinates_server_side(self):
        import pathlib

        source = pathlib.Path(component.__file__).read_text(encoding="utf-8")

        for forbidden in (
            "localStorage",
            "sessionStorage",
            "cookie",
            "chat_repository",
            "database",
            "mysql",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


class FakeComponent:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class RenderGeolocationButtonTests(unittest.TestCase):
    def render(self, result, **kwargs):
        fake = FakeComponent(result)
        with patch.object(
            component,
            "_get_geolocation_component",
            return_value=fake,
        ):
            returned = component.render_geolocation_button(**kwargs)
        return fake.calls[0], returned

    def test_auto_attempt_is_forwarded_to_the_browser(self):
        call, _ = self.render(
            SimpleNamespace(location_result=None, permission_state=None),
            auto_attempt=True,
        )

        self.assertIs(call["data"]["auto_attempt"], True)

    def test_auto_attempt_defaults_to_disabled(self):
        call, _ = self.render(
            SimpleNamespace(location_result=None, permission_state=None),
        )

        self.assertIs(call["data"]["auto_attempt"], False)

    def test_all_transient_values_are_returned(self):
        _, returned = self.render(
            SimpleNamespace(
                location_result={"status": "granted"},
                permission_state="granted",
                location_request_state="finished",
            )
        )

        self.assertEqual(returned.location_result, {"status": "granted"})
        self.assertEqual(returned.permission_state, "granted")
        self.assertEqual(returned.request_state, "finished")

    def test_missing_trigger_values_are_reported_as_none(self):
        _, returned = self.render(SimpleNamespace())

        self.assertIsNone(returned.location_result)
        self.assertIsNone(returned.permission_state)
        self.assertIsNone(returned.request_state)

    def test_all_trigger_callbacks_are_registered(self):
        call, _ = self.render(SimpleNamespace())

        self.assertIn("on_location_result_change", call)
        self.assertIn("on_permission_state_change", call)
        self.assertIn("on_location_request_state_change", call)


if __name__ == "__main__":
    unittest.main()
