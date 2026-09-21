"""Streamlit Components v2 wrapper for one-shot browser geolocation."""

from __future__ import annotations

from typing import Any, NamedTuple

import streamlit as st


# 浏览器 Permissions API 对 geolocation 的四种状态，加上尚未查询过的 unknown。
PERMISSION_STATES = frozenset(
    {"unknown", "granted", "prompt", "denied", "unsupported"}
)
# 浏览器是否真的开始/结束了一次定位请求。只传状态，不传坐标。
REQUEST_STATES = frozenset({"started", "finished"})

GET_LOCATION_LABEL = "获取我的位置"
RELOCATE_LABEL = "重新定位"
DENIED_MESSAGE = (
    "浏览器已禁止位置权限，请在网站权限设置中允许位置访问后重新尝试。"
    "你仍可以手动输入城市或地点。"
)
UNSUPPORTED_MESSAGE = "当前浏览器环境不支持定位。"
TIMEOUT_MESSAGE = "定位超时，请稍后重试。"
UNAVAILABLE_MESSAGE = "暂时无法获取位置，请稍后重试。"


class GeolocationComponentResult(NamedTuple):
    """Transient values reported by one render of the browser component."""

    location_result: Any
    permission_state: Any
    request_state: Any


def should_attempt_auto_location(
    *,
    has_current_location: bool,
    already_attempted: bool,
) -> bool:
    """Return whether this render may arm one automatic geolocation attempt.

    A Streamlit session arms the automatic attempt at most once. Once armed the
    caller records it, so ordinary reruns never trigger GPS again. An existing
    location suppresses the attempt entirely.
    """

    return not has_current_location and not already_attempted


def resolve_request_state_update(
    *,
    stored_pending: bool,
    reported_state: Any,
) -> bool | None:
    """Return the request-pending flag worth storing, or ``None`` unchanged."""

    if not isinstance(reported_state, str) or reported_state not in REQUEST_STATES:
        return None
    pending = reported_state == "started"
    if pending == stored_pending:
        return None
    return pending


def should_consume_auto_attempt(
    *,
    already_attempted: bool,
    has_location_result: bool,
    request_finished: bool,
) -> bool:
    """Return whether this render concludes the one automatic attempt.

    The attempt is consumed only once the browser reports a terminal outcome.
    That keeps ``auto_attempt`` stable while a request is genuinely in flight,
    so a mid-request rerun cannot reconfigure the component and strand the
    browser callback.
    """

    if already_attempted:
        return False
    return has_location_result or request_finished


def resolve_permission_state_update(
    stored_state: Any,
    reported_state: Any,
) -> str | None:
    """Return the permission state worth storing, or ``None`` when unchanged.

    The browser supplies this value, so unknown strings are ignored rather than
    trusted. Returning ``None`` for an unchanged state is what stops a rerun
    loop when the component re-reports the same permission.
    """

    if not isinstance(reported_state, str) or reported_state not in PERMISSION_STATES:
        return None
    if reported_state == stored_state:
        return None
    return reported_state


def location_button_label(
    *,
    attempt_status: str,
    has_current_location: bool,
) -> str:
    """Return the sidebar button label for the current location state."""

    if has_current_location or attempt_status != "idle":
        return RELOCATE_LABEL
    return GET_LOCATION_LABEL


def location_status_message(
    *,
    attempt_status: str,
    permission_state: str,
    has_current_location: bool,
) -> str | None:
    """Return the caption explaining the current location state, if any."""

    if has_current_location:
        return None
    if attempt_status == "denied" or permission_state == "denied":
        return DENIED_MESSAGE
    if attempt_status == "unsupported" or permission_state == "unsupported":
        return UNSUPPORTED_MESSAGE
    if attempt_status == "timeout":
        return TIMEOUT_MESSAGE
    if attempt_status == "unavailable":
        return UNAVAILABLE_MESSAGE
    return None


_COMPONENT_HTML = """
        <div class="location-control">
            <button type="button" data-location-button>获取我的位置</button>
            <span data-location-progress aria-live="polite"></span>
        </div>
    """

_COMPONENT_CSS = """
        .location-control {
            display: grid;
            gap: 0.4rem;
            width: 100%;
        }
        [data-location-button] {
            width: 100%;
            min-height: 2.4rem;
            padding: 0.45rem 0.75rem;
            border: 1px solid var(--st-color-border, rgba(49, 51, 63, 0.2));
            border-radius: 0.5rem;
            background: transparent;
            color: inherit;
            font: inherit;
            cursor: pointer;
        }
        [data-location-button]:hover:not(:disabled) {
            border-color: var(--st-color-primary, #ff4b4b);
        }
        [data-location-button]:disabled {
            cursor: wait;
            opacity: 0.65;
        }
        [data-location-progress] {
            min-height: 1rem;
            color: var(--st-color-text-secondary, #6b7280);
            font-size: 0.78rem;
        }
    """

_COMPONENT_JS = """
        // 上一次上报给 Python 的权限状态。模块只在组件挂载时求值一次，
        // 因此同一个 iframe 内不会重复上报同一个状态，避免触发 rerun 循环。
        let reportedPermissionState = null;
        // 同一个 iframe 内最多自动定位一次。
        let autoAttempted = false;
        // 请求编号与看门狗放在模块作用域：组件被重新调用时也不会丢失，
        // 否则进行中的请求会被误判为过期，反而把加载态留在页面上。
        let requestId = 0;
        let watchdog = null;

        // getCurrentPosition 的 timeout 选项在等待权限决定期间不计时，
        // 浏览器可能既不回调成功也不回调失败。这个看门狗保证任何一次请求
        // 最终都会收尾，UI 不会永久停在进行中状态。
        const REQUEST_WATCHDOG_MS = 15000;

        export default function(component) {
            const { parentElement, data, setTriggerValue } = component;
            const button = parentElement.querySelector('[data-location-button]');
            const progress = parentElement.querySelector('[data-location-progress]');

            button.textContent = data?.button_label || '获取我的位置';

            const supported = window.isSecureContext && !!navigator.geolocation;

            const finish = (id, status, location) => {
                if (id !== requestId) {
                    return;
                }
                if (watchdog !== null) {
                    clearTimeout(watchdog);
                    watchdog = null;
                }
                setTriggerValue(
                    'location_result',
                    location ? { status, location } : { status },
                );
                setTriggerValue('location_request_state', 'finished');
                button.disabled = false;
                progress.textContent = '';
            };

            // 唯一的定位入口：手动点击和自动恢复共用。坐标只通过
            // setTriggerValue 交回 Python，不写入任何浏览器持久存储。
            const requestPosition = () => {
                requestId += 1;
                const id = requestId;

                if (!supported) {
                    finish(id, 'unsupported');
                    return;
                }

                button.disabled = true;
                progress.textContent = '正在获取位置…';
                setTriggerValue('location_request_state', 'started');
                watchdog = setTimeout(() => finish(id, 'timeout'), REQUEST_WATCHDOG_MS);

                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        finish(id, 'granted', {
                            latitude: position.coords.latitude,
                            longitude: position.coords.longitude,
                            accuracy: position.coords.accuracy,
                        });
                    },
                    (error) => {
                        const statuses = {
                            1: 'denied',
                            2: 'unavailable',
                            3: 'timeout',
                        };
                        finish(id, statuses[error.code] || 'unavailable');
                    },
                    {
                        enableHighAccuracy: true,
                        timeout: 10000,
                        maximumAge: 0,
                    },
                );
            };

            // 手动按钮永远可用：即使已有位置也允许用户主动重新定位，
            // 自动定位失败或被看门狗终止后也仍然可以重试。
            button.onclick = requestPosition;

            const reportPermissionState = (state) => {
                if (state === reportedPermissionState) {
                    return;
                }
                reportedPermissionState = state;
                setTriggerValue('permission_state', state);
            };

            if (!supported) {
                reportPermissionState('unsupported');
                return;
            }

            // Permissions API 不可用时保持纯手动模式：既不上报权限状态，
            // 也绝不自动定位。
            const permissions = navigator.permissions;
            if (!permissions || typeof permissions.query !== 'function') {
                return;
            }

            const applyPermissionState = (state) => {
                reportPermissionState(state);
                // 只有浏览器已授权、Python 明确放行、且本 iframe 还没试过时
                // 才自动定位。prompt 状态绝不能自动请求，否则浏览器会弹出权限框。
                if (state === 'granted' && data?.auto_attempt && !autoAttempted) {
                    autoAttempted = true;
                    requestPosition();
                }
            };

            try {
                permissions
                    .query({ name: 'geolocation' })
                    .then((status) => {
                        applyPermissionState(status.state);
                        // 用户在浏览器设置里改动权限时只刷新 UI 状态，
                        // 不自动发起定位请求。
                        status.onchange = () => applyPermissionState(status.state);
                    })
                    .catch(() => {});
            } catch (error) {
                // 查询失败时保持手动按钮可用，不做任何自动请求。
            }
        }
    """

_geolocation_component = None


def _get_geolocation_component():
    """Register the component lazily, after the app has set its page config."""
    global _geolocation_component
    if _geolocation_component is None:
        _geolocation_component = st.components.v2.component(
            "browser_geolocation",
            html=_COMPONENT_HTML,
            css=_COMPONENT_CSS,
            js=_COMPONENT_JS,
            isolate_styles=True,
        )
    return _geolocation_component


def _ignore_location_callback() -> None:
    """Let the trigger rerun Streamlit; processing happens in page code."""


def render_geolocation_button(
    *,
    button_label: str = GET_LOCATION_LABEL,
    auto_attempt: bool = False,
    key: str = "browser_geolocation_component",
) -> GeolocationComponentResult:
    """Render the location control and return transient browser-reported values.

    ``auto_attempt`` only permits a single automatic ``getCurrentPosition`` call
    when the browser already reports ``granted``; it never forces the browser
    permission prompt.
    """

    result = _get_geolocation_component()(
        key=key,
        data={
            "button_label": button_label,
            "auto_attempt": bool(auto_attempt),
        },
        width="stretch",
        height="content",
        on_location_result_change=_ignore_location_callback,
        on_permission_state_change=_ignore_location_callback,
        on_location_request_state_change=_ignore_location_callback,
    )
    return GeolocationComponentResult(
        location_result=getattr(result, "location_result", None),
        permission_state=getattr(result, "permission_state", None),
        request_state=getattr(result, "location_request_state", None),
    )
