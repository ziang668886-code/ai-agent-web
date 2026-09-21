"""Streamlit Components v2 wrapper for one-shot browser geolocation."""

from __future__ import annotations

from typing import Any

import streamlit as st


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
        export default function(component) {
            const { parentElement, data, setTriggerValue } = component;
            const button = parentElement.querySelector('[data-location-button]');
            const progress = parentElement.querySelector('[data-location-progress]');

            button.textContent = data?.button_label || '获取我的位置';
            button.onclick = () => {
                if (!window.isSecureContext || !navigator.geolocation) {
                    setTriggerValue('location_result', { status: 'unsupported' });
                    return;
                }

                button.disabled = true;
                progress.textContent = '正在获取位置…';

                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        setTriggerValue('location_result', {
                            status: 'granted',
                            location: {
                                latitude: position.coords.latitude,
                                longitude: position.coords.longitude,
                                accuracy: position.coords.accuracy,
                            },
                        });
                        button.disabled = false;
                        progress.textContent = '';
                    },
                    (error) => {
                        const statuses = {
                            1: 'denied',
                            2: 'unavailable',
                            3: 'timeout',
                        };
                        setTriggerValue('location_result', {
                            status: statuses[error.code] || 'unavailable',
                        });
                        button.disabled = false;
                        progress.textContent = '';
                    },
                    {
                        enableHighAccuracy: true,
                        timeout: 10000,
                        maximumAge: 0,
                    },
                );
            };
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
    button_label: str = "获取我的位置",
    key: str = "browser_geolocation_component",
) -> Any:
    """Render the one-shot button and return a transient browser result."""

    result = _get_geolocation_component()(
        key=key,
        data={"button_label": button_label},
        width="stretch",
        height="content",
        on_location_result_change=_ignore_location_callback,
    )
    return getattr(result, "location_result", None)
