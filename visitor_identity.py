from datetime import datetime, timedelta
import uuid

import streamlit as st
from streamlit_cookies_controller import CookieController


VISITOR_COOKIE_NAME = "ai_agent_visitor_id"
VISITOR_COOKIE_DAYS = 365


def _normalize_visitor_id(value):
    """Return a canonical UUID string, or None for an invalid cookie value."""
    if not value:
        return None

    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None


def get_or_create_visitor_id():
    """Read a visitor UUID from a cookie or create and persist a new one."""
    visitor_id = _normalize_visitor_id(
        st.context.cookies.get(VISITOR_COOKIE_NAME)
    )
    if visitor_id:
        return visitor_id

    visitor_id = str(uuid.uuid4())

    try:
        cookie_controller = CookieController(key="visitor_cookie_controller")
        cookie_controller.set(
            VISITOR_COOKIE_NAME,
            visitor_id,
            path="/",
            expires=datetime.now() + timedelta(days=VISITOR_COOKIE_DAYS),
            same_site="lax",
        )
    except Exception:
        # Cookie failures should not prevent the current browser session from
        # using its generated visitor ID.
        pass

    return visitor_id
