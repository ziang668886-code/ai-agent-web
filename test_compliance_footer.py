"""Offline checks for public registration links; never import the full app."""

import ast
import base64
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).with_name("ai-agent-web.py")
APP_TREE = ast.parse(APP_PATH.read_text(encoding="utf-8-sig"))
FOOTER = next(
    node for node in APP_TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "render_compliance_footer"
)


class FooterHTML(HTMLParser):
    def __init__(self, body):
        super().__init__()
        self.links = []
        self.images = []
        self.feed(body)

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs))
        elif tag == "img":
            self.images.append(dict(attrs))


class ComplianceFooterTests(unittest.TestCase):
    def render(self):
        html = Mock()
        namespace = {
            "st": SimpleNamespace(html=html), "base64": base64,
            "Path": Path, "__file__": str(APP_PATH),
        }
        exec(compile(ast.Module(body=[FOOTER], type_ignores=[]), str(APP_PATH), "exec"), namespace)
        namespace["render_compliance_footer"]()
        html.assert_called_once()
        return html.call_args.args[0]

    def test_complete_registration_numbers_and_safe_official_links(self):
        body = self.render()
        self.assertEqual(body.count("津ICP备2026014222号"), 1)
        self.assertEqual(body.count("京公网安备11011402057426号"), 1)
        self.assertIn("© 2026 AI 城市生活助手", body)
        links = FooterHTML(body).links[:2]
        self.assertEqual([link["href"] for link in links], [
            "https://beian.miit.gov.cn/",
            "https://beian.mps.gov.cn/#/query/webSearch?code=11011402057426",
        ])
        for link in links:
            self.assertEqual(link["target"], "_blank")
            self.assertEqual(set(link["rel"].split()), {"noopener", "noreferrer"})

    def test_contact_email_is_clickable_and_below_the_registration_links(self):
        body = self.render()
        self.assertIn('联系我们：<a href="mailto:3382783015@qq.com">3382783015@qq.com</a>', body)
        self.assertEqual(FooterHTML(body).links[-1]["href"], "mailto:3382783015@qq.com")
        self.assertGreater(body.index("联系我们："), body.index("京公网安备11011402057426号"))

    def test_icon_uses_the_existing_local_png(self):
        images = FooterHTML(self.render()).images
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0]["src"].startswith("data:image/png;base64,"))
        decoded = base64.b64decode(images[0]["src"].split(",", 1)[1], validate=True)
        self.assertEqual(decoded, (APP_PATH.parent / "assets" / "beian.png").read_bytes())
        self.assertTrue(decoded.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_missing_or_unreadable_icon_keeps_both_text_links(self):
        for error in (FileNotFoundError, PermissionError):
            with self.subTest(error=error.__name__), patch.object(Path, "read_bytes", side_effect=error):
                body = self.render()
            self.assertFalse(FooterHTML(body).images)
            self.assertEqual(len(FooterHTML(body).links), 3)
            self.assertIn("津ICP备2026014222号", body)
            self.assertIn("京公网安备11011402057426号", body)

    def test_responsive_styles_do_not_pin_or_resize_other_content(self):
        body = self.render()
        self.assertIn("flex-wrap: wrap", body)
        self.assertIn("overflow-wrap: anywhere", body)
        self.assertIn("height: 18px", body)
        self.assertIn("width: auto", body)
        self.assertNotIn("position:", body)
        self.assertNotIn("<script", body)
        self.assertNotIn("stSidebar", body)
        self.assertNotIn("stMainBlockContainer", body)

    def test_footer_is_called_once_after_the_shared_view_router(self):
        calls = [node for node in ast.walk(APP_TREE) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "render_compliance_footer"]
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(APP_TREE.body[-2], ast.If)
        self.assertEqual(APP_TREE.body[-1].value, calls[0])

    def test_streamlit_renders_one_footer_across_all_six_view_reruns(self):
        views = ("home", "chat", "explore", "knowledge", "history", "settings")
        script = (
            "import base64\nfrom pathlib import Path\nimport streamlit as st\n"
            f"__file__ = {str(APP_PATH)!r}\n"
            + ast.unparse(FOOTER)
            + "\nconversations = []\n"
        )
        for function in ("render_home", "render_chat", "render_city_explore",
                         "render_knowledge_base", "render_history", "render_settings"):
            script += f"def {function}(*args): st.title({function!r})\n"
        script += (
            f"active_view = st.selectbox('Mock view', {views!r})\n"
            + ast.unparse(APP_TREE.body[-2])
            + "\nrender_compliance_footer()\n"
        )
        app = AppTest.from_string(script).run()
        for view in views:
            app.selectbox[0].set_value(view).run()
            self.assertFalse(app.exception)
            html_elements = app.get("html")
            self.assertEqual(len(html_elements), 1)
            self.assertEqual(html_elements[0].proto.body.count("<footer "), 1)
            self.assertIn("津ICP备2026014222号", html_elements[0].proto.body)
            self.assertIn("京公网安备11011402057426号", html_elements[0].proto.body)
            self.assertIn("mailto:3382783015@qq.com", html_elements[0].proto.body)


if __name__ == "__main__":
    unittest.main()
