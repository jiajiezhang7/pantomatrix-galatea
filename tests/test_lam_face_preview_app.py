from __future__ import annotations

from conftest import load_tool_module


def test_build_app_css_hides_runtime_fps_overlay():
    app = load_tool_module("lam_face_preview_app_test", "lam_face_preview_app.py")

    css = app.build_app_css()

    assert "#fps" in css
    assert "display: none" in css
