#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal LAM face preview app for packaged arkitWithBSData zips.")
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--zip-name", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--width", type=int, default=380)
    parser.add_argument("--height", type=int, default=680)
    return parser.parse_args()


def build_app_css() -> str:
    return "#fps { display: none !important; }"


def main() -> int:
    import gradio as gr
    from gradio_gaussian_render import gaussian_render

    args = parse_args()
    zip_url = f"gradio_api/file={args.assets_dir / args.zip_name}"

    gr.set_static_paths(args.assets_dir)
    with gr.Blocks(analytics_enabled=False, css=build_app_css()) as demo:
        gaussian_render(width=args.width, height=args.height)
        demo.load(
            fn=None,
            js=f"""() => {{
                const zip = '{zip_url}';
                const go = () => {{
                    if (window.start) {{
                        window.start(zip);
                    }} else {{
                        setTimeout(go, 100);
                    }}
                }};
                go();
            }}""",
            queue=False,
        )
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=False,
            inbrowser=False,
            show_error=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
