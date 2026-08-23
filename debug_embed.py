import sys
import os
from pathlib import Path
from streamlit_app.utils.static_embed import render_static_page
import streamlit.components.v1 as components

# Mock components.html to just print
def mock_html(html, height=None, scrolling=None):
    with open("debug_out.html", "w") as f:
        f.write(html)
    print("Wrote debug_out.html with length:", len(html))

components.html = mock_html

ROOT = Path("streamlit_app/Home.py").resolve().parent.parent
original_dir = ROOT / "templatemo_614_quantix_saas"
html_path = str(original_dir / "index.html")
css_path = str(original_dir / "templatemo-quantix-style.css")
js_path = str(original_dir / "templatemo-quantix-script.js")

render_static_page(html_path, css_path, js_path)
