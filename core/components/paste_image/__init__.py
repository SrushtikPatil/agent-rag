import os
import re
import base64
import tempfile
from typing import Optional, Dict, Any
import streamlit as st
import streamlit.components.v1 as components

_component_dir = os.path.dirname(os.path.abspath(__file__))
_paste_component = components.declare_component("paste_image_zone", path=_component_dir)

def render_paste_image_zone(key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Renders an interactive paste dropzone for clipboard images.
    Returns:
        dict with metadata of the saved image file, or None if no image pasted yet.
    """
    result = _paste_component(key=key, default=None)
    if not result or not isinstance(result, dict):
        return None

    data_url = result.get("data_url")
    image_url = result.get("image_url")
    custom_name = result.get("name", "pasted_image.png")

    if data_url and ";base64," in data_url:
        try:
            header, b64_str = data_url.split(";base64,", 1)
            img_bytes = base64.b64decode(b64_str)
            
            ext = "png"
            if "jpeg" in header or "jpg" in header:
                ext = "jpg"
            elif "webp" in header:
                ext = "webp"
            elif "gif" in header:
                ext = "gif"

            temp_dir = tempfile.gettempdir()
            clean_name = f"clipboard_image_{os.urandom(4).hex()}.{ext}"
            temp_path = os.path.join(temp_dir, clean_name)
            with open(temp_path, "wb") as f:
                f.write(img_bytes)

            size_mb = round(len(img_bytes) / (1024 * 1024), 3)
            return {
                "path": temp_path,
                "type": "image",
                "name": clean_name,
                "size_mb": size_mb if size_mb > 0.01 else 0.01
            }
        except Exception:
            return None

    if image_url:
        return {"image_url": image_url}

    return None
