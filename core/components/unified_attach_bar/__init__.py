import os
import re
import base64
import tempfile
from typing import Optional, Dict, Any
import streamlit as st
import streamlit.components.v1 as components

_component_dir = os.path.dirname(os.path.abspath(__file__))
_unified_component = components.declare_component("unified_attach_bar_widget", path=_component_dir)

def render_unified_attach_bar(key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Renders an All-In-One Unified Attachment Bar:
    - Handles file browsing / drag-and-drop (PDF, Excel, Images, Audio, Video)
    - Handles Ctrl+V clipboard image pasting
    - Handles URL link pasting (YouTube, Web, Image, Base64)
    Returns:
        dict with metadata or None.
    """
    result = _unified_component(key=key, default=None)
    if not result or not isinstance(result, dict):
        return None

    res_type = result.get("type")
    
    # 1. File or Image upload via Base64 data_url
    if res_type in ["file", "image"]:
        data_url = result.get("data_url")
        raw_name = result.get("name", "uploaded_file")
        if data_url and ";base64," in data_url:
            try:
                header, b64_str = data_url.split(";base64,", 1)
                file_bytes = base64.b64decode(b64_str)
                
                # Determine extension
                file_ext = raw_name.split(".")[-1].lower() if "." in raw_name else ""
                if not file_ext or len(file_ext) > 5:
                    if "image" in header:
                        file_ext = "png"
                    elif "pdf" in header:
                        file_ext = "pdf"
                    else:
                        file_ext = "bin"

                temp_dir = tempfile.gettempdir()
                clean_name = raw_name if "." in raw_name else f"attachment_{os.urandom(4).hex()}.{file_ext}"
                temp_path = os.path.join(temp_dir, clean_name)
                with open(temp_path, "wb") as f:
                    f.write(file_bytes)

                size_mb = round(len(file_bytes) / (1024 * 1024), 2)
                image_exts = ["png", "jpg", "jpeg", "webp", "bmp", "gif", "svg", "jfif", "tiff", "ico"]
                cat = "image" if file_ext in image_exts or res_type == "image" else \
                      "audio" if file_ext in ["mp3", "wav"] else \
                      "video" if file_ext in ["mp4", "mov"] else \
                      "excel" if file_ext in ["xlsx", "csv"] else \
                      "pdf" if file_ext == "pdf" else "text"

                return {
                    "kind": "file",
                    "path": temp_path,
                    "type": cat,
                    "name": clean_name,
                    "size_mb": size_mb if size_mb > 0.01 else 0.01
                }
            except Exception:
                return None

    # 2. Link or direct URL
    elif res_type == "link":
        url = result.get("url", "").strip()
        if url:
            return {
                "kind": "link",
                "url": url
            }

    return None
