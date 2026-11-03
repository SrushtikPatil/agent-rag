"""
Personal Agentic Multi-Modal Assistant
Executive ChatGPT, Claude & Gemini Professional Interface
"""

import streamlit as st
import os
import tempfile
import importlib
import pandas as pd
import base64
from datetime import datetime
from dotenv import load_dotenv

# Path to .env for permanent storage
ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(ENV_FILE)

# Assets & Cute 3D Robot Avatar Setup
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
ROBOT_AVATAR_PATH = os.path.join(ASSETS_DIR, "robot_avatar.png")
robot_b64 = ""
if os.path.exists(ROBOT_AVATAR_PATH):
    with open(ROBOT_AVATAR_PATH, "rb") as img_f:
        robot_b64 = base64.b64encode(img_f.read()).decode("utf-8")

# Local imports with auto-reload
import core.gemini_service
import core.rag_engine
import core.agentic_router
importlib.reload(core.gemini_service)
importlib.reload(core.rag_engine)
importlib.reload(core.agentic_router)

from database.db import init_db
from database.models import get_user_profile, add_contact_event, get_contact_events, get_festivals
from core.proactive_engine import get_proactive_briefing
from core.agentic_router import AgenticAssistant, AgentState

# Page Configuration - Executive Minimalist Look with Cute Robot Icon
st.set_page_config(
    page_title="Agent Rag",
    page_icon=ROBOT_AVATAR_PATH if os.path.exists(ROBOT_AVATAR_PATH) else "🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database on first run
init_db()

# ---------------------------------------------------------------------------
# Executive Professional Styling (Claude / ChatGPT / Gemini Studio)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0f172a;
    }

    /* Professional Light Blue Background */
    .stApp {
        background: linear-gradient(180deg, #f0f7ff 0%, #f8fafc 40%, #ffffff 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* Header & Toolbar: Transparent and active so sidebar button works */
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 99999 !important;
    }
    div[data-testid="stToolbar"] {
        background: transparent !important;
        display: flex !important;
        visibility: visible !important;
    }
    
    /* Hide top colored decoration bar, Deploy button, and 3-dots menu */
    div[data-testid="stDecoration"],
    .stAppDeployButton,
    #MainMenu {
        display: none !important;
    }

    /* Streamlit Modern Sidebar Expand Button (Appears when sidebar is collapsed) */
    button[data-testid="stExpandSidebarButton"] {
        display: inline-flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: #ffffff !important;
        border: 1.5px solid #0284c7 !important;
        border-radius: 10px !important;
        color: #0284c7 !important;
        box-shadow: 0 3px 12px rgba(2, 132, 199, 0.28) !important;
        margin-top: 8px !important;
        margin-left: 12px !important;
        padding: 6px 10px !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    button[data-testid="stExpandSidebarButton"]:hover {
        background: #f0f9ff !important;
        border-color: #0369a1 !important;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.45) !important;
        transform: scale(1.1) !important;
    }
    button[data-testid="stExpandSidebarButton"] span,
    button[data-testid="stExpandSidebarButton"] svg {
        color: #0284c7 !important;
        fill: #0284c7 !important;
        font-size: 1.3rem !important;
    }

    /* Sidebar Collapse Button (Inside the sidebar header when open) */
    button[data-testid="stSidebarCollapseButton"] {
        color: #0369a1 !important;
        border-radius: 8px !important;
    }
    button[data-testid="stSidebarCollapseButton"]:hover {
        background: #e0f2fe !important;
        color: #0284c7 !important;
    }
    
    /* Main container clean spacing */
    div[data-testid="stMainBlockContainer"],
    .block-container:not([data-testid="stBottomBlockContainer"]) {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 880px !important;
    }

    /* Fixed Bottom Chat Container - Clean Alignment that never overlaps with Sidebar */
    div[data-testid="stBottom"] {
        background: transparent !important;
        z-index: 80 !important;
    }
    div[data-testid="stBottomBlockContainer"] {
        max-width: 880px !important;
        padding-top: 0.5rem !important;
        padding-bottom: 1.2rem !important;
    }

    /* Executive Hero Greeting */
    .hero-wrapper {
        padding: 36px 0 20px 0;
        text-align: left;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        background: #e0f2fe;
        border: 1px solid #bae6fd;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 700;
        color: #0369a1;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(2, 132, 199, 0.08);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.035em;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 40%, #0f172a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #475569;
        font-weight: 400;
        margin-bottom: 22px;
        line-height: 1.5;
    }

    /* Twin Notification Pills - Matching Height & Light Blue / Amber Accents */
    .festival-pill, .birthday-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.84rem;
        font-weight: 500;
        height: 38px;
        min-height: 38px;
        max-height: 38px;
        box-sizing: border-box;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin: 0;
        width: 100%;
    }
    .festival-pill {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
        box-shadow: 0 1px 3px rgba(146, 64, 14, 0.05);
    }
    .birthday-pill {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        box-shadow: 0 1px 3px rgba(29, 78, 216, 0.06);
    }

    /* Executive Document Chip */
    .document-chip {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 8px 18px;
        background: #f0f9ff;
        border: 1px solid #7dd3fc;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #0369a1;
        box-shadow: 0 1px 4px rgba(2, 132, 199, 0.08);
        margin-bottom: 14px;
    }

    /* Refined Light-Blue Action Pill Buttons */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        border-radius: 20px !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        padding: 7px 16px !important;
        border: 1px solid #bae6fd !important;
        background: #ffffff !important;
        color: #0369a1 !important;
        box-shadow: 0 1px 3px rgba(2, 132, 199, 0.06) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        border-color: #0284c7 !important;
        background: #f0f9ff !important;
        color: #0c4a6e !important;
        box-shadow: 0 3px 8px rgba(2, 132, 199, 0.16) !important;
        transform: translateY(-1px) !important;
    }

    /* Chat Messages - Crisp Light Blue Card Styling */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e0f2fe !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.04) !important;
        transition: border-color 0.2s ease !important;
    }
    div[data-testid="stChatMessage"]:hover {
        border-color: #bae6fd !important;
    }
    
    /* Cute 3D AI Robot Avatar Badge */
    div[data-testid="stChatMessageAvatarCustom"],
    div[data-testid="stChatMessageAvatarAssistant"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        width: 38px !important;
        height: 38px !important;
        min-width: 38px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[data-testid="stChatMessageAvatarCustom"] img,
    div[data-testid="stChatMessageAvatarAssistant"] img {
        width: 38px !important;
        height: 38px !important;
        border-radius: 50% !important;
        box-shadow: 0 3px 10px rgba(2, 132, 199, 0.22) !important;
        border: 2px solid #7dd3fc !important;
        object-fit: cover !important;
    }
    
    /* Subtle Citation / Source Footnote */
    .source-footnote {
        font-size: 0.76rem;
        color: #0284c7;
        margin-top: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 500;
    }

    /* Sidebar Light Blue Atmosphere */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f9ff 0%, #f8fafc 100%) !important;
        border-right: 1px solid #bae6fd !important;
        z-index: 100 !important;
    }
    div[data-testid="stSidebarContent"] {
        position: relative !important;
        height: 100vh !important;
        max-height: 100vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        display: block !important;
    }
    div[data-testid="stSidebarUserContent"] {
        padding-bottom: 230px !important; /* ensures scrolling content never touches bottom drawers */
    }

    /* FIRMLY DOCK Milestones & Settings to the VERY BOTTOM / END LINE OF THE SIDEBAR */
    div[data-testid="stElementContainer"]:has(.st-key-sidebar_bottom_box),
    .st-key-sidebar_bottom_box {
        position: absolute !important;
        bottom: 14px !important;
        left: 14px !important;
        right: 14px !important;
        width: calc(100% - 28px) !important;
        margin: 0 !important;
        z-index: 50 !important;
    }
    .st-key-sidebar_bottom_box hr {
        margin: 8px 0 10px 0 !important;
        border-color: #bae6fd !important;
    }
    .st-key-sidebar_bottom_box div[data-testid="stExpander"] {
        margin-bottom: 6px !important;
    }

    section[data-testid="stSidebar"] h4, section[data-testid="stSidebar"] h6 {
        color: #0369a1 !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar Primary New Chat Button */
    div[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.24) !important;
        transition: all 0.2s ease !important;
        padding: 8px 16px !important;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* Light Blue File Uploader */
    div[data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 1.5px dashed #7dd3fc !important;
        border-radius: 12px !important;
        padding: 8px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #0284c7 !important;
        background: #f0f9ff !important;
    }

    /* Light Blue Expander Containers */
    div[data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e0f2fe !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 4px rgba(2, 132, 199, 0.04) !important;
    }

    /* Chat Input Bar - Polished Light Blue Ring with Mic on Left */
    div[data-testid="stChatInput"] {
        position: relative !important;
        border-radius: 28px !important;
        border: 1.5px solid #bae6fd !important;
        background: #ffffff !important;
        box-shadow: 0 4px 18px rgba(2, 132, 199, 0.09) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #0284c7 !important;
        box-shadow: 0 4px 22px rgba(2, 132, 199, 0.22) !important;
    }
    div[data-testid="stChatInput"] textarea {
        padding-left: 48px !important;
        text-decoration: none !important;
        -webkit-text-decorations-in-effect: none !important;
        text-decoration-line: none !important;
        caret-color: #0284c7 !important;
    }
    div[data-testid="stChatInput"] button {
        color: #0284c7 !important;
    }

    /* Completely Remove Any Red Background, Red Borders, or Error Coloring from Chat Input and Microphone */
    div[data-testid="stChatInput"] [data-testid="stTooltipErrorHoverTarget"],
    div[data-testid="stChatInput"] .stTooltipErrorHoverTarget,
    div[data-testid="stChatInput"] div:has(button[data-testid="stChatInputMicButton"]),
    div[data-testid="stChatInput"] div:has([data-testid="stChatInputMicButton"]) {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Executive Microphone Button Inside Search/Chat Bar (Left Side) - Strict Light Blue */
    div[data-testid="stChatInput"] button[data-testid="stChatInputMicButton"] {
        position: absolute !important;
        left: 10px !important;
        bottom: 7px !important;
        z-index: 20 !important;
        margin: 0 !important;
        color: #0284c7 !important;
        background: #f0f9ff !important;
        background-color: #f0f9ff !important;
        border: 1.5px solid #bae6fd !important;
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        min-height: 32px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 1px 3px rgba(2, 132, 199, 0.12) !important;
    }
    div[data-testid="stChatInput"] button[data-testid="stChatInputMicButton"]:hover {
        background: #e0f2fe !important;
        background-color: #e0f2fe !important;
        color: #0369a1 !important;
        border-color: #0284c7 !important;
        transform: scale(1.08) !important;
    }
    div[data-testid="stChatInput"] button[data-testid="stChatInputMicButton"]:active,
    div[data-testid="stChatInput"] button[data-testid="stChatInputMicButton"]:focus {
        background: #e0f2fe !important;
        background-color: #e0f2fe !important;
        outline: none !important;
    }

    /* Never show red icons or red backgrounds even in error, disabled, or recording state */
    div[data-testid="stChatInput"] button[data-testid="stChatInputMicButton"] svg,
    div[data-testid="stChatInput"] button[data-testid="stChatInputMicButton"] * {
        fill: #0284c7 !important;
        color: #0284c7 !important;
    }

    button[data-testid="stChatInputSubmitButton"] {
        color: #0284c7 !important;
    }
    button[data-testid="stChatInputSubmitButton"]:hover {
        color: #0369a1 !important;
    }

    /* Inline Code & Badges */
    code {
        color: #0369a1 !important;
        background-color: #f0f9ff !important;
        padding: 2px 7px !important;
        border-radius: 6px !important;
        border: 1px solid #e0f2fe !important;
        font-size: 0.88em !important;
    }

    /* Executive Voice Mic Widget Styling */
    div[data-testid="stAudioInput"] {
        background: #ffffff !important;
        border: 1.5px solid #bae6fd !important;
        border-radius: 14px !important;
        padding: 6px 12px !important;
        box-shadow: 0 1px 4px rgba(2, 132, 199, 0.05) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stAudioInput"]:hover {
        border-color: #0284c7 !important;
        box-shadow: 0 3px 12px rgba(2, 132, 199, 0.12) !important;
    }
    div[data-testid="stAudioInput"] button {
        border-radius: 50% !important;
        transition: transform 0.15s ease !important;
    }
    div[data-testid="stAudioInput"] button:hover {
        transform: scale(1.08) !important;
    }
    .voice-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #e0f2fe;
        color: #0369a1;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 2px 9px;
        border-radius: 10px;
        border: 1px solid #bae6fd;
        margin-bottom: 6px;
    }

    /* Mobile & Tablet Responsive Optimizations (Show same clean UI on mobile devices) */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 4.5rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        .hero-title {
            font-size: 1.7rem !important;
            line-height: 1.25 !important;
        }
        .hero-subtitle {
            font-size: 0.92rem !important;
            margin-bottom: 14px !important;
        }
        div[data-testid="stBottomBlockContainer"] {
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
            padding-bottom: 0.8rem !important;
        }
        div[data-testid="stChatInput"] {
            border-radius: 22px !important;
        }
        button[data-testid="stExpandSidebarButton"] {
            top: 8px !important;
            left: 8px !important;
            padding: 8px 12px !important;
            font-size: 1.1rem !important;
        }
        div[data-testid="stChatMessage"] {
            padding: 12px 14px !important;
            border-radius: 12px !important;
        }
        .festival-pill, .birthday-pill {
            font-size: 0.78rem !important;
            padding: 4px 10px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization & Auto-Load from .env
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_processed_audio_hash" not in st.session_state:
    st.session_state.last_processed_audio_hash = None

env_key = ""
if os.path.exists(ENV_FILE):
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("GEMINI_API_KEY="):
                env_key = line.split("=", 1)[1].strip()

if "api_key" not in st.session_state or not st.session_state.api_key:
    st.session_state.api_key = env_key or os.getenv("GEMINI_API_KEY", "")

# Always fresh assistant instance with active key
st.session_state.assistant = AgenticAssistant(api_key=st.session_state.api_key)

if "active_file" not in st.session_state:
    st.session_state.active_file = None

if "active_df" not in st.session_state:
    st.session_state.active_df = None

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

if "dismissed_birthday_alerts" not in st.session_state:
    st.session_state.dismissed_birthday_alerts = set()

def on_api_key_update():
    entered_key = st.session_state.api_key_input.strip()
    st.session_state.api_key = entered_key
    st.session_state.assistant.update_api_key(entered_key)
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write(f"GEMINI_API_KEY={entered_key}\n")

# ---------------------------------------------------------------------------
# Refined Executive Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # Upper Section: Workspace & File Upload Controls
    with st.container(key="sidebar_top_box"):
        robot_side = f'<img src="data:image/png;base64,{robot_b64}" width="28" height="28" style="border-radius: 50%; border: 1.5px solid #7dd3fc; vertical-align: middle;">' if robot_b64 else '🤖'
        st.markdown(f'<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">{robot_side}<span style="font-size: 1.15rem; font-weight: 700; color: #0369a1;">Agent Rag</span></div>', unsafe_allow_html=True)
        
        # New Chat
        if st.button("New Chat", key="new_chat_btn", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.write("")

        # Clean Document Uploader
        st.markdown("###### Attach Document")
        uploaded_file = st.file_uploader(
            "PDF, CSV, Excel, Images",
            type=["pdf", "xlsx", "csv", "png", "jpg", "jpeg", "mp3", "wav", "mp4"],
            key=f"uploader_{st.session_state.file_uploader_key}",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            file_ext = uploaded_file.name.split(".")[-1].lower()
            cat = "image" if file_ext in ["png", "jpg", "jpeg"] else \
                  "audio" if file_ext in ["mp3", "wav"] else \
                  "video" if file_ext in ["mp4", "mov"] else \
                  "excel" if file_ext in ["xlsx", "csv"] else \
                  "pdf" if file_ext == "pdf" else "text"

            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.active_file = {
                "path": temp_path,
                "type": cat,
                "name": uploaded_file.name,
                "size_mb": round(uploaded_file.size / (1024 * 1024), 2)
            }

            if cat == "excel":
                try:
                    st.session_state.active_df = pd.read_csv(temp_path) if file_ext == "csv" else pd.read_excel(temp_path)
                except Exception:
                    pass

        if st.session_state.active_file:
            f_info = st.session_state.active_file
            st.caption(f"Active: **{f_info['name']}** ({f_info['size_mb']} MB)")
            if st.button("Detach Document", use_container_width=True):
                st.session_state.active_file = None
                st.session_state.active_df = None
                st.session_state.file_uploader_key += 1
                st.rerun()

    # Bottom Section: Firmly Pinned to the End Line of the Sidebar
    with st.container(key="sidebar_bottom_box"):
        st.divider()

        with st.expander("Milestones & Calendar"):
            from core.festival_calendar import get_festivals_for_window
            all_festivals = get_festivals_for_window(days_window=30)
            if all_festivals:
                for fest in all_festivals[:4]:
                    d = fest['days_remaining']
                    d_str = "Today! 🪔" if d == 0 else f"in {d} days"
                    st.write(f"• **{fest['name']}** ({d_str})")
            
            all_contacts = get_contact_events("usr_default", days_window=30)
            if all_contacts:
                for c in all_contacts[:3]:
                    d = c['days_remaining']
                    d_str = "Today! 🎂" if d == 0 else f"in {d} days"
                    st.write(f"• **{c['contact_name']}** ({c['event_type']} {d_str})")

        with st.expander("Settings"):
            api_key_input = st.text_input(
                "API Key",
                type="password",
                value=st.session_state.api_key,
                key="api_key_input",
                on_change=on_api_key_update,
                label_visibility="collapsed"
            )
            if st.session_state.api_key:
                st.caption("Active · High Quota Multi-Model Pool")

# ---------------------------------------------------------------------------
# Main Canvas
# ---------------------------------------------------------------------------
briefing = get_proactive_briefing("usr_default")
time_info = briefing["time_info"]
user_name = briefing["user_name"]
active_festival = briefing["active_festival"]
contact_events = briefing["contact_events"]

preset_action = None

# If chat is empty, render the classy Claude/Gemini style greeting
if len(st.session_state.messages) == 0:
    robot_tag = f'<img src="data:image/png;base64,{robot_b64}" width="48" height="48" style="border-radius: 50%; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.22); border: 2px solid #7dd3fc;">' if robot_b64 else '<span>🤖</span>'
    st.markdown(f"""
    <div class="hero-wrapper">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            {robot_tag}
            <div class="hero-badge" style="margin-bottom: 0;">AGENT RAG</div>
        </div>
        <div class="hero-title">{time_info['greeting']}, {user_name}</div>
        <div class="hero-subtitle">How can I assist you with your research, documents, or intelligence workflows today?</div>
    </div>
    """, unsafe_allow_html=True)

    # Compact Side-by-Side: Festive Indicator Beside Birthday Reminder
    active_bdays = [
        ev for ev in (contact_events or [])
        if f"{ev.get('contact_name')}_{ev.get('event_type')}_{ev.get('event_month')}_{ev.get('event_day')}" not in st.session_state.dismissed_birthday_alerts
    ]
    top_bday = active_bdays[0] if active_bdays else None

    if active_festival or top_bday:
        col_fest, col_bday = st.columns([1, 1], vertical_alignment="center")

        # Left: Festival Pill
        if active_festival:
            with col_fest:
                f_name = active_festival.get('name') or active_festival.get('festival_name')
                d = active_festival['days_remaining']
                d_str = "Today! 🪔" if d == 0 else f"in {d} days ({active_festival.get('event_date', '')})"
                st.markdown(f"""
                <div class="festival-pill">
                    <span>🪔</span>
                    <span><b>{f_name}</b> is {d_str}</span>
                </div>
                """, unsafe_allow_html=True)

        # Right (Beside Festive Pill): Matching Birthday Pill & Actions
        if top_bday:
            with col_bday:
                ev = top_bday
                ev_id = f"{ev.get('contact_name')}_{ev.get('event_type')}_{ev.get('event_month')}_{ev.get('event_day')}"
                d = ev['days_remaining']
                when_str = "Today" if d == 0 else f"in {d}d"
                
                b_pill, b_btn, b_x = st.columns([2.5, 0.9, 0.45], vertical_alignment="center")
                with b_pill:
                    st.markdown(f"""
                    <div class="birthday-pill">
                        <span>🎂</span>
                        <span><b>{ev['contact_name']}</b> · {when_str}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with b_btn:
                    if st.button("Wish", key=f"wish_{ev_id}", use_container_width=True):
                        preset_action = f"Generate 3 warm and creative {ev['event_type']} wishes for my {ev['relationship']} {ev['contact_name']}."
                with b_x:
                    if st.button("✕", key=f"close_{ev_id}", help="Dismiss", use_container_width=True):
                        st.session_state.dismissed_birthday_alerts.add(ev_id)
                        st.rerun()

    st.write("")

    # Refined Starter Chips (Clean, text-only, no clunky emoji buttons)
    st.caption("Suggested Prompts")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Diwali wishes for colleagues", key="chip_fest", use_container_width=True):
            preset_action = "Draft 3 professional and warm Diwali wishes for my colleagues."
    with c2:
        if st.button("Upcoming birthdays this week", key="chip_bday", use_container_width=True):
            preset_action = "Who has an upcoming birthday this week and what should I send them?"
    with c3:
        if st.button("Explain quantum computing simply", key="chip_ai", use_container_width=True):
            preset_action = "Explain quantum computing in simple terms with a real-world analogy."

# Active Document Toolbar (Refined, elegant pills)
if st.session_state.active_file:
    f_info = st.session_state.active_file
    st.markdown(f"""
    <div class="document-chip">
        <span>📄</span>
        <span><b>{f_info['name']}</b> ({f_info['size_mb']} MB)</span>
    </div>
    """, unsafe_allow_html=True)

    doc_c1, doc_c2, doc_c3, doc_c4 = st.columns(4)
    if doc_c1.button("Summarize", use_container_width=True):
        preset_action = f"Provide a comprehensive summary and key takeaways of '{f_info['name']}'."
    if doc_c2.button("Exam Questions & Answers", use_container_width=True):
        preset_action = f"Generate 5 important questions and detailed answers based on the contents of '{f_info['name']}'."
    if doc_c3.button("Key Concepts", use_container_width=True):
        preset_action = f"Extract the core topics, definitions, and concepts from '{f_info['name']}'."
    if doc_c4.button("Revision Notes", use_container_width=True):
        preset_action = f"Create concise study and revision notes based on '{f_info['name']}'."

# ---------------------------------------------------------------------------
# Chat Feed
# ---------------------------------------------------------------------------
asst_avatar = ROBOT_AVATAR_PATH if os.path.exists(ROBOT_AVATAR_PATH) else "🤖"

for msg in st.session_state.messages:
    is_user = msg["role"] == "user"
    avatar = "👤" if is_user else asst_avatar
    
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        
        # Subtle, professional citation (no loud debug badges)
        if not is_user and msg.get("route") == "MULTIMODAL_RAG":
            st.markdown("""
            <div class="source-footnote">
                <span>📄</span> Referenced from active document
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Integrated Bottom Chat Input (With Built-in Mic on Left Side)
# ---------------------------------------------------------------------------
if st.session_state.active_file:
    f_name = st.session_state.active_file['name']
    short_name = f_name if len(f_name) <= 26 else f_name[:24] + "..."
    chat_placeholder = f"Ask about '{short_name}' or speak with mic..."
else:
    chat_placeholder = "Message Agent Rag or speak using the mic..."

chat_val = st.chat_input(chat_placeholder, accept_audio=True)

user_query = None
is_voice = False

if chat_val:
    if isinstance(chat_val, str):
        user_query = chat_val.strip()
    elif hasattr(chat_val, "get"):
        audio_file = chat_val.get("audio")
        text_val = chat_val.get("text")
        if audio_file is not None:
            raw_audio_bytes = audio_file.read()
            with st.spinner("🎧 Transcribing your voice..."):
                transcribed_text = st.session_state.assistant.gemini.transcribe_audio_bytes(raw_audio_bytes)
                if transcribed_text:
                    user_query = transcribed_text
                    is_voice = True
                else:
                    st.warning("Could not detect clear speech. Please speak closer to the microphone and try again!")
        elif text_val:
            user_query = text_val.strip()
elif preset_action:
    user_query = preset_action.strip()

if user_query:
    # 1. User Message (tag voice-spoken prompts with a clean indicator)
    user_display = f"🎙️ {user_query}" if is_voice else user_query

    st.session_state.messages.append({"role": "user", "content": user_display})
    with st.chat_message("user", avatar="👤"):
        if is_voice:
            st.markdown('<div class="voice-badge"><span>🎙️</span> Spoken via Voice</div>', unsafe_allow_html=True)
        st.markdown(user_query)

    # 2. Assistant State Preparation
    f_info = st.session_state.active_file or {}
    state: AgentState = {
        "user_id": "usr_default",
        "user_name": user_name,
        "query": user_query,
        "attached_file_path": f_info.get("path"),
        "attached_file_type": f_info.get("type"),
        "active_dataframe": st.session_state.active_df,
        "route": None,
        "response": None,
        "mode": "auto"
    }

    # 3. Execution
    with st.chat_message("assistant", avatar=asst_avatar):
        with st.spinner("Generating response..."):
            result = st.session_state.assistant.process_turn(state)
            route = result.get("route", "GENERAL_CHAT")
            response = result.get("response", "No response generated.")

            st.markdown(response)

            if route == "MULTIMODAL_RAG":
                st.markdown("""
                <div class="source-footnote">
                    <span>📄</span> Referenced from active document
                </div>
                """, unsafe_allow_html=True)

    # 4. Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "route": route
    })
