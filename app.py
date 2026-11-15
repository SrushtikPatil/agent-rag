"""
Agent RAG - Commercial Multimodal Cognitive Co-Pilot & Document Intelligence Platform
Architecture: ChatGPT (conversational) + Notion (document management) + Linear (navigation & layout) + Modern AI SaaS.
"""

import os
import tempfile
import importlib
import json
import base64
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import streamlit as st

# Environment setup
ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(ENV_FILE)

# Assets & Robot Avatar Setup
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
from database.models import (
    get_user_profile,
    add_contact_event,
    get_contact_events,
    get_festivals,
    list_documents,
    get_document,
    delete_document as db_delete_document,
    get_document_chunks,
    get_system_stats,
    get_recent_chat_queries
)
from core.proactive_engine import get_proactive_briefing
from core.agentic_router import AgenticAssistant, AgentState

# Page Configuration - Commercial AI SaaS
st.set_page_config(
    page_title="Agent RAG | Cognitive Co-Pilot",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database on first run
init_db()

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_view" not in st.session_state:
    st.session_state.active_view = "dashboard" # "dashboard", "chat", "documents", "search", "analytics"

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "active_file" not in st.session_state:
    st.session_state.active_file = None

if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None

if "search_target_doc" not in st.session_state:
    st.session_state.search_target_doc = "ALL"

if "active_link" not in st.session_state:
    st.session_state.active_link = None

if "active_df" not in st.session_state:
    st.session_state.active_df = None

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

if "dismissed_birthday_alerts" not in st.session_state:
    st.session_state.dismissed_birthday_alerts = set()

env_key = ""
if os.path.exists(ENV_FILE):
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("GEMINI_API_KEY="):
                env_key = line.split("=", 1)[1].strip()

if "api_key" not in st.session_state or not st.session_state.api_key:
    st.session_state.api_key = env_key or os.getenv("GEMINI_API_KEY", "")

# Initialize assistant with API key
st.session_state.assistant = AgenticAssistant(api_key=st.session_state.api_key)

def on_api_key_update():
    entered_key = st.session_state.api_key_input.strip()
    st.session_state.api_key = entered_key
    st.session_state.assistant.update_api_key(entered_key)
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write(f"GEMINI_API_KEY={entered_key}\n")

# ---------------------------------------------------------------------------
# High-End Commercial SaaS Styling (Linear + Nebula Dark Cosmic Design)
# ---------------------------------------------------------------------------
is_dark = st.session_state.dark_mode

bg_main = (
    "radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.15) 0%, transparent 60%), "
    "radial-gradient(circle at 85% 25%, rgba(168, 85, 247, 0.10) 0%, transparent 50%), "
    "linear-gradient(180deg, #0b0d17 0%, #0e1122 45%, #080912 100%)"
) if is_dark else (
    "linear-gradient(180deg, #f0f7ff 0%, #f8fafc 40%, #ffffff 100%)"
)

text_primary = "#f8fafc" if is_dark else "#0f172a"
text_secondary = "#94a3b8" if is_dark else "#475569"
card_bg = "rgba(19, 22, 38, 0.75)" if is_dark else "#ffffff"
card_border = "rgba(255, 255, 255, 0.08)" if is_dark else "#e0f2fe"
card_hover_border = "rgba(168, 85, 247, 0.45)" if is_dark else "#7dd3fc"
sidebar_bg = "#0d0f1a" if is_dark else "#f8fafc"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: {text_primary};
    }}

    .stApp {{
        background: {bg_main} !important;
        background-attachment: fixed !important;
    }}
    
    /* Header & Toolbar: Transparent and clean */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        z-index: 99999 !important;
    }}
    div[data-testid="stToolbar"] {{
        background: transparent !important;
        display: flex !important;
        visibility: visible !important;
    }}
    div[data-testid="stDecoration"],
    .stAppDeployButton,
    #MainMenu {{
        display: none !important;
    }}

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg} !important;
        border-right: 1px solid {card_border} !important;
    }}
    section[data-testid="stSidebar"] div.block-container {{
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
    }}

    /* Expand sidebar button */
    button[data-testid="stExpandSidebarButton"] {{
        background: {card_bg} !important;
        border: 1px solid {card_border} !important;
        color: #a855f7 !important;
        border-radius: 10px !important;
    }}

    /* Container constraints */
    div[data-testid="stMainBlockContainer"],
    .block-container:not([data-testid="stBottomBlockContainer"]) {{
        padding-top: 1.5rem !important;
        padding-bottom: 6rem !important;
        max-width: 980px !important;
    }}

    /* Bottom Chat Container */
    div[data-testid="stBottom"] {{
        background: transparent !important;
        z-index: 90 !important;
    }}
    div[data-testid="stBottomBlockContainer"] {{
        max-width: 980px !important;
        padding-bottom: 1.5rem !important;
    }}

    /* Animated 3D Nebula Orb */
    .nebula-orb-container {{
        text-align: center;
        padding: 24px 0 16px 0;
    }}
    .nebula-orb {{
        width: 76px;
        height: 76px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #f472b6 0%, #a855f7 35%, #6366f1 70%, #0b0d17 100%);
        box-shadow: 0 0 45px rgba(168, 85, 247, 0.6), inset 0 0 16px rgba(255, 255, 255, 0.4);
        margin: 0 auto 16px auto;
        animation: orbPulse 4s ease-in-out infinite alternate;
    }}
    @keyframes orbPulse {{
        0% {{ transform: translateY(0px) scale(1); filter: drop-shadow(0 0 25px rgba(168, 85, 247, 0.5)); }}
        100% {{ transform: translateY(-6px) scale(1.04); filter: drop-shadow(0 0 40px rgba(124, 58, 237, 0.8)); }}
    }}

    .hero-title-nebula {{
        font-size: 2.1rem;
        font-weight: 800;
        color: {text_primary};
        letter-spacing: -0.5px;
        margin-bottom: 6px;
        text-align: center;
    }}
    .hero-subtitle-nebula {{
        font-size: 1.02rem;
        color: {text_secondary};
        text-align: center;
        max-width: 620px;
        margin: 0 auto 28px auto;
        line-height: 1.5;
    }}

    /* Commercial SaaS Cards */
    .saas-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 16px;
        padding: 18px 20px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(16px);
        margin-bottom: 12px;
    }}
    .saas-card:hover {{
        border-color: {card_hover_border};
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.14);
    }}

    /* Stat KPI Card */
    .stat-box {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        transition: all 0.2s ease;
    }}
    .stat-box:hover {{
        border-color: {card_hover_border};
        transform: translateY(-2px);
    }}
    .stat-num {{
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.1;
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .stat-label {{
        font-size: 0.76rem;
        font-weight: 700;
        color: {text_secondary};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }}

    /* Prompt Starter Cards (Matching Uploaded Image) */
    .prompt-box {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 16px;
        padding: 18px;
        min-height: 125px;
        transition: all 0.25s ease;
        cursor: pointer;
    }}
    .prompt-box:hover {{
        border-color: #a855f7;
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.15);
        transform: translateY(-2px);
    }}
    .prompt-tag {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.88rem;
        font-weight: 700;
        color: #c084fc;
        margin-bottom: 8px;
    }}
    .prompt-desc {{
        font-size: 0.82rem;
        color: {text_secondary};
        line-height: 1.4;
    }}

    /* Status Badges */
    .badge-ready {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 9999px;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }}
    .badge-progress {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(99, 102, 241, 0.12);
        color: #818cf8;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 9999px;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }}
    .badge-failed {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 9999px;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }}

    /* Navigation Pill in Sidebar */
    .nav-pill-active {{
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(168, 85, 247, 0.25) 100%);
        border: 1px solid rgba(168, 85, 247, 0.4);
        color: #ffffff !important;
        font-weight: 700;
        border-radius: 12px;
        padding: 10px 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }}
    .nav-pill-inactive {{
        background: transparent;
        color: {text_secondary};
        font-weight: 600;
        border-radius: 12px;
        padding: 10px 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
        transition: all 0.2s ease;
    }}
    .nav-pill-inactive:hover {{
        background: rgba(255, 255, 255, 0.04);
        color: {text_primary};
    }}

    /* Sidebar New Chat Button (Purple Gradient matching Image) */
    .new-chat-btn {{
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 20px !important;
        padding: 10px 20px !important;
        border: none !important;
        box-shadow: 0 4px 18px rgba(168, 85, 247, 0.35) !important;
        transition: all 0.2s ease !important;
    }}
    .new-chat-btn:hover {{
        transform: translateY(-1px) scale(1.02) !important;
        box-shadow: 0 6px 24px rgba(168, 85, 247, 0.5) !important;
    }}

    /* Chat Input Bar - Dark Frosted Glass with Real Blue/Violet Mic */
    div[data-testid="stChatInput"] {{
        position: relative !important;
        border-radius: 28px !important;
        border: 1px solid {card_border} !important;
        background: {card_bg} !important;
        min-height: 58px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35) !important;
        backdrop-filter: blur(20px) !important;
        transition: all 0.2s ease !important;
    }}
    div[data-testid="stChatInput"]:focus-within {{
        border-color: #a855f7 !important;
        box-shadow: 0 8px 35px rgba(168, 85, 247, 0.25) !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        min-height: 48px !important;
        font-size: 1.02rem !important;
        line-height: 1.5 !important;
        padding-top: 14px !important;
        padding-bottom: 14px !important;
        padding-left: 54px !important;
        padding-right: 54px !important;
        caret-color: #a855f7 !important;
        color: {text_primary} !important;
    }}
    div[data-testid="stChatInput"] textarea::placeholder {{
        color: {text_secondary} !important;
    }}

    /* Microphone Button in Chat Input */
    div[data-testid="stChatInput"] button[data-testid="stChatInputMicButton"] {{
        position: absolute !important;
        left: 14px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        z-index: 20 !important;
        background: transparent !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='22' height='22'%3E%3Cpath d='M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z' fill='%23a855f7' stroke='%23a855f7' stroke-width='1.2'/%3E%3Cpath d='M19 10v1a7 7 0 0 1-14 0v-1' fill='none' stroke='%23a855f7' stroke-width='2' stroke-linecap='round'/%3E%3Cline x1='12' y1='18' x2='12' y2='22' stroke='%23a855f7' stroke-width='2' stroke-linecap='round'/%3E%3Cline x1='8' y1='22' x2='16' y2='22' stroke='%23a855f7' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E") !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        background-size: 22px 22px !important;
        border: none !important;
        width: 36px !important;
        height: 36px !important;
        cursor: pointer !important;
        transition: transform 0.15s ease !important;
    }}
    div[data-testid="stChatInput"] button[data-testid="stChatInputMicButton"]:hover {{
        transform: translateY(-50%) scale(1.15) !important;
    }}
    div[data-testid="stChatInput"] button[data-testid="stChatInputMicButton"] > * {{
        display: none !important;
    }}

    /* Submit Button (Purple Gradient Send Arrow) */
    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"] {{
        position: absolute !important;
        right: 14px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        z-index: 20 !important;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        border-radius: 50% !important;
        width: 34px !important;
        height: 34px !important;
        border: none !important;
        color: #ffffff !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 10px rgba(168, 85, 247, 0.4) !important;
        transition: transform 0.15s ease !important;
    }}
    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"]:hover {{
        transform: translateY(-50%) scale(1.1) !important;
    }}
    div[data-testid="stChatInput"] button[data-testid="stChatInputStopButton"] {{
        position: absolute !important;
        right: 14px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        z-index: 20 !important;
        background: transparent !important;
        color: #a855f7 !important;
    }}

    /* Code & Badges */
    code {{
        color: #c084fc !important;
        background-color: rgba(168, 85, 247, 0.1) !important;
        padding: 2px 6px !important;
        border-radius: 6px !important;
        border: 1px solid rgba(168, 85, 247, 0.2) !important;
    }}

    /* Custom Scrollbars */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-thumb {{
        background: rgba(255, 255, 255, 0.12);
        border-radius: 8px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: rgba(168, 85, 247, 0.4);
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Navigation (Linear / ChatGPT Commercial Layout)
# ---------------------------------------------------------------------------
with st.sidebar:
    # 1. Branding Header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 18px; padding: 4px 6px;">
        <div style="width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(168, 85, 247, 0.45); font-size: 1.25rem;">
            🌌
        </div>
        <div>
            <div style="font-weight: 800; font-size: 1.15rem; letter-spacing: -0.3px; line-height: 1.1;">Agent RAG</div>
            <div style="font-size: 0.72rem; color: #a855f7; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px;">Cognitive Co-Pilot</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Primary "+ New Chat" Button
    if st.button("＋ New Conversation", key="sb_new_chat_btn", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.active_doc_id = None
        st.session_state.active_file = None
        st.session_state.active_link = None
        st.session_state.active_view = "chat"
        st.rerun()

    st.write("")

    # 3. Main Navigation Views
    st.caption("PLATFORM")
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("🏠 Overview", use_container_width=True, key="nav_dash_btn"):
            st.session_state.active_view = "dashboard"
            st.rerun()
    with nav_col2:
        if st.button("💬 Co-Pilot", use_container_width=True, key="nav_chat_btn"):
            st.session_state.active_view = "chat"
            st.rerun()

    nav_col3, nav_col4 = st.columns(2)
    with nav_col3:
        if st.button("📑 Documents", use_container_width=True, key="nav_docs_btn"):
            st.session_state.active_view = "documents"
            st.rerun()
    with nav_col4:
        if st.button("🔍 RAG Search", use_container_width=True, key="nav_search_btn"):
            st.session_state.active_view = "search"
            st.rerun()

    if st.button("📊 Analytics & Insights", use_container_width=True, key="nav_analytics_btn"):
        st.session_state.active_view = "analytics"
        st.rerun()

    st.write("")

    # 4. Recent Chat Queries (Real Data from SQLite)
    st.caption("🕒 RECENT CONVERSATIONS")
    recent_queries = get_recent_chat_queries("usr_default", limit=5)
    if recent_queries:
        for idx_q, q_text in enumerate(recent_queries):
            short_q = q_text if len(q_text) <= 28 else q_text[:26] + "..."
            if st.button(f"💬 {short_q}", key=f"rec_q_{idx_q}", use_container_width=True):
                st.session_state.active_view = "chat"
                st.session_state.messages.append({"role": "user", "content": q_text})
                # Process turn directly
                state: AgentState = {
                    "user_id": "usr_default",
                    "user_name": "Aarav Sharma",
                    "query": q_text,
                    "attached_file_path": None,
                    "attached_file_type": None,
                    "attached_link": None,
                    "active_dataframe": None,
                    "route": None,
                    "response": None,
                    "mode": "auto",
                    "active_doc_id": st.session_state.active_doc_id
                }
                res = st.session_state.assistant.process_turn(state)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res.get("response", "No response."),
                    "route": res.get("route")
                })
                st.rerun()
    else:
        st.caption("No previous sessions recorded yet.")

    st.write("")

    # 5. Quick Ingestion Drawer in Sidebar
    with st.expander("⚡ Quick Ingest / Attach", expanded=False):
        sb_url = st.text_input("Ingest Link", placeholder="https://...", key="sb_quick_url", label_visibility="collapsed")
        if st.button("Ingest Link", key="sb_btn_ingest_url", use_container_width=True):
            if sb_url.strip():
                with st.spinner("Processing link..."):
                    u_res = st.session_state.assistant.rag_engine.ingest_url(
                        sb_url.strip(),
                        gemini_service=st.session_state.assistant.gemini
                    )
                    if u_res.get("status") == "Ready":
                        st.success(f"Indexed '{u_res.get('title')[:20]}'!")
                        st.rerun()
                    else:
                        st.error(u_res.get("error"))

        sb_up = st.file_uploader(
            "Upload File",
            type=["pdf", "docx", "pptx", "txt", "csv", "xlsx", "png", "jpg", "mp3", "wav", "mp4"],
            key=f"sb_quick_up_{st.session_state.file_uploader_key}",
            label_visibility="collapsed"
        )
        if sb_up is not None:
            temp_p = os.path.join(tempfile.gettempdir(), sb_up.name)
            with open(temp_p, "wb") as f_out:
                f_out.write(sb_up.getbuffer())
            with st.spinner("Indexing file into ChromaDB..."):
                f_res = st.session_state.assistant.rag_engine.ingest_file(
                    temp_p, sb_up.name, gemini_service=st.session_state.assistant.gemini
                )
                if f_res.get("status") == "Ready":
                    st.success(f"Indexed '{sb_up.name}'!")
                    st.session_state.file_uploader_key += 1
                    st.rerun()

    # 6. Enterprise Pro Card (Matching Screenshot)
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.12) 100%); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 14px; padding: 14px; margin-top: 14px;">
        <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 0.85rem; color: #c084fc;">
            <span>💎</span> Agent RAG Pro
        </div>
        <div style="font-size: 0.74rem; color: #94a3b8; margin: 4px 0 10px 0; line-height: 1.4;">
            Hybrid ChromaDB vector engine, multi-page OCR, and video intelligence active.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 7. User Profile at Bottom
    st.divider()
    user_prof = get_user_profile("usr_default")
    user_name = user_prof.get("full_name", "Aarav Sharma")
    user_email = user_prof.get("email", "aarav@example.com")
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 4px 2px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 34px; height: 34px; border-radius: 50%; background: #4f46e5; color: white; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem;">
                {user_name[0]}
            </div>
            <div>
                <div style="font-weight: 700; font-size: 0.84rem; color: {text_primary};">{user_name}</div>
                <div style="font-size: 0.70rem; color: {text_secondary};">{user_email}</div>
            </div>
        </div>
        <span style="font-size: 0.75rem; color: #34d399;">●</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Settings & API Key"):
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key,
            key="api_key_input",
            on_change=on_api_key_update
        )
        if st.session_state.api_key:
            st.caption("🟢 Active Google Gemini Engine")
        else:
            st.caption("💡 Operating in Autonomous Local Mode")

# ---------------------------------------------------------------------------
# Main Stage Router
# ---------------------------------------------------------------------------
view = st.session_state.active_view
stats = get_system_stats()
preset_action = None

# ===========================================================================
# VIEW 1: DASHBOARD OVERVIEW (Home)
# ===========================================================================
if view == "dashboard":
    # 1. 3D Nebula Orb & Hero Greeting
    st.markdown(f"""
    <div class="nebula-orb-container">
        <div class="nebula-orb"></div>
        <div class="hero-title-nebula">Welcome to your cognitive co-pilot</div>
        <div class="hero-subtitle-nebula">
            I am Agent RAG. You can ask me to analyze multi-page documents, synthesize tabular data, query video timestamps, or search your entire multimodal library.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Starter Prompt Cards (Exact match to uploaded image)
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.markdown("""
        <div class="prompt-box">
            <div class="prompt-tag"><span>&lt;/&gt;</span> Deep Document Analysis</div>
            <div class="prompt-desc">Explain complex architecture, extract security frameworks, and audit citations from uploaded PDFs.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze Documents", key="btn_p1", use_container_width=True):
            preset_action = "Explain the key architecture, security models, and findings from my uploaded documents."
            st.session_state.active_view = "chat"
            st.rerun()

    with p_col2:
        st.markdown("""
        <div class="prompt-box">
            <div class="prompt-tag"><span>⏱</span> Timeline & Transcripts</div>
            <div class="prompt-desc">Query 20-minute video lectures or recorded audio and retrieve answers with exact [MM:SS] timestamps.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Timelines", key="btn_p2", use_container_width=True):
            preset_action = "Summarize the key events, speakers, and topics with exact timestamps from my media."
            st.session_state.active_view = "chat"
            st.rerun()

    with p_col3:
        st.markdown("""
        <div class="prompt-box">
            <div class="prompt-tag"><span>💡</span> Cross-Source Synthesis</div>
            <div class="prompt-desc">Cross-reference Web URLs, Office presentations, and research papers to generate executive summaries.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Synthesize Knowledge", key="btn_p3", use_container_width=True):
            preset_action = "Synthesize key insights across all my uploaded knowledge sources into a strategic briefing."
            st.session_state.active_view = "chat"
            st.rerun()

    st.write("")

    # 3. Real Backend Statistics Cards
    st.markdown("#### 📈 Real-Time Knowledge Repository Statistics")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    with s1:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{stats["total_documents"]}</div><div class="stat-label">Total Documents</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-box"><div class="stat-num" style="background: linear-gradient(135deg, #10b981 0%, #34d399 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{stats["ready_documents"]}</div><div class="stat-label">Ready & Searchable</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-box"><div class="stat-num" style="background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{stats["processing_documents"]}</div><div class="stat-label">In Processing</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="stat-box"><div class="stat-num" style="background: linear-gradient(135deg, #ef4444 0%, #f87171 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{stats["failed_documents"]}</div><div class="stat-label">Failed</div></div>', unsafe_allow_html=True)
    with s5:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{stats["total_chunks"]}</div><div class="stat-label">Indexed Chunks</div></div>', unsafe_allow_html=True)
    with s6:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{stats["total_queries"]}</div><div class="stat-label">Total Queries</div></div>', unsafe_allow_html=True)

    st.write("")

    # 4. Quick Action Control Cards
    st.markdown("#### ⚡ Quick Actions")
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("📤 Upload Documents", use_container_width=True):
            st.session_state.active_view = "documents"
            st.rerun()
    with qa2:
        if st.button("🌐 Ingest Web URL", use_container_width=True):
            st.session_state.active_view = "documents"
            st.rerun()
    with qa3:
        if st.button("🔍 Search Knowledge Base", use_container_width=True):
            st.session_state.active_view = "search"
            st.rerun()
    with qa4:
        if st.button("💬 Start New Chat", use_container_width=True):
            st.session_state.active_view = "chat"
            st.rerun()

# ===========================================================================
# VIEW 2: AI CHAT & CO-PILOT
# ===========================================================================
elif view == "chat":
    # Header bar
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 12px; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="font-weight: 800; font-size: 1.1rem; color: #f8fafc;">Cognitive Co-Pilot</div>
            <span class="badge-ready">● Model Active</span>
        </div>
        <div style="font-size: 0.8rem; color: #94a3b8;">
            Hybrid ChromaDB Retrieval • Anti-Hallucination Guardrail Active
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active targeted document banner
    if st.session_state.get("active_doc_id"):
        target_rec = get_document(st.session_state.active_doc_id)
        if target_rec:
            col_target_info, col_target_clear = st.columns([3.8, 1.2], vertical_alignment="center")
            with col_target_info:
                f_sz = round((target_rec.get("file_size_bytes") or 0) / (1024 * 1024), 2)
                st.markdown(
                    f'<div style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 12px; padding: 10px 16px; color: #c084fc; font-weight: 600; font-size: 0.88rem;">'
                    f'🎯 Targeted Document: <b>{target_rec["filename"]}</b> ({f_sz} MB • {target_rec.get("page_count", 1)} pages)</div>',
                    unsafe_allow_html=True
                )
            with col_target_clear:
                if st.button("Detach Target", key="detach_active_target_btn", use_container_width=True):
                    st.session_state.active_doc_id = None
                    st.session_state.active_file = None
                    st.rerun()

            # Active doc quick actions
            d_c1, d_c2, d_c3, d_c4 = st.columns(4)
            if d_c1.button("Summarize", key="act_sum_doc", use_container_width=True):
                preset_action = f"Provide a comprehensive summary and key takeaways of '{target_rec['filename']}'."
            if d_c2.button("Exam Q&A", key="act_qa_doc", use_container_width=True):
                preset_action = f"Generate 5 important questions and detailed answers based on '{target_rec['filename']}'."
            if d_c3.button("Key Concepts", key="act_con_doc", use_container_width=True):
                preset_action = f"Extract the core topics, definitions, and concepts from '{target_rec['filename']}'."
            if d_c4.button("Revision Notes", key="act_not_doc", use_container_width=True):
                preset_action = f"Create concise study and revision notes based on '{target_rec['filename']}'."

    # Chat Feed
    asst_avatar = ROBOT_AVATAR_PATH if os.path.exists(ROBOT_AVATAR_PATH) else "🌌"

    if len(st.session_state.messages) == 0:
        st.markdown(f"""
        <div style="text-align: center; padding: 36px 0 20px 0;">
            <div style="font-size: 1.5rem; font-weight: 800; color: #f8fafc; margin-bottom: 6px;">How can I assist you today?</div>
            <div style="font-size: 0.9rem; color: #94a3b8; max-width: 500px; margin: 0 auto 20px auto;">
                Ask any question across your uploaded documents, media transcripts, or web knowledge base.
            </div>
        </div>
        """, unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if st.button("What is the CIA Triad?", use_container_width=True):
                preset_action = "What is the CIA Triad in database security and where is it explained?"
        with sc2:
            if st.button("What are Access Control Models?", use_container_width=True):
                preset_action = "Explain Discretionary and Role-Based Access Control models with citations."
        with sc3:
            if st.button("Audit Document Citations", use_container_width=True):
                preset_action = "Show me the top security recommendations across all indexed documents."

    for msg in st.session_state.messages:
        is_user = msg["role"] == "user"
        avatar = "👤" if is_user else asst_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if not is_user and msg.get("route") == "MULTIMODAL_RAG":
                st.markdown("""
                <div style="font-size: 0.76rem; color: #a855f7; margin-top: 8px; font-weight: 600;">
                    ✓ Grounded hybrid retrieval with verified citations
                </div>
                """, unsafe_allow_html=True)

# ===========================================================================
# VIEW 3: DOCUMENT MANAGEMENT DASHBOARD
# ===========================================================================
elif view == "documents":
    st.markdown("### 📑 Document Management Dashboard")
    st.caption("Upload, inspect, reprocess, and manage all your documents and multi-modal knowledge sources.")

    # High-level KPIs
    all_docs = list_documents()
    ready_docs = [d for d in all_docs if d.get("processing_status") == "Ready"]
    proc_docs = [d for d in all_docs if d.get("processing_status") not in ["Ready", "Failed"]]
    failed_docs = [d for d in all_docs if d.get("processing_status") == "Failed"]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{len(all_docs)}</div><div class="stat-label">Total Documents</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="stat-box"><div class="stat-num" style="background: linear-gradient(135deg, #10b981 0%, #34d399 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{len(ready_docs)}</div><div class="stat-label">Ready & Searchable</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="stat-box"><div class="stat-num" style="background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{len(proc_docs)}</div><div class="stat-label">In Processing</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="stat-box"><div class="stat-num" style="background: linear-gradient(135deg, #ef4444 0%, #f87171 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{len(failed_docs)}</div><div class="stat-label">Failed</div></div>', unsafe_allow_html=True)

    st.write("")

    # Multi-File & URL Ingestion Area
    with st.expander("📤 Upload New Documents or Ingest Web Links", expanded=(len(all_docs) == 0)):
        tab_up_file, tab_up_url = st.tabs(["📁 Multi-File Upload", "🌐 Ingest Web URL / YouTube"])
        with tab_up_file:
            uploaded_files = st.file_uploader(
                "Upload Multiple Documents (PDF, DOCX, DOC, PPTX, TXT, CSV, XLSX, PNG, JPG, JPEG, WEBP, MP4, MOV, AVI, MKV, MP3, WAV, M4A)",
                type=["pdf", "docx", "doc", "pptx", "txt", "csv", "xlsx", "xls", "png", "jpg", "jpeg", "webp", "mp4", "mov", "avi", "mkv", "mp3", "wav", "m4a"],
                accept_multiple_files=True,
                key=f"dash_multi_uploader_{st.session_state.file_uploader_key}"
            )
            if uploaded_files:
                if st.button(f"📥 Process & Index {len(uploaded_files)} File(s)", type="primary", use_container_width=True):
                    prog = st.progress(0, text="Starting batch ingestion...")
                    for idx_f, f in enumerate(uploaded_files):
                        prog.progress((idx_f) / len(uploaded_files), text=f"Processing {f.name}...")
                        temp_p = os.path.join(tempfile.gettempdir(), f.name)
                        with open(temp_p, "wb") as out_f:
                            out_f.write(f.getbuffer())
                        res = st.session_state.assistant.rag_engine.ingest_file(
                            src_file_path=temp_p,
                            filename=f.name,
                            gemini_service=st.session_state.assistant.gemini
                        )
                    prog.progress(1.0, text="Batch processing completed!")
                    st.success(f"Indexed {len(uploaded_files)} document(s) successfully!")
                    st.session_state.file_uploader_key += 1
                    st.rerun()

        with tab_up_url:
            col_u1, col_u2 = st.columns([3.5, 1])
            with col_u1:
                dash_url_val = st.text_input("Enter Web URL, YouTube link, or Wikipedia page", placeholder="https://...", key="dash_url_input", label_visibility="collapsed")
            with col_u2:
                if st.button("Ingest Link", use_container_width=True, key="dash_url_btn"):
                    if dash_url_val.strip():
                        with st.spinner("Extracting content and generating embeddings..."):
                            u_res = st.session_state.assistant.rag_engine.ingest_url(
                                dash_url_val.strip(),
                                gemini_service=st.session_state.assistant.gemini
                            )
                            if u_res.get("status") == "Ready":
                                st.success(f"Indexed '{u_res.get('title')}' successfully!")
                                st.rerun()
                            else:
                                st.error(f"Failed to ingest URL: {u_res.get('error')}")

    st.write("")

    # Document Repository List
    st.markdown("#### 📚 Uploaded Documents & Indexes")
    if not all_docs:
        st.info("No documents uploaded yet. Use the upload box above to add your first PDF, Office document, image, audio, video, or web link!")
    else:
        for doc in all_docs:
            d_id = doc["doc_id"]
            status = doc.get("processing_status", "Uploaded")
            f_type = doc.get("file_type", "doc").upper()
            size_mb = round((doc.get("file_size_bytes") or 0) / (1024 * 1024), 2)
            pages = doc.get("page_count", 0)
            chunks = doc.get("chunk_count", 0)
            upload_dt = doc.get("upload_timestamp", "")

            pill_class = "badge-ready" if status == "Ready" else ("badge-failed" if status == "Failed" else "badge-progress")
            status_emoji = "✅" if status == "Ready" else ("❌" if status == "Failed" else "⏳")

            with st.container():
                st.markdown(f"""
                <div class="saas-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="background: rgba(255, 255, 255, 0.08); padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.72rem; color: #c084fc;">{f_type}</span>
                            <span style="font-weight: 700; font-size: 1.05rem; color: #f8fafc;">{doc['filename']}</span>
                        </div>
                        <span class="{pill_class}">{status_emoji} {status}</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 10px;">
                        <span>📦 Size: <b>{size_mb} MB</b></span> &nbsp;•&nbsp;
                        <span>📄 Pages: <b>{pages}</b></span> &nbsp;•&nbsp;
                        <span>🧩 Chunks: <b>{chunks}</b></span> &nbsp;•&nbsp;
                        <span>📅 Uploaded: <b>{upload_dt}</b></span>
                    </div>
                    <div style="font-size: 0.78rem; color: #cbd5e1; background: rgba(255, 255, 255, 0.03); padding: 6px 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 10px;">
                        <span>Extraction: <b>{doc.get('extraction_status', 'Pending')}</b></span> &nbsp;|&nbsp;
                        <span>Embedding: <b>{doc.get('embedding_status', 'Pending')}</b></span> &nbsp;|&nbsp;
                        <span>Indexing: <b>{doc.get('indexing_status', 'Pending')}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if doc.get("error_message"):
                    st.error(f"Error: {doc['error_message']}")

                b_c1, b_c2, b_c3, b_c4 = st.columns([1.5, 1.2, 1.2, 1.2])
                with b_c1:
                    is_active = (st.session_state.active_doc_id == d_id)
                    btn_label = "🎯 Target in Chat" if not is_active else "⭐ Active Target"
                    if st.button(btn_label, key=f"target_{d_id}", use_container_width=True, disabled=is_active):
                        st.session_state.active_doc_id = d_id
                        st.session_state.active_view = "chat"
                        st.rerun()
                with b_c2:
                    if st.button("🔄 Reprocess", key=f"reproc_{d_id}", use_container_width=True):
                        with st.spinner(f"Reprocessing '{doc['filename']}'..."):
                            st.session_state.assistant.rag_engine.reprocess_document(
                                d_id, gemini_service=st.session_state.assistant.gemini
                            )
                            st.rerun()
                with b_c3:
                    if status == "Failed":
                        if st.button("🔁 Retry", key=f"retry_{d_id}", use_container_width=True):
                            with st.spinner(f"Retrying '{doc['filename']}'..."):
                                st.session_state.assistant.rag_engine.reprocess_document(
                                    d_id, gemini_service=st.session_state.assistant.gemini
                                )
                                st.rerun()
                    else:
                        st.write("")
                with b_c4:
                    if st.button("🗑️ Delete", key=f"del_{d_id}", use_container_width=True):
                        st.session_state.assistant.rag_engine.delete_document(d_id)
                        if st.session_state.active_doc_id == d_id:
                            st.session_state.active_doc_id = None
                        st.rerun()

                with st.expander(f"ℹ️ View Details & Indexed Chunks ({chunks})", expanded=False):
                    doc_chunks = get_document_chunks(d_id)
                    st.write(f"**Doc ID:** `{d_id}`")
                    st.write(f"**Storage Path:** `{doc.get('storage_path')}`")
                    if doc_chunks:
                        for idx_c, ch in enumerate(doc_chunks[:6], 1):
                            loc = f"Page {ch.get('page_number', 1)}" if not ch.get("timestamp_str") else f"Timestamp [{ch.get('timestamp_str')}]"
                            st.markdown(f"**Chunk #{idx_c}** ({loc}):")
                            st.markdown(f"> *{ch.get('content')[:250]}...*")
                    else:
                        st.caption("No chunks currently extracted.")

# ===========================================================================
# VIEW 4: RAG SEARCH EXPLORER
# ===========================================================================
elif view == "search":
    st.markdown("### 🔍 Multi-Modal RAG Search Explorer")
    st.caption("Execute hybrid semantic vector + lexical search across your entire repository with cosine similarity scoring.")

    ready_docs_list = [d for d in list_documents() if d.get("processing_status") == "Ready"]
    doc_options = ["All Documents (Global Library Search)"] + [f"{d['filename']} (ID: {d['doc_id']})" for d in ready_docs_list]

    col_scope, _ = st.columns([2.5, 1])
    with col_scope:
        selected_scope = st.selectbox("Search Target Scope:", doc_options, key="rag_search_scope")

    target_doc_id = None
    if selected_scope != "All Documents (Global Library Search)":
        target_doc_id = selected_scope.split("ID: ")[-1].rstrip(")")

    col_s_inp, col_s_btn = st.columns([4, 1.2], vertical_alignment="bottom")
    with col_s_inp:
        search_query_val = st.text_input(
            "Search Query",
            placeholder="🔍 Ask a question or search for concepts, tables, topics, transcripts...",
            key="rag_explorer_query",
            label_visibility="collapsed"
        )
    with col_s_btn:
        do_search = st.button("🔍 Search", type="primary", use_container_width=True, key="rag_explorer_btn")

    if (do_search or search_query_val) and search_query_val.strip():
        with st.spinner("Executing hybrid retrieval & grounded synthesis..."):
            rag_output = st.session_state.assistant.rag_engine.answer_rag_query(
                query=search_query_val.strip(),
                doc_id=target_doc_id,
                gemini_service=st.session_state.assistant.gemini
            )
            raw_matches = st.session_state.assistant.rag_engine.search(
                query=search_query_val.strip(),
                doc_id=target_doc_id,
                top_k=5
            )

        st.markdown("#### 💡 Grounded AI Answer")
        st.markdown(rag_output.get("answer", "No response generated."))

        st.write("")
        st.markdown("#### 📚 Retrieved Chunks & Exact Locations")
        if raw_matches:
            for idx_m, m in enumerate(raw_matches, 1):
                loc = f"Page {m.get('page_number', 1)}" if not m.get("timestamp_str") else f"Timestamp [{m.get('timestamp_str')}]"
                score_pct = int(max(0, min(100, (m.get('score', 0) + 1.0) / 2.0 * 100))) if m.get("score") is not None else 80
                with st.expander(f"Reference #{idx_m}: {m.get('filename')} ({loc}) — {score_pct}% Match", expanded=(idx_m==1)):
                    st.write(f"**Document:** `{m.get('filename')}` | **Type:** `{m.get('file_type')}` | **Location:** `{loc}`")
                    st.markdown(f"> {m.get('content')}")
        else:
            st.info("No matching chunks retrieved for this query.")

# ===========================================================================
# VIEW 5: ANALYTICS & INSIGHTS
# ===========================================================================
elif view == "analytics":
    st.markdown("### 📊 Analytics & System Intelligence")
    st.caption("Repository composition, chunk indexing distribution, and AI routing telemetry.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{stats["total_documents"]}</div><div class="stat-label">Total Documents Ingested</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{stats["total_chunks"]}</div><div class="stat-label">Vector Embeddings Indexed</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{stats["rag_queries"]}</div><div class="stat-label">Grounded RAG Queries</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("#### 📂 Repository Modality Distribution")
    all_d = list_documents()
    if all_d:
        types_count = {}
        for d in all_d:
            t = d.get("file_type", "other").upper()
            types_count[t] = types_count.get(t, 0) + 1
        df_types = pd.DataFrame(list(types_count.items()), columns=["Modality", "Count"])
        st.bar_chart(df_types.set_index("Modality"), color="#a855f7")
    else:
        st.info("No documents uploaded yet to display modality breakdown.")

# ---------------------------------------------------------------------------
# Integrated Fixed Bottom Chat Input Bar (Active Across Views)
# ---------------------------------------------------------------------------
chat_placeholder = "Message Agent RAG, paste link (YouTube/Web), or speak with mic..."
if st.session_state.active_doc_id:
    doc_t = get_document(st.session_state.active_doc_id)
    if doc_t:
        chat_placeholder = f"Ask about '{doc_t['filename'][:24]}' or speak with mic..."

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
            with st.spinner("🎧 Transcribing voice audio..."):
                transcribed_text = st.session_state.assistant.gemini.transcribe_audio_bytes(raw_audio_bytes)
                if transcribed_text:
                    user_query = transcribed_text
                    is_voice = True
                else:
                    st.warning("Could not detect clear speech. Please speak closer to the mic.")
        elif text_val:
            user_query = text_val.strip()
elif preset_action:
    user_query = preset_action.strip()

if user_query:
    st.session_state.active_view = "chat"
    user_display = f"🎙️ {user_query}" if is_voice else user_query

    st.session_state.messages.append({"role": "user", "content": user_display})

    # Prepare Assistant State
    f_info = st.session_state.active_file or {}
    state: AgentState = {
        "user_id": "usr_default",
        "user_name": "Aarav Sharma",
        "query": user_query,
        "attached_file_path": f_info.get("path"),
        "attached_file_type": f_info.get("type"),
        "attached_link": st.session_state.get("active_link"),
        "active_dataframe": st.session_state.active_df,
        "route": None,
        "response": None,
        "mode": "auto",
        "active_doc_id": st.session_state.get("active_doc_id")
    }

    result = st.session_state.assistant.process_turn(state)
    route = result.get("route", "GENERAL_CHAT")
    response = result.get("response", "No response generated.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "route": route
    })
    st.rerun()
