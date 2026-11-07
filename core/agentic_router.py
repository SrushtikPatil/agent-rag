"""
Agentic Router with LangGraph & Google Gemini 1.5 Pro
Determines user intent and routes to the appropriate specialized node.
Handles priority correctly so personal/festive questions are not hijacked by attached files.
"""

import os
import re
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from core.gemini_service import GeminiService
from core.excel_analyzer import ExcelAnalyzer
from core.rag_engine import MultimodalRAGEngine
from core.link_service import LinkService
from database.models import (
    get_contact_events,
    get_festivals,
    get_user_milestones_today,
    get_latest_interaction_summary,
    add_contact_event,
    log_chat_message
)

class AgentState(TypedDict):
    user_id: str
    user_name: str
    query: str
    attached_file_path: Optional[str]
    attached_file_type: Optional[str] # 'pdf', 'excel', 'image', 'audio', 'video', None
    attached_link: Optional[Dict[str, Any]]
    active_dataframe: Optional[Any]
    route: Optional[str]
    response: Optional[str]
    mode: Optional[str] # 'auto', 'doc_qa', 'personal', 'general', 'link'

class AgenticAssistant:
    def __init__(self, api_key: Optional[str] = None):
        self.gemini = GeminiService(api_key=api_key)
        self.rag_engine = MultimodalRAGEngine()
        self.excel_analyzer = ExcelAnalyzer()
        self.link_service = LinkService()

    def update_api_key(self, api_key: str):
        """Allows hot-swapping API key from UI."""
        self.gemini = GeminiService(api_key=api_key)

    def classify_intent(self, query: str, file_type: Optional[str], mode: str = "auto", attached_link: Optional[Dict[str, Any]] = None) -> str:
        """
        Determines which engine should handle the query.
        When a document, video, or data file is attached, the searching bar is FIRMLY ATTACHED to the document!
        When a link (YouTube, webpage, image, movie) is attached or present, routes to LINK_ANALYSIS!
        """
        if mode == "doc_qa" and file_type:
            return "STRUCTURED_DATA" if file_type in ["excel", "csv"] else "MULTIMODAL_RAG"
        elif mode == "link" or attached_link:
            return "LINK_ANALYSIS"
        elif mode == "personal":
            return "PERSONAL_MEMORY"
        elif mode == "general":
            return "GENERAL_CHAT"

        # Check for web / video / photo links inside the query
        if re.search(r'https?://[^\s<>"]+', query):
            return "LINK_ANALYSIS"

        q_lower = query.lower().strip()

        # Pure standalone greetings (only when NO file is attached)
        if not file_type and q_lower in ["hi", "hello", "hey", "hlo", "hii", "who are you"]:
            return "GENERAL_CHAT"

        # Explicit Indian Festival / Calendar triggers (e.g. "when is diwali", "upcoming birthdays")
        explicit_personal = [
            "diwali", "holi", "eid", "navratri", "dussehra", "bhai dooj", "dhanteras",
            "birthday", "anniversary", "friend", "rohan", "priya", "how was my day", "yesterday recap",
            "calendar", "milestones"
        ]
        is_personal = any(w in q_lower for w in explicit_personal)
        has_doc_keywords = any(w in q_lower for w in [
            "summary", "summarize", "document", "video", "audio", "pdf", "file",
            "attached", "takeaways", "notes", "point", "points", "explain", "important",
            "what is this", "tell me about", "exam", "concepts", "key"
        ])

        # If it's a genuine festival/calendar query without referring to a document
        if is_personal and not has_doc_keywords and not file_type:
            return "PERSONAL_MEMORY"

        # -------------------------------------------------------------
        # IF A DOCUMENT OR FILE IS ATTACHED:
        # The searching bar is ATTACHED TO THE ATTACHED FILE!
        # -------------------------------------------------------------
        if file_type:
            if file_type in ["excel", "csv"]:
                return "STRUCTURED_DATA"
            else:
                return "MULTIMODAL_RAG"

        if is_personal:
            return "PERSONAL_MEMORY"

        return "GENERAL_CHAT"

    def handle_personal_memory(self, query: str, user_id: str, user_name: str) -> str:
        """Processes personal memory, festival, relationship, and history queries."""
        q_lower = query.lower()

        # Year-Accurate Indian Festival lookup
        from core.festival_calendar import get_festivals_for_window
        festivals = get_festivals_for_window(days_window=30)
        contacts = get_contact_events(user_id, days_window=14)
        history = get_latest_interaction_summary(user_id)

        # Fallback pre-crafted responses for instant answers even without API key!
        if "diwali" in q_lower and ("wish" in q_lower or "quote" in q_lower):
            return (
                "🪔 **Heartfelt Diwali Wishes for Colleagues & Team:**\n\n"
                "1. **Professional & Warm:**\n"
                "   *\"Wishing you and your family a joyous, safe, and blessed Diwali! May the festival of lights bring abundant prosperity, good health, and continued success to all your endeavors.\"*\n\n"
                "2. **Inspirational & Collaborative:**\n"
                "   *\"May the brilliance of this Diwali illuminate our shared journey with new milestones, great teamwork, and endless positivity. Have a wonderful Diwali celebration!\"*\n\n"
                "3. **Casual & Cheerful:**\n"
                "   *\"Happy Diwali! Wishing you peaceful holidays filled with delicious mithai, warm laughter, and bright moments with your loved ones!\"*"
            )

        if "birthday" in q_lower and "who" in q_lower:
            if contacts:
                upcoming = [f"• **{c['contact_name']}** ({c['relationship']}): {c['event_type'].capitalize()} in {c['days_remaining']} day(s) 🎂" for c in contacts]
                return "📅 **Upcoming Birthdays & Milestones:**\n\n" + "\n".join(upcoming)
            return "📅 No birthdays found in the next 7 days in your calendar."

        if "how was my day" in q_lower or "yesterday" in q_lower or "recap" in q_lower:
            if history:
                return (
                    f"🔍 **Session Memory Recap:**\n\n"
                    f"**Last Session Date:** `{history.get('session_date')}`\n\n"
                    f"**Topics Discussed:** {', '.join(history.get('topics_discussed', []))}\n\n"
                    f"**Summary:** *{history.get('summary_paragraph')}*\n\n"
                    f"💡 *How did that work out for you today? Ready to pick up where we left off?*"
                )

        # If API key is available, generate dynamic response with Gemini
        context_blocks = []
        if festivals:
            fest_names = ", ".join([f"{f.get('name') or f.get('festival_name')} (in {f['days_remaining']} days)" for f in festivals])
            context_blocks.append(f"Upcoming Festivals: {fest_names}")
        if contacts:
            cont_str = ", ".join([f"{c['contact_name']} ({c['event_type']} in {c['days_remaining']} days)" for c in contacts])
            context_blocks.append(f"Upcoming Contacts Events: {cont_str}")
        if history:
            context_blocks.append(f"Last Session Summary: {history.get('summary_paragraph')}")

        memory_context = "\n".join(context_blocks)
        prompt = (
            f"You are Agent Rag, {user_name}'s empathetic personal memory assistant.\n"
            f"Database Context:\n{memory_context}\n\n"
            f"User Question: {query}\n\n"
            "Provide a warm, personalized response. If they want quotes or wishes, give 3 distinct options."
        )

        return self.gemini.generate_chat_response(prompt)

    def process_turn(self, state: AgentState) -> Dict[str, Any]:
        """Main routing and response execution loop."""
        user_id = state.get("user_id", "usr_default")
        user_name = state.get("user_name", "User")
        query = state["query"]
        file_path = state.get("attached_file_path")
        file_type = state.get("attached_file_type")
        df = state.get("active_dataframe")
        mode = state.get("mode", "auto")

        # 1. Routing Decision
        attached_link = state.get("attached_link")
        route = self.classify_intent(query, file_type, mode, attached_link=attached_link)
        state["route"] = route

        # 2. Dispatch
        if route == "LINK_ANALYSIS":
            link_info = attached_link
            url = None
            if not link_info:
                url = self.link_service.extract_url(query)
                if url:
                    link_info = self.link_service.fetch_link_info(url)

            if link_info:
                clean_q = self.link_service.clean_query_around_url(query, url) if url else query
                output = self.link_service.analyze_link(link_info, clean_q, self.gemini)
            else:
                output = "⚠️ Could not resolve a valid web link. Please provide a complete URL starting with `http://` or `https://`."

        elif route == "STRUCTURED_DATA":
            if df is not None:
                output = self.excel_analyzer.answer_query(df, query, self.gemini)
            elif file_path:
                info = self.excel_analyzer.inspect_file(file_path)
                if info["success"]:
                    output = self.excel_analyzer.answer_query(info["dataframe"], query, self.gemini)
                else:
                    output = f"⚠️ Could not read file: {info.get('error')}"
            else:
                output = "Please upload an Excel (.xlsx) or CSV file to perform tabular computations."

        elif route == "MULTIMODAL_RAG":
            if file_path and file_type:
                output = self.gemini.analyze_multimodal_file(file_path, file_type, query)
            else:
                context = self.rag_engine.retrieve_relevant_context(query)
                prompt = (
                    f"You are Agent Rag, an expert document intelligence assistant.\n"
                    f"Retrieved Document Context:\n{context}\n\n"
                    f"User Question: {query}\n\n"
                    "Provide a comprehensive answer citing the context."
                )
                output = self.gemini.generate_chat_response(prompt)

        elif route == "PERSONAL_MEMORY":
            output = self.handle_personal_memory(query, user_id, user_name)

        else: # GENERAL_CHAT
            clean_q = query.strip().lower().rstrip("!.?,")
            greetings = [
                "hi", "hello", "hey", "hlo", "hii", "hiii", "helo", "greeting", "greetings",
                "hello agent rag", "hi agent rag", "hey agent rag", "who are you", "what is your name",
                "introduce yourself", "tell me about yourself"
            ]
            if clean_q in greetings:
                output = "Hello! I am Agent Rag, your personal assistant. How can I help you today?"
            else:
                system_inst = (
                    "You are Agent Rag, your personal assistant. You are helpful, articulate, professional, and knowledgeable. "
                    "Always refer to yourself simply as 'Agent Rag, your personal assistant'. Never use the words 'intelligent' or 'multimodal' in your greeting. "
                    "If the user greets you or asks who you are, always reply exactly: 'Hello! I am Agent Rag, your personal assistant. How can I help you today?'"
                )
                output = self.gemini.generate_chat_response(query, system_instruction=system_inst)
                
                # Sanitize to guarantee the exact requested phrasing
                output = output.replace(
                    "Hello! I am Agent Rag, your intelligent personal multimodal assistant. How can I help you today?",
                    "Hello! I am Agent Rag, your personal assistant. How can I help you today?"
                ).replace(
                    "intelligent personal multimodal assistant",
                    "personal assistant"
                ).replace(
                    "intelligent personal multimodal agentic assistant",
                    "personal assistant"
                ).replace(
                    "personal multimodal assistant",
                    "personal assistant"
                )

        # 3. Log interaction to SQLite
        try:
            log_chat_message(user_id, "user", query, route)
            log_chat_message(user_id, "assistant", output, route)
        except Exception:
            pass

        state["response"] = output
        return state
