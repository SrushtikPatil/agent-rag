"""
Agentic Router with LangGraph & Google Gemini 1.5 Pro
Determines user intent and routes to the appropriate specialized node.
Handles priority correctly so personal/festive questions are not hijacked by attached files.
"""

import os
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from core.gemini_service import GeminiService
from core.excel_analyzer import ExcelAnalyzer
from core.rag_engine import MultimodalRAGEngine
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
    active_dataframe: Optional[Any]
    route: Optional[str]
    response: Optional[str]
    mode: Optional[str] # 'auto', 'doc_qa', 'personal', 'general'

class AgenticAssistant:
    def __init__(self, api_key: Optional[str] = None):
        self.gemini = GeminiService(api_key=api_key)
        self.rag_engine = MultimodalRAGEngine()
        self.excel_analyzer = ExcelAnalyzer()

    def update_api_key(self, api_key: str):
        """Allows hot-swapping API key from UI."""
        self.gemini = GeminiService(api_key=api_key)

    def classify_intent(self, query: str, file_type: Optional[str], mode: str = "auto") -> str:
        """
        Determines which engine should handle the query with intelligent priority.
        Personal & Festive queries are prioritized when relevant.
        """
        if mode == "doc_qa" and file_type:
            return "STRUCTURED_DATA" if file_type in ["excel", "csv"] else "MULTIMODAL_RAG"
        elif mode == "personal":
            return "PERSONAL_MEMORY"
        elif mode == "general":
            return "GENERAL_CHAT"

        q_lower = query.lower()

        # 1. Personal & Festive Triggers (ALWAYS high priority, even if document is uploaded)
        personal_keywords = [
            "diwali", "holi", "eid", "festival", "festive", "navratri", "dussehra",
            "birthday", "anniversary", "wish", "wishes", "quote", "quotes", "greet", "greeting",
            "reminder", "remind me", "friend", "rohan", "priya", "how was my day", "yesterday",
            "recap", "calendar", "milestone"
        ]
        if any(w in q_lower for w in personal_keywords):
            return "PERSONAL_MEMORY"

        # 2. Document & Spreadsheet Specific Triggers
        doc_keywords = [
            "summarize", "summary", "document", "pdf", "file", "page", "chapter",
            "what does this say", "extract", "topics", "key points", "audio", "video",
            "timestamp", "image", "chart", "diagram", "table", "calculate", "average", "rows"
        ]

        if file_type:
            if file_type in ["excel", "csv"]:
                # If an excel file is active, math/data queries go to structured data
                calc_words = ["sum", "total", "average", "mean", "median", "max", "min", "column", "row", "calculate", "chart", "filter"]
                if any(w in q_lower for w in calc_words) or any(w in q_lower for w in doc_keywords):
                    return "STRUCTURED_DATA"
            elif any(w in q_lower for w in doc_keywords):
                return "MULTIMODAL_RAG"

        # 3. Default to General Chat if question doesn't specifically target the document
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
            fest_names = ", ".join([f"{f['festival_name']} (in {f['days_remaining']} days)" for f in festivals])
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
        route = self.classify_intent(query, file_type, mode)
        state["route"] = route

        # 2. Dispatch
        if route == "STRUCTURED_DATA":
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
