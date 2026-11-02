"""
Proactive Personalization & Greeting Engine
Generates time-aware greetings, year-accurate Indian festival celebrations, milestone alerts,
custom wishing quotes, and context-aware check-ins based on past interaction history.
"""

from datetime import datetime, date
import pytz
from typing import Dict, Any, List
from database.models import (
    get_user_profile,
    get_user_milestones_today,
    get_contact_events,
    get_latest_interaction_summary
)
from core.festival_calendar import get_festivals_for_window, get_todays_festival, generate_festive_wishes

def get_time_based_greeting(tz_name: str = "Asia/Kolkata") -> Dict[str, str]:
    """Determines appropriate greeting and icon based on current local hour."""
    try:
        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)
    except Exception:
        now = datetime.now()

    hour = now.hour

    if 4 <= hour < 12:
        return {"greeting": "Good Morning", "icon": "🌅", "time_str": now.strftime("%I:%M %p")}
    elif 12 <= hour < 17:
        return {"greeting": "Good Afternoon", "icon": "☀️", "time_str": now.strftime("%I:%M %p")}
    elif 17 <= hour < 22:
        return {"greeting": "Good Evening", "icon": "🌆", "time_str": now.strftime("%I:%M %p")}
    else:
        return {"greeting": "Good Evening", "icon": "🌙", "time_str": now.strftime("%I:%M %p")}

def generate_wish_quotes(contact_name: str, relationship: str, event_type: str, tone: str = "warm") -> List[str]:
    """Generates three ready-to-use wishing lines in different tones."""
    if event_type == "birthday":
        return [
            f"🎉 Happy Birthday, {contact_name}! Wishing you immense happiness, great health, and breakthrough success this year!",
            f"🎂 Happy Birthday {contact_name}! Cheers to another year of awesome memories and shared laughs!",
            f"✨ Sending warmest thoughts and sincere wishes on your special day, {contact_name}."
        ]
    else:
        return [
            f"💍 Happy Anniversary, {contact_name}! Wishing you both a lifetime of love, laughter, and togetherness.",
            f"🥂 Another year of great partnership and beautiful memories! Happy Anniversary {contact_name}!",
            f"✨ May your bond grow deeper and stronger with each passing season. Happy Anniversary!"
        ]

def get_proactive_briefing(user_id: str = "usr_default") -> Dict[str, Any]:
    """
    Assembles complete proactive intelligence when user opens the app:
    - Time-aware greeting
    - Year-accurate Indian festive calendar triggers
    - Personal milestones (Self)
    - Friends' birthdays & anniversaries with pre-crafted wishes
    - Contextual check-in based on recent chat history
    """
    user = get_user_profile(user_id)
    user_name = user["full_name"].split()[0] if user else "Friend"
    tz_name = user.get("timezone", "Asia/Kolkata") if user else "Asia/Kolkata"

    try:
        tz = pytz.timezone(tz_name)
        today_local = datetime.now(tz).date()
    except Exception:
        today_local = date.today()

    time_info = get_time_based_greeting(tz_name)
    
    # 1. Year-Accurate Indian Festival Check (Window of 14 days)
    festivals = get_festivals_for_window(ref_date=today_local, days_window=14)
    active_festival = festivals[0] if festivals else None

    # 2. User Milestones (Self)
    user_milestones = get_user_milestones_today(user_id)

    # 3. Friends & Contacts Milestones
    contact_events = get_contact_events(user_id, days_window=3)
    for event in contact_events:
        event["suggested_wishes"] = generate_wish_quotes(
            contact_name=event["contact_name"],
            relationship=event["relationship"],
            event_type=event["event_type"],
            tone=event.get("preferred_tone", "warm")
        )

    # 4. Contextual History Check
    latest_history = get_latest_interaction_summary(user_id)
    context_question = None
    if latest_history:
        topics = latest_history.get("topics_discussed", [])
        topic_preview = f"'{topics[0]}'" if topics else "your recent goals"
        context_question = f"How did your work on {topic_preview} go since we last talked?"

    return {
        "user_name": user_name,
        "time_info": time_info,
        "active_festival": active_festival,
        "upcoming_festivals": festivals,
        "user_milestones": user_milestones,
        "contact_events": contact_events,
        "latest_history": latest_history,
        "context_question": context_question
    }
