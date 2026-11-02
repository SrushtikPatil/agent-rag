"""
Data Access Layer for querying personal memory, festivals, and history.
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from database.db import get_connection

def get_user_profile(user_id: str = "usr_default") -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_milestones_today(user_id: str = "usr_default", month: int = None, day: int = None) -> List[Dict[str, Any]]:
    if month is None or day is None:
        now = datetime.now()
        month, day = now.month, now.day

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM user_milestones 
        WHERE user_id = ? AND event_month = ? AND event_day = ?
    """, (user_id, month, day))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_contact_events(user_id: str = "usr_default", days_window: int = 3) -> List[Dict[str, Any]]:
    """
    Returns events for contacts happening today or within the next `days_window` days.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM contact_events WHERE user_id = ?", (user_id,))
    all_contacts = cursor.fetchall()
    conn.close()

    now = datetime.now()
    matched = []

    for contact in all_contacts:
        c_dict = dict(contact)
        c_month = c_dict["event_month"]
        c_day = c_dict["event_day"]

        # Calculate difference in days
        for offset in range(days_window + 1):
            target_date = now + timedelta(days=offset)
            if target_date.month == c_month and target_date.day == c_day:
                c_dict["is_today"] = (offset == 0)
                c_dict["days_remaining"] = offset
                matched.append(c_dict)
                break

    return matched

def get_festivals(days_window: int = 3) -> List[Dict[str, Any]]:
    """
    Returns festivals happening today or within the next `days_window` days.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM festive_calendar WHERE is_active = 1")
    all_festivals = cursor.fetchall()
    conn.close()

    now = datetime.now()
    matched = []

    for fest in all_festivals:
        f_dict = dict(fest)
        f_month = f_dict["event_month"]
        f_day = f_dict["event_day"]

        if isinstance(f_dict.get("sample_wishes"), str):
            try:
                f_dict["wishes_list"] = json.loads(f_dict["sample_wishes"])
            except Exception:
                f_dict["wishes_list"] = []
        else:
            f_dict["wishes_list"] = []

        for offset in range(days_window + 1):
            target_date = now + timedelta(days=offset)
            if target_date.month == f_month and target_date.day == f_day:
                f_dict["is_today"] = (offset == 0)
                f_dict["days_remaining"] = offset
                matched.append(f_dict)
                break

    return matched

def get_latest_interaction_summary(user_id: str = "usr_default") -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM interaction_summaries 
        WHERE user_id = ? 
        ORDER BY id DESC LIMIT 1
    """, (user_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None

    res = dict(row)
    try:
        res["topics_discussed"] = json.loads(res["topics_discussed"])
    except Exception:
        res["topics_discussed"] = []
    return res

def add_contact_event(user_id: str, name: str, relationship: str, event_type: str, 
                      month: int, day: int, tone: str = "warm", notes: str = ""):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO contact_events (user_id, contact_name, relationship, event_type, event_month, event_day, preferred_tone, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, name, relationship, event_type, month, day, tone, notes))
    conn.commit()
    conn.close()

def log_chat_message(user_id: str, role: str, content: str, route: str = "GENERAL_CHAT"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_logs (user_id, role, content, route)
        VALUES (?, ?, ?, ?)
    """, (user_id, role, content, route))
    conn.commit()
    conn.close()

def save_interaction_summary(user_id: str, summary: str, topics: List[str]):
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("""
        INSERT INTO interaction_summaries (user_id, session_date, topics_discussed, key_tasks_pending, mood_sentiment, summary_paragraph)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, now_str, json.dumps(topics), json.dumps([]), "engaged", summary))
    conn.commit()
    conn.close()
