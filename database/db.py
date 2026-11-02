"""
Database initialization and management for the Personal Agentic Assistant.
Uses SQLite for persistent storage of user profiles, milestones, contacts,
Indian festivals, and interaction history.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_FILE = os.path.join(os.path.dirname(__file__), "assistant_memory.db")

def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes tables and seeds default data if tables are empty."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        full_name TEXT NOT NULL,
        email TEXT,
        timezone TEXT DEFAULT 'Asia/Kolkata',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. User Personal Milestones (Own Birthday, Anniversary)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_milestones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        milestone_type TEXT NOT NULL, -- 'birthday', 'anniversary'
        event_month INTEGER NOT NULL,
        event_day INTEGER NOT NULL,
        event_year INTEGER,
        notes TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """)

    # 3. Friends & Family Relationships
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS contact_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        contact_name TEXT NOT NULL,
        relationship TEXT NOT NULL, -- 'friend', 'colleague', 'mother', 'spouse'
        event_type TEXT NOT NULL,   -- 'birthday', 'anniversary'
        event_month INTEGER NOT NULL,
        event_day INTEGER NOT NULL,
        preferred_tone TEXT DEFAULT 'warm', -- 'warm', 'funny', 'poetic', 'formal'
        notes TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """)

    # 4. Festive Calendar (Indian and Global Festivals)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS festive_calendar (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        festival_name TEXT NOT NULL,
        event_month INTEGER NOT NULL,
        event_day INTEGER NOT NULL,
        significance TEXT,
        region TEXT DEFAULT 'Pan-India',
        sample_wishes TEXT, -- JSON string list of wishes & quotes
        is_active INTEGER DEFAULT 1
    );
    """)

    # 5. Interaction Summaries for Contextual Check-ins
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS interaction_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        session_date TEXT NOT NULL,
        topics_discussed TEXT, -- JSON list
        key_tasks_pending TEXT, -- JSON list
        mood_sentiment TEXT DEFAULT 'productive',
        summary_paragraph TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """)

    # 6. Chat Messages Log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        route TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()

    # Seed Default Data if empty
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        seed_default_data(conn)

    conn.close()

def seed_default_data(conn):
    """Populates user, milestones, contacts, festivals, and sample interaction history."""
    cursor = conn.cursor()

    # Seed Default User
    cursor.execute("""
        INSERT INTO users (user_id, full_name, email, timezone)
        VALUES ('usr_default', 'Aarav Sharma', 'aarav@example.com', 'Asia/Kolkata')
    """)

    # Seed User Milestones
    cursor.execute("""
        INSERT INTO user_milestones (user_id, milestone_type, event_month, event_day, event_year, notes)
        VALUES 
        ('usr_default', 'birthday', 11, 15, 1996, 'User 30th milestone approaching'),
        ('usr_default', 'anniversary', 12, 10, 2021, 'Wedding Anniversary')
    """)

    # Seed Contacts (Friends, Family, Colleagues)
    now = datetime.now()
    curr_m = now.month
    curr_d = now.day

    # Create one contact for today or tomorrow so the user immediately experiences the reminder
    next_day = curr_d + 1 if curr_d < 28 else 1
    next_m = curr_m if curr_d < 28 else (curr_m % 12 + 1)

    contacts = [
        ('usr_default', 'Rohan Verma', 'Best Friend', 'birthday', curr_m, curr_d, 'funny', 'Loves tech, gaming, and coffee'),
        ('usr_default', 'Priya Patel', 'Colleague', 'birthday', next_m, next_day, 'formal', 'Senior Project Manager at work'),
        ('usr_default', 'Mom', 'Mother', 'birthday', 5, 20, 'emotional', 'Likes heartfelt blessings and flowers'),
        ('usr_default', 'Ananya & Vikram', 'Friends', 'anniversary', curr_m, (curr_d + 3) % 28 + 1, 'poetic', 'College batchmates')
    ]
    cursor.executemany("""
        INSERT INTO contact_events (user_id, contact_name, relationship, event_type, event_month, event_day, preferred_tone, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, contacts)

    # Seed Festivals with warm wishes and quotes
    festivals = [
        ("Diwali (Deepavali)", 11, 1, 
         "Festival of Lights celebrating victory of light over darkness and knowledge over ignorance.",
         "Pan-India",
         json.dumps([
             "Wishing you and your family a dazzling and blessed Diwali! May your home shine with joy, health, and prosperity.",
             "May the divine light of Diwali illuminate your journey with infinite peace, success, and endless laughter. Happy Diwali!",
             "Let each diya you light be a beacon of hope and happiness. Have a wonderful and safe Diwali!"
         ])),
        ("Bhai Dooj", 11, 3,
         "Celebration of the sacred bond between brothers and sisters.",
         "North & West India",
         json.dumps([
             "Happy Bhai Dooj! Celebrating the timeless bond of protection, love, and sweet childhood memories.",
             "Wishing a joyous Bhai Dooj filled with laughter, blessings, and lifelong camaraderie."
         ])),
        ("Chhath Puja", 11, 7,
         "Ancient Hindu festival dedicated to Surya (the Sun God) and Chhathi Maiya.",
         "Bihar, Jharkhand, UP",
         json.dumps([
             "May the divine rays of Lord Surya bestow abundant health, prosperity, and positivity upon you. Shubh Chhath Puja!",
             "Wishing you peace and purity on the auspicious occasion of Chhath Puja."
         ])),
        ("Guru Nanak Jayanti", 11, 15,
         "Birth of Guru Nanak Dev Ji, founder of Sikhism.",
         "Pan-India",
         json.dumps([
             "Happy Gurpurab! May Guru Nanak Dev Ji inspire you to walk the path of truth, compassion, and universal love.",
             "On this blessed Guru Nanak Jayanti, may your life be filled with righteous peace and serenity."
         ])),
        ("Christmas", 12, 25,
         "Celebration of the Nativity of Jesus.",
         "Global & India",
         json.dumps([
             "Merry Christmas! May your days be merry, your heart be light, and your home be filled with warmth.",
             "Wishing you peace, joy, and goodwill this festive holiday season."
         ])),
        ("New Year", 1, 1,
         "First day of the Gregorian year.",
         "Global & India",
         json.dumps([
             "Happy New Year! Here is to 365 fresh opportunities, growth, breakthroughs, and memorable moments.",
             "May this new year bring boundless happiness, health, and high achievements!"
         ])),
        ("Makar Sankranti / Pongal / Lohri", 1, 14,
         "Harvest festival celebrated across India under varied names.",
         "Pan-India",
         json.dumps([
             "Happy Makar Sankranti and Pongal! May the harvest season bring sweetness, abundance, and sunny skies.",
             "Wishing you a season full of sweet till-gur and soaring kites! Happy Sankranti!"
         ])),
        ("Republic Day", 1, 26,
         "Celebration of the Constitution of India coming into effect.",
         "Pan-India",
         json.dumps([
             "Happy Republic Day! Proud to celebrate the diversity, spirit, and heritage of our great nation.",
             "Saluting the constitution that binds us all. Jai Hind!"
         ])),
        ("Holi", 3, 14,
         "Festival of colors celebrating love, spring, and the triumph of good.",
         "Pan-India",
         json.dumps([
             "Wishing you a vibrant, joyous, and colorful Holi! May your year be painted with laughter and prosperity.",
             "Bura na mano, Holi hai! May your life be filled with the brightest hues of happiness."
         ])),
        ("Independence Day", 8, 15,
         "Commemoration of Indian independence.",
         "Pan-India",
         json.dumps([
             "Happy Independence Day! Let us cherish freedom and contribute to a glorious future.",
             "Honoring the courageous heroes who gave us our freedom. Happy 15th August!"
         ]))
    ]

    cursor.executemany("""
        INSERT INTO festive_calendar (festival_name, event_month, event_day, significance, region, sample_wishes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, festivals)

    # Seed Sample Past Interaction Summary
    cursor.execute("""
        INSERT INTO interaction_summaries (user_id, session_date, topics_discussed, key_tasks_pending, mood_sentiment, summary_paragraph)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        'usr_default',
        now.strftime('%Y-%m-%d'),
        json.dumps(["Multi-modal AI project", "Data analysis presentation", "Sprint goals"]),
        json.dumps(["Review confusion matrix slides", "Push repository updates"]),
        "productive and optimistic",
        "Yesterday we worked on structuring your Multi-Modal RAG architecture and prepared charts for your team presentation. You were excited about integrating audio and video analysis."
    ))

    conn.commit()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_FILE)
