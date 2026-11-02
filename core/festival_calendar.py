"""
Multi-Year Indian Festive Calendar Engine (2024 - 2028)
Accurately maps lunar and solar Hindu, Islamic, Sikh, and National Indian festivals
according to each specific year's calendar.
"""

from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional

# Multi-Year Indian Festival Database
INDIAN_FESTIVAL_CALENDAR = [
    # --- 2024 ---
    {"name": "New Year's Day", "year": 2024, "month": 1, "day": 1, "significance": "First day of the Gregorian year."},
    {"name": "Makar Sankranti / Pongal / Lohri", "year": 2024, "month": 1, "day": 14, "significance": "Harvest festival celebrating the Sun's transit into Capricorn."},
    {"name": "Republic Day", "year": 2024, "month": 1, "day": 26, "significance": "Celebrates the Constitution of India."},
    {"name": "Maha Shivratri", "year": 2024, "month": 3, "day": 8, "significance": "The Great Night of Lord Shiva."},
    {"name": "Holi", "year": 2024, "month": 3, "day": 25, "significance": "Festival of Colors celebrating love and spring."},
    {"name": "Eid al-Fitr", "year": 2024, "month": 4, "day": 11, "significance": "Islamic celebration marking the end of Ramadan."},
    {"name": "Independence Day", "year": 2024, "month": 8, "day": 15, "significance": "Celebration of India's Independence."},
    {"name": "Raksha Bandhan", "year": 2024, "month": 8, "day": 19, "significance": "Celebration of the sacred bond between brothers and sisters."},
    {"name": "Janmashtami", "year": 2024, "month": 8, "day": 26, "significance": "Celebration of the birth of Lord Krishna."},
    {"name": "Ganesh Chaturthi", "year": 2024, "month": 9, "day": 7, "significance": "Welcoming Lord Ganesha, the remover of obstacles."},
    {"name": "Gandhi Jayanti", "year": 2024, "month": 10, "day": 2, "significance": "Birth anniversary of Mahatma Gandhi."},
    {"name": "Dussehra (Vijayadashami)", "year": 2024, "month": 10, "day": 12, "significance": "Triumph of Lord Rama over Ravana."},
    {"name": "Karwa Chauth", "year": 2024, "month": 10, "day": 20, "significance": "Fasting and prayer for marital longevity and happiness."},
    {"name": "Dhanteras", "year": 2024, "month": 10, "day": 29, "significance": "Day of wealth, prosperity, and wellness."},
    {"name": "Diwali (Deepavali)", "year": 2024, "month": 11, "day": 1, "significance": "Festival of Lights celebrating victory of light over darkness."},
    {"name": "Govardhan Puja / Annakut", "year": 2024, "month": 11, "day": 2, "significance": "Honoring Lord Krishna lifting Mount Govardhan."},
    {"name": "Bhai Dooj", "year": 2024, "month": 11, "day": 3, "significance": "Sibling celebration of love and lifelong protection."},
    {"name": "Chhath Puja", "year": 2024, "month": 11, "day": 7, "significance": "Veneration of Surya (the Sun God) and Chhathi Maiya."},
    {"name": "Guru Nanak Jayanti", "year": 2024, "month": 11, "day": 15, "significance": "Birth of Guru Nanak Dev Ji."},
    {"name": "Christmas", "year": 2024, "month": 12, "day": 25, "significance": "Celebration of the Nativity of Jesus Christ."},

    # --- 2025 ---
    {"name": "New Year's Day", "year": 2025, "month": 1, "day": 1, "significance": "First day of the year."},
    {"name": "Makar Sankranti / Pongal / Lohri", "year": 2025, "month": 1, "day": 14, "significance": "Harvest festival."},
    {"name": "Republic Day", "year": 2025, "month": 1, "day": 26, "significance": "Constitution Day of India."},
    {"name": "Maha Shivratri", "year": 2025, "month": 2, "day": 26, "significance": "Great Night of Shiva."},
    {"name": "Holi", "year": 2025, "month": 3, "day": 14, "significance": "Festival of Colors."},
    {"name": "Eid al-Fitr", "year": 2025, "month": 3, "day": 31, "significance": "End of Ramadan fasting."},
    {"name": "Independence Day", "year": 2025, "month": 8, "day": 15, "significance": "78th Independence Day."},
    {"name": "Raksha Bandhan", "year": 2025, "month": 8, "day": 9, "significance": "Sibling protection festival."},
    {"name": "Janmashtami", "year": 2025, "month": 8, "day": 16, "significance": "Birth of Lord Krishna."},
    {"name": "Ganesh Chaturthi", "year": 2025, "month": 8, "day": 27, "significance": "Vinayaka Chaturthi."},
    {"name": "Gandhi Jayanti", "year": 2025, "month": 10, "day": 2, "significance": "Mahatma Gandhi's birth anniversary."},
    {"name": "Dussehra (Vijayadashami)", "year": 2025, "month": 10, "day": 2, "significance": "Victory of good over evil."},
    {"name": "Karwa Chauth", "year": 2025, "month": 10, "day": 10, "significance": "Prayer for marital well-being."},
    {"name": "Dhanteras", "year": 2025, "month": 10, "day": 18, "significance": "Beginning of Diwali festivities."},
    {"name": "Diwali (Deepavali)", "year": 2025, "month": 10, "day": 20, "significance": "Festival of Lights."},
    {"name": "Bhai Dooj", "year": 2025, "month": 10, "day": 22, "significance": "Sibling love celebration."},
    {"name": "Chhath Puja", "year": 2025, "month": 10, "day": 27, "significance": "Sun worship and purification."},
    {"name": "Guru Nanak Jayanti", "year": 2025, "month": 11, "day": 5, "significance": "Gurpurab."},
    {"name": "Christmas", "year": 2025, "month": 12, "day": 25, "significance": "Christmas."},

    # --- 2026 ---
    {"name": "New Year's Day", "year": 2026, "month": 1, "day": 1, "significance": "New Year."},
    {"name": "Makar Sankranti / Pongal / Lohri", "year": 2026, "month": 1, "day": 14, "significance": "Harvest festival."},
    {"name": "Republic Day", "year": 2026, "month": 1, "day": 26, "significance": "Republic Day."},
    {"name": "Maha Shivratri", "year": 2026, "month": 2, "day": 15, "significance": "Night of Shiva."},
    {"name": "Holi", "year": 2026, "month": 3, "day": 4, "significance": "Festival of Colors."},
    {"name": "Eid al-Fitr", "year": 2026, "month": 3, "day": 20, "significance": "Eid celebration."},
    {"name": "Independence Day", "year": 2026, "month": 8, "day": 15, "significance": "Independence Day."},
    {"name": "Raksha Bandhan", "year": 2026, "month": 8, "day": 28, "significance": "Rakhi celebration."},
    {"name": "Janmashtami", "year": 2026, "month": 9, "day": 4, "significance": "Krishna Janmashtami."},
    {"name": "Ganesh Chaturthi", "year": 2026, "month": 9, "day": 14, "significance": "Ganesh Utsav."},
    {"name": "Gandhi Jayanti", "year": 2026, "month": 10, "day": 2, "significance": "Gandhi Jayanti."},
    {"name": "Dussehra (Vijayadashami)", "year": 2026, "month": 10, "day": 20, "significance": "Vijayadashami."},
    {"name": "Karwa Chauth", "year": 2026, "month": 10, "day": 29, "significance": "Karwa Chauth."},
    {"name": "Dhanteras", "year": 2026, "month": 11, "day": 6, "significance": "Auspicious Dhanteras."},
    {"name": "Diwali (Deepavali)", "year": 2026, "month": 11, "day": 8, "significance": "Festival of Lights."},
    {"name": "Bhai Dooj", "year": 2026, "month": 11, "day": 10, "significance": "Bhai Dooj."},
    {"name": "Chhath Puja", "year": 2026, "month": 11, "day": 15, "significance": "Chhath Mahaparv."},
    {"name": "Guru Nanak Jayanti", "year": 2026, "month": 11, "day": 24, "significance": "Gurpurab."},
    {"name": "Christmas", "year": 2026, "month": 12, "day": 25, "significance": "Christmas."},

    # --- 2027 ---
    {"name": "New Year's Day", "year": 2027, "month": 1, "day": 1, "significance": "New Year."},
    {"name": "Makar Sankranti / Pongal / Lohri", "year": 2027, "month": 1, "day": 14, "significance": "Harvest festival."},
    {"name": "Republic Day", "year": 2027, "month": 1, "day": 26, "significance": "Republic Day."},
    {"name": "Maha Shivratri", "year": 2027, "month": 3, "day": 6, "significance": "Night of Shiva."},
    {"name": "Holi", "year": 2027, "month": 3, "day": 22, "significance": "Festival of Colors."},
    {"name": "Eid al-Fitr", "year": 2027, "month": 3, "day": 10, "significance": "Eid celebration."},
    {"name": "Independence Day", "year": 2027, "month": 8, "day": 15, "significance": "Independence Day."},
    {"name": "Raksha Bandhan", "year": 2027, "month": 8, "day": 17, "significance": "Rakhi celebration."},
    {"name": "Janmashtami", "year": 2027, "month": 8, "day": 25, "significance": "Krishna Janmashtami."},
    {"name": "Ganesh Chaturthi", "year": 2027, "month": 9, "day": 4, "significance": "Ganesh Utsav."},
    {"name": "Gandhi Jayanti", "year": 2027, "month": 10, "day": 2, "significance": "Gandhi Jayanti."},
    {"name": "Dussehra (Vijayadashami)", "year": 2027, "month": 10, "day": 9, "significance": "Vijayadashami."},
    {"name": "Dhanteras", "year": 2027, "month": 10, "day": 27, "significance": "Dhanteras."},
    {"name": "Diwali (Deepavali)", "year": 2027, "month": 10, "day": 29, "significance": "Festival of Lights."},
    {"name": "Bhai Dooj", "year": 2027, "month": 10, "day": 31, "significance": "Bhai Dooj."},
    {"name": "Chhath Puja", "year": 2027, "month": 11, "day": 4, "significance": "Chhath Puja."},
    {"name": "Guru Nanak Jayanti", "year": 2027, "month": 11, "day": 14, "significance": "Gurpurab."},
    {"name": "Christmas", "year": 2027, "month": 12, "day": 25, "significance": "Christmas."}
]

def generate_festive_wishes(festival_name: str) -> List[str]:
    """Provides high quality, culturally rich wishes for Indian festivals."""
    f_lower = festival_name.lower()
    if "diwali" in f_lower or "deepavali" in f_lower:
        return [
            "May the divine light of Diwali bring peace, prosperity, and endless joy to you and your loved ones! Happy Diwali!",
            "Wishing you a bright, safe, and sparkling Diwali! May your year ahead be blessed with health and success.",
            "Let the light of the diyas guide your path toward wisdom and high achievements. Shubh Deepavali!"
        ]
    elif "holi" in f_lower:
        return [
            "Wishing you a vibrant, joyful, and colorful Holi! May your life be painted with happiness and laughter.",
            "May this festival of colors bring fresh beginnings and vibrant positivity to your journey. Happy Holi!",
            "Bura na mano, Holi hai! Wishing you a sweet season filled with gujiyas and cheer."
        ]
    elif "eid" in f_lower:
        return [
            "Eid Mubarak! May Allah bless you with peace, happiness, and prosperity today and always.",
            "Wishing you and your family a joyous Eid filled with sweet moments and cherished memories."
        ]
    elif "raksha" in f_lower or "rakhi" in f_lower:
        return [
            "Happy Raksha Bandhan! Celebrating the unbreakable bond of love, protection, and sweet camaraderie.",
            "May the sacred thread of Rakhi bring lifelong happiness and blessings to your sibling bond."
        ]
    elif "ganesh" in f_lower:
        return [
            "Ganpati Bappa Morya! May Lord Ganesha remove all obstacles and bless you with wisdom and prosperity.",
            "Wishing you a blessed Ganesh Chaturthi filled with sweet modaks and divine grace."
        ]
    elif "dussehra" in f_lower or "vijayadashami" in f_lower:
        return [
            "Happy Dussehra! May the triumph of good over evil inspire you to conquer every challenge with courage.",
            "Wishing you victory, righteousness, and prosperity on this auspicious Vijayadashami."
        ]
    elif "chhath" in f_lower:
        return [
            "May Lord Surya and Chhathi Maiya shower you with divine energy, health, and abundant blessings. Shubh Chhath Puja!",
            "Wishing you peace, purity, and spiritual devotion this Chhath Puja."
        ]
    elif "shivratri" in f_lower:
        return [
            "Har Har Mahadev! May Lord Shiva bless you with inner strength, peace, and spiritual growth on Maha Shivratri.",
            "Wishing you a blessed and serene Maha Shivratri."
        ]
    elif "sankranti" in f_lower or "pongal" in f_lower or "lohri" in f_lower:
        return [
            "Happy Makar Sankranti and Pongal! May the harvest season bring soaring kites, sweetness, and prosperity.",
            "Wishing you warm Lohri bonfires and sweet harvest blessings!"
        ]
    elif "new year" in f_lower:
        return [
            "Happy New Year! Here is to 365 fresh opportunities, growth, breakthroughs, and memorable moments.",
            "May this new year bring boundless happiness, health, and high achievements!"
        ]
    else:
        return [
            f"Wishing you a very joyous and blessed {festival_name}! May your day be filled with happiness and celebration.",
            f"Warmest thoughts and sincere wishes on the auspicious occasion of {festival_name}."
        ]

def get_festivals_for_window(ref_date: Optional[date] = None, days_window: int = 14) -> List[Dict[str, Any]]:
    """
    Returns festivals occurring from today through the next `days_window` days,
    accurately matched against the specific year calendar.
    """
    if ref_date is None:
        ref_date = date.today()

    matched_festivals = []

    for fest in INDIAN_FESTIVAL_CALENDAR:
        try:
            fest_date = date(fest["year"], fest["month"], fest["day"])
            diff_days = (fest_date - ref_date).days

            if 0 <= diff_days <= days_window:
                fest_copy = dict(fest)
                fest_copy["event_date"] = fest_date.strftime("%Y-%m-%d")
                fest_copy["days_remaining"] = diff_days
                fest_copy["is_today"] = (diff_days == 0)
                fest_copy["wishes_list"] = generate_festive_wishes(fest["name"])
                matched_festivals.append(fest_copy)
        except Exception:
            continue

    # Sort chronologically by days remaining
    matched_festivals.sort(key=lambda x: x["days_remaining"])
    return matched_festivals

def get_todays_festival(ref_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
    """Checks if today is a festival according to the year calendar."""
    festivals = get_festivals_for_window(ref_date=ref_date, days_window=0)
    return festivals[0] if festivals else None
