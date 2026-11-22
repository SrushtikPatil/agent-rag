"""
Live Real-Time News Service for Agent Rag
Fetches, caches, searches, and formats up-to-the-minute news from verified global and Indian publishers.
Provides chronological analysis from historical background to today's breaking updates.
"""

import time
import email.utils
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*"
}

CATEGORY_FEEDS = {
    "top": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
    "tech": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en",
    "business": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
    "world": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",
    "science": "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-IN&gl=IN&ceid=IN:en"
}

FALLBACK_FEEDS = {
    "world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "tech": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "business": "https://feeds.bbci.co.uk/news/business/rss.xml"
}

# In-memory cache: {cache_key: (timestamp, items)}
_NEWS_CACHE: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}
CACHE_TTL = 300  # 5 minutes


class NewsService:
    def __init__(self):
        pass

    def _format_time_ago(self, pub_date_str: str) -> str:
        """Converts RFC 2822 pubDate string to human-friendly relative time."""
        if not pub_date_str:
            return "Recently"
        try:
            parsed = email.utils.parsedate_to_datetime(pub_date_str)
            now = datetime.now(timezone.utc)
            delta = now - parsed
            seconds = int(delta.total_seconds())

            if seconds < 0:
                return "Just now"
            if seconds < 3600:
                mins = max(1, seconds // 60)
                return f"{mins}m ago"
            if seconds < 86400:
                hrs = seconds // 3600
                return f"{hrs}h ago"
            if seconds < 172800:
                return "Yesterday"
            days = seconds // 86400
            return f"{days}d ago"
        except Exception:
            return pub_date_str[:16]

    def _parse_rss_content(self, xml_content: bytes, category: str, limit: int = 6) -> List[Dict[str, Any]]:
        """Parses RSS XML into clean structured news items."""
        items = []
        try:
            root = ET.fromstring(xml_content)
            for item_elem in root.findall(".//item")[:limit * 2]:
                raw_title = item_elem.find("title").text or ""
                # Split off publisher suffix (e.g., 'Headline - The Hindu')
                parts = raw_title.rsplit(" - ", 1)
                title = parts[0].strip()
                source = parts[1].strip() if len(parts) > 1 else "News"

                link = ""
                link_elem = item_elem.find("link")
                if link_elem is not None and link_elem.text:
                    link = link_elem.text.strip()

                pub_date_elem = item_elem.find("pubDate")
                pub_date = pub_date_elem.text.strip() if pub_date_elem is not None and pub_date_elem.text else ""

                desc_elem = item_elem.find("description")
                desc_text = ""
                if desc_elem is not None and desc_elem.text:
                    # Strip basic HTML tags
                    desc_text = desc_elem.text
                    import re
                    desc_text = re.sub(r"<[^>]+>", "", desc_text).strip()

                time_ago = self._format_time_ago(pub_date)

                # Parse datetime for sorting
                dt_val = 0
                try:
                    dt = email.utils.parsedate_to_datetime(pub_date)
                    dt_val = dt.timestamp()
                except Exception:
                    pass

                items.append({
                    "title": title,
                    "source": source,
                    "link": link,
                    "pub_date": pub_date,
                    "time_ago": time_ago,
                    "timestamp": dt_val,
                    "description": desc_text[:200],
                    "category": category.title()
                })
                if len(items) >= limit:
                    break
        except Exception as e:
            print(f"[NewsService] XML Parse Error: {e}")
        return items

    def fetch_live_news(self, category: str = "top", limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetches live breaking news for a given category with 5-minute caching.
        """
        cache_key = f"cat_{category}_{limit}"
        now = time.time()
        if cache_key in _NEWS_CACHE:
            ts, cached_data = _NEWS_CACHE[cache_key]
            if now - ts < CACHE_TTL and cached_data:
                return cached_data

        feed_url = CATEGORY_FEEDS.get(category.lower(), CATEGORY_FEEDS["top"])
        items = []

        try:
            resp = requests.get(feed_url, headers=HEADERS, timeout=7)
            if resp.status_code == 200:
                items = self._parse_rss_content(resp.content, category, limit=limit)
        except Exception as e:
            print(f"[NewsService] Primary feed request error: {e}")

        # Fallback to secondary feed if needed
        if not items and category.lower() in FALLBACK_FEEDS:
            try:
                resp = requests.get(FALLBACK_FEEDS[category.lower()], headers=HEADERS, timeout=7)
                if resp.status_code == 200:
                    items = self._parse_rss_content(resp.content, category, limit=limit)
            except Exception:
                pass

        if items:
            _NEWS_CACHE[cache_key] = (now, items)
        return items

    def search_news_chronological(self, query: str, limit: int = 6) -> List[Dict[str, Any]]:
        """
        Searches Google News for a specific query and returns chronological articles
        from earlier reports to today's breaking news.
        """
        cache_key = f"search_{query.lower()}_{limit}"
        now = time.time()
        if cache_key in _NEWS_CACHE:
            ts, cached_data = _NEWS_CACHE[cache_key]
            if now - ts < CACHE_TTL and cached_data:
                return cached_data

        encoded_q = urllib.parse.quote(query)
        search_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"
        items = []

        try:
            resp = requests.get(search_url, headers=HEADERS, timeout=8)
            if resp.status_code == 200:
                items = self._parse_rss_content(resp.content, "Search", limit=limit)
        except Exception as e:
            print(f"[NewsService] Search request error: {e}")

        # Sort chronologically (oldest to newest for progression, or keep timestamps visible)
        if items:
            _NEWS_CACHE[cache_key] = (now, items)
        return items

    def synthesize_chronological_news(
        self,
        query: str,
        gemini_service=None
    ) -> str:
        """
        Gathers live news for the user's query and produces a comprehensive
        chronological intelligence report (Old / Background -> Recent -> Today's Updates).
        """
        # 1. Fetch live articles
        items = self.search_news_chronological(query, limit=8)
        if not items:
            items = self.fetch_live_news("top", limit=5)

        if not items:
            return (
                f"### 📰 Live News Updates for: `{query}`\n\n"
                "Could not retrieve live news feeds at this moment. "
                "Please verify your internet connection or paste a specific news article URL."
            )

        # 2. Sort into chronological progression (oldest to today)
        items_sorted = sorted(items, key=lambda x: x.get("timestamp", 0))

        # Format context for synthesis
        articles_md = []
        for idx, it in enumerate(items_sorted, 1):
            t_str = it.get("time_ago", "Recent")
            articles_md.append(f"{idx}. [{t_str}] {it['title']} (Source: {it['source']}) - URL: {it['link']}")

        articles_text = "\n".join(articles_md)

        # 3. LLM Synthesis if available
        if gemini_service and gemini_service.is_configured:
            prompt = (
                f"You are Agent Rag, an executive news and intelligence analyst.\n\n"
                f"USER QUERY: {query}\n\n"
                f"VERIFIED REAL-TIME NEWS ARTICLES (Sorted chronologically from earlier to today):\n"
                f"{articles_text}\n\n"
                "INSTRUCTIONS:\n"
                "1. Provide a well-structured, objective, and comprehensive chronological news brief.\n"
                "2. Structure your response into:\n"
                "   - 📌 **Core Story & Background Context** (Earlier / ongoing context)\n"
                "   - 📈 **Key Developments & Progression**\n"
                "   - 🔴 **Today's Latest Updates & Breaking Points**\n"
                "3. Cite the exact news outlets (e.g. BBC, The Hindu, Reuters, Bloomberg) and provide markdown links to their reports.\n"
                "4. Be articulate, factual, and strictly grounded in the live articles provided above."
            )
            try:
                res = gemini_service.generate_chat_response(prompt)
                if res and len(res) > 50 and not res.startswith("⚠️"):
                    return res
            except Exception as e:
                print(f"[NewsService] Gemini synthesis error: {e}")

        # 4. Fallback structured presentation
        earlier_items = items_sorted[:len(items_sorted)//2] if len(items_sorted) > 2 else []
        latest_items = items_sorted[len(items_sorted)//2:] if len(items_sorted) > 2 else items_sorted

        earlier_md = []
        for it in earlier_items:
            earlier_md.append(f"- **[{it['time_ago']}]** [{it['title']}]({it['link']}) — *{it['source']}*")

        latest_md = []
        for it in latest_items:
            latest_md.append(f"- **[{it['time_ago']}]** 🔴 [{it['title']}]({it['link']}) — *{it['source']}*")

        resp = (
            f"### 📰 Real-Time News & Updates: `{query}`\n\n"
            f"Here is the chronological progression of news from background reports to today's latest updates:\n\n"
        )
        if earlier_md:
            resp += "#### 📌 **Earlier Background & Context**\n" + "\n".join(earlier_md) + "\n\n"

        resp += "#### 🔴 **Today's Latest Updates & Breaking Headlines**\n" + "\n".join(latest_md) + "\n\n"
        resp += "---\n💡 *Live real-time news intelligence powered by Agent Rag.*"
        return resp
