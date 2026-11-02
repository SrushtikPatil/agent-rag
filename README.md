# 🤖 Agent Rag — Personal Agentic Multi-Modal Assistant

An advanced, full-stack **Personal Agentic Multi-Modal RAG AI Companion** built with **Google Gemini**, **LangGraph**, **ChromaDB**, and **Streamlit**.

Agent Rag combines proactive personal memory (Indian festivals & birthday tracking), document & spreadsheet intelligence, multi-modal reasoning, and modern conversational abilities (including Gen-Z slang & meme decoding) wrapped in an executive light-blue interface.

---

## ✨ Key Features & Capabilities

### 1. 📄 Universal Multi-Modal Ingestion & Intelligence
* **PDFs & Documents**: Local, privacy-first text extraction using `pypdf`, bypassing API limits and injecting directly into Gemini's 1-million-token context window.
* **Excel & CSV (`.xlsx`, `.csv`)**: Deterministic analytical calculations, aggregations, and statistics using an integrated Pandas engine.
* **Images, Audio & Video**: Deep visual scene analysis, chart interpretation, audio transcription, and video reasoning.

### 2. 🪔 Proactive Indian Festival & Relationship Engine
* **Year-Accurate Indian Festival Radar (2024–2028)**: Tracks lunar and solar calendar dates for Diwali, Dhanteras, Bhai Dooj, Chhath Puja, Holi, Eid, Raksha Bandhan, etc.
* **Relationship Reminders**: Tracks friends and family birthdays/anniversaries with instant ready-made wishing quotes in multiple tones (Professional, Warm, Casual).
* **Dismissible Twin Notification Pills**: Side-by-side matching indicators on launch with 1-click dismiss (`✕`).

### 3. 🧠 General Intelligence & Gen-Z Slang Coach
* **Full-Spectrum Conversational AI**: Functions like ChatGPT and Gemini for coding, writing, research, and technical Q&A.
* **Gen-Z Slang & Meme Decoder**: Explains internet slang (`rizz`, `no cap`, `delulu`, `ate and left no crumbs`) and decodes counter-intuitive meme emojis (`💀`, `😭`, `🧢`, `🗿`, `💅`, `📉`).

### 4. ⚡ High-Quota Multi-Model Pooling & Failover
* Auto-cascading architecture (`gemini-flash-lite-latest`, `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, etc.) that automatically recovers from `429 ResourceExhausted` errors with zero user interruption.

### 5. 🎨 Executive Light Blue UI
* Modern interface with smooth light-blue accents (`#f0f7ff`), custom cute 3D robot avatar, collapsible left sidebar with bottom-pinned settings, and responsive layout.

---

## 🏛️ System Architecture

```text
                               ┌─────────────────────────────────┐
                               │   Streamlit Web Frontend (UI)   │
                               │    (Executive Light Blue UI)    │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │     Agentic Router Machine      │
                               └──────┬──────────┬──────────┬────┘
                                      │          │          │
                 ┌────────────────────┘          │          └─────────────────────┐
                 ▼                               ▼                                ▼
     ┌───────────────────────┐       ┌───────────────────────┐        ┌───────────────────────┐
     │   PERSONAL MEMORY     │       │     MULTIMODAL RAG    │        │    STRUCTURED DATA    │
     │   • Festival Radar    │       │   • Local pypdf       │        │   • Pandas Analysis   │
     │   • Birthday Tracker  │       │   • ChromaDB Vectors  │        │   • Excel / CSV Calc  │
     │   • SQLite Database   │       │   • Audio/Video/Image │        │   • Deterministic Q&A │
     └───────────┬───────────┘       └───────────┬───────────┘        └───────────┬───────────┘
                 │                               │                                │
                 └───────────────────────┬───────┴────────────────────────────────┘
                                         ▼
                         ┌───────────────────────────────┐
                         │   Gemini Multi-Model Pool     │
                         │  (Auto-Failover / High Quota) │
                         │ • gemini-flash-lite-latest    │
                         │ • gemini-3.1-flash-lite       │
                         │ • gemini-3.5-flash-lite       │
                         └───────────────────────────────┘
```

---

## 📁 Project Structure

```
multimodal_agentic_rag/
├── assets/
│   └── robot_avatar.png      # Custom cute 3D robot avatar badge
├── core/
│   ├── agentic_router.py     # Priority intent classifier & state machine
│   ├── gemini_service.py     # Unified Gemini client with model failover pool
│   ├── festival_calendar.py  # Year-accurate Indian festival engine (2024–2028)
│   ├── proactive_engine.py   # IST time-of-day greeting & relationship tracking
│   ├── rag_engine.py         # ChromaDB retrieval and local document extractor
│   └── excel_analyzer.py     # Deterministic spreadsheet analyst via Pandas
├── database/
│   ├── db.py                 # SQLite initialization & schema
│   └── models.py             # Data access layer for milestones & contacts
├── app.py                    # Streamlit application with custom styling
├── requirements.txt          # Python dependencies
├── .env.example              # Template for API credentials
└── README.md                 # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/agent-rag.git
cd agent-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure your API Key
Create a `.env` file in the root directory:
```bash
GEMINI_API_KEY=your_google_gemini_api_key
```
*(Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey))*

### 4. Launch Agent Rag
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🌐 Free Live Cloud Deployment (No Laptop Required!)

You can deploy this project live on the web for **free** using **Streamlit Community Cloud**:

1. Push this repository to your **GitHub** account.
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account.
3. Click **"New App"** → Select your `agent-rag` repository → Branch `main` → Main file `app.py`.
4. Under **Advanced Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key"
   ```
5. Click **Deploy!**

Within 2 minutes, you will receive a permanent live URL (e.g. `https://agent-rag.streamlit.app`) that can be opened on **any phone, tablet, or classroom computer** to showcase live to your teacher!
