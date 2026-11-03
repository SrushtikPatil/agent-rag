"""
Unified Gemini Service Wrapper
Handles multimodal processing (Images, Audio, Video, PDFs) and router classification.
Features robust multi-model pooling & auto-failover:
If a model hits a 429 quota or 404, it immediately cascades to the next available model
(gemini-flash-lite-latest, gemini-3.1-flash-lite, etc.) with zero downtime.
"""

import os
from typing import Optional, List, Dict, Any
from PIL import Image

DUMMY_KEYS = [
    "aizasyyouractualkeyhere",
    "your_key",
    "your_api_key",
    "placeholder",
    "youractualkeyhere",
    "enter_your_api_key"
]

# Robust pool of high-quota models
PREFERRED_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest"
]

class GeminiService:
    def __init__(self, api_key: Optional[str] = None):
        raw_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        self.api_key = raw_key.strip()

        is_dummy = any(d in self.api_key.lower() for d in DUMMY_KEYS)
        self.is_configured = bool(self.api_key and len(self.api_key) > 20 and not is_dummy)
        self.client = None
        self.active_model_name = "gemini-flash-lite-latest"

        if self.is_configured:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai
            except Exception as e:
                print(f"[GeminiService] Initialization note: {e}")
                self.is_configured = False

    def validate_key_status(self) -> Dict[str, Any]:
        """Returns the status and friendly message for the API key."""
        if not self.api_key:
            return {
                "valid": False,
                "status": "missing",
                "message": "No API key entered. Operating in preview mode."
            }
        
        if any(d in self.api_key.lower() for d in DUMMY_KEYS):
            return {
                "valid": False,
                "status": "placeholder",
                "message": "The example key ('AIzaSyYourActualKeyHere') is currently entered. Please replace it with your personal key."
            }

        if len(self.api_key) < 20:
            return {
                "valid": False,
                "status": "invalid_length",
                "message": "The API key seems too short."
            }

        return {
            "valid": True,
            "status": "configured",
            "message": "API key configured."
        }

    def _format_error(self, error_obj: Exception) -> str:
        """Translates unhandled API errors into clear advice."""
        err_str = str(error_obj)
        if "API_KEY_INVALID" in err_str or "API key not valid" in err_str:
            return (
                "❌ **Invalid Gemini API Key**\n\n"
                "Google rejected this API key. Please check your key at [Google AI Studio](https://aistudio.google.com/app/apikey) "
                "and paste it into the sidebar."
            )
        elif "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
            return (
                "⚠️ **All Gemini Free-Tier Quotas Exceeded (429)**\n\n"
                "You have hit the hourly request threshold across free-tier models. Please wait 60 seconds and try again!"
            )
        else:
            return f"⚠️ **Gemini Notice**: {err_str}"

    def generate_chat_response(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Generates response with automatic model cascading.
        If a model returns 404 or 429 (quota exceeded), it immediately tries the next model!
        """
        key_status = self.validate_key_status()
        if not key_status["valid"]:
            return (
                f"🤖 **[Preview Mode]**\n\n"
                f"{key_status['message']}\n\n"
                f"**Your Query:** *\"{prompt}\"*\n\n"
                "👉 *Please paste your real Gemini API key in the left sidebar to get live AI responses!*"
            )

        last_error = None
        for m_name in PREFERRED_MODELS:
            try:
                model = self.client.GenerativeModel(
                    model_name=m_name,
                    system_instruction=system_instruction
                )
                response = model.generate_content(prompt)
                self.active_model_name = m_name
                return response.text
            except Exception as e:
                last_error = e
                err_str = str(e)
                # If 404 (not found) or 429 (quota limit), smoothly cascade to the next model!
                if "404" in err_str or "not found" in err_str.lower() or "429" in err_str or "quota" in err_str.lower() or "ResourceExhausted" in err_str:
                    continue
                else:
                    return self._format_error(e)

        return self._format_error(last_error or Exception("Unable to connect to any Gemini models."))

    def analyze_multimodal_file(self, file_path: str, file_type: str, user_prompt: str) -> str:
        """
        Multimodal QA:
        - PDF: Extracted locally with pypdf and sent with cascading models.
        - Images: Processed via PIL.Image with model cascade.
        - Audio/Video: Handled with File API or fallback.
        """
        key_status = self.validate_key_status()
        if not key_status["valid"]:
            return (
                f"📁 **[Preview Mode - Multimodal Handler]**\n\n"
                f"File: `{os.path.basename(file_path)}` ({file_type.upper()})\n\n"
                f"{key_status['message']}\n\n"
                "👉 *To process and analyze this document with Gemini, please enter your valid API key in the sidebar.*"
            )

        # 1. PDF Handler (Local parsing - No Discovery API failure)
        if file_type == "pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                total_pages = len(reader.pages)
                pages_text = []

                for idx, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        pages_text.append(f"--- [Page {idx + 1}] ---\n{text.strip()}")

                full_text = "\n\n".join(pages_text)

                if not full_text.strip():
                    return (
                        f"⚠️ **Scanned or Image-Only PDF Detected:** `{os.path.basename(file_path)}` has {total_pages} pages, "
                        "but contains scanned images rather than selectable text."
                    )

                # Google Gemini 1.5 & Flash support a massive 1,000,000-token context window (~3.5M characters).
                # Support 400-500+ pages of full academic/book text in a single prompt!
                max_chars = 1500000
                doc_sample = full_text[:max_chars]
                doc_note = f" (Full {total_pages} pages processed)" if len(full_text) <= max_chars else f" (First {max_chars} characters of {total_pages} pages)"

                prompt = (
                    f"You are an expert academic and professional document analyst examining '{os.path.basename(file_path)}'{doc_note}.\n\n"
                    f"Total Pages: {total_pages}\n\n"
                    f"DOCUMENT CONTENT EXTRACT:\n"
                    f"{doc_sample}\n\n"
                    f"USER REQUEST: {user_prompt}\n\n"
                    "INSTRUCTIONS:\n"
                    "- Ground your answers strictly on the extracted document content above.\n"
                    "- Cite specific pages or sections where applicable.\n"
                    "- Format questions, summaries, or insights clearly with markdown headers and bullet points."
                )
                return self.generate_chat_response(prompt)

            except Exception as e:
                return f"⚠️ Error extracting PDF content: {str(e)}"

        # 2. Image Handler with Model Cascade
        elif file_type == "image":
            for m_name in PREFERRED_MODELS:
                try:
                    model = self.client.GenerativeModel(m_name)
                    img = Image.open(file_path)
                    response = model.generate_content([user_prompt, img])
                    return response.text
                except Exception as e:
                    err_str = str(e)
                    if "404" in err_str or "not found" in err_str.lower() or "429" in err_str or "quota" in err_str.lower():
                        continue
                    return self._format_error(e)

        # 3. Audio / Video Handler
        elif file_type in ["audio", "video"]:
            try:
                uploaded_file = self.client.upload_file(path=file_path)
                # Wait for video processing if video
                if file_type == "video":
                    import time
                    for _ in range(30):
                        if uploaded_file.state.name == "ACTIVE":
                            break
                        elif uploaded_file.state.name == "FAILED":
                            return "⚠️ Google servers could not process this video file. Please ensure it has standard H.264/AAC encoding."
                        time.sleep(2)
                        uploaded_file = self.client.get_file(uploaded_file.name)
            except Exception as e:
                return self._format_error(e)

            instruction = (
                f"You are an expert video and multimedia analyst examining '{os.path.basename(file_path)}'.\n\n"
                f"USER REQUEST: {user_prompt}\n\n"
                "INSTRUCTIONS:\n"
                "- Directly answer the user request based on the actual audio and visual content of the media.\n"
                "- Provide clear, structured bullet points and cite relevant timestamps [MM:SS] where applicable.\n"
                "- Be comprehensive, insightful, and concise."
            )

            for m_name in PREFERRED_MODELS:
                try:
                    model = self.client.GenerativeModel(m_name)
                    response = model.generate_content([instruction, uploaded_file])
                    return response.text
                except Exception as e:
                    err_str = str(e)
                    if "404" in err_str or "429" in err_str or "quota" in err_str.lower():
                        continue
                    return self._format_error(e)

        # 4. Text / Code Handler
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                prompt = f"File Content for '{os.path.basename(file_path)}':\n{content[:200000]}\n\nRequest: {user_prompt}"
                return self.generate_chat_response(prompt)
            except Exception as e:
                return self._format_error(e)

    def transcribe_audio_bytes(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
        """
        Transcribes audio bytes to text using a robust multi-tier pipeline:
        1. Primary: Google Gemini Multimodal Audio (high accuracy, context-aware).
        2. Fallback: Google Speech Recognition (free, zero-config, works without API key).
        """
        if not audio_bytes or len(audio_bytes) < 100:
            return ""

        # Strategy 1: Google Gemini Multimodal (if configured)
        if self.is_configured and self.client:
            for m_name in PREFERRED_MODELS:
                try:
                    model = self.client.GenerativeModel(m_name)
                    audio_part = {
                        "mime_type": mime_type,
                        "data": audio_bytes
                    }
                    prompt = (
                        "Accurately transcribe the spoken words in this audio into clean text. "
                        "Return ONLY the exact transcribed words, with proper punctuation, without any introductory phrases, filler, notes, or quotes."
                    )
                    response = model.generate_content([prompt, audio_part])
                    if response and response.text:
                        cleaned = response.text.strip().strip('"').strip("'")
                        if cleaned:
                            return cleaned
                except Exception as e:
                    err_str = str(e)
                    if "404" in err_str or "429" in err_str or "quota" in err_str.lower():
                        continue
                    # Break out to speech_recognition fallback on key or format failure
                    break

        # Strategy 2: Google Speech Recognition Fallback (Free, zero-config, no API key needed)
        try:
            import speech_recognition as sr
            import io
            r = sr.Recognizer()
            with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data)
                if text and text.strip():
                    return text.strip()
        except Exception:
            # Silence detected, background noise, or no recognizable words
            pass

        return ""
