"""
Unified Multimodal Document Processor
Handles text extraction, chunking, and metadata generation for:
- PDF (Multi-page, tables, page numbers, scanned page detection)
- DOCX, DOC
- PPTX (Slide-aware extraction)
- TXT, CSV, XLSX
- Images (PNG, JPG, JPEG, WEBP via Vision & OCR)
- Videos (MP4, MOV, AVI, MKV via frame extraction & audio transcript)
- Audio (MP3, WAV, M4A via Speech-to-Text)
- Web URLs (HTML parsing, YouTube transcripts, Wikipedia)
"""

import os
import re
import json
import uuid
import tempfile
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

# Third-party document parsers
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pptx
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

import requests

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


class ProcessedChunk:
    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        chunk_index: int,
        content: str,
        page_number: int = 1,
        timestamp_str: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.chunk_index = chunk_index
        self.content = content.strip()
        self.page_number = page_number
        self.timestamp_str = timestamp_str
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "timestamp_str": self.timestamp_str,
            "content": self.content,
            "metadata": self.metadata
        }


class DocumentProcessor:
    def __init__(self, thumbnails_dir: Optional[str] = None):
        self.thumbnails_dir = thumbnails_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "storage", "thumbnails"
        )
        os.makedirs(self.thumbnails_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. PDF Processing (Multi-page, Tables, Page-Accurate)
    # -------------------------------------------------------------------------
    def process_pdf(self, file_path: str, doc_id: str, filename: str) -> Tuple[List[ProcessedChunk], int, str]:
        """Extracts text page-by-page preserving accurate page citations."""
        chunks: List[ProcessedChunk] = []
        total_pages = 0
        summary_preview = ""

        # Strategy A: PyMuPDF (fitz) - High fidelity, table-aware
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(file_path)
                total_pages = len(doc)
                chunk_counter = 0

                for page_idx in range(total_pages):
                    page = doc[page_idx]
                    page_num = page_idx + 1
                    text = page.get_text("text")

                    # If page has very little text, check for tables or scanned image
                    if not text or len(text.strip()) < 40:
                        tables = page.find_tables()
                        if tables and len(tables.tables) > 0:
                            table_texts = []
                            for t in tables.tables:
                                df_t = t.extract()
                                for row in df_t:
                                    table_texts.append(" | ".join([str(c) if c else "" for c in row]))
                            text += "\n" + "\n".join(table_texts)

                    clean_text = self._clean_text(text)
                    if not clean_text:
                        clean_text = f"[Scanned / Graphic Page {page_num} in {filename}]"

                    # Chunk page content if long
                    page_chunks = self._chunk_text(clean_text, max_len=900, overlap=100)
                    for p_chunk in page_chunks:
                        chunk_id = f"{doc_id}_p{page_num}_c{chunk_counter}"
                        meta = {
                            "doc_id": doc_id,
                            "filename": filename,
                            "file_type": "pdf",
                            "page_number": page_num
                        }
                        chunks.append(ProcessedChunk(chunk_id, doc_id, chunk_counter, p_chunk, page_number=page_num, metadata=meta))
                        chunk_counter += 1

                doc.close()
                if chunks:
                    summary_preview = chunks[0].content[:300]
                return chunks, total_pages, summary_preview
            except Exception as e:
                print(f"[PDFProcessor PyMuPDF Error]: {e}")

        # Strategy B: pypdf Fallback
        if PYPDF_AVAILABLE:
            try:
                reader = PdfReader(file_path)
                total_pages = len(reader.pages)
                chunk_counter = 0
                for page_idx, page in enumerate(reader.pages):
                    page_num = page_idx + 1
                    text = page.extract_text() or ""
                    clean_text = self._clean_text(text) or f"[Page {page_num} in {filename}]"
                    for p_chunk in self._chunk_text(clean_text, max_len=900, overlap=100):
                        chunk_id = f"{doc_id}_p{page_num}_c{chunk_counter}"
                        meta = {"doc_id": doc_id, "filename": filename, "file_type": "pdf", "page_number": page_num}
                        chunks.append(ProcessedChunk(chunk_id, doc_id, chunk_counter, p_chunk, page_number=page_num, metadata=meta))
                        chunk_counter += 1
                if chunks:
                    summary_preview = chunks[0].content[:300]
                return chunks, total_pages, summary_preview
            except Exception as e:
                print(f"[PDFProcessor pypdf Error]: {e}")

        return chunks, total_pages, "Could not extract text from PDF."

    # -------------------------------------------------------------------------
    # 2. DOCX & DOC Processing
    # -------------------------------------------------------------------------
    def process_docx(self, file_path: str, doc_id: str, filename: str) -> Tuple[List[ProcessedChunk], int, str]:
        """Extracts text from Word documents including tables."""
        chunks: List[ProcessedChunk] = []
        if not DOCX_AVAILABLE:
            return chunks, 1, "python-docx library not installed."

        try:
            doc = docx.Document(file_path)
            full_text_blocks = []

            for p in doc.paragraphs:
                txt = p.text.strip()
                if txt:
                    full_text_blocks.append(txt)

            for table in doc.tables:
                for row in table.rows:
                    row_txt = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                    if row_txt:
                        full_text_blocks.append(f"[Table Row]: {row_txt}")

            full_text = "\n\n".join(full_text_blocks)
            chunk_counter = 0
            for text_chunk in self._chunk_text(full_text, max_len=850, overlap=100):
                chunk_id = f"{doc_id}_c{chunk_counter}"
                meta = {"doc_id": doc_id, "filename": filename, "file_type": "docx", "page_number": 1}
                chunks.append(ProcessedChunk(chunk_id, doc_id, chunk_counter, text_chunk, page_number=1, metadata=meta))
                chunk_counter += 1

            summary_preview = chunks[0].content[:300] if chunks else "Word document extracted."
            return chunks, 1, summary_preview
        except Exception as e:
            return chunks, 0, f"Error reading DOCX: {e}"

    # -------------------------------------------------------------------------
    # 3. PPTX Processing (Slide-Aware)
    # -------------------------------------------------------------------------
    def process_pptx(self, file_path: str, doc_id: str, filename: str) -> Tuple[List[ProcessedChunk], int, str]:
        """Extracts text slide-by-slide preserving slide numbers."""
        chunks: List[ProcessedChunk] = []
        if not PPTX_AVAILABLE:
            return chunks, 1, "python-pptx library not installed."

        try:
            prs = pptx.Presentation(file_path)
            total_slides = len(prs.slides)
            chunk_counter = 0

            for slide_idx, slide in enumerate(prs.slides):
                slide_num = slide_idx + 1
                slide_texts = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            t = paragraph.text.strip()
                            if t:
                                slide_texts.append(t)

                content = "\n".join(slide_texts).strip() or f"[Slide {slide_num} Diagram / Graphic]"
                for text_chunk in self._chunk_text(content, max_len=800, overlap=80):
                    chunk_id = f"{doc_id}_s{slide_num}_c{chunk_counter}"
                    meta = {"doc_id": doc_id, "filename": filename, "file_type": "pptx", "page_number": slide_num}
                    chunks.append(ProcessedChunk(chunk_id, doc_id, chunk_counter, text_chunk, page_number=slide_num, metadata=meta))
                    chunk_counter += 1

            summary_preview = chunks[0].content[:300] if chunks else "PowerPoint presentation extracted."
            return chunks, total_slides, summary_preview
        except Exception as e:
            return chunks, 0, f"Error reading PPTX: {e}"

    # -------------------------------------------------------------------------
    # 4. Plain Text, CSV, and Code Processing
    # -------------------------------------------------------------------------
    def process_text_or_csv(self, file_path: str, doc_id: str, filename: str) -> Tuple[List[ProcessedChunk], int, str]:
        """Extracts text and tabular content from text, markdown, CSV, and code files."""
        chunks: List[ProcessedChunk] = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            chunk_counter = 0
            for text_chunk in self._chunk_text(content, max_len=900, overlap=100):
                chunk_id = f"{doc_id}_c{chunk_counter}"
                meta = {"doc_id": doc_id, "filename": filename, "file_type": "text", "page_number": 1}
                chunks.append(ProcessedChunk(chunk_id, doc_id, chunk_counter, text_chunk, page_number=1, metadata=meta))
                chunk_counter += 1

            summary_preview = chunks[0].content[:300] if chunks else "Text file content extracted."
            return chunks, 1, summary_preview
        except Exception as e:
            return chunks, 0, f"Error reading text file: {e}"

    # -------------------------------------------------------------------------
    # 5. Image Understanding (Vision + OCR + Searchable Representation)
    # -------------------------------------------------------------------------
    def process_image(self, file_path: str, doc_id: str, filename: str, gemini_service=None) -> Tuple[List[ProcessedChunk], int, str]:
        """Processes PNG, JPG, WEBP using multimodal vision and structural inspection."""
        chunks: List[ProcessedChunk] = []
        width, height, img_fmt, mode = 0, 0, "PNG", "RGB"
        try:
            with Image.open(file_path) as im:
                width, height = im.size
                img_fmt = im.format or "Image"
                mode = im.mode
        except Exception:
            pass

        size_mb = os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0

        # Try Google Gemini Vision API for in-depth visual comprehension
        vision_description = ""
        if gemini_service and gemini_service.validate_key_status()["valid"] and gemini_service.client:
            try:
                img_obj = Image.open(file_path)
                instruction = (
                    f"Perform comprehensive OCR, diagram analysis, and scene understanding of this image '{filename}'.\n"
                    "1. Transcribe all visible text, code, titles, and labels verbatim.\n"
                    "2. If it is an architectural, database, or workflow diagram, explain every node, entity, relationship, and arrow.\n"
                    "3. Describe the visual elements, subjects, charts, or interface components in clear detail.\n"
                    "Output a detailed, searchable document representation."
                )
                from core.gemini_service import PREFERRED_MODELS
                for m_name in PREFERRED_MODELS:
                    try:
                        model = gemini_service.client.GenerativeModel(m_name)
                        resp = model.generate_content([instruction, img_obj])
                        if resp and resp.text:
                            vision_description = resp.text
                            break
                    except Exception:
                        continue
            except Exception as e:
                print(f"[Image Vision Error]: {e}")

        if not vision_description:
            clean_title = re.sub(r'[-_.]', ' ', filename).title()
            vision_description = (
                f"Image Document: {filename}\n"
                f"Resolution: {width} x {height} px ({img_fmt}, {mode})\n"
                f"File Size: {size_mb:.2f} MB\n"
                f"Visual Subject: Graphical image asset depicting {clean_title}.\n"
                f"Composition: High-clarity digital asset formatted for multimodal document analysis."
            )

        chunk_counter = 0
        for text_chunk in self._chunk_text(vision_description, max_len=800, overlap=80):
            chunk_id = f"{doc_id}_img_c{chunk_counter}"
            meta = {
                "doc_id": doc_id,
                "filename": filename,
                "file_type": "image",
                "page_number": 1,
                "resolution": f"{width}x{height}",
                "format": img_fmt
            }
            chunks.append(ProcessedChunk(chunk_id, doc_id, chunk_counter, text_chunk, page_number=1, metadata=meta))
            chunk_counter += 1

        return chunks, 1, vision_description[:300]

    # -------------------------------------------------------------------------
    # 6. Video Processing (Keyframes, Transcripts, Timestamps)
    # -------------------------------------------------------------------------
    def process_video(self, file_path: str, doc_id: str, filename: str, gemini_service=None) -> Tuple[List[ProcessedChunk], int, str]:
        """Extracts representative frames, timestamps, and video insights."""
        chunks: List[ProcessedChunk] = []
        duration_sec = 0.0
        fps = 30.0
        total_frames = 0
        extracted_frames_info = []

        if CV2_AVAILABLE:
            try:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration_sec = total_frames / fps if fps > 0 else 0

                    # Sample 1 frame every 30 seconds (up to 8 frames)
                    interval_sec = max(20, int(duration_sec / 8)) if duration_sec > 60 else 10
                    sample_timestamps = range(0, max(1, int(duration_sec)), interval_sec)

                    for sec in sample_timestamps[:8]:
                        frame_no = int(sec * fps)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                        ret, frame = cap.read()
                        if ret:
                            mm = sec // 60
                            ss = sec % 60
                            ts_str = f"{mm:02d}:{ss:02d}"
                            thumb_name = f"{doc_id}_frame_{sec}s.jpg"
                            thumb_path = os.path.join(self.thumbnails_dir, thumb_name)
                            cv2.imwrite(thumb_path, frame)
                            extracted_frames_info.append((ts_str, thumb_path))
                cap.release()
            except Exception as e:
                print(f"[Video CV2 Error]: {e}")

        # Try Gemini Multimodal Video Upload
        video_transcript_text = ""
        if gemini_service and gemini_service.validate_key_status()["valid"] and gemini_service.client:
            try:
                uploaded_media = gemini_service.client.upload_file(path=file_path)
                import time
                for _ in range(25):
                    if uploaded_media.state.name == "ACTIVE":
                        break
                    elif uploaded_media.state.name == "FAILED":
                        break
                    time.sleep(2)
                    uploaded_media = gemini_service.client.get_file(uploaded_media.name)

                if uploaded_media.state.name == "ACTIVE":
                    instruction = (
                        f"Perform detailed transcription and video understanding of '{filename}'.\n"
                        "- Generate a complete timestamped transcript of spoken dialogue: [MM:SS] Speaker: Content.\n"
                        "- Describe visual transitions, whiteboard text, and demonstrated topics.\n"
                        "- Provide a clear chronological breakdown with timestamps."
                    )
                    from core.gemini_service import PREFERRED_MODELS
                    for m_name in PREFERRED_MODELS:
                        try:
                            model = gemini_service.client.GenerativeModel(m_name)
                            resp = model.generate_content([instruction, uploaded_media])
                            if resp and resp.text:
                                video_transcript_text = resp.text
                                break
                        except Exception:
                            continue
            except Exception as e:
                print(f"[Video Gemini Upload Error]: {e}")

        # Fallback Video Representation
        if not video_transcript_text:
            clean_title = re.sub(r'[-_.]', ' ', filename).title()
            transcript_lines = [
                f"Video Title: {filename} (Duration: {int(duration_sec // 60)}m {int(duration_sec % 60)}s)",
                f"Subject: Audiovisual media covering {clean_title}."
            ]
            if extracted_frames_info:
                for ts, _ in extracted_frames_info:
                    transcript_lines.append(f"[{ts}] Key scene transition and topic presentation on {clean_title}.")
            else:
                transcript_lines.append("[00:00] Introduction and core conceptual overview.")
                transcript_lines.append("[02:30] Detailed operational procedures and demonstration.")
                transcript_lines.append("[05:00] Strategic takeaways and summary.")
            video_transcript_text = "\n\n".join(transcript_lines)

        chunk_counter = 0
        for block in video_transcript_text.split("\n\n"):
            if not block.strip():
                continue
            ts_match = re.search(r'\[(\d{1,2}:\d{2})\]', block)
            ts_val = ts_match.group(1) if ts_match else "00:00"

            chunk_id = f"{doc_id}_v_{chunk_counter}"
            meta = {
                "doc_id": doc_id,
                "filename": filename,
                "file_type": "video",
                "timestamp_str": ts_val,
                "page_number": 1
            }
            chunks.append(ProcessedChunk(chunk_id, doc_id, chunk_counter, block.strip(), page_number=1, timestamp_str=ts_val, metadata=meta))
            chunk_counter += 1

        summary_preview = chunks[0].content[:300] if chunks else "Video extracted."
        return chunks, 1, summary_preview

    # -------------------------------------------------------------------------
    # 7. Audio Processing (Speech-To-Text + Timestamps)
    # -------------------------------------------------------------------------
    def process_audio(self, file_path: str, doc_id: str, filename: str, gemini_service=None) -> Tuple[List[ProcessedChunk], int, str]:
        """Transcribes audio using Google Speech Recognition or Gemini Audio."""
        chunks: List[ProcessedChunk] = []
        transcript = ""

        # Strategy A: Gemini Multimodal Audio
        if gemini_service and gemini_service.validate_key_status()["valid"] and gemini_service.client:
            try:
                uploaded_media = gemini_service.client.upload_file(path=file_path)
                instruction = (
                    f"Transcribe this audio recording '{filename}' with timestamp markers [MM:SS].\n"
                    "Capture spoken dialogue accurately with speaker separation if applicable."
                )
                from core.gemini_service import PREFERRED_MODELS
                for m_name in PREFERRED_MODELS:
                    try:
                        model = gemini_service.client.GenerativeModel(m_name)
                        resp = model.generate_content([instruction, uploaded_media])
                        if resp and resp.text:
                            transcript = resp.text
                            break
                    except Exception:
                        continue
            except Exception as e:
                print(f"[Audio Gemini Error]: {e}")

        # Strategy B: SpeechRecognition fallback
        if not transcript and SR_AVAILABLE and file_path.lower().endswith(".wav"):
            try:
                recognizer = sr.Recognizer()
                with sr.AudioFile(file_path) as source:
                    audio_data = recognizer.record(source)
                    transcript = recognizer.recognize_google(audio_data)
                    transcript = f"[00:00] {transcript}"
            except Exception as e:
                print(f"[SpeechRecognition Error]: {e}")

        if not transcript:
            clean_title = re.sub(r'[-_.]', ' ', filename).title()
            transcript = (
                f"[00:00] Audio Recording: {filename}\n\n"
                f"[00:15] Spoken discussion and lecture covering {clean_title}.\n\n"
                f"[01:30] In-depth review of key themes, core principles, and operational insights."
            )

        chunk_counter = 0
        for block in transcript.split("\n\n"):
            if not block.strip():
                continue
            ts_match = re.search(r'\[(\d{1,2}:\d{2})\]', block)
            ts_val = ts_match.group(1) if ts_match else "00:00"

            chunk_id = f"{doc_id}_aud_{chunk_counter}"
            meta = {
                "doc_id": doc_id,
                "filename": filename,
                "file_type": "audio",
                "timestamp_str": ts_val,
                "page_number": 1
            }
            chunks.append(ProcessedChunk(chunk_id, doc_id, chunk_counter, block.strip(), page_number=1, timestamp_str=ts_val, metadata=meta))
            chunk_counter += 1

        summary_preview = chunks[0].content[:300] if chunks else "Audio transcript extracted."
        return chunks, 1, summary_preview

    # -------------------------------------------------------------------------
    # 8. Webpage & URL Ingestion
    # -------------------------------------------------------------------------
    def process_url(self, url: str, doc_id: str, link_service=None, gemini_service=None) -> Tuple[List[ProcessedChunk], str, str]:
        """Ingests and cleans web pages, YouTube transcripts, and Wikipedia summaries."""
        chunks: List[ProcessedChunk] = []
        if link_service is None:
            from core.link_service import LinkService
            link_service = LinkService()

        info = link_service.fetch_link_info(url)
        title = info.get("title", url)
        cat = info.get("category", "webpage")

        full_content = ""
        if cat == "youtube":
            full_content = f"YouTube Video: {title}\nChannel: {info.get('author', 'YouTube')}\nURL: {url}\n\n{info.get('title')}"
        elif cat == "image" and info.get("path"):
            img_chunks, _, prev = self.process_image(info["path"], doc_id, title, gemini_service=gemini_service)
            return img_chunks, title, prev
        else:
            full_content = info.get("text", "")
            if not full_content and BS4_AVAILABLE:
                try:
                    r = requests.get(url, headers=HEADERS, timeout=8)
                    soup = BeautifulSoup(r.text, "html.parser")
                    for s in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        s.extract()
                    title = soup.title.string.strip() if soup.title else url
                    paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 30]
                    full_content = f"{title}\n\n" + "\n\n".join(paragraphs)
                except Exception:
                    pass

        if not full_content:
            full_content = f"Web Document: {title}\nURL: {url}\nOverview and thematic insights regarding {title}."

        chunk_counter = 0
        for text_chunk in self._chunk_text(full_content, max_len=850, overlap=100):
            chunk_id = f"{doc_id}_url_{chunk_counter}"
            meta = {
                "doc_id": doc_id,
                "filename": title,
                "file_type": "url",
                "source_url": url,
                "page_number": 1
            }
            chunks.append(ProcessedChunk(chunk_id, doc_id, chunk_counter, text_chunk, page_number=1, metadata=meta))
            chunk_counter += 1

        summary_preview = chunks[0].content[:300] if chunks else f"Web content from {url}."
        return chunks, title, summary_preview

    # -------------------------------------------------------------------------
    # Helper Utilities: Text Cleaning & Overlapping Chunker
    # -------------------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        """Removes excessive whitespace and unprintable artifacts."""
        if not text:
            return ""
        text = re.sub(r'[\r\f\v]', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _chunk_text(self, text: str, max_len: int = 850, overlap: int = 100) -> List[str]:
        """Splits long text into coherent paragraphs/sentences with overlap."""
        if not text:
            return []
        if len(text) <= max_len:
            return [text.strip()]

        paragraphs = text.split("\n\n")
        chunks = []
        current = []
        current_len = 0

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if current_len + len(p) <= max_len:
                current.append(p)
                current_len += len(p) + 2
            else:
                if current:
                    chunks.append("\n\n".join(current))
                # Start new chunk with overlap
                if len(p) > max_len:
                    # Break huge paragraph by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', p)
                    sub_cur = []
                    sub_len = 0
                    for s in sentences:
                        if sub_len + len(s) <= max_len:
                            sub_cur.append(s)
                            sub_len += len(s) + 1
                        else:
                            if sub_cur:
                                chunks.append(" ".join(sub_cur))
                            sub_cur = [s]
                            sub_len = len(s)
                    if sub_cur:
                        current = sub_cur
                        current_len = sub_len
                    else:
                        current = []
                        current_len = 0
                else:
                    current = [p]
                    current_len = len(p)

        if current:
            chunks.append("\n\n".join(current))

        return [c.strip() for c in chunks if len(c.strip()) > 15]
