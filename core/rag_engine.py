"""
Multi-Modal RAG Engine
Handles text/PDF document chunking, ChromaDB vector indexing, and multimodal retrieval.
"""

import os
from typing import List, Dict, Any, Optional

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

class MultimodalRAGEngine:
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or os.path.join(os.path.dirname(__file__), "chroma_db")
        self.client = None
        self.collection = None

        if CHROMA_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(path=self.persist_dir)
                self.collection = self.client.get_or_create_collection(name="multimodal_rag")
            except Exception as e:
                print(f"[RAG Engine] ChromaDB init note: {e}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts text content from PDF file."""
        if not PYPDF_AVAILABLE:
            return "PyPDF is not installed. Unable to parse PDF content directly."

        try:
            reader = PdfReader(pdf_path)
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(f"--- Page {i + 1} ---\n{text}")
            return "\n\n".join(pages_text)
        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    def index_document(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        """Splits text and adds chunks into vector database."""
        if not self.collection:
            return

        # Simple character chunking
        chunk_size = 1000
        overlap = 150
        chunks = []
        metas = []
        ids = []

        start = 0
        chunk_idx = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            metas.append({**metadata, "chunk_index": chunk_idx})
            ids.append(f"{doc_id}_chunk_{chunk_idx}")
            start += (chunk_size - overlap)
            chunk_idx += 1

        if chunks:
            self.collection.add(
                documents=chunks,
                metadatas=metas,
                ids=ids
            )

    def retrieve_relevant_context(self, query: str, n_results: int = 3) -> str:
        """Searches vector DB for chunks matching the query."""
        if not self.collection:
            return ""

        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            docs = results.get("documents", [[]])[0]
            return "\n\n".join(docs)
        except Exception:
            return ""
