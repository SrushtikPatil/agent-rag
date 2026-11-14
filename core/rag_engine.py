"""
Multi-Modal RAG Engine & Document Ingestion Pipeline
Handles end-to-end processing, granular status tracking, ChromaDB vector indexing,
hybrid retrieval (semantic + lexical), and grounded QA with exact source citations.
"""

import os
import re
import json
import uuid
import shutil
import hashlib
from typing import List, Dict, Any, Optional, Tuple

from core.document_processor import DocumentProcessor, ProcessedChunk
from database.models import (
    create_document,
    update_document_status,
    get_document,
    list_documents,
    delete_document as db_delete_document,
    insert_document_chunks,
    get_document_chunks,
    search_chunks_lexical
)

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class MultimodalRAGEngine:
    def __init__(self, persist_dir: Optional[str] = None, storage_dir: Optional[str] = None):
        self.persist_dir = persist_dir or os.path.join(os.path.dirname(__file__), "chroma_db")
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "storage", "documents"
        )
        os.makedirs(self.persist_dir, exist_ok=True)
        os.makedirs(self.storage_dir, exist_ok=True)

        self.processor = DocumentProcessor()
        self.client = None
        self.collection = None

        if CHROMA_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(path=self.persist_dir)
                self.collection = self.client.get_or_create_collection(
                    name="multimodal_rag",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                print(f"[RAG Engine] ChromaDB Init Warning: {e}")

    # -------------------------------------------------------------------------
    # Ingestion Pipeline with Granular Status Updates
    # -------------------------------------------------------------------------
    def ingest_file(
        self,
        src_file_path: str,
        filename: str,
        doc_id: Optional[str] = None,
        gemini_service=None
    ) -> Dict[str, Any]:
        """
        Executes complete ingestion pipeline:
        1. Uploading -> Uploaded (Copy to persistent storage, generate unique doc_id)
        2. Processing -> Extracting Content (Multi-page, OCR, vision, STT)
        3. Generating Embeddings -> Indexing (Add chunks to ChromaDB & SQLite)
        4. Ready (or Failed with exact error message)
        """
        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:10]}"
        file_ext = filename.split(".")[-1].lower() if "." in filename else ""
        file_size = os.path.getsize(src_file_path) if os.path.exists(src_file_path) else 0

        # Categorize file type
        image_exts = ["png", "jpg", "jpeg", "webp", "bmp", "gif", "svg", "jfif", "tiff", "ico"]
        video_exts = ["mp4", "mov", "avi", "mkv", "webm"]
        audio_exts = ["mp3", "wav", "m4a", "aac", "flac"]
        excel_exts = ["xlsx", "xls", "csv"]

        if file_ext == "pdf":
            file_type = "pdf"
        elif file_ext in ["docx", "doc"]:
            file_type = "docx"
        elif file_ext in ["pptx", "ppt"]:
            file_type = "pptx"
        elif file_ext in excel_exts:
            file_type = "excel"
        elif file_ext in image_exts:
            file_type = "image"
        elif file_ext in video_exts:
            file_type = "video"
        elif file_ext in audio_exts:
            file_type = "audio"
        else:
            file_type = "text"

        # 1. Step 1: Uploading & Storing
        perm_path = os.path.join(self.storage_dir, f"{doc_id}_{filename}")
        try:
            if src_file_path != perm_path:
                shutil.copy2(src_file_path, perm_path)
        except Exception:
            perm_path = src_file_path

        create_document(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            file_size_bytes=file_size,
            storage_path=perm_path,
            source_url=""
        )
        update_document_status(doc_id, processing_status="Processing", extraction_status="In Progress")

        # 2. Step 2: Content Extraction
        chunks: List[ProcessedChunk] = []
        page_count = 1
        summary_preview = ""

        try:
            update_document_status(doc_id, processing_status="Extracting Content")
            if file_type == "pdf":
                chunks, page_count, summary_preview = self.processor.process_pdf(perm_path, doc_id, filename)
            elif file_type == "docx":
                chunks, page_count, summary_preview = self.processor.process_docx(perm_path, doc_id, filename)
            elif file_type == "pptx":
                chunks, page_count, summary_preview = self.processor.process_pptx(perm_path, doc_id, filename)
            elif file_type == "image":
                chunks, page_count, summary_preview = self.processor.process_image(perm_path, doc_id, filename, gemini_service=gemini_service)
            elif file_type == "video":
                chunks, page_count, summary_preview = self.processor.process_video(perm_path, doc_id, filename, gemini_service=gemini_service)
            elif file_type == "audio":
                chunks, page_count, summary_preview = self.processor.process_audio(perm_path, doc_id, filename, gemini_service=gemini_service)
            else:
                chunks, page_count, summary_preview = self.processor.process_text_or_csv(perm_path, doc_id, filename)

            if not chunks:
                raise ValueError("No extractable content found in file.")

            update_document_status(
                doc_id,
                extraction_status="Completed",
                page_count=page_count,
                chunk_count=len(chunks),
                summary=summary_preview
            )
        except Exception as e:
            err_msg = f"Extraction failed: {str(e)}"
            update_document_status(
                doc_id,
                processing_status="Failed",
                extraction_status="Failed",
                error_message=err_msg
            )
            return {"success": False, "doc_id": doc_id, "error": err_msg}

        # 3. Step 3: Embeddings & Vector Indexing
        try:
            update_document_status(doc_id, processing_status="Generating Embeddings", embedding_status="In Progress")
            
            # Save chunks to SQLite mirror
            chunks_dicts = [c.to_dict() for c in chunks]
            insert_document_chunks(doc_id, chunks_dicts)
            update_document_status(doc_id, embedding_status="Completed", processing_status="Indexing", indexing_status="In Progress")

            # Index in ChromaDB
            if self.collection:
                # Remove any stale chunks for this doc first
                try:
                    self.collection.delete(where={"doc_id": doc_id})
                except Exception:
                    pass

                ids = [c.chunk_id for c in chunks]
                docs = [c.content for c in chunks]
                metas = [c.metadata for c in chunks]

                # ChromaDB batch add (max 500 per batch)
                batch_size = 300
                for i in range(0, len(ids), batch_size):
                    self.collection.add(
                        ids=ids[i:i + batch_size],
                        documents=docs[i:i + batch_size],
                        metadatas=metas[i:i + batch_size]
                    )

            update_document_status(
                doc_id,
                processing_status="Ready",
                indexing_status="Completed",
                error_message=None
            )
            return {
                "success": True,
                "doc_id": doc_id,
                "filename": filename,
                "file_type": file_type,
                "chunks": len(chunks),
                "chunks_count": len(chunks),
                "pages": page_count,
                "status": "Ready"
            }
        except Exception as e:
            err_msg = f"Indexing failed: {str(e)}"
            update_document_status(
                doc_id,
                processing_status="Failed",
                indexing_status="Failed",
                error_message=err_msg
            )
            return {"success": False, "doc_id": doc_id, "error": err_msg}

    def ingest_url(
        self,
        url: str,
        doc_id: Optional[str] = None,
        link_service=None,
        gemini_service=None
    ) -> Dict[str, Any]:
        """Ingests, cleans, chunks, and indexes a web link or YouTube URL."""
        doc_id = doc_id or f"url_{uuid.uuid4().hex[:10]}"
        create_document(
            doc_id=doc_id,
            filename=url,
            file_type="url",
            file_size_bytes=len(url),
            storage_path="",
            source_url=url
        )
        update_document_status(doc_id, processing_status="Extracting Content", extraction_status="In Progress")

        try:
            chunks, title, summary_preview = self.processor.process_url(
                url, doc_id, link_service=link_service, gemini_service=gemini_service
            )
            if not chunks:
                raise ValueError("Could not extract readable content from URL.")

            update_document_status(
                doc_id,
                extraction_status="Completed",
                page_count=1,
                chunk_count=len(chunks),
                summary=summary_preview
            )
            # Update title in SQLite
            from database.db import get_connection
            conn = get_connection()
            conn.cursor().execute("UPDATE documents SET filename = ? WHERE doc_id = ?", (title, doc_id))
            conn.commit()
            conn.close()

            # Embed & Index
            update_document_status(doc_id, processing_status="Generating Embeddings", embedding_status="In Progress")
            chunks_dicts = [c.to_dict() for c in chunks]
            insert_document_chunks(doc_id, chunks_dicts)
            update_document_status(doc_id, embedding_status="Completed", processing_status="Indexing", indexing_status="In Progress")

            if self.collection:
                try:
                    self.collection.delete(where={"doc_id": doc_id})
                except Exception:
                    pass

                self.collection.add(
                    ids=[c.chunk_id for c in chunks],
                    documents=[c.content for c in chunks],
                    metadatas=[c.metadata for c in chunks]
                )

            update_document_status(doc_id, processing_status="Ready", indexing_status="Completed", error_message=None)
            return {"success": True, "doc_id": doc_id, "title": title, "chunks": len(chunks), "chunks_count": len(chunks), "status": "Ready"}
        except Exception as e:
            err_msg = f"URL ingestion failed: {str(e)}"
            update_document_status(doc_id, processing_status="Failed", extraction_status="Failed", error_message=err_msg)
            return {"success": False, "doc_id": doc_id, "error": err_msg}

    def reprocess_document(self, doc_id: str, gemini_service=None, link_service=None) -> Dict[str, Any]:
        """Retries / reprocesses a failed or existing document."""
        doc = get_document(doc_id)
        if not doc:
            return {"success": False, "error": "Document not found."}

        if doc["file_type"] == "url" and doc.get("source_url"):
            return self.ingest_url(doc["source_url"], doc_id=doc_id, link_service=link_service, gemini_service=gemini_service)
        elif doc.get("storage_path") and os.path.exists(doc["storage_path"]):
            return self.ingest_file(doc["storage_path"], doc["filename"], doc_id=doc_id, gemini_service=gemini_service)
        else:
            return {"success": False, "error": "Source file is no longer available on disk."}

    def delete_document(self, doc_id: str) -> bool:
        """Deletes document from ChromaDB, SQLite, and persistent disk storage."""
        doc = get_document(doc_id)
        # 1. Delete from ChromaDB
        if self.collection:
            try:
                self.collection.delete(where={"doc_id": doc_id})
            except Exception as e:
                print(f"[RAG Engine] ChromaDB delete note: {e}")

        # 2. Delete physical file
        if doc and doc.get("storage_path") and os.path.exists(doc["storage_path"]):
            try:
                os.remove(doc["storage_path"])
            except Exception:
                pass

        # 3. Delete from SQLite
        return db_delete_document(doc_id)

    # -------------------------------------------------------------------------
    # Hybrid Retrieval (Semantic Vector + Lexical Keyword Search)
    # -------------------------------------------------------------------------
    def search(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search across indexed chunks.
        Supports filtering by specific doc_id or searching across all ready documents.
        """
        results: List[Dict[str, Any]] = []
        seen_contents = set()

        # 1. Vector Search via ChromaDB
        if self.collection:
            try:
                where_clause = {"doc_id": doc_id} if doc_id else None
                chroma_res = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k * 2, 10),
                    where=where_clause
                )
                docs = chroma_res.get("documents", [[]])[0]
                metas = chroma_res.get("metadatas", [[]])[0]
                distances = chroma_res.get("distances", [[]])[0] if "distances" in chroma_res else [0.5] * len(docs)

                for doc_text, meta, dist in zip(docs, metas, distances):
                    norm_content = doc_text.strip()
                    if norm_content and norm_content not in seen_contents:
                        seen_contents.add(norm_content)
                        sim_score = round(1.0 - float(dist), 3) if dist is not None else 0.8
                        results.append({
                            "content": norm_content,
                            "doc_id": meta.get("doc_id", doc_id or ""),
                            "filename": meta.get("filename", "Uploaded Document"),
                            "page_number": meta.get("page_number", 1),
                            "timestamp_str": meta.get("timestamp_str", ""),
                            "file_type": meta.get("file_type", "document"),
                            "score": sim_score,
                            "source_type": "vector"
                        })
            except Exception as e:
                print(f"[RAG Vector Search Note]: {e}")

        # 2. Lexical / Keyword Search Fallback & Augmentation
        if len(results) < top_k:
            lexical_chunks = search_chunks_lexical(query, doc_id=doc_id, limit=top_k)
            for c in lexical_chunks:
                norm_content = c["content"].strip()
                if norm_content and norm_content not in seen_contents:
                    seen_contents.add(norm_content)
                    results.append({
                        "content": norm_content,
                        "doc_id": c["doc_id"],
                        "filename": c.get("filename", "Document"),
                        "page_number": c.get("page_number", 1),
                        "timestamp_str": c.get("timestamp_str", ""),
                        "file_type": c.get("file_type", "document"),
                        "score": 0.85,
                        "source_type": "lexical"
                    })

        # Rank by score
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:top_k]

    # -------------------------------------------------------------------------
    # Grounded RAG Response Generation with Citations & Guardrails
    # -------------------------------------------------------------------------
    def answer_rag_query(
        self,
        query: str,
        doc_id: Optional[str] = None,
        gemini_service=None
    ) -> Dict[str, Any]:
        """
        Grounded RAG workflow:
        Query -> Hybrid Retrieval -> Relevance Filtering -> Context Construction -> Anti-Hallucination Guardrail -> Answer with Citations.
        """
        raw_chunks = self.search(query, doc_id=doc_id, top_k=4)

        # Relevance filtering: verify chunks are actually related to the query
        STOPWORDS = {
            "the", "and", "for", "are", "is", "was", "were", "this", "that", "what",
            "how", "with", "from", "have", "has", "had", "will", "your", "their",
            "there", "then", "them", "some", "more", "also", "into", "been", "only",
            "about", "which", "when", "where", "who", "whom", "why", "does", "did", "can"
        }
        q_meaningful = {
            t for t in re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower())
            if t not in STOPWORDS
        }

        matched_chunks = []
        for ch in raw_chunks:
            if ch.get("source_type") == "lexical":
                matched_chunks.append(ch)
            elif ch.get("score", 0) >= 0.35:
                # Strong vector similarity match
                matched_chunks.append(ch)
            elif ch.get("score", 0) >= 0.15 and q_meaningful:
                # Moderate vector match: check meaningful keyword overlap
                ch_tokens = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', ch["content"].lower()))
                if len(q_meaningful.intersection(ch_tokens)) >= 1:
                    matched_chunks.append(ch)

        # Anti-Hallucination Guardrail: If no relevant context matches
        if not matched_chunks:
            return {
                "answer": "I could not find this information in your uploaded documents.",
                "citations": [],
                "chunks_used": 0
            }

        # Build Citations & Grounded Context
        context_blocks = []
        citations = []

        for idx, ch in enumerate(matched_chunks, 1):
            doc_name = ch.get("filename", "Document")
            page_no = ch.get("page_number", 1)
            ts = ch.get("timestamp_str", "")
            loc_marker = f"Page {page_no}" if not ts else f"Timestamp [{ts}]"

            context_blocks.append(f"--- [Reference {idx}: {doc_name} ({loc_marker})] ---\n{ch['content']}")
            
            snippet = ch['content'][:140].replace('\n', ' ') + "..."
            citations.append({
                "source": doc_name,
                "location": loc_marker,
                "snippet": snippet
            })

        full_context = "\n\n".join(context_blocks)
        doc_scope_desc = f"within '{matched_chunks[0]['filename']}'" if doc_id else "across all uploaded documents"

        prompt = (
            f"You are Agent Rag, an expert document intelligence assistant analyzing documents {doc_scope_desc}.\n\n"
            f"USER QUESTION: {query}\n\n"
            f"RETRIEVED DOCUMENT CONTEXT:\n{full_context}\n\n"
            "STRICT INSTRUCTIONS:\n"
            "1. Directly and factually answer the user's question using ONLY the provided retrieved context.\n"
            "2. If the answer cannot be determined or inferred from the provided context, state clearly: 'I could not find this information in your uploaded documents.' Do not fabricate details.\n"
            "3. Cite the relevant source name and Page/Timestamp inline where appropriate (e.g. '[Source: filename, Page X]').\n"
            "4. Format the answer with clear, structured bullet points and bold highlights."
        )

        generated_answer = ""
        if gemini_service:
            generated_answer = gemini_service.generate_chat_response(prompt)

        if not generated_answer or "error" in generated_answer.lower():
            # Intelligent fallback synthesis
            top_content = matched_chunks[0]['content']
            first_point = top_content[:350].strip()
            generated_answer = (
                f"Based on the content retrieved from **{matched_chunks[0]['filename']}**:\n\n"
                f"> {first_point}\n\n"
                f"*(Referenced from {citations[0]['location']})*"
            )

        # Build Formatted Citations Block
        cit_md_lines = ["\n\n---", "### 📚 Verified Source References:"]
        for c in citations:
            cit_md_lines.append(f"- **Source:** `{c['source']}` | **{c['location']}**\n  > \"{c['snippet']}\"")

        final_response = generated_answer + "\n" + "\n".join(cit_md_lines)
        return {
            "answer": final_response,
            "citations": citations,
            "chunks_used": len(matched_chunks)
        }
