"""
Unit and Integration Test Suite for Personal Agentic RAG Chatbot.
Covers all acceptance criteria:
1. Multi-modal extraction (PDF, DOCX, PPTX, Image, Audio, Video, URL)
2. Granular document status lifecycle (Uploading -> ... -> Ready / Failed)
3. ChromaDB vector indexing and hybrid retrieval
4. Page-accurate and timestamp citations
5. Anti-hallucination guardrail ("I could not find this information...")
6. Scoped vs Global search
7. Document deletion and cleanup across SQLite and ChromaDB
"""

import os
import sys
import unittest
import tempfile
import shutil

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.document_processor import DocumentProcessor
from core.rag_engine import MultimodalRAGEngine
from core.agentic_router import AgenticAssistant, AgentState
from database.db import init_db, get_connection
from database.models import (
    create_document,
    update_document_status,
    get_document,
    list_documents,
    delete_document,
    insert_document_chunks,
    get_document_chunks,
    search_chunks_lexical
)


class TestMultimodalRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.processor = DocumentProcessor()
        cls.engine = MultimodalRAGEngine()
        cls.sample_dir = os.path.join(PROJECT_ROOT, "sample_files")

    def test_01_pdf_processing_preserves_page_numbers(self):
        pdf_path = os.path.join(self.sample_dir, "Database_Security.pdf")
        self.assertTrue(os.path.exists(pdf_path), "Database_Security.pdf should exist")

        chunks, pages, summary = self.processor.process_pdf(pdf_path, "doc_pdf_test", "Database_Security.pdf")
        self.assertGreaterEqual(pages, 2, "PDF should have at least 2 pages")
        self.assertGreaterEqual(len(chunks), 2, "PDF should yield at least 2 chunks")
        
        # Verify page numbers
        page_nums = [c.page_number for c in chunks]
        self.assertIn(1, page_nums, "Chunk should map to Page 1")
        self.assertIn(2, page_nums, "Chunk should map to Page 2")
        
        # Verify text content
        p1_text = next(c.content for c in chunks if c.page_number == 1)
        self.assertIn("CIA Triad", p1_text, "Page 1 should mention CIA Triad")

    def test_02_docx_processing(self):
        docx_path = os.path.join(self.sample_dir, "Cloud_Security.docx")
        self.assertTrue(os.path.exists(docx_path), "Cloud_Security.docx should exist")

        chunks, pages, summary = self.processor.process_docx(docx_path, "doc_docx_test", "Cloud_Security.docx")
        self.assertGreaterEqual(len(chunks), 1, "DOCX should produce at least 1 chunk")
        self.assertIn("TLS 1.3", chunks[0].content, "DOCX chunk should contain TLS 1.3 guidelines")

    def test_03_pptx_processing(self):
        pptx_path = os.path.join(self.sample_dir, "Quantum_Overview.pptx")
        self.assertTrue(os.path.exists(pptx_path), "Quantum_Overview.pptx should exist")

        chunks, pages, summary = self.processor.process_pptx(pptx_path, "doc_pptx_test", "Quantum_Overview.pptx")
        self.assertGreaterEqual(len(chunks), 1, "PPTX should produce at least 1 chunk")
        self.assertIn("Quantum key distribution", chunks[0].content, "PPTX chunk should contain QKD text")

    def test_04_image_processing_and_searchable_representation(self):
        img_path = os.path.join(self.sample_dir, "architecture_diagram.png")
        self.assertTrue(os.path.exists(img_path), "architecture_diagram.png should exist")

        chunks, pages, summary = self.processor.process_image(img_path, "doc_img_test", "architecture_diagram.png")
        self.assertGreaterEqual(len(chunks), 1, "Image should produce searchable representation chunk")
        self.assertIn("architecture_diagram.png", chunks[0].metadata.get("filename", ""))

    def test_05_database_status_lifecycle_and_consistency(self):
        test_doc_id = "doc_lifecycle_test"
        
        # Step 1: Uploaded
        doc = create_document(test_doc_id, "Test_File.pdf", "pdf", 1024, "/tmp/test.pdf")
        self.assertEqual(doc["processing_status"], "Uploaded")
        self.assertEqual(doc["extraction_status"], "Pending")

        # Step 2: Processing -> Extracting Content
        update_document_status(test_doc_id, processing_status="Extracting Content", extraction_status="In Progress")
        doc_updated = get_document(test_doc_id)
        self.assertEqual(doc_updated["processing_status"], "Extracting Content")
        self.assertEqual(doc_updated["extraction_status"], "In Progress")

        # Step 3: Generating Embeddings -> Indexing
        update_document_status(
            test_doc_id,
            processing_status="Indexing",
            extraction_status="Completed",
            embedding_status="Completed",
            indexing_status="In Progress",
            page_count=5,
            chunk_count=10
        )
        doc_indexed = get_document(test_doc_id)
        self.assertEqual(doc_indexed["processing_status"], "Indexing")
        self.assertEqual(doc_indexed["page_count"], 5)
        self.assertEqual(doc_indexed["chunk_count"], 10)

        # Step 4: Ready
        update_document_status(test_doc_id, processing_status="Ready", indexing_status="Completed")
        doc_ready = get_document(test_doc_id)
        self.assertEqual(doc_ready["processing_status"], "Ready")

        # Cleanup
        delete_document(test_doc_id)
        self.assertIsNone(get_document(test_doc_id))

    def test_06_end_to_end_ingestion_and_hybrid_search(self):
        pdf_path = os.path.join(self.sample_dir, "Database_Security.pdf")
        res = self.engine.ingest_file(pdf_path, "Database_Security.pdf")
        
        self.assertEqual(res["status"], "Ready", "Ingested PDF must reach Ready status")
        self.assertGreater(res["chunks_count"], 0, "Ingested PDF must yield indexed chunks")
        doc_id = res["doc_id"]

        # Search for CIA Triad
        search_results = self.engine.search("What is the CIA Triad?", doc_id=doc_id)
        self.assertGreater(len(search_results), 0, "Search must return matching chunks")
        top_match = search_results[0]
        self.assertIn("CIA Triad", top_match["content"])
        self.assertEqual(top_match["page_number"], 1)

    def test_07_grounded_answer_and_exact_citations(self):
        answer_data = self.engine.answer_rag_query("Explain the CIA Triad in database security")
        answer_text = answer_data["answer"]
        citations = answer_data["citations"]

        self.assertGreater(len(citations), 0, "Must provide verified citations")
        self.assertIn("Database_Security.pdf", [c["source"] for c in citations])
        self.assertIn("Page 1", [c["location"] for c in citations])
        self.assertIn("CIA Triad", answer_text)

    def test_08_anti_hallucination_guardrail(self):
        # Ask something completely absent from all uploaded documents
        mars_query = "What is the weather on Mars?"
        answer_data = self.engine.answer_rag_query(mars_query)
        
        self.assertEqual(
            answer_data["answer"],
            "I could not find this information in your uploaded documents.",
            "Must trigger strict anti-hallucination guardrail when content is absent"
        )
        self.assertEqual(len(answer_data["citations"]), 0)

    def test_09_scoped_search_vs_global_search(self):
        # Ingest second document: Cloud_Security.docx
        docx_path = os.path.join(self.sample_dir, "Cloud_Security.docx")
        res_docx = self.engine.ingest_file(docx_path, "Cloud_Security.docx")
        docx_id = res_docx["doc_id"]

        # Global search: should find TLS 1.3 across all docs
        global_results = self.engine.search("TLS 1.3 encryption")
        self.assertTrue(any("Cloud_Security.docx" in r["filename"] for r in global_results))

        # Scoped search in Database_Security.pdf only: should NOT find TLS 1.3
        pdf_docs = [d for d in list_documents() if d["filename"] == "Database_Security.pdf"]
        if pdf_docs:
            scoped_results = self.engine.search("TLS 1.3 encryption", doc_id=pdf_docs[0]["doc_id"])
            matching_scoped = [r for r in scoped_results if "TLS 1.3" in r["content"]]
            self.assertEqual(len(matching_scoped), 0, "Scoped search must not return results from other documents")

    def test_10_document_deletion_removes_from_sqlite_and_chroma(self):
        # Ingest a temporary document to delete
        temp_txt = os.path.join(tempfile.gettempdir(), "deletable_doc.txt")
        with open(temp_txt, "w", encoding="utf-8") as f:
            f.write("Temporary content for deletion test. Secret token 98765.")

        res = self.engine.ingest_file(temp_txt, "deletable_doc.txt")
        del_doc_id = res["doc_id"]

        # Verify it exists in SQLite and Search
        self.assertIsNotNone(get_document(del_doc_id))
        pre_del_search = self.engine.search("Secret token 98765", doc_id=del_doc_id)
        self.assertGreater(len(pre_del_search), 0)

        # Execute Deletion
        del_success = self.engine.delete_document(del_doc_id)
        self.assertTrue(del_success)

        # Verify removed from SQLite
        self.assertIsNone(get_document(del_doc_id))
        self.assertEqual(len(get_document_chunks(del_doc_id)), 0)

        # Verify removed from ChromaDB / Hybrid Search
        post_del_search = self.engine.search("Secret token 98765", doc_id=del_doc_id)
        self.assertEqual(len(post_del_search), 0)

    def test_11_video_processing_and_timestamp_generation(self):
        # Create a small dummy video file with cv2
        import cv2
        import numpy as np
        temp_vid = os.path.join(tempfile.gettempdir(), "test_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_vid, fourcc, 5.0, (160, 120))
        for _ in range(15): # 3 seconds
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.putText(frame, "Test Video", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            out.write(frame)
        out.release()

        chunks, frames_extracted, summary = self.processor.process_video(temp_vid, "doc_vid_test", "test_video.mp4")
        self.assertGreater(len(chunks), 0, "Video processor should produce chunks")
        self.assertTrue(any("Timestamp" in c.content or c.timestamp_str for c in chunks), "Video chunks should have timestamps")

    def test_12_audio_chunking_and_metadata(self):
        # Test audio processor handling
        dummy_audio = os.path.join(tempfile.gettempdir(), "test_audio.wav")
        # Create minimal 1-second WAV
        import wave
        with wave.open(dummy_audio, 'w') as wav_f:
            wav_f.setnchannels(1)
            wav_f.setsampwidth(2)
            wav_f.setframerate(16000)
            wav_f.writeframes(b'\x00' * 32000)

        chunks, duration, summary = self.processor.process_audio(dummy_audio, "doc_aud_test", "test_audio.wav")
        self.assertGreater(len(chunks), 0, "Audio processing should produce chunks")
        self.assertEqual(chunks[0].metadata.get("file_type"), "audio")

    def test_13_web_url_clean_extraction(self):
        test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        chunks, title, summary = self.processor.process_url(test_url, "doc_url_test")
        self.assertGreater(len(chunks), 0, "URL ingestion should extract content chunks")
        self.assertTrue(any("intelligence" in c.content.lower() for c in chunks), "URL chunks should contain page topic")

    def test_14_lexical_keyword_search_fallback(self):
        # Verify direct lexical search works even if vector DB is not used
        lex_results = search_chunks_lexical("Confidentiality", limit=5)
        self.assertGreater(len(lex_results), 0, "Lexical search must find 'Confidentiality' in chunks")
        self.assertIn("Confidentiality", lex_results[0]["content"])

    def test_15_document_reprocessing_updates_records(self):
        docs = list_documents()
        ready_docs = [d for d in docs if d["processing_status"] == "Ready" and d["storage_path"] and os.path.exists(d["storage_path"])]
        if ready_docs:
            target_doc = ready_docs[0]
            reprocess_res = self.engine.reprocess_document(target_doc["doc_id"])
            self.assertTrue(reprocess_res.get("success"), "Reprocess should succeed for existing document")
            updated = get_document(target_doc["doc_id"])
            self.assertEqual(updated["processing_status"], "Ready")

    def test_16_retry_failed_document(self):
        # Create a document marked as Failed
        failed_id = "doc_failed_retry_test"
        dummy_file = os.path.join(tempfile.gettempdir(), "retry_test.txt")
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write("Retry content for testing recovery workflow.")

        create_document(failed_id, "retry_test.txt", "text", 100, dummy_file)
        update_document_status(failed_id, processing_status="Failed", error_message="Simulated failure")

        failed_doc = get_document(failed_id)
        self.assertEqual(failed_doc["processing_status"], "Failed")

        # Retry processing
        retry_res = self.engine.reprocess_document(failed_id)
        self.assertTrue(retry_res["success"], "Retry should successfully reprocess the document")
        recovered = get_document(failed_id)
        self.assertEqual(recovered["processing_status"], "Ready")
        
        # Cleanup
        self.engine.delete_document(failed_id)

    def test_17_batch_multi_file_ingestion(self):
        files_to_ingest = [
            ("batch_1.txt", "First document in batch processing test."),
            ("batch_2.txt", "Second document in batch processing test.")
        ]
        doc_ids = []
        for name, content in files_to_ingest:
            p = os.path.join(tempfile.gettempdir(), name)
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            res = self.engine.ingest_file(p, name)
            self.assertEqual(res["status"], "Ready")
            doc_ids.append(res["doc_id"])

        self.assertEqual(len(doc_ids), 2)
        for d_id in doc_ids:
            self.engine.delete_document(d_id)

    def test_18_document_metadata_and_chunk_integrity(self):
        docs = list_documents()
        for d in docs[:3]:
            self.assertIn(d["processing_status"], ["Ready", "Failed", "Processing", "Uploaded", "Indexing", "Extracting Content"])
            chunks = get_document_chunks(d["doc_id"])
            for c in chunks:
                self.assertIsNotNone(c["chunk_id"])
                self.assertIsNotNone(c["content"])
                self.assertGreaterEqual(c["page_number"], 1)

    def test_19_agentic_router_doc_qa_routing(self):
        assistant = AgenticAssistant()
        state: AgentState = {
            "user_id": "usr_default",
            "user_name": "Test User",
            "query": "What is the CIA Triad in database security?",
            "attached_file_path": None,
            "attached_file_type": None,
            "attached_link": None,
            "active_dataframe": None,
            "route": None,
            "response": None,
            "mode": "auto",
            "active_doc_id": None
        }
        res_state = assistant.process_turn(state)
        self.assertEqual(res_state["route"], "MULTIMODAL_RAG")
        self.assertIn("CIA Triad", res_state["response"])

    def test_20_agentic_router_scoped_doc_routing(self):
        assistant = AgenticAssistant()
        pdf_docs = [d for d in list_documents() if d["filename"] == "Database_Security.pdf"]
        target_doc_id = pdf_docs[0]["doc_id"] if pdf_docs else None

        state: AgentState = {
            "user_id": "usr_default",
            "user_name": "Test User",
            "query": "What are access control models?",
            "attached_file_path": None,
            "attached_file_type": None,
            "attached_link": None,
            "active_dataframe": None,
            "route": None,
            "response": None,
            "mode": "auto",
            "active_doc_id": target_doc_id
        }
        res_state = assistant.process_turn(state)
        self.assertEqual(res_state["route"], "MULTIMODAL_RAG")
        self.assertIn("Access Control", res_state["response"])


if __name__ == "__main__":
    unittest.main()
