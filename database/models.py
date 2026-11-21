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

# ---------------------------------------------------------------------------
# Document Management & Chunk Tracking DAL
# ---------------------------------------------------------------------------

def create_document(
    doc_id: str,
    filename: str,
    file_type: str,
    file_size_bytes: int = 0,
    storage_path: str = "",
    source_url: str = "",
    metadata_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Registers a newly uploaded document or URL in SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    meta_json = json.dumps(metadata_dict or {})
    
    cursor.execute("""
        INSERT OR REPLACE INTO documents (
            doc_id, filename, file_type, file_size_bytes, storage_path, 
            source_url, processing_status, extraction_status, embedding_status, 
            indexing_status, page_count, chunk_count, error_message, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, 'Uploaded', 'Pending', 'Pending', 'Pending', 0, 0, NULL, ?)
    """, (doc_id, filename, file_type, file_size_bytes, storage_path, source_url, meta_json))
    
    conn.commit()
    cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else {}

def update_document_status(
    doc_id: str,
    processing_status: Optional[str] = None,
    extraction_status: Optional[str] = None,
    embedding_status: Optional[str] = None,
    indexing_status: Optional[str] = None,
    page_count: Optional[int] = None,
    chunk_count: Optional[int] = None,
    error_message: Optional[str] = None,
    summary: Optional[str] = None
):
    """Updates the granular processing, extraction, embedding, and indexing status of a document."""
    conn = get_connection()
    cursor = conn.cursor()
    
    fields = []
    values = []
    
    if processing_status is not None:
        fields.append("processing_status = ?")
        values.append(processing_status)
    if extraction_status is not None:
        fields.append("extraction_status = ?")
        values.append(extraction_status)
    if embedding_status is not None:
        fields.append("embedding_status = ?")
        values.append(embedding_status)
    if indexing_status is not None:
        fields.append("indexing_status = ?")
        values.append(indexing_status)
    if page_count is not None:
        fields.append("page_count = ?")
        values.append(page_count)
    if chunk_count is not None:
        fields.append("chunk_count = ?")
        values.append(chunk_count)
    if error_message is not None:
        fields.append("error_message = ?")
        values.append(error_message)
    if summary is not None:
        fields.append("summary = ?")
        values.append(summary)

    if fields:
        values.append(doc_id)
        query = f"UPDATE documents SET {', '.join(fields)} WHERE doc_id = ?"
        cursor.execute(query, tuple(values))
        conn.commit()
    conn.close()

def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves document record by doc_id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def list_documents(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists registered documents, optionally filtered by processing status."""
    conn = get_connection()
    cursor = conn.cursor()
    if status:
        cursor.execute("SELECT * FROM documents WHERE processing_status = ? ORDER BY upload_timestamp DESC", (status,))
    else:
        cursor.execute("SELECT * FROM documents ORDER BY upload_timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_document(doc_id: str) -> bool:
    """Deletes a document and its chunks from SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,))
    cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return True

def insert_document_chunks(doc_id: str, chunks: List[Dict[str, Any]]):
    """Inserts a batch of extracted chunks into SQLite."""
    if not chunks:
        return
    conn = get_connection()
    cursor = conn.cursor()
    # Delete existing chunks for this doc first to prevent duplication
    cursor.execute("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,))
    
    rows = []
    for c in chunks:
        meta_str = json.dumps(c.get("metadata", {}))
        rows.append((
            c["chunk_id"],
            doc_id,
            c.get("chunk_index", 0),
            c.get("page_number", 1),
            c.get("timestamp_str", ""),
            c.get("content", ""),
            meta_str
        ))
        
    cursor.executemany("""
        INSERT INTO document_chunks (chunk_id, doc_id, chunk_index, page_number, timestamp_str, content, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()

def get_document_chunks(doc_id: str) -> List[Dict[str, Any]]:
    """Retrieves all chunks for a specific document."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM document_chunks WHERE doc_id = ? ORDER BY chunk_index ASC", (doc_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def search_chunks_lexical(query: str, doc_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    High-speed keyword / lexical search across document chunks.
    Provides robust fallback when vector embeddings are unavailable or offline.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    STOPWORDS = {
        "the", "and", "for", "are", "is", "was", "were", "this", "that", "what",
        "how", "with", "from", "have", "has", "had", "will", "your", "their",
        "there", "then", "them", "some", "more", "also", "into", "been", "only",
        "about", "which", "when", "where", "who", "whom", "why", "does", "did", "can"
    }
    tokens = [t.strip().lower() for t in query.split() if len(t.strip()) > 2 and t.strip().lower() not in STOPWORDS]
    if not tokens:
        tokens = [t.strip().lower() for t in query.split() if len(t.strip()) > 2]
    if not tokens:
        tokens = [query.strip().lower()]
        
    like_clauses = " OR ".join(["LOWER(content) LIKE ?" for _ in tokens])
    params = [f"%{t}%" for t in tokens]
    
    if doc_id:
        sql = f"""
            SELECT c.*, d.filename, d.file_type 
            FROM document_chunks c 
            JOIN documents d ON c.doc_id = d.doc_id 
            WHERE c.doc_id = ? AND ({like_clauses})
            LIMIT ?
        """
        full_params = [doc_id] + params + [limit]
    else:
        sql = f"""
            SELECT c.*, d.filename, d.file_type 
            FROM document_chunks c 
            JOIN documents d ON c.doc_id = d.doc_id 
            WHERE ({like_clauses})
            LIMIT ?
        """
        full_params = params + [limit]
        
    try:
        cursor.execute(sql, tuple(full_params))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        conn.close()
        return []

def get_system_stats() -> Dict[str, Any]:
    """Retrieves real-time system and repository statistics directly from SQLite and ChromaDB."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM documents")
    total_docs = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM documents WHERE processing_status = 'Ready'")
    ready_docs = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM documents WHERE processing_status NOT IN ('Ready', 'Failed')")
    proc_docs = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM documents WHERE processing_status = 'Failed'")
    failed_docs = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM document_chunks")
    total_chunks = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT file_type) FROM documents")
    source_types_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM chat_logs WHERE role = 'user'")
    total_queries = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM chat_logs WHERE route = 'MULTIMODAL_RAG'")
    rag_queries = cursor.fetchone()[0]
    
    conn.close()
    return {
        "total_documents": total_docs,
        "ready_documents": ready_docs,
        "processing_documents": proc_docs,
        "failed_documents": failed_docs,
        "total_chunks": total_chunks,
        "source_types_count": source_types_count,
        "total_queries": total_queries,
        "rag_queries": rag_queries
    }

def get_recent_chat_queries(user_id: str = "usr_default", limit: int = 8) -> List[str]:
    """Retrieves recent unique user queries for the sidebar chat history."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT content FROM chat_logs 
        WHERE user_id = ? AND role = 'user' 
        ORDER BY id DESC LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows if r[0]]

