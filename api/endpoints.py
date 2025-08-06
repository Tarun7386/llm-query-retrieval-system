# api/endpoints.py
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.auth import verify_token
from config import settings
from database import get_db
from models.database_models import DocumentChunk
from models.pydantic_models import (
    ClauseMatch,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    ErrorResponse,
    HackRXRequest,
    HackRXResponse,
    HackRXResult,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from services.adaptive_extractor import AdaptiveAnswerExtractor
from services.document_processor import DocumentProcessor
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ──────────────────────────  service singletons ─────────────────────────
processor = DocumentProcessor()
vectors = VectorStore()
extractor = AdaptiveAnswerExtractor()

# ─────────────────────────────── routers ───────────────────────────────
router = APIRouter()
hackrx_router = APIRouter(prefix="/hackrx", tags=["HackRX"])
router.include_router(hackrx_router)

# ────────────────────────────── health   ──────────────────────────────
@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        stats = vectors.describe()
        return HealthResponse(
            status="healthy",
            service="LLM Query-Retrieval System",
            version="1.0.0",
            timestamp=datetime.now(),
            database_status="healthy",
            vector_store_status="healthy" if stats["total_vectors"] >= 0 else "unhealthy",
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Health check failed: %s", exc)
        return HealthResponse(
            status="unhealthy",
            service="LLM Query-Retrieval System",
            version="1.0.0",
            timestamp=datetime.now(),
            database_status="unhealthy",
            vector_store_status="unknown",
        )

# ─────────────────────────── Generic cleanup function ──────────────────
def _clean_answer(raw_answer: str) -> str:
    """Enhanced answer cleanup that produces human-readable responses."""
    
    if not raw_answer or raw_answer.strip() == "":
        return "No relevant information found."
    
    # Aggressive cleanup
    answer = raw_answer.strip()
    
    # Remove document artifacts
    answer = re.sub(r'--- Page \d+ ---', '', answer)
    answer = re.sub(r'Page \d+ of \d+', '', answer)
    answer = re.sub(r'Cbd -\d+.*?Policy', '', answer)  # Remove address headers
    answer = re.sub(r'Uin.*?\d+', '', answer)  # Remove UIN codes
    answer = re.sub(r'[A-Z]{2,}\s*-\s*[A-Z0-9]+', '', answer)  # Remove reference codes
    
    # Remove list markers and incomplete sentences
    answer = re.sub(r'^[a-z]\)\s*', '', answer)  # Remove (a), (b), etc.
    answer = re.sub(r'^\d+\.\s*', '', answer)   # Remove 1., 2., etc.
    answer = re.sub(r'^[A-Z]\)\s*', '', answer) # Remove A), B), etc.
    
    # Clean up spacing and formatting
    answer = re.sub(r'\s+', ' ', answer)
    answer = answer.strip()
    
    # Extract meaningful content
    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
    
    if sentences:
        # Take the most complete sentence
        best_sentence = max(sentences, key=len)
        
        # Ensure proper capitalization
        if best_sentence and best_sentence[0].islower():
            best_sentence = best_sentence[0].upper() + best_sentence[1:]
        
        # Ensure proper ending
        if not best_sentence.endswith(('.', '!', '?')):
            best_sentence += '.'
            
        return best_sentence
    
    return "Information found but requires manual review."

    """Generic answer cleanup that works for any document type."""
    
    if not raw_answer or raw_answer.strip() == "":
        return "No relevant information found."
    
    # Basic cleanup
    answer = raw_answer.strip()
    answer = re.sub(r'--- Page \d+ ---', '', answer)  # Remove page markers
    answer = re.sub(r'\s+', ' ', answer)  # Normalize whitespace
    answer = answer.strip()
    
    # Ensure proper sentence structure
    if answer and not answer.endswith(('.', '!', '?')):
        answer += '.'
    
    # Capitalize first letter if it's lowercase
    if answer and answer[0].islower():
        answer = answer[0].upper() + answer[1:]
    
    return answer

# ─────────────────────────── HackRX endpoint  ──────────────────────────
@hackrx_router.post("/run")
async def run_hackrx(req: HackRXRequest, db: Session = Depends(get_db)):
    """
    • Downloads any document type (PDF/DOCX/TXT/EML)
    • Splits it → embeds → upserts to Pinecone  
    • Answers questions using adaptive extraction that works across document types
    • Returns simple answers array
    """
    t0 = time.perf_counter()
    logger.info("Processing HackRX request with %s questions", len(req.questions))

    # ── validate input ──────────────────────────────────────────────────
    if not req.documents.lower().startswith(("http://", "https://")):
        raise HTTPException(400, "Document URL must be http/https")
    if len(req.questions) > 20:
        raise HTTPException(400, "Maximum 20 questions allowed")

    # ── 1. Process the document (any type) ──────────────────────────────
    document = await processor.process_document_from_url(req.documents, db)

    # fetch the chunks just created
    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )

    # ── 2. Embed & upsert chunks to Pinecone ────────────────────────────
    await vectors.add_document_chunks(chunks, db)

    # ── 3. Analyze document for adaptive extraction ─────────────────────
    chunk_contents = [chunk.content for chunk in chunks]
    extractor.analyze_document(chunk_contents, document.id)

    # ── 4. Answer each question with adaptive extraction ────────────────
    answers = []
    # In your run_hackrx function, replace the answer extraction section:

    for q in req.questions:
        hits = await vectors.search_similar_chunks(
            query=q,
            k=5,
            similarity_threshold=0.45,
            document_ids=[document.id],
            db=db,
        )

        if not hits:
            answers.append("No relevant information found.")
            continue

    # Try multiple chunks for better answers
        best_answer = "No relevant information found."
        best_score = 0
    
        for chunk, score in hits[:3]:  # Try top 3 chunks
        # Clean the chunk content first
            clean_content = re.sub(r'--- Page \d+ ---.*?(?=\w)', '', chunk.content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        
        # Extract meaningful sentences
            sentences = [s.strip() for s in clean_content.split('.') if len(s.strip()) > 20]
        
        # Find the best sentence for this question
            question_words = set(q.lower().split())
        
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(question_words.intersection(sentence_words))
            
            # Sentence quality score
                quality_score = overlap * score
            
            # Bonus for complete information
                if any(word in sentence.lower() for word in ['days', 'months', 'years', 'covered', 'benefit']):
                    quality_score += 0.1
            
                if quality_score > best_score and len(sentence) > 30:
                    best_score = quality_score
                    best_answer = sentence.strip() + "."
    
    # Final cleanup
    clean_answer = _clean_answer(best_answer)
    answers.append(clean_answer)

    

    # ── 5. Wrap up ──────────────────────────────────────────────────────
    total_time = time.perf_counter() - t0
    logger.info("HackRX processing completed in %.2fs", total_time)

    # Return simple answers array that adapts to any document
    return {"answers": answers}
