# models/pydantic_models.py
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, validator


# ──────────────────────────── enums ──────────────────────────────
class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    EMAIL = "email"


class QueryType(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


# ─────────────────────── upload / doc models ─────────────────────
class DocumentUploadRequest(BaseModel):
    filename: str
    content_type: str
    metadata: Optional[Dict[str, Any]] = {}


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str
    processing_time: float


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    document_type: DocumentType
    file_size: int
    upload_timestamp: datetime
    processing_status: str
    chunk_count: int
    metadata: Dict[str, Any]


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_count: int


# ─────────────────────────── health/error ───────────────────────
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: datetime
    database_status: str
    vector_store_status: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None


# ─────────────────────────── query models ────────────────────────
class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1_000)
    document_ids: Optional[List[str]] = None
    query_type: QueryType = QueryType.SEMANTIC
    max_results: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    include_metadata: bool = True


class QueryResponse(BaseModel):
    query: str
    results: List[DocumentChunk]
    total_results: int
    processing_time: float
    query_type: str
    explanation: Optional[str] = None


# ─────────────────────────── HackRX models ───────────────────────
class HackRXRequest(BaseModel):
    documents: str = Field(
        ...,
        description="HTTPS URL pointing to a PDF/DOCX/TXT/EML document",
    )
    questions: List[str] = Field(
        ...,
        min_items=1,
        max_items=20,
        description="Batch of questions (max 20) to answer against the document",
    )

    @validator("documents")
    def _validate_url(cls, v: str) -> str:  # noqa: N805
        p = urlparse(v)
        if p.scheme not in {"http", "https"} or not p.netloc:
            raise ValueError("documents must be an absolute HTTP/HTTPS URL")
        return v

    @validator("questions", each_item=True)
    def _strip_q(cls, q: str) -> str:  # noqa: N805
        q = q.strip()
        if not q:
            raise ValueError("question text cannot be empty")
        return q


class ClauseMatch(BaseModel):
    clause_text: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    page_reference: Optional[str] = None
    section_title: Optional[str] = None


class QueryResult(BaseModel):  # generic result model
    question: str
    answer: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    supporting_clauses: List[ClauseMatch]
    explanation: str
    processing_time: float = Field(..., ge=0.0)


# Alias for convenience in endpoints
HackRXResult = QueryResult


class HackRXResponse(BaseModel):
    document_id: str
    document_title: str
    results: List[QueryResult]  # or List[HackRXResult]
    total_processing_time: float = Field(..., ge=0.0)
    token_usage: Dict[str, int] = Field(
        ...,
        description="Rough token counts (e.g. {'estimated': 1234})",
    )
    status: Literal["completed"] = "completed"
