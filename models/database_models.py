# models/database_models.py
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from database import Base
from config import settings


# Helper to pick JSONB for Postgres, JSON otherwise
JsonType = JSONB if settings.DATABASE_URL.startswith("postgresql") else JSON


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(256), nullable=False)
    original_filename = Column(String(256), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(128), nullable=False)
    document_type = Column(String(32), nullable=False)
    upload_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    processing_status = Column(String(32), default="pending")
    processing_error = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    document_metadata = Column(JsonType, default=dict)

    # ── Relationships ─────────────────────────────────────────────
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Document {self.id} ({self.original_filename})>"


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    embedding_vector = Column(JsonType, nullable=True)          # stored for analytics only
    chunk_metadata = Column(JsonType, default=dict)
    created_timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # ── Relationships ─────────────────────────────────────────────
    document = relationship("Document", back_populates="chunks")

    # Composite index speeds up retrieval (document_id, chunk_index)
    __table_args__ = (
        Index("ix_chunk_doc_idx", "document_id", "chunk_index"),
    )

    def __repr__(self) -> str:
        return f"<Chunk {self.id} doc={self.document_id} idx={self.chunk_index}>"


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query_text = Column(Text, nullable=False)
    query_type = Column(String(32), nullable=False)
    document_ids = Column(JsonType, nullable=True)
    results_count = Column(Integer, default=0)
    processing_time = Column(Float, nullable=False)
    similarity_threshold = Column(Float, nullable=False)
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    user_id = Column(String(64), nullable=True)  # reserved for future auth system

    def __repr__(self) -> str:
        preview = self.query_text.replace("\n", " ")[:40]
        return f"<QueryLog {self.id} '{preview}...'>"
