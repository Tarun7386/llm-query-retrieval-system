# services/vector_store.py
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from pinecone import Pinecone, ServerlessSpec
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from config import settings
from models.database_models import DocumentChunk
from services.gemini_service import GeminiService  # your async embedder

logger = logging.getLogger(__name__)

# ──────────────────────────  Pinecone client  ──────────────────────────
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

if settings.PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=settings.PINECONE_INDEX_NAME,
        dimension=settings.VECTOR_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    logger.info("✅ Created Pinecone index %s", settings.PINECONE_INDEX_NAME)

_INDEX = pc.Index(settings.PINECONE_INDEX_NAME)
_NAMESPACE = "default"  # single namespace simplifies everything


# ─────────────────────────  VectorStore class  ─────────────────────────
class VectorStore:
    """Thin wrapper around Pinecone v3 using Gemini embeddings."""

    def __init__(self) -> None:
        self._index = _INDEX
        self._ns = _NAMESPACE
        self._embedder = GeminiService()
        self._dim = settings.VECTOR_DIMENSION

    # ── internal helper ────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8))
    async def _embed(self, text: str) -> List[float]:
        vec = await self._embedder.get_embedding(text)
        if len(vec) != self._dim:
            raise ValueError(f"Expected {self._dim}-D embedding, got {len(vec)}")
        return vec

    # ── public API ─────────────────────────────────────────────────────
    async def add_document_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        db: Session,
        batch_size: int = 100,
    ) -> None:
        """
        Embed each chunk and upsert to Pinecone in batches.
        Also stores the embedding back into the chunk row.
        """
        batch: List[Dict] = []
        total = len(chunks)
        logger.info("⌛ Embedding %s chunks …", total)

        for i, chunk in enumerate(chunks, start=1):
            try:
                vec = await self._embed(chunk.content)
            except RetryError as exc:
                logger.error("❌ Embedding failed for chunk %s: %s", i - 1, exc)
                continue

            # Store in DB (optional analytics)
            chunk.embedding_vector = vec
            db.add(chunk)

            # Prepare vector for Pinecone
            batch.append(
                {
                    "id": str(chunk.id),
                    "values": vec,
                    "metadata": {
                        "document_id": str(chunk.document_id),
                        "chunk_index": chunk.chunk_index,
                        "chunk_text": chunk.content[:1_000],
                    },
                }
            )

            # Flush full batch
            if len(batch) >= batch_size:
                self._flush(batch)
                logger.info("   ↳ upserted %s/%s", i, total)
                batch.clear()

        # Flush remaining vectors
        if batch:
            self._flush(batch)
            logger.info("   ↳ upserted %s/%s", total, total)

        db.commit()
        logger.info("✅ Finished upserting %s vectors", total)

    def _flush(self, vectors: List[Dict]) -> None:
        """Send a batch to Pinecone."""
        if vectors:
            self._index.upsert(vectors=vectors, namespace=self._ns)

    # ───────────────────────────────────────────────────────────────────
    async def search_similar_chunks(
        self,
        query: str,
        k: int = 5,
        similarity_threshold: float = 0.45,
        document_ids: Optional[List[str]] = None,
        db: Optional[Session] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Return [(chunk, score), …] whose score ≥ similarity_threshold.
        """
        try:
            qvec = await self._embed(query)
        except RetryError as exc:
            logger.error("❌ Query embedding failed: %s", exc)
            return []

        resp = self._index.query(
            vector=qvec,
            top_k=k * 10,            # get plenty of candidates
            include_metadata=True,
            namespace=self._ns,
        )

        hits: List[Tuple[DocumentChunk, float]] = []
        if not db:
            return hits

        for match in resp.matches:
            if match.score < similarity_threshold:
                continue

            chunk = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.id == match.id)
                .first()
            )
            if not chunk:
                continue
            if document_ids and chunk.document_id not in document_ids:
                continue

            hits.append((chunk, match.score))
            if len(hits) >= k:
                break

        logger.info("🔎 Returned %s chunks ≥ %.2f", len(hits), similarity_threshold)
        return hits

    # ───────────────────────────────────────────────────────────────────
    def describe(self) -> Dict:
        """Simple helper for debugging."""
        s = self._index.describe_index_stats()
        return {
            "total_vectors": s.get("total_vector_count", 0),
            "dimension": self._dim,
            "namespaces": list(s.get("namespaces", {}).keys()),
        }
