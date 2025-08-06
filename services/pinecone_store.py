# services/vector_store.py  ← rename file everywhere to keep naming consistent
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pinecone
from sqlalchemy.orm import Session
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from config import settings
from models.database_models import DocumentChunk
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

# ─────────────────────────  Pinecone bootstrap ─────────────────────────
pinecone.init(
    api_key=settings.PINECONE_API_KEY,
    environment=settings.PINECONE_ENVIRONMENT,
)

if settings.PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=settings.PINECONE_INDEX_NAME,
        dimension=settings.VECTOR_DIMENSION,
        metric="cosine",
        pods=1,
        pod_type="p1.x1",
    )
    logger.info("✅ Created Pinecone index %s", settings.PINECONE_INDEX_NAME)

_INDEX = pinecone.Index(settings.PINECONE_INDEX_NAME)
_NAMESPACE = "default"


# ───────────────────────────  VectorStore  ─────────────────────────────
class VectorStore:
    """Async-friendly wrapper around Pinecone for the whole app."""

    def __init__(self) -> None:
        self._index = _INDEX
        self._ns = _NAMESPACE
        self._embedder = GeminiService()
        self._dim = settings.VECTOR_DIMENSION

    # ─────────────────  private helpers  ──────────────────
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8))
    async def _embed(self, text: str) -> List[float]:
        emb = await self._embedder.get_embedding(text)
        if len(emb) != self._dim:
            raise ValueError("Bad embedding dimension")
        return emb

    # ─────────────────  public API  ───────────────────────
    async def add_document_chunks(
        self,
        chunks: List[DocumentChunk],
        db: Session,
        batch_size: int = 100,
    ) -> None:
        """Embed & upsert all chunks. Stores vectors in Pinecone and embeddings in DB."""
        vectors: List[Dict] = []

        for chunk in chunks:
            emb = await self._embed(chunk.content)
            chunk.embedding_vector = emb  # optional but nice for analytics
            db.add(chunk)

            vectors.append(
                {
                    "id": str(chunk.id),
                    "values": emb,
                    "metadata": {
                        "document_id": str(chunk.document_id),
                        "chunk_index": chunk.chunk_index,
                    },
                }
            )

            if len(vectors) >= batch_size:
                self._index.upsert(vectors=vectors, namespace=self._ns)
                vectors.clear()

        if vectors:
            self._index.upsert(vectors=vectors, namespace=self._ns)

        db.commit()
        logger.info("🔼 Upserted %s chunks to Pinecone", len(chunks))

    async def search_similar_chunks(
        self,
        query: str,
        k: int = 5,
        similarity_threshold: float = 0.7,
        document_ids: Optional[List[str]] = None,
        db: Optional[Session] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Return (chunk, score) pairs filtered by threshold and doc-ids."""
        try:
            emb = await self._embed(query)
        except RetryError as exc:
            logger.error("Embedding failed: %s", exc)
            return []

        rsp = self._index.query(
            vector=emb,
            top_k=k * 3,  # ask for more, filter later
            include_metadata=True,
            namespace=self._ns,
        )

        results: List[Tuple[DocumentChunk, float]] = []
        if not db:
            return results  # we need DB to hydrate chunks

        for match in rsp.matches:
            score = match.score
            if score < similarity_threshold:
                continue

            chunk = db.query(DocumentChunk).filter(DocumentChunk.id == match.id).first()
            if not chunk:
                continue
            if document_ids and chunk.document_id not in document_ids:
                continue

            results.append((chunk, score))
            if len(results) >= k:
                break
        return results

    async def remove_document_chunks(self, document_id: str, db: Session) -> None:
        """Delete all vectors belonging to a document."""
        ids = [
            c.id
            for c in db.query(DocumentChunk.id).filter(
                DocumentChunk.document_id == document_id
            )
        ]
        if ids:
            self._index.delete(ids=ids, namespace=self._ns)
            logger.info("🗑️  Deleted %s vectors for doc %s", len(ids), document_id)

    def get_index_stats(self) -> Dict[str, int | str | List[str]]:
        """Quick stats used by /health and /stats endpoints."""
        stats = self._index.describe_index_stats()
        return {
            "total_vectors": stats.get("total_vector_count", 0),
            "dimension": self._dim,
            "index_type": "pinecone",
            "namespaces": list(stats.get("namespaces", {}).keys()),
        }
