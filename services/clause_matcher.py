# services/clause_matcher.py
from __future__ import annotations

import json
import logging
import time
from typing import List, Tuple

from sqlalchemy.orm import Session

from config import settings
from models.database_models import DocumentChunk
from services.gemini_service import GeminiService
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ClauseMatcher:
    """Locate, score and explain clauses that answer a question."""

    _JSON_KEYS = {"answer", "confidence", "explanation"}
    _MAX_CONTEXT_CHARS = 6_000                      # keep LLM prompt cost sane

    def __init__(self) -> None:
        self._vector_store = VectorStore()
        self._llm = GeminiService()

    # ──────────────────────────────────────────────────────────────
    async def match(
        self,
        question: str,
        document_ids: List[str],
        db: Session,
    ) -> Tuple[str, float, List[dict]]:
        """
        Returns
        -------
        answer_text : str
        confidence  : float  (0–1)
        supporting  : list[dict]  [{clause_text, similarity_score, page_reference, section_title}]
        """
        tic = time.perf_counter()

        # 1️⃣  Semantic search → top-K chunks
        top_chunks = await self._vector_store.search_similar_chunks(
            query=question,
            k=settings.MAX_RESULTS_PER_QUERY,
            similarity_threshold=settings.SIMILARITY_THRESHOLD,
            document_ids=document_ids,
            db=db,
        )

        if not top_chunks:
            return "No relevant information found.", 0.0, []

        # 2️⃣  Build LLM context (truncate if necessary)
        context_parts: List[str] = []
        supporting: List[dict] = []

        consumed_chars = 0
        for chunk, score in top_chunks:
            snippet = chunk.content[:1_000]                     # keep each clause short
            part = f"\n<CLAUSE score={score:.3f}>\n{snippet}\n</CLAUSE>"
            if consumed_chars + len(part) > self._MAX_CONTEXT_CHARS:
                break
            context_parts.append(part)
            consumed_chars += len(part)

            supporting.append(
                {
                    "clause_text": snippet,
                    "similarity_score": round(score, 3),
                    "page_reference": chunk.chunk_metadata.get("page_number"),
                    "section_title": chunk.chunk_metadata.get("section_title"),
                }
            )

        context = "".join(context_parts)

        # 3️⃣  Compose prompt
        prompt = f"""
You are an expert insurance-policy analyst.

Question:
{question}

Relevant policy clauses:
{context}

Tasks:
1. Give a concise answer (max 2 sentences).
2. Provide a confidence score between 0 and 1 (one decimal place).
3. Explain briefly which clauses justify the answer.

Respond strictly in JSON with keys: answer, confidence, explanation.
"""

        # 4️⃣  Call Gemini – retry once if rate-limited
        llm_rsp = await self._llm.generate_text(prompt, max_tokens=settings.EXPLANATION_MAX_TOKENS)

        # 5️⃣  Parse JSON safely
        try:
            data = json.loads(llm_rsp)
            if not self._JSON_KEYS.issubset(data):
                raise ValueError("Missing keys in LLM response")

            answer = str(data["answer"]).strip()
            confidence = float(data["confidence"])
            explanation = str(data["explanation"]).strip()
        except Exception as exc:
            logger.warning("LLM JSON parse failed (%s); falling back to raw text", exc)
            answer = llm_rsp.strip()[:1_000]
            confidence = 0.5
            explanation = "Explanation unavailable."

        toc = time.perf_counter()
        logger.info(
            "ClauseMatcher: '%s' processed in %.2fs (confidence %.2f)",
            question[:40],
            toc - tic,
            confidence,
        )

        explanation_item = {"explanation": explanation}
        if supporting:
            return answer, confidence, supporting + [explanation_item]
        else:
            return answer, confidence, [explanation_item]
