# services/query_engine.py
from __future__ import annotations

import asyncio
import logging
import time
from typing import List

from sqlalchemy.orm import Session

from config import settings
from models.database_models import Document 
from models.pydantic_models import (
    HackRXRequest,
    HackRXResponse,
    QueryResult,
    ClauseMatch,
)
from services.clause_matcher import ClauseMatcher
from services.document_processor import DocumentProcessor
from utils.helpers import log_performance

logger = logging.getLogger(__name__)


class QueryEngine:
    """High-level orchestration for the /hackrx/run endpoint."""

    _ANS_CAP = 4_000          # cap answer chars when estimating tokens
    _Q_TIMEOUT = 120          # seconds per question

    def __init__(self) -> None:
        self._matcher = ClauseMatcher()
        self._processor = DocumentProcessor()
        self._doc_cache: dict[str, str] = {}  # url → document_id (per request)

    # ─────────────────────────── public ────────────────────────────
    async def run_hackrx(self, req: HackRXRequest, db: Session) -> HackRXResponse:
        """Download/ingest doc (once) then answer all questions."""
        t_start = time.perf_counter()

        # 1️⃣  ingest (download & chunk) – cache by URL for this request
        if req.documents not in self._doc_cache:
            doc = await self._processor.process_document_from_url(req.documents, db)
            self._doc_cache[req.documents] = doc.id
        else:
            # fetch document record from DB (quick)
            doc_id = self._doc_cache[req.documents]
            doc = db.query(Document).filter(Document.id == doc_id).first()

        doc_id = doc.id
        results: List[QueryResult] = []
        total_tokens = 0

        # 2️⃣  answer questions concurrently (but bounded)
        async def _solve(q: str) -> QueryResult:
            t0 = time.perf_counter()
            try:
                answer, conf, support = await asyncio.wait_for(
                    self._matcher.match(q, [doc_id], db),
                    timeout=self._Q_TIMEOUT,
                )
            except Exception as exc:                           # noqa
                logger.error("Q failed: %s → %s", q[:40], exc)
                answer, conf, support = "Error", 0.0, [{"explanation": str(exc)}]

            dt = time.perf_counter() - t0
            est_tok = (len(q) + len(answer[: self._ANS_CAP])) // 4

            return QueryResult(
                question=q,
                answer=answer,
                confidence_score=round(conf, 2),
                supporting_clauses=[
                    ClauseMatch(**c) for c in support if "clause_text" in c
                ],
                explanation=support[-1].get("explanation", "") if support else "",

                processing_time=dt,
            ), est_tok

        tasks = [_solve(q) for q in req.questions]
        for coro in asyncio.as_completed(tasks):
            qr, toks = await coro
            results.append(qr)
            total_tokens += toks

        total_dt = time.perf_counter() - t_start
        log_performance("hackrx.run", total_dt, questions=len(results))

        return HackRXResponse(
            document_id=doc.id,
            document_title=doc.original_filename,
            results=results,
            total_processing_time=total_dt,
            token_usage={"estimated": total_tokens},
        )
