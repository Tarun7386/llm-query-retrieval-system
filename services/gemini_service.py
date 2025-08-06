import asyncio
import logging
from typing import List

import google.generativeai as genai
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from config import settings

logger = logging.getLogger(__name__)

# Configure Gemini once
genai.configure(api_key=settings.GEMINI_API_KEY)

class GeminiService:
    _EMB_DIM = settings.VECTOR_DIMENSION

    def __init__(self) -> None:
        self._model = genai.GenerativeModel(settings.GEMINI_MODEL)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _embed_sync(self, text: str) -> List[float]:
        """Blocking call used inside run_in_executor."""
        rsp = genai.embed_content(
            model=settings.EMBEDDING_MODEL,  # Now uses models/text-embedding-004
            content=text,
            task_type="retrieval_document",
        )
        return rsp["embedding"]

    async def get_embedding(self, text: str) -> List[float]:
        """Async, resilient embedding fetch."""
        if not text.strip():
            return [0.0] * self._EMB_DIM

        try:
            emb = await asyncio.to_thread(self._embed_sync, text)
            if len(emb) != self._EMB_DIM:
                logger.warning(f"Expected {self._EMB_DIM}D embedding, got {len(emb)}D")
                # Pad or truncate to expected dimension
                if len(emb) > self._EMB_DIM:
                    emb = emb[:self._EMB_DIM]
                else:
                    emb.extend([0.0] * (self._EMB_DIM - len(emb)))
            return emb
        except (RetryError, Exception) as exc:
            logger.error("Gemini embedding failed: %s", exc)
            return [0.0] * self._EMB_DIM
    # ─────────────────────────── Completion ────────────────────────────
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _generate_sync(self, prompt: str, max_tokens: int) -> str:
        rsp = self._model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.3,
                top_p=0.95,
            ),
        )
        return rsp.text or ""

    async def generate_text(self, prompt: str, max_tokens: int = 150) -> str:
        """Async text generation with retries and timeout."""
        try:
            return await asyncio.to_thread(self._generate_sync, prompt, max_tokens)
        except (RetryError, Exception) as exc:
            logger.error("Gemini completion failed: %s", exc)
            return "Unable to generate response."
