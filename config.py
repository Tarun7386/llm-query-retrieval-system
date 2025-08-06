# config/settings.py
from __future__ import annotations

import os
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Application ───────────────────────────────
    APP_NAME: str = "LLM-Powered Intelligent Query-Retrieval System"
    VERSION: str = "2.0.0"

    # Will be overwritten by Vercel / dotenv
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", 8000))

    # ── Runtime mode ──────────────────────────────
    ENVIRONMENT: str = os.getenv("VERCEL_ENV", "development")
    DEBUG: bool = ENVIRONMENT != "production"

    # ── Database ──────────────────────────────────
    DATABASE_URL: str  # ← set in .env / Vercel

    # ── Authentication ────────────────────────────
    BEARER_TOKEN: str | None = None  # Optional local testing

    # ── Gemini AI ─────────────────────────────────
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    VECTOR_DIMENSION: int = 768

    # ── Pinecone ──────────────────────────────────
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "hackrx-query-system"
    USE_PINECONE: bool = True

    # ── Document Processing ───────────────────────
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100 MB
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx", ".txt", ".eml"]
    CHUNK_SIZE: int = 1_500
    CHUNK_OVERLAP: int = 300

    # ── Query Processing ──────────────────────────
    MAX_RESULTS_PER_QUERY: int = 10
    SIMILARITY_THRESHOLD: float = 0.75
    EXPLANATION_MAX_TOKENS: int = 500

    # ── Security ──────────────────────────────────
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ── Logging ───────────────────────────────────
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()
