import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # ── Application ─────────────────────────────────────
    APP_NAME: str = "LLM-Powered Intelligent Query-Retrieval System"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Database ────────────────────────────────────────
    DATABASE_URL: str = "postgresql://postgres:mGJOjNUsNMgtwLOCYSWObWqIpsyvgTAo@shuttle.proxy.rlwy.net:15139/hacrxDB"

    # ── Authentication ──────────────────────────────────
    BEARER_TOKEN: str = "bf787797d87f5efa023148d772c502d020e57fe3e912a9528010a0f662b7dc2c"

    # ── Gemini AI ───────────────────────────────────────
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    VECTOR_DIMENSION: int = 768

    # ── Pinecone ────────────────────────────────────────
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "hackrx-query-system"
    USE_PINECONE: bool = True

    # ── Document Processing ─────────────────────────────
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB for large policies
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx", ".txt", ".eml"]
    CHUNK_SIZE: int = 1500                   # Larger for legal documents
    CHUNK_OVERLAP: int = 300                 # More overlap for context

    # ── Query Processing ────────────────────────────────
    MAX_RESULTS_PER_QUERY: int = 10
    SIMILARITY_THRESHOLD: float = 0.75       # Higher for accuracy
    EXPLANATION_MAX_TOKENS: int = 500        # Detailed explanations



    # ── Security ────────────────────────────────────────
    SECRET_KEY: str = "hackrx-super-secret-key-2025"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30  # <- This was missing
    
    # ── Logging ─────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
