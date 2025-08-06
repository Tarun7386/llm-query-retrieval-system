from __future__ import annotations

import logging
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from api.endpoints import router as core_router
from api.endpoints import hackrx_router
from config import settings
from database import engine, get_db
from models import database_models
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL)

# Vercel-specific optimizations
if os.getenv("VERCEL_ENV"):
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

# Create database tables (only in non-production environments)
try:
    if os.getenv("VERCEL_ENV") != "production":
        database_models.Base.metadata.create_all(bind=engine)
except Exception as e:
    logger.warning(f"Database table creation skipped: {e}")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Intelligent Query-Retrieval System for Insurance & Legal Documents",
    version=settings.VERSION,
    docs_url="/docs" if os.getenv("VERCEL_ENV") != "production" else None,
    redoc_url="/redoc" if os.getenv("VERCEL_ENV") != "production" else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(core_router, prefix="/api/v1")
app.include_router(hackrx_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup - Vercel optimized"""
    try:
        # Skip database connection in production serverless environment
        if os.getenv("VERCEL_ENV") != "production":
            db = next(get_db())
            db.execute(text("SELECT 1"))
            logger.info("✅ Database connection OK")
        
        # Initialize Pinecone (essential for the service)
        vector_store = VectorStore()
        logger.info("✅ Pinecone vector store initialized")
        
        logger.info("🚀 Application startup complete")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Don't crash in production - let Vercel handle gracefully
        if os.getenv("VERCEL_ENV") != "production":
            raise

@app.get("/", tags=["Meta"])
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "environment": os.getenv("VERCEL_ENV", "development"),
        "docs": "/docs" if os.getenv("VERCEL_ENV") != "production" else "disabled",
        "hackrx_endpoint": "/api/v1/hackrx/run"
    }

@app.get("/health", tags=["Meta"])
async def health_check():
    """Simple health check for Vercel"""
    return {
        "status": "healthy",
        "timestamp": "2025-01-06T17:30:00Z",
        "environment": os.getenv("VERCEL_ENV", "development")
    }

# This is important for Vercel
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=port,
        reload=False,  # Always disable reload in production
        log_level=settings.LOG_LEVEL.lower(),
    )
