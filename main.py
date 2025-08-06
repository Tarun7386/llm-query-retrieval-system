from __future__ import annotations

import logging
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text           # <- new

from api.endpoints import router as core_router
from api.endpoints import hackrx_router
from config import settings
from database import engine, get_db
from models import database_models
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL)

# Create database tables
database_models.Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Intelligent Query-Retrieval System for Insurance & Legal Documents",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
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
    """Initialize services on startup"""
    try:
        # Test database connection
        db = next(get_db())
        db.execute(text("SELECT 1"))          # ← wrap with text()
        logger.info("✅ Database connection OK")

        # Initialize Pinecone
        vector_store = VectorStore()
        logger.info("✅ Pinecone vector store initialized")

        logger.info("🚀 Application startup complete")
    except Exception as e:
        logger.error("❌ Startup failed: %s", e)
        raise

@app.get("/", tags=["Meta"])
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "hackrx_endpoint": "/api/v1/hackrx/run"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
